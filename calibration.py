"""
calibration.py — Geriye dönük test ve kural kalibrasyonu
========================================================

Bir sigortacının GEÇMİŞ hasar kararlarını (uzman/eksper kararı) platformun karar motoruyla
yeniden oynatır; uyum, kritik hata ve finansal sapmayı ölçer; eşikleri geçmişe uyacak şekilde
önerir. Böylece "kurallarınız geçmiş kararlarınızı ne kadar yeniden üretiyor?" sorusu, sektörde
çoğu rakibin göstermediği biçimde ölçülebilir kanıtla yanıtlanır.

Tasarım ilkeleri
----------------
* Karar mantığı tek yerde durur: ``data_engine.InsuranceSimulator.decide_codes``. Burada
  yeniden yazılmaz; test edilen kural ile sahada uygulanan kural ayrışamaz.
* Eşikler yalnızca EĞİTİM bölümünde aranır; sonuç, hiç görülmemiş TEST bölümünde raporlanır
  (aşırı uyumu gizlememek için). Uyum oranı için Wilson %95 güven aralığı verilir.
* Hata iki türe ayrılır. KRİTİK hata: platform ödeme kararı verirken uzman REDDETMİŞ
  (haksız ödeme riski) ya da platform REDDEDERKEN uzman ÖDEMİŞ (haksız ret). EKSPERTİZ'e
  düşmek hata değil, otomasyon kaybıdır; ayrı ölçülür.
* Arama hedefi: kritik hata oranı ``critical_limit``'in altında kalırken tam uyumu azami
  yapmak; eşit sonuçlarda daha yüksek otomasyon, sonra mevcut kurala en yakın eşik seçilir.
* Sentetik örnek geçmiş (``sample_history``) yalnızca mekanizmayı göstermek içindir; gerçek
  bir sigortacı verisiyle çalıştırılana dek doğruluk iddiası TAŞIMAZ.

Bu modül Streamlit'e bağlı değildir; yalnızca numpy ve pandas kullanır.

Geliştirici: Elif Turut
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import data_engine as de
from data_engine import DECISION_LABELS, ClaimRules

# --------------------------------------------------------------------------- #
# Sabitler
# --------------------------------------------------------------------------- #
#: Kalibre edilen eşiklerin izin verilen aralığı (arayüzdeki kaydırıcılarla aynıdır).
BOUNDS: Dict[str, Tuple[float, float]] = {
    "approve_dnbr": (0.30, 0.70),
    "partial_dnbr": (0.15, 0.43),
    "review_dnbr": (0.02, 0.20),
    "min_confidence": (0.50, 0.99),
}
PARAM_LABELS: Dict[str, str] = {
    "approve_dnbr": "Tam onay eşiği (dNBR ≥)",
    "partial_dnbr": "Kısmi onay eşiği (dNBR ≥)",
    "review_dnbr": "Ret eşiği (dNBR <)",
    "min_confidence": "Min. gözlem güveni",
}
REQUIRED_COLUMNS: Tuple[str, ...] = ("dnbr", "guven", "gercek_karar")
OPTIONAL_COLUMNS: Tuple[str, ...] = ("police_aktif", "sigorta_bedeli_tl", "gercek_odeme_tl")
MIN_ROWS_CALIBRATE: int = 30
SMALL_SAMPLE_ROWS: int = 100
MAX_ROWS: int = 200_000
MAX_BYTES: int = 10_000_000
SEARCH_ROW_CAP: int = 8_000  # eşik araması için eğitim satırı üst sınırı (aşılırsa deterministik örnekleme)
_RET, _EXPERT, _PARTIAL, _APPROVE = 0, 1, 2, 3
_GAP = 0.01  # eşikler arası en küçük fark (arayüzdeki otomatik düzeltmeyle aynı)
SIGNIFICANCE: float = 0.10  # iyileşmenin "belirgin" sayılması için işaret testi p eşiği

_TR_FOLD = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
_COLUMN_ALIASES: Dict[str, str] = {
    "dnbr": "dnbr", "d_nbr": "dnbr",
    "guven": "guven", "guven_skoru": "guven", "confidence": "guven",
    "gercek_karar": "gercek_karar", "karar": "gercek_karar", "uzman_karari": "gercek_karar",
    "eksper_karari": "gercek_karar", "adjuster_decision": "gercek_karar",
    "police_aktif": "police_aktif", "aktif": "police_aktif",
    "sigorta_bedeli_tl": "sigorta_bedeli_tl", "sigorta_bedeli": "sigorta_bedeli_tl",
    "gercek_odeme_tl": "gercek_odeme_tl", "odenen_tl": "gercek_odeme_tl", "odeme_tl": "gercek_odeme_tl",
}
_DECISION_ALIASES: Dict[str, str] = {
    "ONAY": "ONAY", "TAM ONAY": "ONAY", "ONAYLANDI": "ONAY",
    "KISMI ONAY": "KISMİ ONAY", "KISMI": "KISMİ ONAY", "AVANS": "KISMİ ONAY",
    "EKSPERTIZ": "EKSPERTİZ", "EKSPER": "EKSPERTİZ", "KESIF": "EKSPERTİZ",
    "RET": "RET", "RED": "RET", "REDDEDILDI": "RET",
}
_TRUE_WORDS = frozenset({"true", "1", "evet", "e", "aktif", "yes", "y"})
_FALSE_WORDS = frozenset({"false", "0", "hayir", "h", "pasif", "no", "n"})


class CalibrationError(ValueError):
    """Geçmiş veri okunamadığında ya da kalibrasyon için yetersiz olduğunda fırlatılır."""


# --------------------------------------------------------------------------- #
# Veri sınıfları
# --------------------------------------------------------------------------- #
@dataclass
class HistoryTable:
    """Doğrulanmış geçmiş karar tablosu.

    Attributes:
        frame: Temiz tablo. Sütunlar: ``dnbr``, ``guven``, ``police_aktif``,
            ``sigorta_bedeli_tl``, ``gercek_karar``, ``gercek_kod`` (0=RET … 3=ONAY),
            ``gercek_odeme_tl`` (yoksa NaN).
        warnings: Kullanıcıya gösterilecek uyarılar (atılan satırlar, varsayımlar).
        dropped: Geçersiz olduğu için atılan satır sayısı.
        has_financials: Finansal karşılaştırma için bedel ve ödeme sütunları var mı.
    """

    frame: pd.DataFrame
    warnings: List[str] = field(default_factory=list)
    dropped: int = 0
    has_financials: bool = False


@dataclass(frozen=True)
class Metrics:
    """Bir kural setinin geçmiş kararlara karşı ölçümleri.

    Attributes:
        n: Kayıt sayısı.
        agreement: Tam uyum oranı (0–1): platform kararı = uzman kararı.
        agreement_low: Uyum oranı Wilson %95 alt sınırı.
        agreement_high: Uyum oranı Wilson %95 üst sınırı.
        false_approval: Haksız ödeme riski adedi (platform öder, uzman reddetmiş).
        wrongful_denial: Haksız ret adedi (platform reddeder, uzman ödemiş).
        critical_rate: Kritik hata oranı (0–1).
        expert_share: Platformun EKSPERTİZ'e bıraktığı oran (0–1).
        confusion: ``4×4`` karışıklık matrisi; satır = uzman, sütun = platform
            (sıra: ``DECISION_LABELS``).
        precision: Sınıf bazlı kesinlik (platform sütunu bazında; boşsa NaN).
        recall: Sınıf bazlı duyarlılık (uzman satırı bazında; boşsa NaN).
    """

    n: int
    agreement: float
    agreement_low: float
    agreement_high: float
    false_approval: int
    wrongful_denial: int
    critical_rate: float
    expert_share: float
    confusion: np.ndarray
    precision: Tuple[float, ...]
    recall: Tuple[float, ...]

    @property
    def automation(self) -> float:
        """Otomatik karar oranı (EKSPERTİZ dışı), 0–1."""
        return 1.0 - self.expert_share

    def confusion_frame(self) -> pd.DataFrame:
        """Karışıklık matrisini etiketli tablo olarak verir (satır: uzman, sütun: platform)."""
        return pd.DataFrame(self.confusion, index=list(DECISION_LABELS), columns=list(DECISION_LABELS))


@dataclass(frozen=True)
class FinancialImpact:
    """Test bölümünde platform ödemeleri ile uzman ödemelerinin toplamı (TL).

    Attributes:
        rows: Hesaba giren kayıt sayısı.
        human_tl: Uzmanların ödediği toplam.
        base_paid_tl: Mevcut kurallarla otomatik ödenecek toplam.
        base_pending_tl: Mevcut kurallarla ekspertize bırakılan/bekleyen toplam.
        tuned_paid_tl: Önerilen kurallarla otomatik ödenecek toplam.
        tuned_pending_tl: Önerilen kurallarla bekleyen toplam.
    """

    rows: int
    human_tl: float
    base_paid_tl: float
    base_pending_tl: float
    tuned_paid_tl: float
    tuned_pending_tl: float

    @property
    def base_gap_tl(self) -> float:
        """Mevcut kurallarda (ödenecek + bekleyen) − uzman toplamı."""
        return self.base_paid_tl + self.base_pending_tl - self.human_tl

    @property
    def tuned_gap_tl(self) -> float:
        """Önerilen kurallarda (ödenecek + bekleyen) − uzman toplamı."""
        return self.tuned_paid_tl + self.tuned_pending_tl - self.human_tl


@dataclass
class CalibrationResult:
    """Kalibrasyonun tüm çıktıları.

    Attributes:
        base_rules: Başlangıç (mevcut) kurallar.
        tuned_rules: Önerilen kurallar (yalnızca dört eşik değişir).
        n_train: Eğitim bölümü satır sayısı.
        n_test: Test bölümü satır sayısı.
        critical_limit: İstenen azami kritik hata oranı (0–1).
        feasible: Eğitim bölümünde kritik hata sınırı sağlanabildi mi.
        train_before: Mevcut kurallar, eğitim bölümü.
        train_after: Önerilen kurallar, eğitim bölümü.
        test_before: Mevcut kurallar, test bölümü.
        test_after: Önerilen kurallar, test bölümü.
        changes: Parametre değişiklik tablosu.
        sensitivity: Parametre → (deger, uyum_pct, kritik_pct) eğrisi (eğitim bölümü).
        financial: Test bölümü finansal etkisi (veri yoksa ``None``).
        test_wins: Test bölümünde yalnızca önerilen kuralın doğru bildiği kayıt sayısı.
        test_losses: Test bölümünde yalnızca mevcut kuralın doğru bildiği kayıt sayısı.
        p_value: Bu farkın işaret testi p değeri (iki yönlü).
        significant: Fark istatistiksel olarak belirgin mi (``p < SIGNIFICANCE``).
        recommended: Önerinin uygulanması tavsiye edilir mi (test bölümünde kazanç > kayıp ve kritik hata sınırda).
        notes: Okuyucuya yönelik dürüstlük notları.
    """

    base_rules: ClaimRules
    tuned_rules: ClaimRules
    n_train: int
    n_test: int
    critical_limit: float
    feasible: bool
    train_before: Metrics
    train_after: Metrics
    test_before: Metrics
    test_after: Metrics
    changes: pd.DataFrame
    sensitivity: Dict[str, pd.DataFrame]
    financial: Optional[FinancialImpact]
    test_wins: int
    test_losses: int
    p_value: float
    significant: bool
    recommended: bool
    notes: List[str]


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def consistent_thresholds(approve: float, partial: float, review: float) -> Tuple[float, float, float, bool]:
    """Eşikleri ``ret < kısmi onay < tam onay`` olacak biçimde düzeltir.

    Args:
        approve: Tam onay eşiği.
        partial: Kısmi onay eşiği.
        review: Ret eşiği.

    Returns:
        ``(approve, partial, review, düzeltildi_mi)``.
    """
    partial_eff = min(partial, round(approve - _GAP, 2))
    review_eff = min(review, round(partial_eff - _GAP, 2))
    return approve, partial_eff, review_eff, (partial_eff, review_eff) != (partial, review)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Oran için Wilson güven aralığı (küçük örneklemde normal yaklaşımdan güvenilirdir).

    Args:
        successes: Başarı sayısı.
        total: Toplam gözlem.
        z: Standart normal kritik değer (varsayılan %95).

    Returns:
        ``(alt, üst)``; ``total == 0`` ise ``(0.0, 1.0)``.
    """
    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _normal_sign_p(wins: int, losses: int) -> float:
    """Büyük örneklem için işaret testinin normal yaklaşımı (süreklilik düzeltmeli, iki yönlü)."""
    n = wins + losses
    if n == 0:
        return 1.0
    z = max(abs(wins - losses) - 1.0, 0.0) / math.sqrt(n)
    return float(math.erfc(z / math.sqrt(2.0)))


def sign_test_p(wins: int, losses: int) -> float:
    """Eşleştirilmiş kayıtlar için iki yönlü işaret testi p değeri (McNemar'ın kesin biçimi).

    ``wins``: yalnızca yeni kuralın doğru bildiği kayıtlar; ``losses``: yalnızca eski kuralın
    doğru bildiği kayıtlar. İkisinin de doğru ya da yanlış bildiği kayıtlar teste girmez.
    ``n ≤ 200`` için kesin iki terimli toplam, üstü için normal yaklaşım kullanılır.

    Args:
        wins: Yeni kuralın kazandığı kayıt sayısı.
        losses: Yeni kuralın kaybettiği kayıt sayısı.

    Returns:
        0–1 arası p değeri (fark yoksa 1).
    """
    n = wins + losses
    if n == 0:
        return 1.0
    if n > 200:
        return _normal_sign_p(wins, losses)
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / 2.0 ** n)


def _fold_name(name: object) -> str:
    """Sütun adını Türkçe karakter ve ayraçlardan arındırıp ``snake_case`` yapar."""
    text = str(name).strip().translate(_TR_FOLD).lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _normalise_decision(value: object) -> Optional[str]:
    """Serbest yazılmış kararı ``DECISION_LABELS`` değerlerinden birine çevirir (bilinmiyorsa ``None``)."""
    text = str(value).strip().translate(_TR_FOLD).upper()
    text = re.sub(r"[\s_\-]+", " ", text).strip()
    return _DECISION_ALIASES.get(text)


def _parse_number(value: object) -> float:
    """TR/EN biçimli sayıyı çözer (``0,52`` ve ``0.52``; ``1.234,5`` ve ``1,234.5``). Çözülemezse NaN.

    Tek noktalı değerler (``1.234``) ondalık sayılır; binlik ayırıcıyla yazılmış tek noktalı
    tutarlar için şablonda binlik ayırıcı kullanılmaması istenir.
    """
    text = str(value).strip().replace(" ", "").replace("₺", "")
    if not text:
        return math.nan
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return math.nan
    return number if math.isfinite(number) else math.nan


def _parse_bool(value: object) -> Optional[bool]:
    """``evet/hayır/aktif/pasif/true/false/1/0`` değerlerini mantıksala çevirir (bilinmiyorsa ``None``)."""
    text = str(value).strip().translate(_TR_FOLD).lower()
    if text in _TRUE_WORDS:
        return True
    if text in _FALSE_WORDS:
        return False
    return None


def _decode(raw: Union[bytes, str]) -> str:
    """Bayt dizisini UTF-8 (BOM'lu olabilir), olmazsa Windows-1254 ile çözer."""
    if isinstance(raw, str):
        return raw
    if len(raw) > MAX_BYTES:
        raise CalibrationError(f"Dosya çok büyük ({len(raw) / 1e6:.1f} MB); sınır {MAX_BYTES // 1_000_000} MB.")
    for encoding in ("utf-8-sig", "cp1254"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _sniff_separator(text: str) -> str:
    """Başlık satırına bakarak ``;``, ``,`` ya da sekme ayracını seçer."""
    header = text.lstrip("﻿").splitlines()[0] if text.strip() else ""
    counts = {sep: header.count(sep) for sep in (";", ",", "\t")}
    best = max(counts, key=lambda s: counts[s])
    return best if counts[best] > 0 else ";"


# --------------------------------------------------------------------------- #
# Okuma ve doğrulama
# --------------------------------------------------------------------------- #
def parse_history(raw: Union[bytes, str]) -> HistoryTable:
    """CSV içeriğini okuyup doğrulanmış geçmiş karar tablosuna çevirir.

    Zorunlu sütunlar: ``dnbr``, ``guven``, ``gercek_karar``. İsteğe bağlı: ``police_aktif``
    (yoksa hepsi aktif sayılır), ``sigorta_bedeli_tl`` ve ``gercek_odeme_tl`` (ikisi birlikte
    varsa finansal karşılaştırma yapılır). Sütun adları Türkçe karakter/büyük-küçük harf
    farkına bakmaz; ``;`` ya da ``,`` ayracı ve ondalık virgül kabul edilir.

    Args:
        raw: CSV baytları ya da metni.

    Returns:
        ``HistoryTable``.

    Raises:
        CalibrationError: Okunamayan dosya, eksik zorunlu sütun ya da geçerli satır kalmaması.
    """
    text = _decode(raw)
    if not text.strip():
        raise CalibrationError("Dosya boş.")
    try:
        table = pd.read_csv(io.StringIO(text), sep=_sniff_separator(text), dtype=str,
                            keep_default_na=False, skipinitialspace=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise CalibrationError(f"CSV okunamadı: {exc}") from exc
    if len(table) > MAX_ROWS:
        raise CalibrationError(f"Satır sayısı sınırı aşıldı ({len(table):,} > {MAX_ROWS:,}).")

    renamed: Dict[str, str] = {}
    for original in table.columns:
        target = _COLUMN_ALIASES.get(_fold_name(original))
        if target is None:
            continue
        if target in renamed.values():
            raise CalibrationError(f"'{target}' sütunu birden fazla kez geçiyor.")
        renamed[original] = target
    table = table.rename(columns=renamed)
    missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        raise CalibrationError(
            f"Zorunlu sütun(lar) eksik: {', '.join(missing)}. Bulunan sütunlar: "
            f"{', '.join(map(str, table.columns)) or '—'}.")

    warnings: List[str] = []
    n_in = len(table)
    dnbr = table["dnbr"].map(_parse_number).to_numpy(dtype=float)
    conf = table["guven"].map(_parse_number).to_numpy(dtype=float)
    finite_conf = conf[np.isfinite(conf)]
    if finite_conf.size and np.median(finite_conf) > 1.0 and finite_conf.max() <= 100.0:
        conf = conf / 100.0
        warnings.append("Güven değerlerinin çoğu 1'in üzerinde olduğundan yüzde olarak yorumlandı (÷100).")
    decisions = table["gercek_karar"].map(_normalise_decision)

    ok = np.isfinite(dnbr) & np.isfinite(conf) & decisions.notna().to_numpy()
    ok &= (dnbr >= -0.5) & (dnbr <= 1.5) & (conf >= 0.0) & (conf <= 1.0)

    if "police_aktif" in table.columns:
        flags = [_parse_bool(v) for v in table["police_aktif"]]
        unknown_flag = np.array([f is None for f in flags], dtype=bool)
        if unknown_flag.any():
            warnings.append(f"{int(unknown_flag.sum())} satırda 'police_aktif' anlaşılamadı; satırlar atıldı.")
        ok &= ~unknown_flag
        active = np.array([True if f is None else f for f in flags], dtype=bool)
    else:
        active = np.ones(n_in, dtype=bool)
        warnings.append("'police_aktif' sütunu yok; tüm poliçeler aktif varsayıldı.")

    insured = (table["sigorta_bedeli_tl"].map(_parse_number).to_numpy(dtype=float)
               if "sigorta_bedeli_tl" in table.columns else np.full(n_in, math.nan))
    paid = (table["gercek_odeme_tl"].map(_parse_number).to_numpy(dtype=float)
            if "gercek_odeme_tl" in table.columns else np.full(n_in, math.nan))

    bad_decisions = sorted({str(v) for v, d in zip(table["gercek_karar"], decisions) if pd.isna(d)})
    if bad_decisions:
        shown = ", ".join(repr(v) for v in bad_decisions[:5])
        warnings.append(f"Tanınmayan karar değerleri atıldı: {shown}{' …' if len(bad_decisions) > 5 else ''}.")
    dropped = int((~ok).sum())
    if dropped:
        warnings.append(f"{dropped} / {n_in} satır geçersiz değer nedeniyle atıldı "
                        "(dNBR −0,5…1,5 ve güven 0…1 aralığında, karar tanınır olmalı).")
    if not ok.any():
        raise CalibrationError("Geçerli satır kalmadı; sütun değerlerini ve karar etiketlerini kontrol edin.")

    codes = np.array([DECISION_LABELS.index(d) for d in decisions[ok]], dtype=np.int8)
    frame = pd.DataFrame({
        "dnbr": dnbr[ok],
        "guven": conf[ok],
        "police_aktif": active[ok],
        "sigorta_bedeli_tl": insured[ok],
        "gercek_karar": [DECISION_LABELS[c] for c in codes],
        "gercek_kod": codes,
        "gercek_odeme_tl": paid[ok],
    }).reset_index(drop=True)

    has_fin = bool(np.isfinite(frame["sigorta_bedeli_tl"]).sum() >= 5 and np.isfinite(frame["gercek_odeme_tl"]).sum() >= 5)
    counts = np.bincount(codes, minlength=4) / max(len(codes), 1)
    thin = [DECISION_LABELS[i] for i in range(4) if counts[i] < 0.05]
    if thin:
        warnings.append(f"Dengesiz sınıflar (<%5): {', '.join(thin)}. Bu sınıflar için kalibrasyon güvenilmez olabilir.")
    if len(frame) < SMALL_SAMPLE_ROWS:
        warnings.append(f"Küçük örneklem ({len(frame)} kayıt): uyum oranlarının güven aralığı geniştir.")
    return HistoryTable(frame, warnings, dropped, has_fin)


# --------------------------------------------------------------------------- #
# Ölçüm
# --------------------------------------------------------------------------- #
def predict_codes(frame: pd.DataFrame, rules: ClaimRules) -> np.ndarray:
    """Geçmiş kayıtlar için platform kararlarını (kod olarak) üretir.

    Args:
        frame: ``HistoryTable.frame``.
        rules: Karar kuralları.

    Returns:
        ``int8`` dizisi (0=RET … 3=ONAY).
    """
    return de.InsuranceSimulator.decide_codes(
        frame["dnbr"].to_numpy(dtype=float), frame["guven"].to_numpy(dtype=float),
        frame["police_aktif"].to_numpy(dtype=bool), rules)


def _confusion(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """``4×4`` karışıklık matrisi (satır uzman, sütun platform)."""
    flat = np.bincount(actual.astype(np.int64) * 4 + predicted.astype(np.int64), minlength=16)
    return flat.reshape(4, 4)


def score(actual: np.ndarray, predicted: np.ndarray) -> Metrics:
    """İki karar dizisini karşılaştırıp ``Metrics`` üretir.

    Args:
        actual: Uzman karar kodları.
        predicted: Platform karar kodları.

    Returns:
        ``Metrics``.
    """
    n = int(len(actual))
    conf = _confusion(actual, predicted)
    exact = int(np.trace(conf))
    false_app = int(conf[_RET, _PARTIAL] + conf[_RET, _APPROVE])
    wrong_den = int(conf[_PARTIAL, _RET] + conf[_APPROVE, _RET])
    col = conf.sum(axis=0).astype(float)
    row = conf.sum(axis=1).astype(float)
    diag = np.diag(conf).astype(float)
    precision = tuple(float(d / c) if c > 0 else math.nan for d, c in zip(diag, col))
    recall = tuple(float(d / r) if r > 0 else math.nan for d, r in zip(diag, row))
    low, high = wilson_interval(exact, n)
    return Metrics(
        n=n, agreement=exact / n if n else 0.0, agreement_low=low, agreement_high=high,
        false_approval=false_app, wrongful_denial=wrong_den,
        critical_rate=(false_app + wrong_den) / n if n else 0.0,
        expert_share=float(conf[:, _EXPERT].sum()) / n if n else 0.0,
        confusion=conf, precision=precision, recall=recall)


def backtest(history: pd.DataFrame, rules: ClaimRules) -> Metrics:
    """Kuralları geçmiş kararlara karşı oynatır.

    Args:
        history: ``HistoryTable.frame``.
        rules: Test edilecek kurallar.

    Returns:
        ``Metrics``.
    """
    return score(history["gercek_kod"].to_numpy(dtype=np.int64), predict_codes(history, rules))


def disagreements(history: pd.DataFrame, rules: ClaimRules) -> pd.DataFrame:
    """Platform ile uzmanın ayrıştığı kayıtları, hata türüyle birlikte listeler.

    Args:
        history: ``HistoryTable.frame``.
        rules: Test edilecek kurallar.

    Returns:
        Sapmalı satırlar; ``tur`` sütunu: ``Haksız ödeme riski``, ``Haksız ret``,
        ``Ekspertize düştü`` ya da ``Diğer sapma``. Kritik hatalar üstte sıralanır.
    """
    predicted = predict_codes(history, rules)
    actual = history["gercek_kod"].to_numpy(dtype=np.int64)
    off = actual != predicted
    out = history.loc[off, ["dnbr", "guven", "police_aktif", "gercek_karar"]].copy()
    p, a = predicted[off], actual[off]
    out["platform_karar"] = [DECISION_LABELS[c] for c in p]
    kind = np.select(
        [(a == _RET) & (p >= _PARTIAL), (p == _RET) & (a >= _PARTIAL), p == _EXPERT],
        ["Haksız ödeme riski", "Haksız ret", "Ekspertize düştü"], default="Diğer sapma")
    out["tur"] = kind
    order = {"Haksız ödeme riski": 0, "Haksız ret": 1, "Diğer sapma": 2, "Ekspertize düştü": 3}
    out = out.assign(_o=out["tur"].map(order)).sort_values(["_o", "dnbr"], ascending=[True, False])
    return out.drop(columns="_o").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Kalibrasyon (eşik araması)
# --------------------------------------------------------------------------- #
class _Evaluator:
    """Eşik aramasında tekrar tekrar kullanılan, ön hesaplı hızlı değerlendirici."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.dnbr = frame["dnbr"].to_numpy(dtype=float)
        self.conf = frame["guven"].to_numpy(dtype=float)
        self.active = frame["police_aktif"].to_numpy(dtype=bool)
        self.actual = frame["gercek_kod"].to_numpy(dtype=np.int64)
        self.n = len(frame)

    def counts(self, rules: ClaimRules) -> Tuple[int, int, int]:
        """``(tam uyum adedi, kritik hata adedi, ekspertize düşen adet)`` döndürür."""
        pred = de.InsuranceSimulator.decide_codes(self.dnbr, self.conf, self.active, rules).astype(np.int64)
        flat = np.bincount(self.actual * 4 + pred, minlength=16)
        exact = int(flat[0] + flat[5] + flat[10] + flat[15])
        critical = int(flat[2] + flat[3] + flat[8] + flat[12])
        expert = int(flat[1] + flat[5] + flat[9] + flat[13])
        return exact, critical, expert


def _grid(low: float, high: float, step: float) -> List[float]:
    """``low`` … ``high`` aralığını ``step`` adımıyla (2 ondalığa yuvarlı) üretir."""
    steps = int(round((high - low) / step))
    return [round(low + i * step, 2) for i in range(steps + 1)]


def _ordered(approve: float, partial: float, review: float) -> bool:
    """Eşikler ``ret < kısmi < tam`` (en az ``_GAP`` farkla) mı."""
    return partial <= round(approve - _GAP, 2) and review <= round(partial - _GAP, 2)


def _rank(evaluator: _Evaluator, cand: ClaimRules, base: ClaimRules, limit_count: float) -> Tuple[float, ...]:
    """Adayın sıralama anahtarını üretir (büyük olan iyidir).

    Kritik hata sınırını sağlayanlar: (1, tam uyum, otomatik adet, −uzaklık); sağlamayanlar:
    (0, −kritik hata, tam uyum, −uzaklık). Uzaklık, mevcut kurala olan mutlak eşik farklarının
    toplamıdır; eşitlikte en az değişiklik tercih edilir.
    """
    exact, critical, expert = evaluator.counts(cand)
    dist = (abs(cand.approve_dnbr - base.approve_dnbr) + abs(cand.partial_dnbr - base.partial_dnbr)
            + abs(cand.review_dnbr - base.review_dnbr) + abs(cand.min_confidence - base.min_confidence))
    if critical <= limit_count:
        return 1.0, float(exact), float(evaluator.n - expert), -dist
    return 0.0, -float(critical), float(exact), -dist


def _search(evaluator: _Evaluator, base: ClaimRules, critical_limit: float,
            approve_vals: Sequence[float], partial_vals: Sequence[float],
            review_vals: Sequence[float], conf_vals: Sequence[float]) -> Tuple[ClaimRules, bool]:
    """Verilen ızgarada en iyi kural setini bulur (sıralama: ``_rank``).

    Returns:
        ``(en iyi kural, sınır sağlandı mı)``.
    """
    limit_count = critical_limit * evaluator.n
    best_key: Optional[Tuple[float, ...]] = None
    best_rules = base
    for a in approve_vals:
        for p in partial_vals:
            for r in review_vals:
                if not _ordered(a, p, r):
                    continue
                for c in conf_vals:
                    cand = replace(base, approve_dnbr=a, partial_dnbr=p, review_dnbr=r, min_confidence=c)
                    key = _rank(evaluator, cand, base, limit_count)
                    if best_key is None or key > best_key:
                        best_key, best_rules = key, cand
    if best_key is None:
        return base, False
    return best_rules, best_key[0] == 1.0


def _clip_grid(centre: float, key: str, radius: float, step: float) -> List[float]:
    """Merkez çevresinde sınırlara kırpılmış ince ızgara üretir."""
    low, high = BOUNDS[key]
    return _grid(max(low, round(centre - radius, 2)), min(high, round(centre + radius, 2)), step)


def search_rules(train: pd.DataFrame, base: ClaimRules, critical_limit: float) -> Tuple[ClaimRules, bool]:
    """Dört eşiği (tam onay, kısmi onay, ret, güven) eğitim kayıtlarında arar.

    İki aşamalıdır: kaba ızgara (dNBR için 0,02; güven için 0,05) ve en iyi nokta çevresinde
    0,01 adımlı ince ızgara. Muafiyet, teminat ve avans poliçe şartıdır; kalibre edilmez.

    Args:
        train: Eğitim kayıtları.
        base: Başlangıç kuralları (diğer alanlar korunur).
        critical_limit: İstenen azami kritik hata oranı (0–1).

    Returns:
        ``(önerilen kurallar, kritik hata sınırı sağlandı mı)``.
    """
    ev = _Evaluator(train)
    coarse_conf = _grid(BOUNDS["min_confidence"][0], 0.95, 0.05) + [BOUNDS["min_confidence"][1]]
    best, _ = _search(
        ev, base, critical_limit,
        _grid(*BOUNDS["approve_dnbr"], 0.02), _grid(*BOUNDS["partial_dnbr"], 0.02),
        _grid(*BOUNDS["review_dnbr"], 0.02), coarse_conf)
    refined, _ = _search(
        ev, base, critical_limit,
        _clip_grid(best.approve_dnbr, "approve_dnbr", 0.02, 0.01),
        _clip_grid(best.partial_dnbr, "partial_dnbr", 0.02, 0.01),
        _clip_grid(best.review_dnbr, "review_dnbr", 0.02, 0.01),
        _clip_grid(best.min_confidence, "min_confidence", 0.04, 0.01))
    # Mevcut kuralın kendisi de aday sayılır: sonuç hiçbir zaman mevcut kuraldan kötü olmaz.
    limit_count = critical_limit * ev.n
    refined_key = _rank(ev, refined, base, limit_count)
    base_key = _rank(ev, base, base, limit_count)
    chosen = base if base_key >= refined_key else refined
    return chosen, max(base_key, refined_key)[0] == 1.0


def split_history(history: pd.DataFrame, test_fraction: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Kayıtları tekrarlanabilir biçimde eğitim ve test bölümlerine ayırır.

    Args:
        history: ``HistoryTable.frame``.
        test_fraction: Test payı (0,1–0,5 aralığına kırpılır).
        seed: Tohum.

    Returns:
        ``(eğitim, test)``.
    """
    fraction = float(np.clip(test_fraction, 0.1, 0.5))
    order = de.make_rng(seed, 77).permutation(len(history))
    n_test = max(int(round(len(history) * fraction)), 1)
    test_idx, train_idx = order[:n_test], order[n_test:]
    return (history.iloc[np.sort(train_idx)].reset_index(drop=True),
            history.iloc[np.sort(test_idx)].reset_index(drop=True))


def describe_changes(base: ClaimRules, tuned: ClaimRules) -> pd.DataFrame:
    """Eşik değişikliklerini tablolaştırır.

    Args:
        base: Mevcut kurallar.
        tuned: Önerilen kurallar.

    Returns:
        ``parametre``, ``mevcut``, ``onerilen``, ``fark`` sütunlu tablo.
    """
    rows = []
    for key, label in PARAM_LABELS.items():
        before, after = float(getattr(base, key)), float(getattr(tuned, key))
        rows.append({"parametre": label, "mevcut": round(before, 2), "onerilen": round(after, 2),
                     "fark": round(after - before, 2)})
    return pd.DataFrame(rows)


def sensitivity_curves(train: pd.DataFrame, tuned: ClaimRules) -> Dict[str, pd.DataFrame]:
    """Her eşik için, diğerleri önerilen değerde sabitken uyum ve kritik hata eğrisini üretir.

    Args:
        train: Eğitim kayıtları.
        tuned: Önerilen kurallar.

    Returns:
        Parametre adı → ``deger``, ``uyum_pct``, ``kritik_pct`` sütunlu tablo.
    """
    ev = _Evaluator(train)
    curves: Dict[str, pd.DataFrame] = {}
    for key, (low, high) in BOUNDS.items():
        rows = []
        for value in _grid(low, high, 0.01):
            a, p, r = tuned.approve_dnbr, tuned.partial_dnbr, tuned.review_dnbr
            c = tuned.min_confidence
            if key == "approve_dnbr":
                a = value
            elif key == "partial_dnbr":
                p = value
            elif key == "review_dnbr":
                r = value
            else:
                c = value
            a, p, r, _ = consistent_thresholds(a, p, r)
            exact, critical, _ = ev.counts(replace(tuned, approve_dnbr=a, partial_dnbr=p, review_dnbr=r,
                                                   min_confidence=c))
            rows.append({"deger": value, "uyum_pct": 100.0 * exact / ev.n, "kritik_pct": 100.0 * critical / ev.n})
        curves[key] = pd.DataFrame(rows)
    return curves


def financial_impact(frame: pd.DataFrame, base: ClaimRules, tuned: ClaimRules) -> Optional[FinancialImpact]:
    """Bedel ve uzman ödemesi bilinen kayıtlarda toplam ödeme farkını hesaplar.

    Args:
        frame: Genellikle test bölümü.
        base: Mevcut kurallar.
        tuned: Önerilen kurallar.

    Returns:
        ``FinancialImpact``; yeterli (≥5) bedelli ve ödemeli kayıt yoksa ``None``.
    """
    usable = frame[np.isfinite(frame["sigorta_bedeli_tl"]) & np.isfinite(frame["gercek_odeme_tl"])]
    if len(usable) < 5:
        return None
    base_claims = de.InsuranceSimulator.evaluate_claims(usable, base)
    tuned_claims = de.InsuranceSimulator.evaluate_claims(usable, tuned)
    return FinancialImpact(
        rows=len(usable), human_tl=float(usable["gercek_odeme_tl"].sum()),
        base_paid_tl=float(base_claims["odenecek_tl"].sum()), base_pending_tl=float(base_claims["bekleyen_tl"].sum()),
        tuned_paid_tl=float(tuned_claims["odenecek_tl"].sum()), tuned_pending_tl=float(tuned_claims["bekleyen_tl"].sum()))


def calibrate(history: pd.DataFrame, base_rules: ClaimRules, critical_limit: float = 0.02,
              test_fraction: float = 0.30, seed: int = 2026) -> CalibrationResult:
    """Eşikleri eğitim bölümünde arar, sonucu test bölümünde sınar.

    Args:
        history: ``HistoryTable.frame``.
        base_rules: Mevcut (başlangıç) kurallar.
        critical_limit: İstenen azami kritik hata oranı (0–1).
        test_fraction: Test payı.
        seed: Bölme tohumu.

    Returns:
        ``CalibrationResult``.

    Raises:
        CalibrationError: Kayıt sayısı ``MIN_ROWS_CALIBRATE``'ten azsa.
    """
    if len(history) < MIN_ROWS_CALIBRATE:
        raise CalibrationError(
            f"Kalibrasyon için en az {MIN_ROWS_CALIBRATE} geçerli kayıt gerekir (bulunan: {len(history)}).")
    limit = float(np.clip(critical_limit, 0.0, 0.5))
    train, test = split_history(history, test_fraction, seed)
    notes: List[str] = []
    searched = train
    if len(train) > SEARCH_ROW_CAP:
        keep = np.sort(de.make_rng(seed, 78).choice(len(train), size=SEARCH_ROW_CAP, replace=False))
        searched = train.iloc[keep].reset_index(drop=True)
        notes.append(f"Eşik araması hız için {SEARCH_ROW_CAP:,} eğitim kaydı üzerinde yapıldı; "
                     "tüm ölçümler tam kayıt kümesiyle hesaplandı.")
    tuned, feasible = search_rules(searched, base_rules, limit)

    train_before, train_after = backtest(train, base_rules), backtest(train, tuned)
    test_before, test_after = backtest(test, base_rules), backtest(test, tuned)
    ok_before = predict_codes(test, base_rules) == test["gercek_kod"].to_numpy()
    ok_after = predict_codes(test, tuned) == test["gercek_kod"].to_numpy()
    wins, losses = int((ok_after & ~ok_before).sum()), int((ok_before & ~ok_after).sum())
    p_value = sign_test_p(wins, losses)
    significant = bool(wins > losses and p_value < SIGNIFICANCE)
    improved = wins > losses
    safe = test_after.critical_rate <= max(limit, test_before.critical_rate)
    recommended = bool(improved and safe)

    gap = train_after.agreement - test_after.agreement
    if gap > 0.08:
        notes.append(f"Eğitim ({train_after.agreement:.0%}) ile test ({test_after.agreement:.0%}) uyumu arasındaki "
                     "fark büyük: eşikler eğitim kayıtlarına fazla uymuş olabilir; daha çok kayıtla yineleyin.")
    if not feasible:
        notes.append(f"Eğitim bölümünde kritik hata sınırı (%{limit * 100:.1f}) sağlanamadı; en düşük kritik hatalı "
                     "kural seçildi. Karar kuralının yapısı (yalnızca dNBR ve güven) bu geçmişi açıklamıyor olabilir.")
    if not improved:
        notes.append("Test bölümünde uyum artmadı: mevcut kuralları koruyun ya da daha fazla kayıt getirin.")
    elif not safe:
        notes.append("Test bölümünde uyum arttı ancak kritik hata oranı sınırı aştı; öneri uygulanmamalı.")
    elif not significant:
        notes.append(f"Test bölümündeki iyileşme istatistiksel olarak belirgin değil (kazanılan {wins}, kaybedilen "
                     f"{losses} kayıt; p={p_value:.2f}): daha çok kayıtla doğrulamadan güvenmeyin.")
    if test_after.n < 40:
        notes.append(f"Test bölümü küçük ({test_after.n} kayıt); sonuç belirsizdir (güven aralığına bakın).")

    return CalibrationResult(
        base_rules=base_rules, tuned_rules=tuned, n_train=len(train), n_test=len(test),
        critical_limit=limit, feasible=feasible,
        train_before=train_before, train_after=train_after, test_before=test_before, test_after=test_after,
        changes=describe_changes(base_rules, tuned), sensitivity=sensitivity_curves(train, tuned),
        financial=financial_impact(test, base_rules, tuned), test_wins=wins, test_losses=losses,
        p_value=p_value, significant=significant, recommended=recommended, notes=notes)


# --------------------------------------------------------------------------- #
# Örnek (SENTETİK) geçmiş ve şablon
# --------------------------------------------------------------------------- #
#: Örnek geçmişi üreten varsayımsal eksperin kuralları (platform varsayılanlarından FARKLIDIR).
ADJUSTER_RULES = ClaimRules(approve_dnbr=0.50, partial_dnbr=0.31, review_dnbr=0.13, min_confidence=0.72)


#: Örnek geçmişteki üç ayrı olayın rüzgâr yönleri (derece); olaylar parsel konumu ve şiddet deseni bakımından farklıdır.
SAMPLE_EVENT_BEARINGS: Tuple[float, ...] = (70.0, 25.0, 115.0)


def sample_history(seed: int = 2026, rows: int = 18, cols: int = 24) -> pd.DataFrame:
    """SENTETİK bir geçmiş karar tablosu üretir (mekanizmayı göstermek için).

    Geçmiş, üç ayrı sentetik olaydan oluşur (``SAMPLE_EVENT_BEARINGS``). Varsayımsal eksper,
    platformdan farklı eşiklerle (``ADJUSTER_RULES``) karar verir; dNBR'yi hafif gürültülü "okur"
    ve kararların %6'sını komşu sınıfa kaydırır. Bu nedenle sonuç gerçek dünya doğruluğunu
    KANITLAMAZ.

    Args:
        seed: Tohum.
        rows: Her olayın ızgara satırı.
        cols: Her olayın ızgara sütunu.

    Returns:
        ``dnbr, guven, police_aktif, sigorta_bedeli_tl, gercek_karar, gercek_odeme_tl`` sütunlu tablo.
    """
    parts: List[pd.DataFrame] = []
    for event, bearing in enumerate(SAMPLE_EVENT_BEARINGS):
        parcels = de.InsuranceSimulator(seed=seed + 1000 + event, rows=rows, cols=cols,
                                        wind_bearing_deg=bearing).simulate().parcels
        rng = de.make_rng(seed, 91 + event)
        n = len(parcels)
        perceived = parcels["dnbr"].to_numpy(dtype=float) + rng.normal(0.0, 0.035, n)
        codes = de.InsuranceSimulator.decide_codes(
            perceived, parcels["guven"].to_numpy(dtype=float), parcels["police_aktif"].to_numpy(dtype=bool),
            ADJUSTER_RULES).astype(np.int64)
        flip = rng.random(n) < 0.06
        shift = rng.choice([-1, 1], size=n)
        codes = np.where(flip, np.clip(codes + shift, 0, 3), codes)

        seen = parcels.assign(dnbr=perceived)
        seen_claims = de.InsuranceSimulator.evaluate_claims(seen, ADJUSTER_RULES)
        net = (seen_claims["odenecek_tl"] + seen_claims["bekleyen_tl"]).to_numpy(dtype=float)
        factor = np.array([0.0, 0.6, 1.0, 1.0])[codes]
        payment = net * factor * rng.lognormal(0.0, 0.05, n)
        parts.append(pd.DataFrame({
            "dnbr": parcels["dnbr"].round(3), "guven": parcels["guven"].round(2),
            "police_aktif": parcels["police_aktif"].to_numpy(dtype=bool),
            "sigorta_bedeli_tl": parcels["sigorta_bedeli_tl"].round(0),
            "gercek_karar": [DECISION_LABELS[c] for c in codes],
            "gercek_odeme_tl": np.round(payment, 0),
        }))
    return pd.concat(parts, ignore_index=True)


def template_csv(seed: int = 2026, sample_rows: int = 6) -> bytes:
    """Yüklenecek CSV için şablon üretir (başlık + birkaç SENTETİK örnek satır).

    Args:
        seed: Örnek satırlar için tohum.
        sample_rows: Örnek satır sayısı.

    Returns:
        UTF-8 BOM'lu, ``;`` ayraçlı, ondalık virgüllü CSV baytları.
    """
    frame = sample_history(seed).head(sample_rows)
    return frame.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def history_csv(frame: pd.DataFrame) -> bytes:
    """Bir tabloyu TR uyumlu (``;`` ve ondalık virgül, UTF-8 BOM) CSV baytlarına çevirir.

    Args:
        frame: Dışa aktarılacak tablo.

    Returns:
        CSV baytları.
    """
    return frame.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


__all__ = [
    "ADJUSTER_RULES", "BOUNDS", "CalibrationError", "CalibrationResult", "FinancialImpact", "HistoryTable",
    "Metrics", "PARAM_LABELS", "REQUIRED_COLUMNS", "OPTIONAL_COLUMNS", "MIN_ROWS_CALIBRATE", "backtest",
    "calibrate", "consistent_thresholds", "describe_changes", "disagreements", "financial_impact",
    "history_csv", "parse_history", "predict_codes", "sample_history", "score", "search_rules",
    "sensitivity_curves", "sign_test_p", "split_history", "template_csv", "wilson_interval",
]
