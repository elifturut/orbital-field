"""
thresholds.py — Hasar eşiklerinin dayanağı ve doğrulaması (dNBR 0,27 / 0,44 …)
==============================================================================

Üç soruya yanıt verir:

1. **Hangi indeks, eşikler neden bu değer?** ``evidence_table`` her eşiğin dayanağını ve *doğrulanma durumunu* açıkça yazar
   (literatür / sezgisel başlangıç; "bu projede yerel olarak doğrulanmadı" dahil).
2. **Eşikler gerçek hasarlı/hasarsız alanlarla nasıl karşılaşıyor?** ``evaluate_thresholds`` kullanıcının saha doğrulama tablosuyla
   (parsel başına dNBR + hasarlı mı) karışıklık matrisi, duyarlılık/özgüllük, **yanlış pozitif / yanlış negatif oranları**
   (Wilson güven aralıklı), ROC/AUC ve Youden J'ye göre en iyi eşiği (bootstrap aralıklı) hesaplar.
3. **Yöntem kendi içinde tutarlı mı?** ``synthetic_validation``, bilinen "gerçek" yanma alanına sahip sentetik sahnelerde aynı
   değerlendirmeyi çalıştırır. Bu bir **ölçüm zinciri sınamasıdır** (gürültü, bulut maskesi, dNBR → sınıf), gerçek hasar doğrulaması
   DEĞİLDİR; arayüzde öyle etiketlenir.

Literatür dayanağı
------------------
dNBR şiddet sınıfları (×1000 ölçeğinde 100 / 270 / 440 / 660) USGS FIREMON'un (Key & Benson, 2006) önerisidir; ABD ormanlarında
Composite Burn Index (CBI) saha plotlarıyla ve Landsat görüntüleriyle türetilmiştir. Sentinel-2'ye ve **tarım arazisine** aktarımı
bu projede doğrulanmamıştır. Uygulamadaki karar eşikleri (0,44 onay, 0,27 kısmi onay, 0,10 ret) bu sınıf sınırlarına hizalanmış
**politika seçimleridir**; sigorta şartnamesi değildir ve kullanıcı tarafından ayarlanabilir.

Geliştirici: Elif Turut
"""

from __future__ import annotations

import datetime as dt
import io
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import analysis as an
import data_engine as de
import demo_imagery as di

__all__ = ["ThresholdError", "OperatingPoint", "ThresholdReport", "evidence_table", "confusion_at", "roc_curve", "auc_score", "best_threshold",
           "wilson", "evaluate_thresholds", "ground_truth_template_csv", "load_ground_truth_csv", "synthetic_validation", "synthetic_sweep", "SYNTHETIC_VALIDATION_NOTE",
           "MIN_LABELLED", "DEFAULT_THRESHOLDS"]

DEFAULT_THRESHOLDS: Tuple[float, ...] = (0.10, 0.27, 0.44, 0.66)
MIN_LABELLED = 20
MIN_PER_CLASS = 5
SYNTHETIC_VALIDATION_NOTE = ("SENTETİK DOĞRULAMA — bilinen yanma alanına sahip üretilmiş sahnelerde ölçüm zincirinin (gürültü, bulut maskesi, dNBR) "
                             "sınanmasıdır. Gerçek hasarlı/hasarsız parsellerle karşılaştırma DEĞİLDİR; sahada doğrulama yerine geçmez.")


class ThresholdError(ValueError):
    """Doğrulama verisi ya da parametre geçersiz."""


# --------------------------------------------------------------------------- #
# Dayanak tablosu
# --------------------------------------------------------------------------- #
def evidence_table(rules: de.ClaimRules) -> pd.DataFrame:
    """Uygulamada kullanılan her eşiğin dayanağı ve doğrulama durumu.

    Args:
        rules: Geçerli karar kuralları (ayarlanmış değerleri göstermek için).

    Returns:
        Sütunlar: ``alan, esik, deger, dayanak, rol, dogrulama`` (``dogrulama`` ∈ Literatür / Sezgisel / Politika).
    """
    FB = "Key & Benson (2006), USGS FIREMON — CBI saha plotları + Landsat, ABD ormanı"
    rows = [
        ("Yangın (dNBR)", "Yanmamış / düşük sınırı", rules.review_dnbr, FB + " (0,10)", "Altı: hasar yok → RET", "Literatür; yerel doğrulama yok"),
        ("Yangın (dNBR)", "Orta-düşük şiddet başlangıcı", rules.partial_dnbr, FB + " (0,27)", "Üstü: KISMİ ONAY (güven yeterliyse)", "Literatür sınırına hizalı politika"),
        ("Yangın (dNBR)", "Orta-yüksek şiddet başlangıcı", rules.approve_dnbr, FB + " (0,44)", "Üstü: ONAY (güven yeterliyse)", "Literatür sınırına hizalı politika"),
        ("Yangın (dNBR)", "Yüksek şiddet / tam hasar", de.FULL_DAMAGE_DNBR, FB + " (0,66)", "Hasar oranı bu değerde %100 sayılır (doğrusal ara değer)", "Literatür sınırı; hasar oranı eşlemesi varsayım"),
        ("Yangın (dNBR)", "Gözlem güveni alt sınırı", rules.min_confidence, "Sezgisel: kullanılabilir piksel × karo bulutu bileşeni", "Altı: otomatik onay yok → EKSPERTİZ", "Sezgisel"),
        ("Sel (MNDWI)", "Su eşiği", 0.0, "Xu (2006) MNDWI; 0 yaygın varsayılan değerdir", "MNDWI > 0 → su", "Yaygın varsayılan; yerel ayar gerekir"),
        ("Kuraklık (ΔNDVI bağıl)", "Hafif / orta / şiddetli düşüş", -0.10, "Sezgisel başlangıç (−%10 / −%25 / −%45); kalibre edilmedi", "Sınıflandırma ve ağırlıklı hasar oranı", "Sezgisel"),
        ("Tarım (ΔNDVI)", "Değişmedi / belirgin", 0.05, "Sezgisel başlangıç (±0,05 / ±0,15); ölçüm gürültüsünün üstü olacak biçimde seçildi", "Sınıflandırma", "Sezgisel"),
        ("Bulut", "Güvenilirlik uyarısı", an.CLOUD_WARN_PCT, "Kullanıcı gereksinimi / yaygın uygulama (%20)", "Üstü: 'Güvenilirlik Düşük' uyarısı", "Politika"),
        ("Hava kanıtı", "Aşırı yağış / sıcak / rüzgâr", 40.0, "Ayarlanabilir başlangıç (3 gün ≥ 40 mm, Tmax ≥ 30–32 °C, rüzgâr ≥ 30 km/sa); resmî eşik değildir", "Olay koşulu 'destekler / desteklemez'", "Sezgisel"),
        ("Güvenilirlik puanı", "Bileşen ağırlıkları", 0.45, "Sezgisel veri-kalitesi puanı (berraklık .45, bulut .15, zamanlama .25, örnek .15); istatistiksel güven aralığı değildir", "Düşük/Orta/Yüksek", "Sezgisel"),
    ]
    return pd.DataFrame(rows, columns=["alan", "esik", "deger", "dayanak", "rol", "dogrulama"])


# --------------------------------------------------------------------------- #
# Ölçütler
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Binom oranı için Wilson güven aralığı (n = 0 ise ``(nan, nan)``)."""
    if n <= 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class OperatingPoint:
    """Bir eşikteki ölçütler (``dNBR ≥ eşik`` ⇒ "hasarlı" tahmini).

    Attributes:
        threshold: Eşik.
        tp: Doğru pozitif.
        fp: Yanlış pozitif (hasarsız parsele hasar dendi).
        tn: Doğru negatif.
        fn: Yanlış negatif (hasarlı parsel kaçırıldı).
        sensitivity: Duyarlılık = TP/(TP+FN).
        specificity: Özgüllük = TN/(TN+FP).
        fpr: Yanlış pozitif oranı = FP/(FP+TN).
        fnr: Yanlış negatif oranı = FN/(FN+TP).
        precision: Kesinlik = TP/(TP+FP).
        accuracy: Doğruluk.
        f1: F1.
        youden: Youden J = duyarlılık + özgüllük − 1.
        fpr_ci: FPR Wilson %95 aralığı.
        fnr_ci: FNR Wilson %95 aralığı.
    """

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    sensitivity: float
    specificity: float
    fpr: float
    fnr: float
    precision: float
    accuracy: float
    f1: float
    youden: float
    fpr_ci: Tuple[float, float]
    fnr_ci: Tuple[float, float]


def _div(a: float, b: float) -> float:
    return a / b if b else math.nan


def confusion_at(dnbr: np.ndarray, damaged: np.ndarray, thr: float) -> OperatingPoint:
    """Tek bir eşik için karışıklık matrisi ve türev ölçütler.

    Args:
        dnbr: Parsel dNBR değerleri.
        damaged: Gerçek durum (``True`` = hasarlı).
        thr: Eşik.
    """
    d = np.asarray(dnbr, dtype=float)
    y = np.asarray(damaged, dtype=bool)
    pred = d >= thr
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    tn, fn = int((~pred & ~y).sum()), int((~pred & y).sum())
    sens, spec = _div(tp, tp + fn), _div(tn, tn + fp)
    prec = _div(tp, tp + fp)
    f1 = _div(2 * prec * sens, prec + sens) if not (math.isnan(prec) or math.isnan(sens)) else math.nan
    return OperatingPoint(float(thr), tp, fp, tn, fn, sens, spec, _div(fp, fp + tn), _div(fn, fn + tp), prec, _div(tp + tn, len(d)), f1,
                          sens + spec - 1.0, wilson(fp, fp + tn), wilson(fn, fn + tp))


def roc_curve(dnbr: np.ndarray, damaged: np.ndarray) -> pd.DataFrame:
    """ROC eğrisi: her benzersiz dNBR değeri bir eşiktir (azalan sırada), ``(0,0)`` ve ``(1,1)`` uçları dahil.

    Returns:
        Sütunlar ``esik, tpr, fpr``.
    """
    d = np.asarray(dnbr, dtype=float)
    y = np.asarray(damaged, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ThresholdError("ROC için hem hasarlı hem hasarsız örnek gerekir.")
    thr = np.unique(d)[::-1]
    pos, neg = np.sort(d[y]), np.sort(d[~y])
    # #(d >= t) = n - #(d < t); sıralı dizide arama O(n log n) (eşik başına tarama O(n²) olurdu)
    tpr = (n_pos - np.searchsorted(pos, thr, side="left")) / n_pos
    fpr = (n_neg - np.searchsorted(neg, thr, side="left")) / n_neg
    return pd.DataFrame({"esik": np.r_[np.inf, thr], "tpr": np.r_[0.0, tpr], "fpr": np.r_[0.0, fpr]})


def auc_score(dnbr: np.ndarray, damaged: np.ndarray) -> float:
    """ROC alanı (Mann-Whitney U / rank toplamı; eşit değerler ortalama rank alır)."""
    d = np.asarray(dnbr, dtype=float)
    y = np.asarray(damaged, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ThresholdError("AUC için hem hasarlı hem hasarsız örnek gerekir.")
    ranks = pd.Series(d).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def best_threshold(dnbr: np.ndarray, damaged: np.ndarray) -> float:
    """Youden J'yi (duyarlılık + özgüllük − 1) en büyükleyen eşik; eşitlikte en küçük eşik."""
    roc = roc_curve(dnbr, damaged)
    j = roc["tpr"] - roc["fpr"]
    finite = np.isfinite(roc["esik"].to_numpy())
    j = j.where(finite, -np.inf)
    return float(roc.loc[j.idxmax(), "esik"]) if j.max() > -np.inf else math.nan


@dataclass(frozen=True)
class ThresholdReport:
    """Eşik değerlendirme raporu.

    Attributes:
        points: Eşik başına ölçüt tablosu.
        roc: ROC eğrisi.
        auc: ROC alanı.
        auc_ci: AUC bootstrap %95 aralığı.
        best: Youden J eşiği.
        best_ci: En iyi eşiğin bootstrap %95 aralığı.
        n: Etiketli örnek sayısı.
        n_pos: Hasarlı sayısı.
        n_neg: Hasarsız sayısı.
        dropped: Eksik değer nedeniyle atılan satır.
        notes: Uyarılar.
    """

    points: pd.DataFrame
    roc: pd.DataFrame
    auc: float
    auc_ci: Tuple[float, float]
    best: float
    best_ci: Tuple[float, float]
    n: int
    n_pos: int
    n_neg: int
    dropped: int
    notes: Tuple[str, ...]


def evaluate_thresholds(df: pd.DataFrame, thresholds: Tuple[float, ...] = DEFAULT_THRESHOLDS, n_boot: int = 400, seed: int = 1) -> ThresholdReport:
    """Etiketli parsel tablosundan eşik değerlendirmesi.

    Args:
        df: Sütunlar ``dnbr`` (sayı) ve ``hasarli`` (bool/0-1).
        thresholds: Raporlanacak eşikler.
        n_boot: Bootstrap yineleme sayısı (≥ 50).
        seed: Tohum.

    Raises:
        ThresholdError: Sütun eksik, örnek yetersiz ya da tek sınıf varsa.
    """
    if not {"dnbr", "hasarli"} <= set(df.columns):
        raise ThresholdError("Tabloda 'dnbr' ve 'hasarli' sütunları olmalı.")
    if n_boot < 50:
        raise ThresholdError("n_boot en az 50 olmalı.")
    d0 = pd.to_numeric(df["dnbr"], errors="coerce")
    y0 = df["hasarli"].astype(bool)
    keep = d0.notna() & np.isfinite(d0)
    dropped = int((~keep).sum())
    d, y = d0[keep].to_numpy(dtype=float), y0[keep].to_numpy(dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if len(d) < MIN_LABELLED or min(n_pos, n_neg) < MIN_PER_CLASS:
        raise ThresholdError(f"En az {MIN_LABELLED} etiketli parsel ve her sınıftan en az {MIN_PER_CLASS} örnek gerekir (var: {n_pos} hasarlı, {n_neg} hasarsız).")
    points = pd.DataFrame([vars(confusion_at(d, y, t)) for t in thresholds])
    points["fpr_alt"], points["fpr_ust"] = zip(*points["fpr_ci"])
    points["fnr_alt"], points["fnr_ust"] = zip(*points["fnr_ci"])
    points = points.drop(columns=["fpr_ci", "fnr_ci"])
    rng = np.random.default_rng(seed)
    aucs, bests = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        yb = y[idx]
        if yb.all() or (~yb).all():
            continue
        aucs.append(auc_score(d[idx], yb))
        bests.append(best_threshold(d[idx], yb))
    notes: List[str] = []
    if min(n_pos, n_neg) < 30:
        notes.append("Bir sınıfta 30'dan az örnek var; oranların güven aralıkları geniştir.")
    if dropped:
        notes.append(f"{dropped} satır eksik/sonlu olmayan dNBR nedeniyle atıldı.")
    if len(aucs) < 0.9 * n_boot:
        notes.append("Bootstrap örneklerinin bir kısmı tek sınıflı çıktığından atlandı.")
    qa = (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))) if aucs else (math.nan, math.nan)
    qb = (float(np.nanpercentile(bests, 2.5)), float(np.nanpercentile(bests, 97.5))) if bests else (math.nan, math.nan)
    return ThresholdReport(points, roc_curve(d, y), auc_score(d, y), qa, best_threshold(d, y), qb, len(d), n_pos, n_neg, dropped, tuple(notes))


# --------------------------------------------------------------------------- #
# Kullanıcının saha doğrulama tablosu
# --------------------------------------------------------------------------- #
_TRUE = {"1", "true", "evet", "e", "yes", "y", "hasarli", "hasarlı", "var"}
_FALSE = {"0", "false", "hayir", "hayır", "h", "no", "n", "hasarsiz", "hasarsız", "yok"}


def ground_truth_template_csv() -> str:
    """Saha doğrulama CSV şablonu."""
    return "parsel_id,dnbr,hasarli\nP001,0.52,1\nP002,0.08,0\n"


def load_ground_truth_csv(text: str) -> pd.DataFrame:
    """Saha doğrulama CSV'sini okur: ``parsel_id`` (isteğe bağlı), ``dnbr``, ``hasarli`` (1/0, evet/hayır).

    Raises:
        ThresholdError: Sütun eksik, dNBR sayı değil ya da etiket tanınmıyorsa.
    """
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:  # noqa: BLE001
        raise ThresholdError(f"CSV okunamadı ({type(exc).__name__}).") from None
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in ("dnbr", "hasarli") if c not in df.columns]
    if missing:
        raise ThresholdError(f"CSV'de eksik sütun: {missing}")
    df["dnbr"] = pd.to_numeric(df["dnbr"], errors="coerce")
    if df["dnbr"].isna().all():
        raise ThresholdError("'dnbr' sütunu sayısal değil.")

    def lab(v: object) -> bool:
        s = str(v).strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        raise ThresholdError(f"'hasarli' değeri tanınmadı: {v!r} (1/0, evet/hayır kullanın).")

    df["hasarli"] = df["hasarli"].map(lab)
    return df


# --------------------------------------------------------------------------- #
# Sentetik ölçüm zinciri sınaması
# --------------------------------------------------------------------------- #
def synthetic_validation(n_scenes: int = 12, block_px: int = 8, damage_cut: float = 0.50, seed: int = 100,
                         min_valid: float = 0.5) -> pd.DataFrame:
    """Bilinen yanma alanına sahip sentetik sahnelerde blok-bazlı dNBR ve gerçek durum tablosu.

    Her sahne farklı tohumla (farklı arazi örtüsü, yanma izi, bulut/gölge, gürültü) üretilir; ızgara ``block_px`` kenarlı bloklara
    ("parsel") bölünür. Blok *gerçekte hasarlı* sayılır: gerçek yanma şiddetinin blok ortalaması ``damage_cut`` ve üstü.
    Ölçülen dNBR blokta bulut/gölge dışı piksellerin ortalamasıdır; kullanılabilir piksel oranı ``min_valid`` altındaki bloklar atılır.

    Args:
        n_scenes: Sahne sayısı.
        block_px: Blok kenarı (piksel).
        damage_cut: "Gerçekte hasarlı" sayılması için blok ortalama gerçek şiddet eşiği (0–1).
        seed: İlk tohum.
        min_valid: Blokta en az kullanılabilir piksel oranı.

    Returns:
        Sütunlar ``sahne, blok, dnbr, hasarli, gercek_siddet, gecerli_oran``.
    """
    if not (1 <= n_scenes <= 40 and 4 <= block_px <= 32 and 0.0 < damage_cut < 1.0 and 0.0 < min_valid <= 1.0):
        raise ThresholdError("Parametreler aralık dışında.")
    base = di.preset("yangin")
    rows = []
    for i in range(n_scenes):
        scn = di.DemoScenario("yangin", base.region, base.event_date, base.start, base.end, seed + i, base.title)
        grid = an.make_grid(scn.region.bbox, 96)
        cat = di.catalog(scn)
        pre_c = [c for c in cat if c.date <= scn.event_date - dt.timedelta(days=3) and c.cloud_pct < 40]
        post_c = [c for c in cat if c.date >= scn.event_date + dt.timedelta(days=7) and c.cloud_pct < 40]
        if not pre_c or not post_c:
            continue
        # sahne çiftini kümelere yay: bazen bulutlu sahne kullanılsın (gerçek koşullara yakın)
        pre_m, post_m = pre_c[-1], post_c[i % min(len(post_c), 3)]
        pre, post = di.load_scene(scn, grid, pre_m), di.load_scene(scn, grid, post_m)
        res = an.analyse_change("yangin", pre, post, scn.region, grid, scn.event_date)
        truth = di.truth_masks(scn, grid, post_m.date)["burn"]
        for bi, r0 in enumerate(range(0, grid.rows - block_px + 1, block_px)):
            for bj, c0 in enumerate(range(0, grid.cols - block_px + 1, block_px)):
                sl = (slice(r0, r0 + block_px), slice(c0, c0 + block_px))
                inside = res.region_mask[sl]
                if inside.mean() < 0.99:
                    continue
                valid = res.cls[sl] >= 0
                vf = float(valid.mean())
                if vf < min_valid:
                    continue
                dn = float(res.delta[sl][valid].mean())
                tr = float(truth[sl].mean())
                rows.append((i, f"{bi}-{bj}", dn, tr >= damage_cut, tr, vf))
    return pd.DataFrame(rows, columns=["sahne", "blok", "dnbr", "hasarli", "gercek_siddet", "gecerli_oran"])


def synthetic_sweep(cuts: Tuple[float, ...] = (0.3, 0.5, 0.7), n_scenes: int = 12, seed: int = 100, base: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """"Hasarlı" tanımı değiştikçe eşik başarımının nasıl kaydığını gösterir (sentetik).

    Sahneler bir kez üretilir, etiketler her ``damage_cut`` için yeniden verilir. En iyi (Youden J) eşiğin tanıma göre kaymasını ve
    uygulamadaki 0,27 / 0,44 eşiklerinin yanlış pozitif / yanlış negatif oranlarını döndürür.

    Args:
        cuts: "Hasarlı" sayılma eşikleri (gerçek şiddet).
        n_scenes: Sahne sayısı (``base`` verilmezse).
        seed: İlk tohum (``base`` verilmezse).
        base: Önceden üretilmiş ``synthetic_validation`` çıktısı (aynı sahneleri yeniden üretmemek için).

    Returns:
        Sütunlar ``hasarli_tanimi, n_hasarli, n, auc, en_iyi_esik, fpr_027, fnr_027, fpr_044, fnr_044``.
    """
    if base is None:
        base = synthetic_validation(n_scenes=n_scenes, damage_cut=0.5, seed=seed)
    rows = []
    for cut in cuts:
        if not 0.0 < cut < 1.0:
            raise ThresholdError("damage_cut 0 ile 1 arasında olmalı.")
        d = base["dnbr"].to_numpy()
        y = (base["gercek_siddet"] >= cut).to_numpy()
        if y.all() or (~y).all():
            continue
        a, b = confusion_at(d, y, 0.27), confusion_at(d, y, 0.44)
        rows.append((cut, int(y.sum()), len(y), auc_score(d, y), best_threshold(d, y), a.fpr, a.fnr, b.fpr, b.fnr))
    return pd.DataFrame(rows, columns=["hasarli_tanimi", "n_hasarli", "n", "auc", "en_iyi_esik", "fpr_027", "fnr_027", "fpr_044", "fnr_044"])
