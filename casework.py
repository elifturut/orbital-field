"""Müşteri dosyası ve sigortacı kararı: doğrulama, tutar hesabı ve kararın tabloya uygulanması.

Saf Python (numpy/pandas); Streamlit içe aktarmaz. Otomatik karar motoru (``data_engine.evaluate_claims``) bir
**öneri** üretir; sigortacı bir poliçe için bu öneriyi onaylayabilir, değiştirebilir ya da geri alabilir. Burada:

* Karar kaydı (:class:`Decision`) ve girdi doğrulaması (not uzunluğu, tutar sınırı, geçerli karar),
* Karara göre ödenecek/bekleyen tutarın hesabı (:func:`plan_decision`; otomatik motorla aynı formül),
* Kayıtlardan geçerli kararın bulunması (:func:`effective`) ve karar tablosuna uygulanması (:func:`apply_decisions`),
* Sigortacının müşteri listesi ve karar kaydı tablosu

bulunur. Kalıcılık ``store.py``'dedir.

Gizlilik: müşteriye yalnızca ``musteri_notu`` (herkese açık not) ve karar kaynağı gider. **İç not** ve kararı veren
kullanıcının adı karar tablosuna HİÇ girmez; yalnızca sigortacı ekranlarındaki kayıt tablosunda görünür.
"""
from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import data_engine as de
import portal as pt
import tenancy
from data_engine import ClaimRules

__all__ = [
    "ACTION_DECIDE", "ACTION_REVERT", "SOURCE_AUTO", "SOURCE_INSURER", "REASON_INSURER", "NOTE_MAX", "INTERNAL_MIN",
    "CaseworkError", "Decision", "Draft", "clean_text", "plan_decision", "validate_record", "effective",
    "apply_decisions", "summarize_final", "customer_list", "log_frame", "csv_safe", "LOG_COLUMNS",
]

ACTION_DECIDE, ACTION_REVERT = "karar", "geri_al"
SOURCE_AUTO, SOURCE_INSURER = "Otomatik", "Sigortacı"
#: Sigortacı kararında ``gerekce`` sütununa yazılan iç gerekçe etiketi (müşteriye gitmez).
REASON_INSURER = "Sigortacı kararı"
NOTE_MAX = 500
#: İç gerekçe (denetim izi) en az bu kadar karakter olmalıdır.
INTERNAL_MIN = 5
_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f  ]+")


class CaseworkError(ValueError):
    """Karar girdisi geçersiz (kullanıcıya gösterilebilir Türkçe ileti taşır)."""


@dataclass(frozen=True)
class Decision:
    """Kalıcı karar kaydı (eklemeli kayıt defteri satırı).

    Attributes:
        id: Kayıt numarası (veritabanı verir; yeni kayıtta ``0``).
        company: Kararı veren şirketin kimliği (``tenancy``).
        police_no: Poliçe numarası.
        seed: Kararın verildiği veri tohumu (başka tohumda uygulanmaz).
        action: ``karar`` ya da ``geri_al`` (son kayıt "geri_al" ise otomatik karara dönülür).
        decision: ``RET`` | ``EKSPERTİZ`` | ``KISMİ ONAY`` | ``ONAY`` (geri almada boş).
        paid_tl: Kararla ödenecek tutar.
        pending_tl: Kararla bekleyen tutar.
        public_note: Müşterinin görebileceği not.
        internal_note: Yalnızca sigortacının gördüğü iç gerekçe.
        decided_by: Kararı veren kullanıcı adı.
        decided_by_name: Kararı veren kişinin görünen adı.
        decided_at: ISO 8601 zaman damgası (saat dilimli).
        auto_decision: Kayıt anında otomatik motorun kararı.
        auto_paid_tl: Kayıt anında otomatik motorun ödenecek tutarı.
    """

    company: str
    police_no: str
    seed: int
    action: str
    decision: str
    paid_tl: float
    pending_tl: float
    public_note: str
    internal_note: str
    decided_by: str
    decided_by_name: str
    decided_at: str
    auto_decision: str
    auto_paid_tl: float
    id: int = 0


@dataclass(frozen=True)
class Draft:
    """Doğrulanmış karar taslağı (henüz kaydedilmemiş).

    Attributes:
        decision: Karar etiketi.
        paid_tl: Ödenecek tutar.
        pending_tl: Bekleyen tutar.
        public_note: Temizlenmiş herkese açık not.
        internal_note: Temizlenmiş iç gerekçe.
    """

    decision: str
    paid_tl: float
    pending_tl: float
    public_note: str
    internal_note: str


# --------------------------------------------------------------------------- #
# Girdi doğrulama
# --------------------------------------------------------------------------- #
def clean_text(value: Any, label: str, max_len: int = NOTE_MAX, min_len: int = 0) -> str:
    """Serbest metni tek satıra indirir (denetim karakterleri atılır, boşluklar birleşir) ve uzunluğunu sınar.

    Args:
        value: Girilen metin.
        label: Hata iletisinde görünen alan adı.
        max_len: En çok karakter.
        min_len: En az karakter (temizlendikten sonra).

    Returns:
        Temiz metin.

    Raises:
        CaseworkError: Metin değilse, çok uzunsa ya da çok kısaysa.
    """
    if not isinstance(value, str):
        raise CaseworkError(f"{label}: metin bekleniyordu.")
    text = " ".join(_CTRL.sub(" ", value).split())
    if len(text) > max_len:
        raise CaseworkError(f"{label}: en çok {max_len} karakter olabilir (şu an {len(text)}).")
    if len(text) < min_len:
        raise CaseworkError(f"{label}: en az {min_len} karakter yazın.")
    return text


def _rounded(value: float) -> float:
    """Tutarı motorla aynı biçimde tam liraya yuvarlar (``-0.0`` olmaz)."""
    return float(round(max(float(value), 0.0)))


def plan_decision(row: Mapping[str, Any], decision: str, rules: ClaimRules, manual_tl: Optional[float],
                  public_note: Any, internal_note: Any) -> Draft:
    """Bir poliçe için sigortacı kararının tutarlarını hesaplar ve girdileri doğrular.

    Tutar kuralları (otomatik motorla aynı formül: :func:`data_engine.net_claim_tl`):

    * ``ONAY``: ödenecek = hesaplanan tazminat (ya da elle girilen tutar), bekleyen 0.
    * ``KISMİ ONAY``: ödenecek = avans (hesaplananın ``advance_pct`` kadarı ya da elle girilen tutar),
      bekleyen = hesaplanan − ödenecek (en az 0).
    * ``EKSPERTİZ``: ödenecek 0, bekleyen = hesaplanan tazminat.
    * ``RET``: ikisi de 0.

    Elle tutar yalnızca ``ONAY`` ve ``KISMİ ONAY`` için verilebilir ve 0 ile sigorta bedeli arasında olmalıdır.

    Args:
        row: Karar tablosu satırı (``dnbr``, ``sigorta_bedeli_tl`` okunur).
        decision: İstenen karar.
        rules: Tutarı hesaplayan kurallar (sigortacının geçerli kuralları).
        manual_tl: İsteğe bağlı elle tutar.
        public_note: Müşteriye görünecek not (boş olabilir).
        internal_note: İç gerekçe (en az :data:`INTERNAL_MIN` karakter).

    Returns:
        ``Draft``.

    Raises:
        CaseworkError: Karar geçersizse, tutar sınır dışıysa ya da notlar kuralları sağlamıyorsa.
    """
    if decision not in de.DECISION_LABELS:
        raise CaseworkError("Geçersiz karar.")
    pub = clean_text(public_note, "Müşteri notu")
    internal = clean_text(internal_note, "İç gerekçe", min_len=INTERNAL_MIN)
    insured, dnbr = float(row["sigorta_bedeli_tl"]), float(row["dnbr"])
    net = float(de.net_claim_tl(np.array([dnbr]), np.array([insured]), rules)[0])
    manual: Optional[float] = None
    if manual_tl is not None:
        if decision not in ("ONAY", "KISMİ ONAY"):
            raise CaseworkError("Elle tutar yalnızca ONAY ve KISMİ ONAY kararlarında girilebilir.")
        if isinstance(manual_tl, bool) or not isinstance(manual_tl, (int, float, np.integer, np.floating)) \
                or not math.isfinite(float(manual_tl)):
            raise CaseworkError("Tutar geçerli bir sayı olmalı.")
        manual = float(manual_tl)
        if manual < 0 or manual > insured:
            raise CaseworkError("Tutar 0 ile sigorta bedeli arasında olmalı.")
    if decision == "ONAY":
        paid, pending = (net if manual is None else manual), 0.0
    elif decision == "KISMİ ONAY":
        paid = net * (rules.advance_pct / 100.0) if manual is None else manual
        pending = max(net - paid, 0.0)
    elif decision == "EKSPERTİZ":
        paid, pending = 0.0, net
    else:
        paid, pending = 0.0, 0.0
    return Draft(decision, _rounded(paid), _rounded(pending), pub, internal)


def validate_record(d: Decision) -> None:
    """Kayıt defterine yazılacak kaydı son kez doğrular (savunma amaçlı; arayüz atlanmış olsa da çalışır).

    Raises:
        CaseworkError: Alan geçersizse.
    """
    if not tenancy.valid_company(d.company):
        raise CaseworkError("Bilinmeyen şirket.")
    if d.action not in (ACTION_DECIDE, ACTION_REVERT):
        raise CaseworkError("Geçersiz işlem.")
    if d.action == ACTION_DECIDE and d.decision not in de.DECISION_LABELS:
        raise CaseworkError("Geçersiz karar.")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", str(d.police_no)):
        raise CaseworkError("Geçersiz poliçe numarası.")
    if isinstance(d.seed, bool) or not isinstance(d.seed, int):
        raise CaseworkError("Geçersiz tohum.")
    for value in (d.paid_tl, d.pending_tl, d.auto_paid_tl):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise CaseworkError("Tutarlar sonlu ve eksi olmayan sayı olmalı.")
    clean_text(d.public_note, "Müşteri notu")
    clean_text(d.internal_note, "İç gerekçe", min_len=INTERNAL_MIN)         # geri almada da gerekçe şart (denetim izi)
    if not d.decided_by or len(d.decided_by) > 64 or len(d.decided_by_name) > 120:
        raise CaseworkError("Kararı veren bilgisi geçersiz.")
    try:
        dt.datetime.fromisoformat(d.decided_at)
    except (TypeError, ValueError) as exc:
        raise CaseworkError("Zaman damgası geçersiz.") from exc


# --------------------------------------------------------------------------- #
# Geçerli karar ve uygulama
# --------------------------------------------------------------------------- #
def effective(history_oldest_first: Iterable[Decision]) -> Dict[str, Decision]:
    """Kayıt defterinden her poliçenin GEÇERLİ kararını bulur: son kayıt kazanır, ``geri_al`` kararı kaldırır.

    Args:
        history_oldest_first: Eskiden yeniye sıralı kayıtlar.

    Returns:
        ``{police_no: son karar}``; son kaydı geri alma olan poliçeler yoktur.
    """
    current: Dict[str, Decision] = {}
    for d in history_oldest_first:
        if d.action == ACTION_DECIDE:
            current[d.police_no] = d
        else:
            current.pop(d.police_no, None)
    return current


def apply_decisions(claims: pd.DataFrame, decisions: Mapping[str, Decision]) -> pd.DataFrame:
    """Sigortacı kararlarını otomatik karar tablosuna uygular (girdi değiştirilmez).

    Eklenen sütunlar: ``otomatik_karar``, ``otomatik_odenecek_tl``, ``karar_kaynagi`` (``Otomatik``/``Sigortacı``),
    ``musteri_notu`` (herkese açık not). Kararı olan satırlarda ``karar``, ``odenecek_tl``, ``bekleyen_tl`` kararın
    değerlerini alır ve ``gerekce`` :data:`REASON_INSURER` olur. İç not ve kararı veren kullanıcı tabloya GİRMEZ.

    Şirketi uyuşmayan kayıtlar (başka şirketin poliçesine yazılmış ya da sahte kayıt) yok sayılır.

    Args:
        claims: ``evaluate_claims`` çıktısı (``sirket`` sütunlu).
        decisions: ``{police_no: Decision}`` (yalnızca ``karar`` işlemleri uygulanır).

    Returns:
        Yeni tablo.
    """
    df = claims.copy()
    karar = df["karar"].to_numpy(dtype=object).copy()
    paid = df["odenecek_tl"].to_numpy(dtype=float).copy()
    pending = df["bekleyen_tl"].to_numpy(dtype=float).copy()
    reason = df["gerekce"].to_numpy(dtype=object).copy()
    source = np.full(len(df), SOURCE_AUTO, dtype=object)
    note = np.full(len(df), "", dtype=object)
    df["otomatik_karar"] = df["karar"].to_numpy(dtype=object)
    df["otomatik_odenecek_tl"] = df["odenecek_tl"].to_numpy(dtype=float)
    where = {str(p): i for i, p in enumerate(df["police_no"])}
    owner = df["sirket"].to_numpy(dtype=object)
    for police_no, d in decisions.items():
        i = where.get(str(police_no))
        if i is None or d.action != ACTION_DECIDE or d.police_no != police_no or d.company != owner[i] \
                or d.decision not in de.DECISION_LABELS:
            continue
        karar[i], paid[i], pending[i] = d.decision, d.paid_tl, d.pending_tl
        reason[i], source[i], note[i] = REASON_INSURER, SOURCE_INSURER, d.public_note
    df["karar"], df["odenecek_tl"], df["bekleyen_tl"], df["gerekce"] = karar, paid, pending, reason
    df["karar_kaynagi"], df["musteri_notu"] = source, note
    return df


def summarize_final(final: pd.DataFrame) -> Dict[str, float]:
    """``summarize_claims`` özeti; "otomasyon oranı" yalnızca sigortacı müdahalesi olmayan kararları sayar.

    Args:
        final: ``apply_decisions`` çıktısı (ya da şirket süzmesi).

    Returns:
        ``data_engine`` özetine ek olarak ``manual`` (sigortacı kararı sayısı).
    """
    s = de.InsuranceSimulator.summarize_claims(final)
    manual = final["karar_kaynagi"].to_numpy() == SOURCE_INSURER
    auto_done = (~manual) & final["karar"].isin(["ONAY", "KISMİ ONAY", "RET"]).to_numpy()
    s["automation_pct"] = 100.0 * float(auto_done.sum()) / max(len(final), 1)
    s["manual"] = float(manual.sum())
    return s


# --------------------------------------------------------------------------- #
# Sigortacı tabloları
# --------------------------------------------------------------------------- #
def customer_list(mine: pd.DataFrame) -> pd.DataFrame:
    """Sigortacının müşteri listesi: yalnızca verilen (şirket süzülmüş) tablodaki müşteriler.

    Args:
        mine: Şirketin poliçelerinden oluşan karar tablosu (``apply_decisions`` çıktısı).

    Returns:
        ``musteri``, ``musteri_no``, ``police_sayisi``, ``ekspertiz``, ``sigortaci_karari``, ``bedel_tl``,
        ``odenecek_tl``, ``bekleyen_tl`` sütunlu tablo; önce ekspertiz bekleyenler, sonra ada göre.
    """
    if mine.empty:
        return pd.DataFrame(columns=["musteri", "musteri_no", "police_sayisi", "ekspertiz", "sigortaci_karari",
                                     "bedel_tl", "odenecek_tl", "bekleyen_tl"])
    g = mine.assign(_rev=(mine["karar"] == "EKSPERTİZ").astype(int),
                    _man=(mine["karar_kaynagi"] == SOURCE_INSURER).astype(int)).groupby("sigortali", sort=False)
    out = pd.DataFrame({
        "police_sayisi": g.size(), "ekspertiz": g["_rev"].sum(), "sigortaci_karari": g["_man"].sum(),
        "bedel_tl": g["sigorta_bedeli_tl"].sum(), "odenecek_tl": g["odenecek_tl"].sum(),
        "bekleyen_tl": g["bekleyen_tl"].sum(),
    }).reset_index().rename(columns={"sigortali": "musteri"})
    out["musteri_no"] = [pt.customer_no(n) for n in out["musteri"]]
    out = out.sort_values(["ekspertiz", "musteri"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    return out[["musteri", "musteri_no", "police_sayisi", "ekspertiz", "sigortaci_karari", "bedel_tl", "odenecek_tl",
                "bekleyen_tl"]]


#: Karar kaydı tablosunun sütunları (sigortacıya özel; iç not içerir).
LOG_COLUMNS: Sequence[str] = ("no", "zaman", "poliçe", "işlem", "karar", "ödenecek_tl", "bekleyen_tl", "otomatik_karar",
                              "müşteri_notu", "iç_gerekçe", "veren")


def csv_safe(value: Any) -> Any:
    """CSV/Excel formül enjeksiyonunu önler: ``= + - @`` ya da sekme ile başlayan metnin başına kesme işareti koyar."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def log_frame(decisions: Sequence[Decision], safe: bool = False) -> pd.DataFrame:
    """Karar kaydını tabloya çevirir.

    Args:
        decisions: Kayıtlar (verilen sırada).
        safe: ``True`` ise metin hücreleri :func:`csv_safe` ile korunur (dışa aktarma için).

    Returns:
        :data:`LOG_COLUMNS` sütunlu tablo.
    """
    fix = csv_safe if safe else (lambda v: v)
    rows: List[Dict[str, Any]] = []
    for d in decisions:
        rows.append({
            "no": d.id, "zaman": d.decided_at.replace("T", " ")[:16], "poliçe": d.police_no,
            "işlem": "Karar" if d.action == ACTION_DECIDE else "Geri alma", "karar": fix(d.decision or "—"),
            "ödenecek_tl": d.paid_tl, "bekleyen_tl": d.pending_tl, "otomatik_karar": fix(d.auto_decision),
            "müşteri_notu": fix(d.public_note), "iç_gerekçe": fix(d.internal_note),
            "veren": fix(f"{d.decided_by_name} (@{d.decided_by})"),
        })
    return pd.DataFrame(rows, columns=list(LOG_COLUMNS))
