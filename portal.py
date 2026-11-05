"""Müşteri portalı: müşteriye gösterilebilecek veriyi süzer, sade dille anlatır, karar bildirimi üretir.

Saf Python (numpy/pandas); Streamlit içe aktarmaz. Müşteriye giden HER ŞEY buradaki beyaz listeden geçer:

* **Satır süzme:** yalnızca ``sigortali`` adı müşteriyle BİREBİR eşleşen poliçeler (alt dize eşleşmesi yok).
* **Sütun süzme:** :data:`CUSTOMER_COLUMNS` dışındaki hiçbir sütun (risk skoru, ham ölçümler, gözlem güveni,
  iç gerekçe metni) müşteri tablosuna girmez.
* **Metin:** gerekçeler eşik ya da skor sayısı içermez; kural izinden yalnızca neden kodu alınır.
* **Belge:** karar bildirimi yalnızca müşterinin kendi parselini çizer (komşu parsel yok) ve sahiplik denetler.

Karar ve tutar, ``reports.verify`` ile motordan bağımsız yeniden hesaplanır; uyuşmazsa bildirim üretilmez.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import data_engine as de
import reports as rep
from auth import Principal, Role
from data_engine import ClaimRules, format_tr

__all__ = [
    "CUSTOMER_COLUMNS", "INTERNAL_COLUMNS", "STATUS_LABELS", "HEADLINES", "REASON_TEXT", "VISIBILITY",
    "Step", "CustomerView", "customer_ranking", "customers_table", "customer_no", "resolve_customer", "build_view",
    "steps_for", "own_geojson", "own_parcel_svg", "build_notice_html", "notice_filename", "NOTICE_CSS",
    "PRODUCTIVITY_COLUMNS", "ProductivityView", "build_productivity", "productivity_text",
]

#: Müşteri tablosuna girebilen sütunlar (beyaz liste).
CUSTOMER_COLUMNS: Tuple[str, ...] = (
    "police_no", "parsel_id", "varlik_turu", "alan_ha", "sigorta_bedeli_tl", "police_aktif", "siddet", "karar",
    "hasar_yuzdesi", "odenecek_tl", "bekleyen_tl", "enlem", "boylam",
)
#: Müşteriye ASLA gitmeyen sütunlar (testler bunların sızmadığını doğrular).
INTERNAL_COLUMNS: Tuple[str, ...] = (
    "sigortali", "risk_skoru", "risk_sinifi", "guven", "dnbr", "nbr_once", "nbr_sonra", "ndvi_once", "ndvi_sonra",
    "siddet_no", "gerekce",
)
#: Verimlilik tablosuna girebilen sütunlar (beyaz liste); ham ölçüm, risk ve eşik içermez.
PRODUCTIVITY_COLUMNS: Tuple[str, ...] = (
    "police_no", "parsel_id", "varlik_turu", "alan_ha", "verim_skoru", "olay_kaybi_pct", "beklenen_t_ha",
    "tahmini_t_ha", "fark_pct", "beklenen_toplam_t", "tahmini_toplam_t", "seyir",
)
STATUS_LABELS: Dict[str, str] = {
    "ONAY": "Onaylandı", "KISMİ ONAY": "Kısmen onaylandı (avans)", "EKSPERTİZ": "Ekspertiz bekleniyor",
    "RET": "Reddedildi",
}
HEADLINES: Dict[str, str] = {
    "ONAY": "Hasarınız onaylandı",
    "KISMİ ONAY": "Hasarınız kısmen onaylandı; avans ödenecek",
    "EKSPERTİZ": "Ekspertiz incelemesi gerekiyor",
    "RET": "Bu olay için ödeme yapılmayacak",
}
#: ``Trace.reason_key`` → müşteriye gösterilen cümle. Hiçbir sayı ya da eşik içermez.
REASON_TEXT: Dict[str, str] = {
    "inactive": "Poliçeniz olay tarihinde dönem dışında ya da pasif olduğundan bu olay için ödeme yapılmaz.",
    "below_review": "Uydu ölçümlerinde parselinizde poliçe kapsamında tazminat gerektirecek düzeyde hasar saptanmadı.",
    "approve": "Uydu ölçümleri parselinizdeki hasarı net biçimde doğruladı; tazminatınız hesaplandı.",
    "partial": "Parselinizde orta düzeyde hasar saptandı. Avans ödenir; kalan tutar ekspertiz sonrası belirlenir.",
    "low_confidence": "Uydu görüntüsü bulut ya da gölge nedeniyle yeterince net değil; yerinde inceleme gerekir.",
    "borderline": "Hasar saptandı ancak otomatik onay için yeterince net değil; ekspertiz incelemesi gerekir.",
}
#: Rol görünürlük tablosu: (bilgi, müşteri görür mü, sigortacı görür mü).
VISIBILITY: Tuple[Tuple[str, bool, bool], ...] = (
    ("Kendi poliçeleri ve parselleri", True, True),
    ("Karar, ödenecek ve bekleyen tutar", True, True),
    ("Sade dille gerekçe ve takip adımları", True, True),
    ("Kendi parselinin uydu haritası", True, True),
    ("Kendi parsellerinin bitki gelişimi ve tahmini verim (beklenene oranla)", True, True),
    ("Karar bildirimi (kendi poliçesi)", True, True),
    ("Diğer müşterilerin poliçe ve kararları", False, True),
    ("Poliçe risk skoru ve risk sınıfı", False, True),
    ("Ham uydu ölçümleri (dNBR, NBR, NDVI, gözlem güveni)", False, True),
    ("Karar eşikleri ve kural paneli", False, True),
    ("Geriye dönük test ve kalibrasyon", False, True),
    ("Karar kanıt paketi (iç ölçüm ve eşikler)", False, True),
    ("Tarım, belediye ve kurtarma modülleri", False, True),
)

STEP_DONE, STEP_CURRENT, STEP_WAITING, STEP_CLOSED = "done", "current", "waiting", "closed"


@dataclass(frozen=True)
class Step:
    """Hasar takip adımı.

    Attributes:
        label: Adımın adı.
        detail: Kısa açıklama.
        state: ``done`` | ``current`` | ``waiting`` | ``closed``.
    """

    label: str
    detail: str
    state: str


@dataclass(frozen=True)
class CustomerView:
    """Bir müşterinin görebildiği her şey.

    Attributes:
        name: Müşteri adı (``sigortali``).
        number: Görüntülenen müşteri numarası.
        policies: Müşteri sütunlu poliçe tablosu (+ ``durum``, ``aciklama``).
        positions: ``policies`` satırlarının geometri/tablo konumları (aynı sırada).
        total_insured: Toplam sigorta bedeli (TL).
        total_paid: Ödenecek toplam (TL).
        total_pending: Bekleyen toplam (TL).
        counts: Karara göre poliçe sayıları.
    """

    name: str
    number: str
    policies: pd.DataFrame
    positions: Tuple[int, ...]
    total_insured: float
    total_paid: float
    total_pending: float
    counts: Dict[str, int]


def _e(value: object) -> str:
    """HTML kaçışı."""
    return html.escape(str(value), quote=True)


def _tl(value: float) -> str:
    """Türk lirası tutarı."""
    return "₺" + format_tr(float(value), 0)


# --------------------------------------------------------------------------- #
# Müşteri eşlemesi
# --------------------------------------------------------------------------- #
def customer_ranking(claims: pd.DataFrame) -> List[str]:
    """Müşterileri demo hesaplarına eşlemek için sıralar: önce karar çeşitliliği, sonra poliçe sayısı, sonra ad.

    Böylece 1. müşteri gösterimde en çok karar türünü (onay, ekspertiz, ret…) bir arada görür.

    Args:
        claims: Karar tablosu.

    Returns:
        Müşteri adları listesi (1. müşteri = en çeşitli kararlara sahip olan).
    """
    grouped = claims.groupby("sigortali")["karar"].agg(["nunique", "size"])
    return sorted(grouped.index.tolist(), key=lambda n: (-int(grouped.at[n, "nunique"]), -int(grouped.at[n, "size"]), str(n)))


def customers_table(claims: pd.DataFrame) -> pd.DataFrame:
    """Sigortacının müşteri seçici için kullandığı özet: ad, poliçe sayısı, müşteri no (sıralamayla aynı sırada).

    Args:
        claims: Karar tablosu.

    Returns:
        ``musteri``, ``police_sayisi``, ``musteri_no`` sütunlu tablo (sıralamayla aynı sırada).
    """
    counts = claims["sigortali"].value_counts()
    names = customer_ranking(claims)
    return pd.DataFrame({"musteri": names, "police_sayisi": [int(counts[n]) for n in names],
                         "musteri_no": [customer_no(n) for n in names]})


def customer_no(name: str) -> str:
    """Ad'dan kararlı, yalnızca görüntü amaçlı müşteri numarası türetir (arama anahtarı değildir)."""
    digest = hashlib.sha256(str(name).encode("utf-8")).hexdigest()
    return f"MÜŞ-{int(digest[:8], 16) % 100_000:05d}"


def resolve_customer(claims: pd.DataFrame, principal: Principal) -> Optional[str]:
    """Müşteri hesabını tablodaki müşteri adına eşler.

    Args:
        claims: Karar tablosu.
        principal: Oturum açmış müşteri.

    Returns:
        Müşteri adı; eşleşme yoksa (ya da rol müşteri değilse) ``None``.
    """
    if principal.role is not Role.CUSTOMER:
        return None
    if principal.customer_name is not None:
        return principal.customer_name if bool((claims["sigortali"] == principal.customer_name).any()) else None
    ranking = customer_ranking(claims)
    rank = principal.customer_rank
    return ranking[rank - 1] if rank is not None and 1 <= rank <= len(ranking) else None


# --------------------------------------------------------------------------- #
# Görünüm
# --------------------------------------------------------------------------- #
def build_view(claims: pd.DataFrame, name: str, rules: ClaimRules) -> CustomerView:
    """Müşterinin görebileceği tabloyu kurar (satır ve sütun süzmesi burada yapılır).

    Args:
        claims: Karar tablosu (tüm müşteriler; geometriyle aynı sıra).
        name: Müşteri adı; ``sigortali`` ile birebir eşleşen satırlar alınır.
        rules: Kararı üreten kurallar (neden kodu için).

    Returns:
        ``CustomerView``.

    Raises:
        KeyError: Bu ada ait poliçe yoksa.
        ValueError: Kural izi ya da tutar motor çıktısıyla uyuşmazsa.
    """
    positions = np.flatnonzero((claims["sigortali"] == name).to_numpy())
    if positions.size == 0:
        raise KeyError(f"Bu müşteriye ait poliçe yok: {name}")
    own = claims.iloc[positions]
    reasons: List[str] = []
    for _, row in own.iterrows():
        trace, _ = rep.verify(row, rules)
        reasons.append(trace.reason_key)
    frame = own[list(CUSTOMER_COLUMNS)].reset_index(drop=True)
    frame["durum"] = frame["karar"].map(STATUS_LABELS)
    frame["aciklama"] = [REASON_TEXT[k] for k in reasons]
    counts = {k: int((frame["karar"] == k).sum()) for k in STATUS_LABELS}
    return CustomerView(
        name=name, number=customer_no(name), policies=frame, positions=tuple(int(p) for p in positions),
        total_insured=float(frame["sigorta_bedeli_tl"].sum()), total_paid=float(frame["odenecek_tl"].sum()),
        total_pending=float(frame["bekleyen_tl"].sum()), counts=counts)


@dataclass(frozen=True)
class ProductivityView:
    """Müşterinin kendi parsellerinin verimlilik takibi (yalnızca kendi satırları; ham ölçüm yok).

    Attributes:
        table: ``PRODUCTIVITY_COLUMNS`` sütunlu tablo (``CustomerView.policies`` ile aynı sırada).
        dates: Gözlem tarihleri; sonuncusu olay sonrası gözlem.
        index: ``{police_no: gelişim indeksi dizisi}``; yalnızca takip edilen parseller (100 = beklenen düzey).
        event_index: Olay sonrası gözlemin konumu.
        tracked: Takip edilen parsel sayısı.
        mean_score: Alan ağırlıklı sezon seyri skoru (%); takip edilen parsel yoksa ``None``.
        mean_loss: Alan ağırlıklı olay kaybı (%); takip edilen parsel yoksa ``None``.
        forecast_index: Alan ağırlıklı tahmini verim / beklenen verim (%); takip edilen parsel yoksa ``None``.
    """

    table: pd.DataFrame
    dates: pd.DatetimeIndex
    index: Dict[str, np.ndarray]
    event_index: int
    tracked: int
    mean_score: Optional[float]
    mean_loss: Optional[float]
    forecast_index: Optional[float]


def build_productivity(prod: de.ProductivityDataset, view: CustomerView) -> ProductivityView:
    """Tüm parsellerin verimlilik verisinden YALNIZCA müşterinin kendi satırlarını ve eğrilerini alır.

    Args:
        prod: ``data_engine.simulate_productivity`` çıktısı (tüm parseller; müşteriye doğrudan verilmez).
        view: ``build_view`` çıktısı (``positions`` ve ``policies`` kendi parsellerinin konumudur).

    Returns:
        ``ProductivityView``.

    Raises:
        ValueError: ``prod`` ile ``view`` uyuşmuyorsa (konum aralık dışı).
    """
    pos = list(view.positions)
    if len(prod.table) <= max(pos):
        raise ValueError("Verimlilik verisi parsel sayısıyla uyuşmuyor.")
    own = prod.table.iloc[pos].reset_index(drop=True)
    pol = view.policies.reset_index(drop=True)
    table = pd.concat([pol[["police_no", "parsel_id", "varlik_turu", "alan_ha"]],
                       own[[c for c in PRODUCTIVITY_COLUMNS if c in own.columns]]], axis=1)[list(PRODUCTIVITY_COLUMNS)]
    mask = own["takip"].to_numpy(dtype=bool)
    index = {str(pol.at[i, "police_no"]): prod.index[p].copy() for i, p in enumerate(pos) if mask[i]}
    area = pol["alan_ha"].to_numpy(dtype=float)
    if mask.any():
        w = area[mask]
        expected = own["beklenen_t_ha"].to_numpy(dtype=float)[mask]
        forecast = own["tahmini_t_ha"].to_numpy(dtype=float)[mask]
        mean_score = float(np.average(own["verim_skoru"].to_numpy(dtype=float)[mask], weights=w))
        mean_loss = float(np.average(own["olay_kaybi_pct"].to_numpy(dtype=float)[mask], weights=w))
        forecast_index = float(np.average(100.0 * forecast / expected, weights=w))
    else:
        mean_score = mean_loss = forecast_index = None
    return ProductivityView(table=table, dates=prod.dates, index=index, event_index=prod.event_index,
                            tracked=int(mask.sum()), mean_score=mean_score, mean_loss=mean_loss,
                            forecast_index=forecast_index)


def productivity_text(row: pd.Series) -> str:
    """Bir parselin verimlilik durumunu sade dille anlatır (ham ölçüm ve eşik içermez).

    Args:
        row: ``ProductivityView.table`` satırı.

    Returns:
        Kısa açıklama cümleleri.
    """
    if str(row["seyir"]) == de.NOT_TRACKED:
        return f'{row["varlik_turu"]} bitki üretmediği için bu varlık türünde verim takibi yapılmaz.'
    parts = [f'Sezon boyunca bitki gelişimi beklenen seyre göre %{format_tr(float(row["verim_skoru"]), 0)} düzeyinde '
             f'kaldı ({str(row["seyir"]).lower()}).']
    loss = float(row["olay_kaybi_pct"])
    if loss >= 1.0:
        parts.append(f"Olay nedeniyle tahmini verim ayrıca %{format_tr(loss, 0)} azaldı.")
    parts.append(f'Sezon sonu tahmini verim {format_tr(float(row["tahmini_t_ha"]), 1)} ton/ha; '
                 f'normal bir sezonda beklenen {format_tr(float(row["beklenen_t_ha"]), 1)} ton/ha.')
    return " ".join(parts)


def steps_for(decision: str, event_date: dt.date, paid: float, pending: float, advance_pct: float) -> Tuple[Step, ...]:
    """Kararın durumundan hasar takip adımlarını türetir (gerçek ödeme/ekspertiz kaydı değildir).

    Args:
        decision: ``ONAY`` | ``KISMİ ONAY`` | ``EKSPERTİZ`` | ``RET``.
        event_date: Olay sonrası görüntü tarihi.
        paid: Ödenecek tutar (TL).
        pending: Bekleyen tutar (TL).
        advance_pct: Kısmi onayda avans oranı (%).

    Returns:
        Adımlar.

    Raises:
        KeyError: Bilinmeyen karar.
    """
    head = (
        Step("Olay kaydedildi", f"Olay sonrası uydu görüntüsü: {event_date:%d.%m.%Y}", STEP_DONE),
        Step("Uydu analizi tamamlandı", "Parselinizdeki hasar ölçüldü", STEP_DONE),
        Step("Karar verildi", HEADLINES[decision], STEP_DONE),
    )
    cap = f" (en çok {_tl(pending)})" if pending >= 0.5 else ""      # tutar 0'a yuvarlanıyorsa "en çok ₺0" demeyiz
    tail = {
        "ONAY": (Step("Ödeme", f"{_tl(paid)} ödeme sürecine alınır", STEP_CURRENT),),
        "KISMİ ONAY": (
            Step("Avans ödemesi", f"{_tl(paid)} (hesaplanan tutarın %{format_tr(advance_pct, 0)}'i)", STEP_CURRENT),
            Step("Ekspertiz ve kalan ödeme", f"Kalan tutar ekspertiz sonrası belirlenir{cap}", STEP_WAITING)),
        "EKSPERTİZ": (
            Step("Ekspertiz", "Yerinde inceleme için eksper görevlendirilir", STEP_CURRENT),
            Step("Ödeme", f"Ekspertiz sonrası belirlenir{cap}", STEP_WAITING)),
        "RET": (Step("Dosya kapandı", "Bu olay için ödeme yapılmaz", STEP_CLOSED),),
    }[decision]
    return head + tail


# --------------------------------------------------------------------------- #
# Harita verisi ve çizim (yalnızca müşterinin kendi parselleri)
# --------------------------------------------------------------------------- #
def own_geojson(geometry: de.ParcelGeometry, view: CustomerView, colors: Sequence[str]) -> dict:
    """Müşterinin kendi parselleri için GeoJSON üretir; özellikler beyaz listeden gelir.

    Başka müşterinin parseli ya da iç alan (risk, dNBR, güven) çıktıya girmez.

    Args:
        geometry: Tüm parsellerin geometrisi.
        view: ``build_view`` çıktısı.
        colors: Şiddet sınıfı renkleri.

    Returns:
        GeoJSON ``FeatureCollection``.
    """
    sev_names = list(de.SEVERITY_LABELS)
    features: List[dict] = []
    for pos, (_, row) in zip(view.positions, view.policies.iterrows()):
        ring = [[float(lo), float(la)] for la, lo in zip(geometry.corners_lat[pos], geometry.corners_lon[pos])]
        ring.append(ring[0])
        severity = sev_names.index(str(row["siddet"])) if str(row["siddet"]) in sev_names else 0
        features.append({
            "type": "Feature", "id": int(pos),
            "properties": {
                "police_no": str(row["police_no"]), "varlik_turu": str(row["varlik_turu"]),
                "alan_txt": f'{format_tr(float(row["alan_ha"]), 2)} ha', "bedel_txt": _tl(row["sigorta_bedeli_tl"]),
                "siddet": str(row["siddet"]), "durum": str(row["durum"]), "odeme_txt": _tl(row["odenecek_tl"]),
                "bekleyen_txt": _tl(row["bekleyen_tl"]), "renk": str(colors[severity]),
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    return {"type": "FeatureCollection", "features": features}


def own_parcel_svg(geometry: de.ParcelGeometry, position: int, color: str, label: str) -> str:
    """Yalnızca müşterinin parselini çizen SVG (komşu parsel yok).

    Args:
        geometry: Parsel geometrisi.
        position: Parselin geometri indeksi.
        color: Dolgu rengi (şiddet sınıfı).
        label: Parsel etiketi.

    Returns:
        ``<svg>`` metni.
    """
    lat0, lon0 = float(geometry.centroid_lat[position]), float(geometry.centroid_lon[position])
    kx = math.cos(math.radians(lat0)) * de.KM_PER_DEG_LAT
    xs = (geometry.corners_lon[position] - lon0) * kx
    ys = -(geometry.corners_lat[position] - lat0) * de.KM_PER_DEG_LAT
    pad = 0.06
    x_min, x_max, y_min, y_max = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad
    width = 300.0
    height = width * (y_max - y_min) / max(x_max - x_min, 1e-9)
    points = " ".join(f"{(x - x_min) / (x_max - x_min) * width:.1f},{(y - y_min) / (y_max - y_min) * height:.1f}"
                      for x, y in zip(xs, ys))
    return (f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" role="img" aria-label="Parseliniz">'
            f'<rect width="{width:.0f}" height="{height:.0f}" fill="#f4f5f7"/>'
            f'<polygon points="{points}" fill="{_e(color)}" fill-opacity="0.9" stroke="#14181f" stroke-width="2.4"/>'
            f'<text x="{width / 2:.1f}" y="{height / 2 + 4:.1f}" text-anchor="middle" font-size="12" font-weight="700" '
            f'fill="#14181f" stroke="#ffffff" stroke-width="3" paint-order="stroke">{_e(label)}</text></svg>')


# --------------------------------------------------------------------------- #
# Karar bildirimi (HTML)
# --------------------------------------------------------------------------- #
NOTICE_CSS = rep.PACK_CSS + """
ol.steps{list-style:none;margin:0;padding:0}
ol.steps li{display:flex;gap:12px;padding:8px 0;border-bottom:1px solid var(--line)}
ol.steps .dot{flex:0 0 22px;height:22px;border:2px solid #9aa2af;border-radius:50%;text-align:center;font-size:12px;line-height:18px;color:#9aa2af}
ol.steps .done .dot{background:#1f8a5b;border-color:#1f8a5b;color:#fff}
ol.steps .current .dot{border-color:#2563c9;color:#2563c9}
ol.steps .closed .dot{background:#5b6472;border-color:#5b6472;color:#fff}
ol.steps b{display:block;font-size:13.5px}ol.steps span.d{color:var(--muted);font-size:12.5px}
"""
_REQUIRED = ("police_no", "parsel_id", "sigortali", "varlik_turu", "alan_ha", "sigorta_bedeli_tl", "police_aktif",
             "siddet", "siddet_no", "karar", "hasar_yuzdesi", "odenecek_tl", "bekleyen_tl")
_ASCII_FOLD = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def notice_filename(police_no: str, decision: str) -> str:
    """Bildirim için ASCII-güvenli dosya adı (ör. ``hasar_bildirimi_POL-2026-10007_ONAY.html``)."""
    safe_no = re.sub(r"[^A-Za-z0-9_-]+", "-", str(police_no).translate(_ASCII_FOLD))
    safe_dec = re.sub(r"[^A-Za-z0-9_-]+", "-", str(decision).translate(_ASCII_FOLD))
    return f"hasar_bildirimi_{safe_no}_{safe_dec}.html"


def build_notice_html(row: Mapping[str, Any], owner: str, rules: ClaimRules, event_label: str, event_date: dt.date,
                      generated_at: dt.datetime, geometry: de.ParcelGeometry, position: int,
                      colors: Sequence[str], synthetic: bool = True) -> str:
    """Müşteriye verilecek tek poliçelik "Hasar karar bildirimi" belgesini üretir.

    Belge risk skoru, ham ölçüm, gözlem güveni, eşik ve komşu parsel içermez.

    Args:
        row: ``evaluate_claims`` çıktısından TAM satır (yalnızca beyaz listedeki alanlar okunur).
        owner: Belgeyi isteyen müşterinin adı; satırın sahibiyle aynı olmalı.
        rules: Kararı üreten kurallar.
        event_label: Olay açıklaması.
        event_date: Olay sonrası görüntü tarihi.
        generated_at: Düzenleme zamanı (dışarıdan verilir).
        geometry: Parsel geometrisi.
        position: Satırın geometri indeksi.
        colors: Şiddet sınıfı renkleri.
        synthetic: ``True`` ise "DEMO · SENTETİK VERİ" bandı eklenir.

    Returns:
        Tam HTML belgesi.

    Raises:
        PermissionError: Satır ``owner`` müşterisine ait değilse.
        ValueError: Alan eksikse ya da karar/tutar motor çıktısıyla uyuşmuyorsa.
    """
    missing = [f for f in _REQUIRED if f not in row]
    if missing:
        raise ValueError(f"Satırda alan(lar) eksik: {', '.join(missing)}")
    if str(row["sigortali"]) != str(owner):
        raise PermissionError("Bu poliçe istenen müşteriye ait değil.")
    trace, pay = rep.verify(row, rules)     # tam satır gerekir: karar, ölçümlerden bağımsız yeniden hesaplanır
    decision = str(row["karar"])
    colour = rep.BADGE_COLORS[decision]
    steps = steps_for(decision, event_date, float(row["odenecek_tl"]), float(row["bekleyen_tl"]), pay.advance_pct)
    marks = {STEP_DONE: "✓", STEP_CURRENT: "●", STEP_WAITING: "○", STEP_CLOSED: "■"}
    step_html = "".join(
        f'<li class="{s.state}"><span class="dot">{marks[s.state]}</span><div><b>{_e(s.label)}</b>'
        f'<span class="d">{_e(s.detail)}</span></div></li>' for s in steps)
    policy_rows = "".join(f'<tr><td>{_e(a)}</td><td class="num">{_e(b)}</td></tr>' for a, b in [
        ("Poliçe no", row["police_no"]), ("Parsel", row["parsel_id"]), ("Sigortalı", row["sigortali"]),
        ("Müşteri no", customer_no(str(row["sigortali"]))), ("Varlık türü", row["varlik_turu"]),
        ("Alan", f'{format_tr(float(row["alan_ha"]), 2)} ha'), ("Sigorta bedeli", _tl(row["sigorta_bedeli_tl"])),
        ("Poliçe durumu", "Aktif" if bool(row["police_aktif"]) else "Pasif"),
        ("Uydu hasar sınıfı", row["siddet"]),
    ])
    if decision == "RET":
        pay_html = '<p class="note">Karar RET olduğundan ödeme hesaplanmaz (₺0).</p>'
    else:
        lines = [
            ("Sigorta bedeli", _tl(pay.insured)),
            (f"× teminat oranı (%{format_tr(pay.coverage_pct, 0)}) × hasar oranı (%{format_tr(pay.ratio * 100, 1)})",
             _tl(pay.gross)),
            (f"− muafiyet (%{format_tr(rules.deductible_pct, 1)})", _tl(pay.deductible)),
            ("= hesaplanan tazminat", _tl(pay.net)),
        ]
        if decision == "KISMİ ONAY":
            lines += [(f"Avans (%{format_tr(pay.advance_pct, 0)}) — ilk ödeme", _tl(pay.paid)),
                      ("Kalan — ekspertiz sonrası", _tl(pay.pending))]
        elif decision == "ONAY":
            lines += [("Ödenecek tutar", _tl(pay.paid))]
        else:
            lines += [("Şu an ödenecek", _tl(0)), ("Ekspertiz sonrası belirlenecek (en çok)", _tl(pay.pending))]
        body = "".join(f'<tr><td>{_e(a)}</td><td class="num">{_e(b)}</td></tr>' for a, b in lines)
        pay_html = f'<table class="t"><tr><th>Adım</th><th class="num">Tutar</th></tr>{body}</table>'
    code = hashlib.sha256("|".join([str(row["police_no"]), decision, f'{float(row["odenecek_tl"]):.0f}',
                                    f'{float(row["bekleyen_tl"]):.0f}', str(owner)]).encode("utf-8")).hexdigest()[:10].upper()
    band = ('<div class="band">DEMO · SENTETİK VERİ — Bu belge gerçek bir poliçe, parsel ya da hasar kaydı değildir; '
            'yalnızca belge biçimini göstermek içindir.</div>') if synthetic else ""
    svg = own_parcel_svg(geometry, position, colors[int(row["siddet_no"])], str(row["parsel_id"]))
    stamp = generated_at.strftime("%d.%m.%Y %H:%M")
    return f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hasar Karar Bildirimi — {_e(row["police_no"])}</title><style>{NOTICE_CSS}</style></head>
<body><div class="page">
<div class="noprint"><button type="button" onclick="window.print()">Yazdır / PDF olarak kaydet</button></div>
{band}
<div class="eyebrow">Hasar karar bildirimi</div>
<h1>Poliçe {_e(row["police_no"])}</h1>
<p class="sub">{_e(event_label)} · olay sonrası görüntü {event_date:%d.%m.%Y} · düzenleme {_e(stamp)}</p>

<div class="decision"><div><span class="badge" style="background:{colour}">{_e(STATUS_LABELS[decision])}</span></div>
<div style="flex:1"><div class="big">{_e(HEADLINES[decision])}</div>
<p class="note" style="font-size:13.5px;color:#14181f">{_e(REASON_TEXT[trace.reason_key])}</p>
<div class="kv"><div><span>HASAR ORANI</span><b>%{format_tr(float(row["hasar_yuzdesi"]), 1)}</b></div>
<div><span>ÖDENECEK</span><b>{_e(_tl(row["odenecek_tl"]))}</b></div>
<div><span>BEKLEYEN</span><b>{_e(_tl(row["bekleyen_tl"]))}</b></div></div></div></div>

<h2>1 · Poliçe ve parsel</h2>
<div class="grid2"><table class="t">{policy_rows}</table><div>{svg}<p class="note">Yalnızca size ait parsel gösterilir.</p></div></div>

<h2>2 · Tutar nasıl hesaplandı</h2>
{pay_html}

<h2>3 · Takip adımları</h2>
<ol class="steps">{step_html}</ol>
<p class="note">Adımlar karardan türetilir; ödeme ve ekspertiz tarihleri bu belgede yer almaz.</p>

<h2>4 · Karara itiraz ve bilgi</h2>
<p class="note">Kararla ilgili bilgi almak ya da itiraz etmek için sigortacınıza yazılı başvurabilirsiniz.
Bu belge karar destek çıktısıdır; nihai karar sigortacının yetkili birimine aittir.</p>
<p class="note"><b>Bildirim kodu:</b> <span class="mono">BLD-{code}</span></p>
<div class="foot">ORBITAL FIELD · Çok Yönlü Uydu ve Saha Analiz Platformu · Prototip</div>
</div></body></html>"""
