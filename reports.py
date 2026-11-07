"""
reports.py — Karar kanıt paketi
================================

Her otomatik tazminat kararı için, bir eksperin, müşterinin ya da denetçinin tek başına
okuyup yeniden hesaplayabileceği bağımsız bir belge (HTML; tarayıcıdan PDF olarak yazdırılır).

Bir paket şunları içerir:

* karar, gerekçe ve ödenecek/bekleyen tutar;
* uydu ölçümleri (NBR, dNBR, NDVI, gözlem güveni) ve USGS şiddet sınıfı;
* **kural izi**: kararın hangi kontrolden, hangi eşikle çıktığı (adım adım);
* **tutar hesabı**: sigorta bedeli → teminat → hasar oranı → muafiyet → avans;
* **karar sınırına uzaklık**: dNBR ya da güven ne kadar değişse karar değişir;
* parsel ve çevresinin konum çizimi (SVG; harici harita/ağ gerekmez);
* kural seti, veri kaynağı, kalibrasyon bilgisi ve içeriğin SHA-256 özeti;
* eksper onay/imza alanı.

Dürüstlük ilkeleri
------------------
* Kural izi, motordan BAĞIMSIZ olarak ölçümlerden yeniden yürütülür; sonucu motorun
  kararıyla (ve tutar hesabı motorun ödemesiyle) eşleşmezse belge ÜRETİLMEZ (``ValueError``).
* Demo/sentetik veriden üretilen her belge bunu görünür bir bantla belirtir.
* SHA-256 özeti yalnızca içeriğin sonradan değişmediğini göstermek içindir; kimlik doğrulama
  ya da elektronik imza DEĞİLDİR.
* Tüm metin ``html.escape`` ile kaçışlanır; belge dış kaynak yüklemez.

Bu modül Streamlit'e bağlı değildir.

Geliştirici: Elif Turut
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import html
import io
import json
import math
import re
import zipfile
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import data_engine as de
from data_engine import DECISION_LABELS, ClaimRules, format_tr

# --------------------------------------------------------------------------- #
# Sabitler
# --------------------------------------------------------------------------- #
MAX_PACK_FILES: int = 500
FRAGILE_DELTA: float = 0.03        # bu kadar (ya da daha az) değişimde karar değişiyorsa "sınırda"
NEIGHBOUR_RADIUS: int = 3          # konum çiziminde seçili parselin çevresindeki hücre yarıçapı
#: Açık (baskı) temada karar rozeti renkleri.
BADGE_COLORS: Dict[str, str] = {
    "ONAY": "#1f8a5b", "KISMİ ONAY": "#2563c9", "EKSPERTİZ": "#b7791f", "RET": "#c8321a",
}
_ASCII_FOLD = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")

REQUIRED_ROW_FIELDS: Tuple[str, ...] = (
    "parsel_id", "police_no", "sigortali", "varlik_turu", "alan_ha", "sigorta_bedeli_tl", "police_aktif",
    "ndvi_once", "ndvi_sonra", "nbr_once", "nbr_sonra", "dnbr", "siddet", "siddet_no", "guven",
    "risk_skoru", "risk_sinifi", "enlem", "boylam", "karar", "gerekce", "hasar_yuzdesi",
    "odenecek_tl", "bekleyen_tl",
)


def _e(value: object) -> str:
    """Değeri HTML için kaçışlar."""
    return html.escape(str(value), quote=True)


def _n(value: float, decimals: int = 0) -> str:
    """Türkçe sayı biçimi."""
    return format_tr(float(value), decimals)


def _tl(value: float) -> str:
    """Türk lirası tutarı."""
    return "₺" + format_tr(float(value), 0)


# --------------------------------------------------------------------------- #
# Veri sınıfları
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvidenceContext:
    """Bir paket kümesinin ortak bağlamı.

    Attributes:
        rules: Kararı üreten kural seti.
        event_label: Olay açıklaması.
        event_date: Olay sonrası görüntü tarihi.
        generated_at: Üretim zamanı (dışarıdan verilir; işlev saf kalır).
        severity_colors: dNBR şiddet sınıfı renkleri (0=yanmamış … 4=yüksek).
        synthetic: ``True`` ise belge "DEMO · SENTETİK VERİ" bandı taşır.
        provenance: Veri kaynağı açıklaması.
        calibration_lines: İsteğe bağlı kalibrasyon bilgisi satırları.
    """

    rules: ClaimRules
    event_label: str
    event_date: dt.date
    generated_at: dt.datetime
    severity_colors: Tuple[str, ...]
    synthetic: bool = True
    provenance: str = "Sentetik demo verisi (Sentinel-2 L2A benzetimi)"
    calibration_lines: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceStep:
    """Kural izinin tek adımı.

    Attributes:
        label: Kontrolün adı.
        observed: Ölçülen değer (metin).
        operator: Karşılaştırma işareti.
        threshold: Eşik (metin).
        passed: Geçti mi; ``None`` ise karar önceki adımda verildiği için değerlendirilmedi.
    """

    label: str
    observed: str
    operator: str
    threshold: str
    passed: Optional[bool]


@dataclass(frozen=True)
class Trace:
    """Bağımsız olarak yürütülmüş kural izi.

    Attributes:
        steps: Adımlar.
        decision: İzin vardığı karar.
        reason_key: Kararın nedeni (kısa kod).
    """

    steps: Tuple[TraceStep, ...]
    decision: str
    reason_key: str


@dataclass(frozen=True)
class Payment:
    """Tutar hesabının adımları (TL).

    Attributes:
        insured: Sigorta bedeli.
        coverage_pct: Teminat oranı (%).
        ratio: Hasar oranı (0–1).
        gross: Brüt tazminat.
        deductible: Muafiyet tutarı.
        net: Net tazminat.
        paid: Ödenecek (hemen).
        pending: Bekleyen.
        advance_pct: Avans oranı (%).
    """

    insured: float
    coverage_pct: float
    ratio: float
    gross: float
    deductible: float
    net: float
    paid: float
    pending: float
    advance_pct: float


@dataclass(frozen=True)
class FlipPoint:
    """Kararın değiştiği en küçük değişim.

    Attributes:
        delta: Ölçümdeki mutlak değişim (dNBR ya da güven birimiyle).
        to_decision: Yeni karar.
    """

    delta: float
    to_decision: str


# --------------------------------------------------------------------------- #
# Kural izi, tutar, sınır analizi
# --------------------------------------------------------------------------- #
def _check_row(row: Mapping[str, Any]) -> None:
    """Satırda gerekli alanların bulunduğunu doğrular."""
    missing = [f for f in REQUIRED_ROW_FIELDS if f not in row]
    if missing:
        raise ValueError(f"Karar satırında alan(lar) eksik: {', '.join(missing)}")


def build_trace(row: Mapping[str, Any], rules: ClaimRules) -> Trace:
    """Kararı ölçümlerden, motor kodunu çağırmadan adım adım yeniden yürütür.

    Sıra motorla aynıdır: pasif poliçe → ret eşiği → tam onay → kısmi onay → ekspertiz.

    Args:
        row: ``evaluate_claims`` çıktısından bir satır.
        rules: Kararı üreten kurallar.

    Returns:
        ``Trace``.
    """
    _check_row(row)
    dnbr, conf, active = float(row["dnbr"]), float(row["guven"]), bool(row["police_aktif"])
    trusted = conf >= rules.min_confidence
    steps: List[TraceStep] = []

    def add(label: str, observed: str, op: str, threshold: str, passed: Optional[bool]) -> None:
        steps.append(TraceStep(label, observed, op, threshold, passed))

    def skip_rest(items: Sequence[Tuple[str, str, str, str]]) -> None:
        for label, observed, op, threshold in items:
            add(label, observed, op, threshold, None)

    obs_dnbr, obs_conf = _n(dnbr, 3), _n(conf, 2)
    later_all = [
        ("dNBR ≥ ret eşiği (hasar var mı)", obs_dnbr, "≥", _n(rules.review_dnbr, 2)),
        ("dNBR ≥ tam onay eşiği", obs_dnbr, "≥", _n(rules.approve_dnbr, 2)),
        ("Gözlem güveni ≥ asgari güven", obs_conf, "≥", _n(rules.min_confidence, 2)),
        ("dNBR ≥ kısmi onay eşiği", obs_dnbr, "≥", _n(rules.partial_dnbr, 2)),
    ]
    add("Poliçe aktif", "Evet" if active else "Hayır", "=", "Evet", active)
    if not active:
        skip_rest(later_all)
        return Trace(tuple(steps), "RET", "inactive")

    has_damage = dnbr >= rules.review_dnbr
    add(later_all[0][0], obs_dnbr, "≥", _n(rules.review_dnbr, 2), has_damage)
    if not has_damage:
        skip_rest(later_all[1:])
        return Trace(tuple(steps), "RET", "below_review")

    is_high = dnbr >= rules.approve_dnbr
    add(later_all[1][0], obs_dnbr, "≥", _n(rules.approve_dnbr, 2), is_high)
    add(later_all[2][0], obs_conf, "≥", _n(rules.min_confidence, 2), trusted)
    if is_high and trusted:
        add(later_all[3][0], obs_dnbr, "≥", _n(rules.partial_dnbr, 2), None)
        return Trace(tuple(steps), "ONAY", "approve")

    is_mid = dnbr >= rules.partial_dnbr
    add(later_all[3][0], obs_dnbr, "≥", _n(rules.partial_dnbr, 2), is_mid)
    if is_mid and trusted:
        return Trace(tuple(steps), "KISMİ ONAY", "partial")
    return Trace(tuple(steps), "EKSPERTİZ", "low_confidence" if not trusted else "borderline")


def build_payment(row: Mapping[str, Any], rules: ClaimRules, decision: str) -> Payment:
    """Tutar hesabını motor formülünden bağımsız olarak adım adım yeniden hesaplar.

    Args:
        row: Karar satırı.
        rules: Kurallar.
        decision: ``build_trace`` kararı.

    Returns:
        ``Payment``.
    """
    insured = float(row["sigorta_bedeli_tl"])
    span = max(0.66 - rules.review_dnbr, 1e-6)
    ratio = min(max((float(row["dnbr"]) - rules.review_dnbr) / span, 0.0), 1.0)
    gross = insured * (rules.coverage_pct / 100.0) * ratio
    deductible = insured * (rules.deductible_pct / 100.0)
    net = max(gross - deductible, 0.0)
    if decision == "ONAY":
        paid, pending = net, 0.0
    elif decision == "KISMİ ONAY":
        paid = net * (rules.advance_pct / 100.0)
        pending = net - paid
    elif decision == "EKSPERTİZ":
        paid, pending = 0.0, net
    else:
        paid, pending = 0.0, 0.0
    return Payment(insured, rules.coverage_pct, ratio, gross, deductible, net, paid, pending, rules.advance_pct)


def flip_points(row: Mapping[str, Any], rules: ClaimRules, max_delta: float = 0.30) -> Dict[str, Optional[FlipPoint]]:
    """dNBR ya da güven ne kadar değişirse kararın değiştiğini bulur.

    Motorun gerçek karar işleviyle, 0,005 (dNBR) ve 0,01 (güven) adımlarıyla taranır.

    Args:
        row: Karar satırı.
        rules: Kurallar.
        max_delta: Taranacak en büyük değişim.

    Returns:
        ``dnbr_down``, ``dnbr_up``, ``guven_down``, ``guven_up`` anahtarlı sözlük (değişmiyorsa ``None``).
    """
    dnbr, conf, active = float(row["dnbr"]), float(row["guven"]), bool(row["police_aktif"])
    base = int(de.InsuranceSimulator.decide_codes(np.array([dnbr]), np.array([conf]), np.array([active]), rules)[0])
    out: Dict[str, Optional[FlipPoint]] = {}
    for axis, step in (("dnbr", 0.005), ("guven", 0.01)):
        deltas = np.arange(1, int(round(max_delta / step)) + 1) * step
        for sign, name in ((-1.0, "down"), (1.0, "up")):
            # 6 haneye yuvarlama: eşiğe tam eşit değerlerde kayan nokta gürültüsü kararı bozmasın
            moved = np.round(dnbr + sign * deltas if axis == "dnbr" else np.clip(conf + sign * deltas, 0.0, 1.0), 6)
            codes = de.InsuranceSimulator.decide_codes(
                moved if axis == "dnbr" else np.full_like(deltas, dnbr),
                np.full_like(deltas, conf) if axis == "dnbr" else moved,
                np.full(deltas.shape, active), rules)
            changed = np.flatnonzero(codes != base)
            out[f"{axis}_{name}"] = (
                FlipPoint(round(float(deltas[changed[0]]), 3), DECISION_LABELS[int(codes[changed[0]])])
                if changed.size else None)
    return out


def fragility(flips: Mapping[str, Optional[FlipPoint]]) -> Tuple[str, Optional[float]]:
    """Karar kırılganlığını özetler.

    Args:
        flips: ``flip_points`` çıktısı.

    Returns:
        ``("Sınırda" | "Sağlam" | "Etkisiz", en küçük değişim)``. ``Etkisiz``: ölçüm
        değişimi kararı hiç etkilemiyor (ör. pasif poliçe).
    """
    deltas = [f.delta for f in flips.values() if f is not None]
    if not deltas:
        return "Etkisiz", None
    smallest = min(deltas)
    return ("Sınırda" if smallest <= FRAGILE_DELTA + 1e-9 else "Sağlam"), smallest


def verify(row: Mapping[str, Any], rules: ClaimRules) -> Tuple[Trace, Payment]:
    """Kural izini ve tutarı yeniden hesaplar; motor çıktısıyla uyuşmazsa hata verir.

    Args:
        row: Karar satırı.
        rules: Kurallar.

    Returns:
        ``(Trace, Payment)``.

    Raises:
        ValueError: İz kararı ya da tutar hesabı motor çıktısından ±1 TL'den fazla ayrışırsa.
    """
    trace = build_trace(row, rules)
    if trace.decision != row["karar"]:
        raise ValueError(f"{row['police_no']}: kural izi '{trace.decision}' ancak motor '{row['karar']}' dedi.")
    payment = build_payment(row, rules, trace.decision)
    if abs(payment.paid - float(row["odenecek_tl"])) > 1.0 or abs(payment.pending - float(row["bekleyen_tl"])) > 1.0:
        raise ValueError(f"{row['police_no']}: tutar hesabı motor çıktısıyla uyuşmuyor.")
    return trace, payment


# --------------------------------------------------------------------------- #
# Özet kodu
# --------------------------------------------------------------------------- #
def _canonical(value: Any) -> Any:
    """JSON'a güvenli, kararlı (yuvarlanmış) değere çevirir."""
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return str(value)


def evidence_digest(row: Mapping[str, Any], ctx: EvidenceContext) -> str:
    """Kararın girdilerinden ve çıktılarından SHA-256 özeti üretir.

    Özet; ölçümleri, kural setini, kararı, tutarları, veri kaynağını ve olay tarihini kapsar.
    Üretim zamanını KAPSAMAZ: aynı girdi her zaman aynı özeti verir.

    Args:
        row: Karar satırı.
        ctx: Bağlam.

    Returns:
        64 karakterlik onaltılık özet.
    """
    _check_row(row)
    payload = {
        "alanlar": {f: row[f] for f in REQUIRED_ROW_FIELDS},
        "kurallar": asdict(ctx.rules),
        "olay_tarihi": ctx.event_date,
        "kaynak": ctx.provenance,
        "sentetik": ctx.synthetic,
    }
    text = json.dumps(_canonical(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# SVG çizimleri
# --------------------------------------------------------------------------- #
def threshold_ruler_svg(dnbr: float, rules: ClaimRules) -> str:
    """dNBR'nin karar eşiklerine göre konumunu gösteren yatay cetvel (SVG).

    Args:
        dnbr: Parselin dNBR değeri.
        rules: Kurallar.

    Returns:
        ``<svg>`` metni.
    """
    lo, hi, width, left, top, bar_h = -0.10, 0.80, 640.0, 12.0, 26.0, 18.0
    span_px = width - 2 * left

    def px(v: float) -> float:
        return left + (min(max(v, lo), hi) - lo) / (hi - lo) * span_px

    bands = [
        (lo, rules.review_dnbr, "#e7d9d4", "RET"),
        (rules.review_dnbr, rules.partial_dnbr, "#f1e3bd", "EKSPERTİZ"),
        (rules.partial_dnbr, rules.approve_dnbr, "#cfe0f7", "KISMİ"),
        (rules.approve_dnbr, hi, "#cdebdc", "ONAY"),
    ]
    parts = [f'<svg viewBox="0 0 {width:.0f} 84" width="100%" role="img" '
             f'aria-label="dNBR eşik cetveli">']
    for a, b, colour, text in bands:
        x0, x1 = px(a), px(b)
        parts.append(f'<rect x="{x0:.1f}" y="{top}" width="{max(x1 - x0, 0):.1f}" height="{bar_h}" fill="{colour}"/>')
        if x1 - x0 > 46:
            parts.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{top + 12.5}" text-anchor="middle" font-size="9.5" '
                         f'fill="#3b4350" font-weight="600">{_e(text)}</text>')
    for value in (rules.review_dnbr, rules.partial_dnbr, rules.approve_dnbr):
        x = px(value)
        parts.append(f'<line x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{top + bar_h + 4}" stroke="#5b6472" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + bar_h + 16}" text-anchor="middle" font-size="9.5" fill="#5b6472">{_e(_n(value, 2))}</text>')
    x = px(dnbr)
    parts.append(f'<path d="M{x - 6:.1f},{top - 12} L{x + 6:.1f},{top - 12} L{x:.1f},{top - 2} Z" fill="#14181f"/>')
    anchor = "end" if x > width - 70 else ("start" if x < 70 else "middle")
    parts.append(f'<text x="{x:.1f}" y="{top - 15}" text-anchor="{anchor}" font-size="10.5" fill="#14181f" '
                 f'font-weight="700">dNBR {_e(_n(dnbr, 3))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def parcel_context_svg(geometry: de.ParcelGeometry, frame: pd.DataFrame, position: int,
                       colors: Sequence[str], radius: int = NEIGHBOUR_RADIUS) -> str:
    """Seçili parseli ve çevresindeki parselleri şiddet rengiyle çizen SVG üretir.

    Harici harita kullanılmaz: yerel eşdikdörtgen izdüşümü (km) ile çokgenler doğrudan çizilir.

    Args:
        geometry: Parsel geometrisi.
        frame: Karar tablosu (``siddet_no``, ``parsel_id`` sütunlu; geometriyle aynı sıra).
        position: Seçili parselin geometri indeksi.
        colors: Şiddet sınıfı renkleri.
        radius: Çevre yarıçapı (hücre).

    Returns:
        ``<svg>`` metni.
    """
    r0, c0 = int(geometry.row[position]), int(geometry.col[position])
    near = np.flatnonzero((np.abs(geometry.row - r0) <= radius) & (np.abs(geometry.col - c0) <= radius))
    lat0, lon0 = float(geometry.centroid_lat[position]), float(geometry.centroid_lon[position])
    kx = math.cos(math.radians(lat0)) * de.KM_PER_DEG_LAT
    ky = de.KM_PER_DEG_LAT

    def project(la: np.ndarray, lo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (lo - lon0) * kx, -(la - lat0) * ky

    xs, ys = project(geometry.corners_lat[near], geometry.corners_lon[near])
    pad = 0.05
    x_min, x_max, y_min, y_max = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad
    width_px = 420.0
    height_px = width_px * (y_max - y_min) / max(x_max - x_min, 1e-9)

    def sx(v: float) -> float:
        return (v - x_min) / (x_max - x_min) * width_px

    def sy(v: float) -> float:
        return (v - y_min) / (y_max - y_min) * height_px

    parts = [f'<svg viewBox="0 0 {width_px:.0f} {height_px:.0f}" width="100%" role="img" '
             f'aria-label="Parsel ve çevresi">',
             f'<rect width="{width_px:.0f}" height="{height_px:.0f}" fill="#f4f5f7"/>']
    ordered = sorted(near.tolist(), key=lambda i: i == position)  # seçili en üstte
    for i in ordered:
        px_, py_ = project(geometry.corners_lat[i], geometry.corners_lon[i])
        points = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(px_, py_))
        severity = int(frame["siddet_no"].iloc[i])
        selected = i == position
        parts.append(
            f'<polygon points="{points}" fill="{_e(colors[severity])}" fill-opacity="{0.95 if selected else 0.62}" '
            f'stroke="{"#14181f" if selected else "#ffffff"}" stroke-width="{2.6 if selected else 0.8}"/>')
    cx, cy = sx(0.0), sy(0.0)
    parts.append(f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-size="11" font-weight="700" '
                 f'fill="#14181f" stroke="#ffffff" stroke-width="3" paint-order="stroke">'
                 f'{_e(frame["parsel_id"].iloc[position])}</text>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# HTML belge
# --------------------------------------------------------------------------- #
PACK_CSS = """
:root{--ink:#14181f;--muted:#5b6472;--line:#d9dde4;--bg:#eef0f3;--paper:#fff;--soft:#f6f7f9}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Inter,"Segoe UI","Helvetica Neue",Arial,sans-serif}
.page{max-width:820px;margin:18px auto;background:var(--paper);border:1px solid var(--line);padding:32px 40px 28px}
.band{padding:7px 12px;border:1px solid #e0b4ab;background:#fbeeeb;color:#8f2a17;font-size:12px;font-weight:600;
  letter-spacing:.02em;margin:0 0 18px}
.band.real{border-color:var(--line);background:var(--soft);color:var(--muted)}
.eyebrow{font-size:11px;letter-spacing:.14em;color:var(--muted);text-transform:uppercase}
h1{margin:4px 0 2px;font-size:24px;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 18px}
h2{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:26px 0 8px;
  padding-bottom:5px;border-bottom:1px solid var(--line)}
.decision{display:flex;gap:18px;align-items:stretch;border:1px solid var(--line);padding:16px 18px;background:var(--soft)}
.badge{display:inline-block;color:#fff;font-weight:700;letter-spacing:.04em;padding:6px 14px;font-size:15px}
.decision .big{font-size:16px;font-weight:650;line-height:1.4}
.kv{display:grid;grid-template-columns:repeat(3,1fr);gap:10px 18px;margin-top:12px}
.kv div span{display:block;font-size:11px;color:var(--muted);letter-spacing:.04em}
.kv div b{font-size:14.5px}
table.t{width:100%;border-collapse:collapse;font-size:13px}
table.t th{text-align:left;font-weight:600;color:var(--muted);font-size:11.5px;border-bottom:1px solid var(--ink);padding:5px 8px 5px 0}
table.t td{padding:6px 8px 6px 0;border-bottom:1px solid var(--line);vertical-align:top}
table.t td.num,table.t th.num{text-align:right;font-variant-numeric:tabular-nums;padding-right:0}
table.t td.gap,table.t th.gap{padding-left:16px}
.ok{color:#1f8a5b;font-weight:700}.no{color:#c8321a;font-weight:700}.na{color:#9aa2af}
.grid2{display:grid;grid-template-columns:1.1fr .9fr;gap:22px;align-items:start}
.legend{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:11px;color:var(--muted);margin-top:6px}
.legend i{display:inline-block;width:10px;height:10px;margin-right:5px;vertical-align:-1px}
.note{font-size:12px;color:var(--muted);margin:6px 0 0}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11.5px;word-break:break-all}
.sign{display:grid;grid-template-columns:1fr 1fr;gap:34px;margin-top:22px}
.sign div{border-top:1px solid var(--ink);padding-top:5px;font-size:12px;color:var(--muted);height:54px}
.foot{margin-top:24px;padding-top:10px;border-top:1px solid var(--line);font-size:11.5px;color:var(--muted)}
.noprint{margin:0 0 14px;text-align:right}
.noprint button{font:600 12px Inter,Arial,sans-serif;padding:6px 12px;border:1px solid var(--ink);background:#fff;cursor:pointer}
@page{size:A4;margin:13mm}
@media print{body{background:#fff}.page{border:0;margin:0;padding:0;max-width:none}.noprint{display:none}
  h2{break-after:avoid}table,.decision,.grid2{break-inside:avoid}}
@media (max-width:640px){.page{padding:20px 16px;margin:8px auto}.kv{grid-template-columns:1fr 1fr}.grid2{grid-template-columns:1fr}.decision{flex-direction:column;gap:12px}
  table.t{font-size:12px}table.t th,table.t td{padding-right:6px;overflow-wrap:anywhere}table.t td.gap,table.t th.gap{padding-left:8px}.sign{gap:18px}h1{font-size:21px}}
"""


def _reason_text(trace: Trace, rules: ClaimRules) -> str:
    """Kural izinden okunur gerekçe cümlesi üretir."""
    return {
        "inactive": "Poliçe dönem dışında ya da pasif olduğundan hasar ödemesi yapılmaz.",
        "below_review": f"Ölçülen dNBR, hasar (ret) eşiği olan {_n(rules.review_dnbr, 2)} değerinin altında.",
        "approve": "dNBR tam onay eşiğini aştı ve gözlem güveni yeterli; tam tazminat hesaplandı.",
        "partial": "dNBR kısmi onay eşiğini aştı ancak tam onay eşiğinin altında; avans ödenir, kalan ekspertiz sonrası belirlenir.",
        "low_confidence": "Gözlem güveni asgari eşiğin altında (bulut/gölge olasılığı); yerinde keşif gerekir.",
        "borderline": "Hasar var ancak onay eşiklerinin altında kalan sınır bir değer; eksper incelemesi gerekir.",
    }[trace.reason_key]


def _trace_table(trace: Trace) -> str:
    """Kural izi tablosunun HTML'i."""
    rows = []
    for i, step in enumerate(trace.steps, start=1):
        if step.passed is None:
            mark = '<span class="na">— değerlendirilmedi</span>'
        elif step.passed:
            mark = '<span class="ok">✓ sağlandı</span>'
        else:
            mark = '<span class="no">✕ sağlanmadı</span>'
        rows.append(f"<tr><td>{i}</td><td>{_e(step.label)}</td><td class=\"num\">{_e(step.observed)}</td>"
                    f"<td class=\"gap\">{_e(step.operator)}</td><td class=\"num\">{_e(step.threshold)}</td>"
                    f"<td class=\"gap\">{mark}</td></tr>")
    return ('<table class="t"><tr><th>#</th><th>Kontrol</th><th class="num">Ölçülen</th><th class="gap"></th>'
            '<th class="num">Eşik</th><th class="gap">Sonuç</th></tr>' + "".join(rows) + "</table>")


def _payment_table(pay: Payment, decision: str) -> str:
    """Tutar hesabı tablosunun HTML'i."""
    if decision == "RET":
        return '<p class="note">Karar RET olduğundan ödeme hesaplanmaz (₺0).</p>'
    lines = [
        ("Sigorta bedeli", _tl(pay.insured)),
        (f"× teminat oranı (%{_n(pay.coverage_pct, 0)}) × hasar oranı (%{_n(pay.ratio * 100, 1)})", _tl(pay.gross)),
        ("− muafiyet", _tl(pay.deductible)),
        ("= net tazminat", _tl(pay.net)),
    ]
    if decision == "KISMİ ONAY":
        lines += [(f"Avans (%{_n(pay.advance_pct, 0)}) — hemen ödenir", _tl(pay.paid)),
                  ("Kalan — ekspertiz sonrası", _tl(pay.pending))]
    elif decision == "ONAY":
        lines += [("Ödenecek (tam)", _tl(pay.paid))]
    else:
        lines += [("Ödenecek", _tl(0)), ("Bekleyen — ekspertiz sonrası belirlenir", _tl(pay.pending))]
    body = "".join(f'<tr><td>{_e(a)}</td><td class="num">{_e(b)}</td></tr>' for a, b in lines)
    return f'<table class="t"><tr><th>Adım</th><th class="num">Tutar</th></tr>{body}</table>'


def _flip_table(flips: Mapping[str, Optional[FlipPoint]]) -> str:
    """Karar sınırına uzaklık tablosunun HTML'i."""
    items = [
        ("dNBR düşerse", "dnbr_down", "−", 3), ("dNBR artarsa", "dnbr_up", "+", 3),
        ("Gözlem güveni düşerse", "guven_down", "−", 2), ("Gözlem güveni artarsa", "guven_up", "+", 2),
    ]
    rows = []
    for label, key, sign, decimals in items:
        flip = flips[key]
        if flip is None:
            rows.append(f'<tr><td>{_e(label)}</td><td class="na gap" colspan="2">±0,30 içinde karar değişmiyor</td></tr>')
        else:
            rows.append(f'<tr><td>{_e(label)}</td><td class="num">{sign}{_e(_n(flip.delta, decimals))}</td>'
                        f'<td class="gap">karar <b>{_e(flip.to_decision)}</b> olur</td></tr>')
    return ('<table class="t"><tr><th>Değişim</th><th class="num">En küçük etki eden fark</th><th class="gap"></th></tr>'
            + "".join(rows) + "</table>")


def render_claim_html(row: Mapping[str, Any], ctx: EvidenceContext, geometry: de.ParcelGeometry,
                      frame: pd.DataFrame, position: int) -> str:
    """Tek bir karar için bağımsız kanıt paketi HTML belgesi üretir.

    Args:
        row: ``evaluate_claims`` çıktısından bir satır.
        ctx: Ortak bağlam.
        geometry: Parsel geometrisi (konum çizimi için).
        frame: Tüm karar tablosu (çevre parseller için).
        position: Satırın geometri/tablo içindeki konumu.

    Returns:
        Tam HTML belgesi (``<!doctype html>`` ile).

    Raises:
        ValueError: Kural izi ya da tutar hesabı motor çıktısıyla uyuşmazsa.
    """
    trace, pay = verify(row, ctx.rules)
    flips = flip_points(row, ctx.rules)
    level, smallest = fragility(flips)
    digest = evidence_digest(row, ctx)
    decision = str(row["karar"])
    colour = BADGE_COLORS[decision]
    rules = ctx.rules

    if level == "Sınırda":
        fragile_text = f"Sınırda — {_n(smallest or 0, 3)} birimlik değişim kararı değiştirir"
    elif level == "Sağlam":
        fragile_text = f"Sağlam — ±{_n(FRAGILE_DELTA, 2)} içinde karar değişmez"
    else:
        fragile_text = "Ölçüm değişimi kararı etkilemez"

    band_cls, band_text = (
        ("band", "DEMO · SENTETİK VERİ — Bu belge gerçek bir poliçe, parsel ya da hasar kaydı değildir; "
                 "yalnızca belge biçimini göstermek içindir.")
        if ctx.synthetic else ("band real", f"Veri kaynağı: {ctx.provenance}"))

    legend = "".join(f'<span><i style="background:{_e(c)}"></i>{_e(label)}</span>'
                     for c, label in zip(ctx.severity_colors, de.SEVERITY_LABELS))
    measures = [
        ("NBR (olay öncesi)", _n(row["nbr_once"], 3)), ("NBR (olay sonrası)", _n(row["nbr_sonra"], 3)),
        ("dNBR = NBR öncesi − NBR sonrası", _n(row["dnbr"], 3)), ("NDVI (öncesi → sonrası)",
                                                                 f'{_n(row["ndvi_once"], 3)} → {_n(row["ndvi_sonra"], 3)}'),
        ("Gözlem güveni (bulut/gölge)", _n(row["guven"], 2)), ("USGS şiddet sınıfı", str(row["siddet"])),
        ("Poliçe risk skoru", f'{_n(row["risk_skoru"], 1)} / 100 ({row["risk_sinifi"]})'),
    ]
    measure_rows = "".join(f'<tr><td>{_e(a)}</td><td class="num">{_e(b)}</td></tr>' for a, b in measures)
    policy_rows = "".join(f'<tr><td>{_e(a)}</td><td class="num">{_e(b)}</td></tr>' for a, b in [
        ("Poliçe no", row["police_no"]), ("Parsel", row["parsel_id"]), ("Sigortalı", row["sigortali"]),
        ("Varlık türü", row["varlik_turu"]), ("Alan", f'{_n(row["alan_ha"], 2)} ha'),
        ("Sigorta bedeli", _tl(row["sigorta_bedeli_tl"])), ("Poliçe durumu", "Aktif" if row["police_aktif"] else "Pasif"),
        ("Konum", f'{_n(row["enlem"], 5)}, {_n(row["boylam"], 5)}'),
    ])
    rule_rows = "".join(f'<tr><td>{_e(a)}</td><td class="num">{_e(b)}</td></tr>' for a, b in [
        ("Tam onay eşiği (dNBR ≥)", _n(rules.approve_dnbr, 2)), ("Kısmi onay eşiği (dNBR ≥)", _n(rules.partial_dnbr, 2)),
        ("Ret eşiği (dNBR <)", _n(rules.review_dnbr, 2)), ("Asgari gözlem güveni", _n(rules.min_confidence, 2)),
        ("Muafiyet", f"%{_n(rules.deductible_pct, 1)}"), ("Teminat oranı", f"%{_n(rules.coverage_pct, 0)}"),
        ("Kısmi onayda avans", f"%{_n(rules.advance_pct, 0)}"),
    ])
    calib = "".join(f"<li>{_e(line)}</li>" for line in ctx.calibration_lines)
    calib_html = (f'<p class="note"><b>Kural seti kalibrasyonu</b></p><ul class="note">{calib}</ul>' if calib else
                  '<p class="note">Bu kural seti geçmiş kararlarla kalibre edilmemiştir (platform varsayılanı / elle ayarlı).</p>')
    stamp = ctx.generated_at.strftime("%d.%m.%Y %H:%M")

    return f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Karar Kanıt Paketi — {_e(row["police_no"])}</title><style>{PACK_CSS}</style></head>
<body><div class="page">
<div class="noprint"><button type="button" onclick="window.print()">Yazdır / PDF olarak kaydet</button></div>
<div class="{band_cls}">{_e(band_text)}</div>
<div class="eyebrow">Karar kanıt paketi · Otomatik tazminat kararı</div>
<h1>Poliçe {_e(row["police_no"])}</h1>
<p class="sub">{_e(ctx.event_label)} · olay sonrası görüntü {ctx.event_date:%d.%m.%Y} · düzenleme {_e(stamp)}</p>

<div class="decision"><div><span class="badge" style="background:{colour}">{_e(decision)}</span></div>
<div style="flex:1"><div class="big">{_e(_reason_text(trace, rules))}</div>
<div class="kv"><div><span>HASAR ORANI</span><b>%{_n(row["hasar_yuzdesi"], 1)}</b></div>
<div><span>ÖDENECEK</span><b>{_e(_tl(row["odenecek_tl"]))}</b></div>
<div><span>BEKLEYEN</span><b>{_e(_tl(row["bekleyen_tl"]))}</b></div>
<div style="grid-column:1/-1"><span>KARARIN KIRILGANLIĞI</span><b>{_e(fragile_text)}</b></div></div></div></div>

<h2>1 · Poliçe ve parsel</h2>
<div class="grid2"><table class="t">{policy_rows}</table>
<div>{parcel_context_svg(geometry, frame, position, ctx.severity_colors)}<div class="legend">{legend}</div>
<p class="note">Seçili parsel kalın çerçeveli; çevresindeki parseller şiddet rengiyle gösterilir.</p></div></div>

<h2>2 · Uydu ölçümleri</h2>
<table class="t">{measure_rows}</table>
{threshold_ruler_svg(float(row["dnbr"]), rules)}
<p class="note">Bantlar yalnızca dNBR eşiklerini gösterir; onay için gözlem güveni eşiği de sağlanmalıdır.</p>

<h2>3 · Kural izi — karar nasıl çıktı</h2>
{_trace_table(trace)}
<p class="note">İz, ölçümlerden bağımsız olarak yeniden yürütülmüş ve motor kararıyla eşleşmiştir.</p>

<h2>4 · Tutar hesabı</h2>
{_payment_table(pay, decision)}

<h2>5 · Karar sınırına uzaklık</h2>
{_flip_table(flips)}
<p class="note">Motorun gerçek karar işleviyle taranmıştır (dNBR adımı 0,005; güven adımı 0,01; en çok ±0,30).</p>

<h2>6 · Kural seti, kaynak ve doğrulama</h2>
<div class="grid2"><table class="t">{rule_rows}</table>
<div>{calib_html}<p class="note"><b>Veri kaynağı:</b> {_e(ctx.provenance)}</p>
<p class="note"><b>Yöntem:</b> dNBR = NBR<sub>öncesi</sub> − NBR<sub>sonrası</sub>; şiddet sınıfları USGS FIREMON eşikleriyle.</p></div></div>
<p class="note"><b>İçerik özeti (SHA-256)</b></p><p class="mono">{digest}</p>
<p class="note">Özet; bu belgedeki ölçümleri, kural setini, kararı ve tutarları kapsar (üretim zamanı hariç). İçeriğin sonradan
değişmediğini göstermek içindir; kimlik doğrulama ya da elektronik imza değildir.</p>

<h2>7 · Onay</h2>
<p class="note">Bu belge karar destek çıktısıdır; nihai karar yetkili kişiye aittir.</p>
<div class="sign"><div>Eksper / yetkili adı, imza</div><div>Tarih</div></div>
<div class="foot">NYXE · ORBITAL FIELD · Çok Yönlü Uydu ve Saha Analiz Platformu · Prototip · Kanıt kimliği {digest[:12]}</div>
</div></body></html>"""


# --------------------------------------------------------------------------- #
# Toplu paket (ZIP)
# --------------------------------------------------------------------------- #
def claim_filename(row: Mapping[str, Any]) -> str:
    """Belge için ASCII-güvenli dosya adı üretir (ör. ``kanit_POL-2026-10007_ONAY.html``).

    Args:
        row: Karar satırı.

    Returns:
        Dosya adı.
    """
    safe_no = re.sub(r"[^A-Za-z0-9_-]+", "-", str(row["police_no"]).translate(_ASCII_FOLD))
    safe_dec = re.sub(r"[^A-Za-z0-9_-]+", "-", str(row["karar"]).translate(_ASCII_FOLD))
    return f"kanit_{safe_no}_{safe_dec}.html"


def _positions(claims: pd.DataFrame) -> Dict[str, int]:
    """``parsel_id`` → tablo/geometri konumu eşlemesi."""
    ids = claims["parsel_id"].tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("parsel_id değerleri benzersiz değil.")
    return {pid: i for i, pid in enumerate(ids)}


def build_claim_document(claims: pd.DataFrame, police_no: str, ctx: EvidenceContext,
                         geometry: de.ParcelGeometry) -> Tuple[str, str]:
    """Poliçe numarasına göre tek belge üretir.

    Args:
        claims: ``evaluate_claims`` çıktısı (geometriyle aynı sıra).
        police_no: İstenen poliçe.
        ctx: Bağlam.
        geometry: Parsel geometrisi.

    Returns:
        ``(dosya adı, html)``.

    Raises:
        KeyError: Poliçe bulunamazsa.
        ValueError: Tablo ile geometri uyuşmazsa ya da doğrulama başarısızsa.
    """
    if len(claims) != geometry.count:
        raise ValueError("Karar tablosu satır sayısı parsel sayısıyla uyuşmuyor.")
    hit = np.flatnonzero((claims["police_no"] == police_no).to_numpy())
    if hit.size == 0:
        raise KeyError(f"Poliçe bulunamadı: {police_no}")
    pos = int(hit[0])
    row = claims.iloc[pos]
    return claim_filename(row), render_claim_html(row, ctx, geometry, claims, pos)


def build_pack_zip(claims: pd.DataFrame, police_nos: Sequence[str], ctx: EvidenceContext,
                   geometry: de.ParcelGeometry) -> bytes:
    """Seçili poliçeler için ZIP paket üretir: her karar için HTML + ``ozet.csv`` + ``manifest.json``.

    ZIP içindeki zaman damgaları ``ctx.generated_at`` değerinden alınır; aynı girdi aynı
    baytları verir.

    Args:
        claims: ``evaluate_claims`` çıktısı (geometriyle aynı sıra).
        police_nos: Pakete girecek poliçeler (``MAX_PACK_FILES`` ile sınırlı).
        ctx: Bağlam.
        geometry: Parsel geometrisi.

    Returns:
        ZIP baytları.

    Raises:
        ValueError: Liste boşsa ya da ``MAX_PACK_FILES`` aşılırsa.
    """
    wanted = list(dict.fromkeys(police_nos))
    if not wanted:
        raise ValueError("Pakete eklenecek poliçe yok.")
    if len(wanted) > MAX_PACK_FILES:
        raise ValueError(f"Bir pakette en çok {MAX_PACK_FILES} belge olabilir (istenen: {len(wanted)}).")
    stamp = (max(ctx.generated_at.year, 1980), ctx.generated_at.month, ctx.generated_at.day,
             ctx.generated_at.hour, ctx.generated_at.minute, ctx.generated_at.second)
    entries: List[Dict[str, Any]] = []
    buffer = io.BytesIO()
    summary = io.StringIO()
    writer = csv.writer(summary, delimiter=";", lineterminator="\n")
    writer.writerow(["police_no", "sigortali", "karar", "dnbr", "guven", "odenecek_tl", "bekleyen_tl",
                     "kirilganlik", "dosya", "sha256"])
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        def put(name: str, data: bytes) -> None:
            info = zipfile.ZipInfo(name, date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)

        for police_no in wanted:
            name, document = build_claim_document(claims, police_no, ctx, geometry)
            row = claims.loc[claims["police_no"] == police_no].iloc[0]
            level, _ = fragility(flip_points(row, ctx.rules))
            digest = evidence_digest(row, ctx)
            put(name, document.encode("utf-8"))
            writer.writerow([police_no, row["sigortali"], row["karar"], _n(row["dnbr"], 3), _n(row["guven"], 2),
                             f'{float(row["odenecek_tl"]):.0f}', f'{float(row["bekleyen_tl"]):.0f}', level, name, digest])
            entries.append({"dosya": name, "poliçe": police_no, "karar": str(row["karar"]), "sha256": digest,
                            "belge_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest()})
        manifest = {
            "olusturma": ctx.generated_at.isoformat(), "adet": len(entries), "kaynak": ctx.provenance,
            "sentetik": ctx.synthetic, "olay_tarihi": ctx.event_date.isoformat(),
            "kural_seti": _canonical(asdict(ctx.rules)), "kalibrasyon": list(ctx.calibration_lines),
            "dosyalar": entries,
        }
        put("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
        put("ozet.csv", ("﻿" + summary.getvalue()).encode("utf-8"))
        readme = ("KARAR KANIT PAKETİ\n"
                  "==================\n"
                  "Her .html dosyası tek bir poliçenin bağımsız kanıt belgesidir; tarayıcıda açıp\n"
                  "'Yazdır / PDF olarak kaydet' ile PDF'e çevirebilirsiniz.\n"
                  "manifest.json: kural seti, veri kaynağı ve her belgenin SHA-256 özeti.\n"
                  "ozet.csv: tüm kararların tek tablosu.\n"
                  + ("\nDİKKAT: Bu paket DEMO · SENTETİK veriden üretilmiştir; gerçek kayıt değildir.\n" if ctx.synthetic else ""))
        put("OKUBENI.txt", readme.encode("utf-8"))
    return buffer.getvalue()


__all__ = [
    "BADGE_COLORS", "EvidenceContext", "FlipPoint", "MAX_PACK_FILES", "Payment", "Trace", "TraceStep",
    "build_claim_document", "build_pack_zip", "build_payment", "build_trace", "claim_filename",
    "evidence_digest", "flip_points", "fragility", "parcel_context_svg", "render_claim_html",
    "threshold_ruler_svg", "verify",
]
