"""
workflow_report.py — Analiz çıktısından PDF rapor (ReportLab)
=============================================================

Bir ``workflow.AnalysisRun``'dan, sigorta hasar raporu ya da tarım gelişim raporu üretir. Rapor şunları içerir: parsel koordinatları,
önce/sonra gerçek renk görüntüleri ve değişim haritası, indeks zaman serisi grafiği, sınıf alanları, güvenilirlik göstergeleri, hava ve
yangın noktası kanıtı, (tarımda) rekolte tahmini, yöntem ve sınırlar, uzman kararı ile **imza ve tarih alanları**; isteğe bağlı saha fotoğrafı eki.

Türkçe karakterler için paketle gelen DejaVu Sans yazı tipi (``fonts/``) gömülür; bulunamazsa Helvetica'ya ve ASCII dönüşümüne düşülür.

Dürüstlük notları (rapora da yazılır): sınıflandırma eşikleri literatür sınırlarına hizalı politika seçimleridir ve yerel olarak doğrulanmamıştır;
güvenilirlik puanı bir veri kalitesi göstergesidir; hava verisi ERA5 tabanlı yeniden-analizdir (istasyon ölçümü değil); belge, imzalanana kadar
bir karar belgesi değildir.

Geliştirici: Elif Turut
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape as _xml_escape

import numpy as np
import pandas as pd
from PIL import Image
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus import Image as RLImage

import analysis as an
import thresholds as th
import workflow as wf

__all__ = ["ReportError", "FieldPhoto", "YieldSummary", "ReportMeta", "REPORT_TYPES", "build_pdf", "prepare_photo", "haversine_m"]

REPORT_TYPES: Tuple[str, ...] = ("sigorta", "tarim")
MAX_PHOTO_BYTES = 10 * 1024 * 1024
MAX_PHOTO_PIXELS = 40_000_000
MAX_TEXT = 200
MAX_NOTE = 1500
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

C_INK = colors.HexColor("#1b2330")
C_MUTE = colors.HexColor("#5b6678")
C_LINE = colors.HexColor("#c9d0db")
C_HEAD = colors.HexColor("#1f3a5f")
C_SOFT = colors.HexColor("#f2f4f8")
C_WARN = colors.HexColor("#b36b00")
C_BAD = colors.HexColor("#b3261e")
C_OK = colors.HexColor("#1b7f4c")

_TR_ASCII = str.maketrans({"ğ": "g", "Ğ": "G", "ş": "s", "Ş": "S", "ı": "i", "İ": "I", "ö": "o", "Ö": "O", "ü": "u", "Ü": "U", "ç": "c", "Ç": "C",
                           "₺": "TL", "→": "->", "≥": ">=", "≤": "<=", "−": "-", "—": "-", "–": "-", "·": "-", "✓": "+", "✗": "x", "Δ": "d", "°": " ", "²": "2"})


class ReportError(ValueError):
    """Rapor girdisi geçersiz (ileti kullanıcıya gösterilebilir)."""


@dataclass(frozen=True)
class FieldPhoto:
    """Uzmanın yüklediği saha fotoğrafı ve elle girdiği konum.

    Attributes:
        data: Görüntü baytları (JPEG/PNG); ``prepare_photo`` ile doğrulanıp küçültülür.
        lat: Enlem (°).
        lon: Boylam (°).
        caption: Açıklama.
        taken: Çekim zamanı (serbest metin; boş olabilir).
    """

    data: bytes
    lat: float
    lon: float
    caption: str = ""
    taken: str = ""


@dataclass(frozen=True)
class YieldSummary:
    """Rapora giren rekolte tahmini özeti.

    Attributes:
        crop: Ürün.
        stage_label: Fenolojik evre adı.
        value: Tahmin (t/ha).
        lo: Alt sınır (t/ha).
        hi: Üst sınır (t/ha).
        level: Aralık düzeyi (0–1).
        prior_yield: Geçmiş verim girdisi (t/ha).
        loso_rmse: Sezon-dışı RMSE (t/ha).
        loso_mape: Sezon-dışı MAPE (%).
        synthetic: Model sentetik veriyle mi eğitildi.
        warnings: Uyarılar.
    """

    crop: str
    stage_label: str
    value: float
    lo: float
    hi: float
    level: float
    prior_yield: float
    loso_rmse: float
    loso_mape: float
    synthetic: bool
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportMeta:
    """Raporun kullanıcı girdileri.

    Attributes:
        report_type: ``sigorta`` ya da ``tarim``.
        parcel_id: Parsel kimliği.
        policy_no: Poliçe numarası (sigorta).
        insured_name: Sigortalı / üretici adı (isteğe bağlı).
        expert_name: Ekspertiz uzmanı adı.
        expert_reg_no: Uzman sicil / ruhsat numarası.
        expert_decision: Uzman kararı (serbest metin; boşsa "beklemede").
        expert_note: Uzman notu.
        field_photo: Saha fotoğrafı eki.
        yield_summary: Rekolte özeti (tarım raporu).
        generated_at: Oluşturma zamanı (``None`` = şimdi, UTC).
    """

    report_type: str
    parcel_id: str = ""
    policy_no: str = ""
    insured_name: str = ""
    expert_name: str = ""
    expert_reg_no: str = ""
    expert_decision: str = ""
    expert_note: str = ""
    field_photo: Optional[FieldPhoto] = None
    yield_summary: Optional[YieldSummary] = None
    generated_at: Optional[dt.datetime] = None


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """İki nokta arası büyük daire uzaklığı (m)."""
    r = 6_371_008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def prepare_photo(raw: bytes, max_side: int = 1600) -> bytes:
    """Fotoğrafı doğrular, yönünü düzeltir, küçültür ve JPEG olarak yeniden kodlar (EXIF/GPS verisi atılır).

    Args:
        raw: Yüklenen dosya baytları.
        max_side: En uzun kenar (piksel).

    Raises:
        ReportError: Dosya görüntü değilse, çok büyükse ya da bozuksa.
    """
    from PIL import ImageOps

    if not raw:
        raise ReportError("Fotoğraf dosyası boş.")
    if len(raw) > MAX_PHOTO_BYTES:
        raise ReportError(f"Fotoğraf en çok {MAX_PHOTO_BYTES // (1024 * 1024)} MB olabilir.")
    try:
        probe = Image.open(io.BytesIO(raw))
        if probe.format not in ("JPEG", "PNG"):
            raise ReportError("Yalnızca JPEG ve PNG fotoğraflar desteklenir.")
        if probe.width * probe.height > MAX_PHOTO_PIXELS:
            raise ReportError("Fotoğraf çözünürlüğü çok yüksek.")
        probe.verify()
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except ReportError:
        raise
    except Exception:  # noqa: BLE001 — bozuk/desteklenmeyen dosya
        raise ReportError("Fotoğraf okunamadı (bozuk ya da desteklenmeyen dosya).") from None
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85, optimize=True)
    return out.getvalue()


def _png(arr: np.ndarray, min_width: int = 480) -> bytes:
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    img = Image.fromarray(a, "RGBA" if a.shape[2] == 4 else "RGB")
    if img.width < min_width:
        k = int(math.ceil(min_width / img.width))
        img = img.resize((img.width * k, img.height * k), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _clean(text: str, limit: int) -> str:
    t = " ".join(str(text or "").replace("\x00", " ").split()) if limit <= MAX_TEXT else str(text or "").replace("\x00", " ").strip()
    return t[:limit]


class _Fonts:
    """Yazı tipi kaydı (bir kez)."""

    regular = "Helvetica"
    bold = "Helvetica-Bold"
    unicode_ok = False
    _done = False

    @classmethod
    def setup(cls) -> None:
        if cls._done:
            return
        cands = [FONT_DIR]
        try:
            import matplotlib  # yalnızca yedek yazı tipi dizini için

            cands.append(os.path.join(matplotlib.get_data_path(), "fonts", "ttf"))
        except Exception:  # noqa: BLE001
            pass
        for d in cands:
            reg, bold = os.path.join(d, "DejaVuSans.ttf"), os.path.join(d, "DejaVuSans-Bold.ttf")
            if os.path.isfile(reg) and os.path.isfile(bold):
                pdfmetrics.registerFont(TTFont("DejaVu", reg))
                pdfmetrics.registerFont(TTFont("DejaVu-Bold", bold))
                registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu", boldItalic="DejaVu-Bold")
                cls.regular, cls.bold, cls.unicode_ok = "DejaVu", "DejaVu-Bold", True
                break
        cls._done = True


def _t(text: object) -> str:
    """Yazı tipi Türkçe karakter taşımıyorsa ASCII'ye indirger; taşıyorsa olduğu gibi bırakır."""
    s = str(text)
    return s if _Fonts.unicode_ok else s.translate(_TR_ASCII)


def _e(text: object) -> str:
    """Paragraph işaretlemesi için kaçışlı metin."""
    return _xml_escape(_t(text))


def _fmt(x: float, d: int = 1) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def _styles() -> Dict[str, ParagraphStyle]:
    f, fb = _Fonts.regular, _Fonts.bold
    base = ParagraphStyle("base", fontName=f, fontSize=8.6, leading=11.8, textColor=C_INK)
    return {
        "base": base,
        "small": ParagraphStyle("small", parent=base, fontSize=7.4, leading=9.8, textColor=C_MUTE),
        "h1": ParagraphStyle("h1", parent=base, fontName=fb, fontSize=17, leading=21, textColor=C_HEAD, spaceAfter=2),
        "h2": ParagraphStyle("h2", parent=base, fontName=fb, fontSize=10.5, leading=14, textColor=C_HEAD, spaceBefore=11, spaceAfter=4),
        "sub": ParagraphStyle("sub", parent=base, fontSize=9.4, textColor=C_MUTE, spaceAfter=6),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8.2, leading=10.6),
        "cellb": ParagraphStyle("cellb", parent=base, fontName=fb, fontSize=8.2, leading=10.6),
        "banner": ParagraphStyle("banner", parent=base, fontName=fb, fontSize=8.6, leading=11.4, textColor=C_BAD, borderColor=C_BAD, borderWidth=0.8,
                                 borderPadding=5, spaceAfter=8, spaceBefore=2),
        "warn": ParagraphStyle("warn", parent=base, fontSize=8.2, textColor=C_WARN),
    }


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def _kv_table(rows: Sequence[Tuple[str, str]], st: Dict[str, ParagraphStyle], widths: Tuple[float, float]) -> Table:
    data = [[_p(f"<b>{_e(k)}</b>", st["cell"]), _p(_e(v), st["cell"])] for k, v in rows]
    t = Table(data, colWidths=list(widths))
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, C_LINE), ("BACKGROUND", (0, 0), (0, -1), C_SOFT),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def _grid_table(head: Sequence[str], rows: Sequence[Sequence[str]], st: Dict[str, ParagraphStyle], widths: Sequence[float], right: Sequence[int] = ()) -> Table:
    right_style = ParagraphStyle("r", parent=st["cell"], alignment=2)
    right_head = ParagraphStyle("rh", parent=st["cellb"], alignment=2)
    data: List[List[Any]] = [[_p(_e(h), right_head if i in right else st["cellb"]) for i, h in enumerate(head)]]
    for r in rows:
        data.append([_p(_e(c), right_style if i in right else st["cell"]) for i, c in enumerate(r)])
    t = Table(data, colWidths=list(widths), repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), C_SOFT), ("LINEBELOW", (0, 0), (-1, -1), 0.4, C_LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
    return t


# --------------------------------------------------------------------------- #
# Grafik (yerel çizim)
# --------------------------------------------------------------------------- #
def _series_chart(series: pd.DataFrame, index_label: str, event: Optional[dt.date], pre: dt.date, post: dt.date, width: float, height: float) -> Drawing:
    """İndeks zaman serisi: ortalama (mavi), medyan (yeşil), %2–%98 dilimleri (gri kesikli), olay (kırmızı), önce/sonra işaretleri."""
    d = Drawing(width, height)
    s = series.dropna(subset=["ortalama"])
    f = _Fonts.regular
    if s.empty:
        d.add(String(8, height / 2, _t("Ölçülebilir sahne yok"), fontName=f, fontSize=8, fillColor=C_MUTE))
        return d
    left, right, top, bottom = 34.0, 10.0, 10.0, 24.0
    pw, ph = width - left - right, height - top - bottom
    t0, t1 = s["tarih"].min(), s["tarih"].max()
    for x in (event, pre, post):
        if x is not None:
            t0, t1 = min(t0, pd.Timestamp(x)), max(t1, pd.Timestamp(x))
    span = max((t1 - t0).days, 1)
    vals = pd.concat([s["minimum"], s["maksimum"], s["ortalama"], s["medyan"]]).dropna()
    lo, hi = float(vals.min()), float(vals.max())
    pad = max((hi - lo) * 0.06, 0.02)
    lo, hi = lo - pad, hi + pad

    def X(t: Any) -> float:
        return left + pw * ((pd.Timestamp(t) - t0).days / span)

    def Y(v: float) -> float:
        return bottom + ph * (v - lo) / (hi - lo)

    d.add(Rect(left, bottom, pw, ph, strokeColor=C_LINE, fillColor=None, strokeWidth=0.5))
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        d.add(Line(left, Y(v), left + pw, Y(v), strokeColor=C_LINE, strokeWidth=0.25))
        d.add(String(left - 3, Y(v) - 2.5, _fmt(v, 2), fontName=f, fontSize=6.2, fillColor=C_MUTE, textAnchor="end"))
    for k in range(5):
        t = t0 + pd.Timedelta(days=round(span * k / 4))
        d.add(String(X(t), bottom - 10, f"{t:%d.%m.%y}", fontName=f, fontSize=6.2, fillColor=C_MUTE, textAnchor="middle"))

    def line(key: str, col: Any, w: float, dash: Optional[List[float]] = None) -> None:
        pts: List[float] = []
        for _, r in s.iterrows():
            if math.isfinite(r[key]):
                pts += [X(r["tarih"]), Y(r[key])]
        if len(pts) >= 4:
            d.add(PolyLine(pts, strokeColor=col, strokeWidth=w, strokeDashArray=dash))

    line("maksimum", colors.HexColor("#8a94a6"), 0.5, [2, 2])
    line("minimum", colors.HexColor("#8a94a6"), 0.5, [2, 2])
    line("medyan", C_OK, 0.9)
    line("ortalama", colors.HexColor("#2563c9"), 1.3)
    for _, r in s.iterrows():
        d.add(Circle(X(r["tarih"]), Y(r["ortalama"]), 1.4, fillColor=colors.HexColor("#2563c9"), strokeColor=None))
    if event is not None:
        x = X(event)
        d.add(Line(x, bottom, x, bottom + ph, strokeColor=C_BAD, strokeWidth=0.8, strokeDashArray=[3, 2]))
        d.add(String(x + 2, bottom + ph - 7, _t(f"olay {event:%d.%m}"), fontName=f, fontSize=6.2, fillColor=C_BAD))
    for dd, lab, col in ((pre, "önce", C_WARN), (post, "sonra", C_OK)):
        row = s[s["tarih"] == pd.Timestamp(dd)]
        if not row.empty:
            x, y = X(dd), Y(float(row["ortalama"].iloc[0]))
            d.add(Circle(x, y, 4, fillColor=None, strokeColor=col, strokeWidth=1))
            d.add(String(x, y + 6, _t(lab), fontName=_Fonts.bold, fontSize=6.2, fillColor=col, textAnchor="middle"))
    d.add(String(left, height - 8, _t(index_label), fontName=_Fonts.bold, fontSize=6.8, fillColor=C_MUTE))
    lx = left + 60
    for col, name in ((colors.HexColor("#2563c9"), "ortalama"), (C_OK, "medyan"), (colors.HexColor("#8a94a6"), "%2–%98")):
        d.add(Line(lx, height - 6, lx + 10, height - 6, strokeColor=col, strokeWidth=1.2))
        d.add(String(lx + 13, height - 8.2, _t(name), fontName=f, fontSize=6.2, fillColor=C_MUTE))
        lx += 52
    return d


def _change_map_png(res: an.ChangeResult, base_rgb: Optional[np.ndarray]) -> bytes:
    """Değişim sınıflarını (varsa gerçek renkli görüntü üstüne) boyar; bölge dışı soluk bırakılır."""
    ov = an.colorize(res.delta, 0.0, 1.0, mask=res.region_mask, classes=[c[1] for c in res.kind.classes], class_idx=res.cls, alpha=215)
    if base_rgb is not None and base_rgb.shape[:2] == ov.shape[:2]:
        base = Image.fromarray(np.asarray(base_rgb, dtype=np.uint8), "RGB").convert("RGBA")
    else:
        base = Image.new("RGBA", (ov.shape[1], ov.shape[0]), (205, 210, 218, 255))
    comp = Image.alpha_composite(base, Image.fromarray(ov, "RGBA")).convert("RGB")
    return _png(np.asarray(comp), 520)


# --------------------------------------------------------------------------- #
# Sayfa düzeni
# --------------------------------------------------------------------------- #
def _page_deco(report_id: str, demo: bool, title: str):
    def draw(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        w, h = A4
        canvas.setFillColor(C_HEAD)
        canvas.rect(0, h - 11 * mm, w, 11 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont(_Fonts.bold, 8.5)
        canvas.drawString(15 * mm, h - 7.2 * mm, "ORBITAL FIELD")
        canvas.setFont(_Fonts.regular, 8)
        canvas.drawRightString(w - 15 * mm, h - 7.2 * mm, _t(title))
        if demo:
            canvas.setFillColor(C_BAD)
            canvas.setFont(_Fonts.bold, 7.4)
            canvas.drawCentredString(w / 2, h - 16 * mm, _t("SENTETİK DEMO VERİ — GERÇEK BİR OLAYI TEMSİL ETMEZ"))
        canvas.setStrokeColor(C_LINE)
        canvas.line(15 * mm, 12 * mm, w - 15 * mm, 12 * mm)
        canvas.setFillColor(C_MUTE)
        canvas.setFont(_Fonts.regular, 7)
        canvas.drawString(15 * mm, 8 * mm, _t(f"Rapor no: {report_id}  ·  Geliştirici: Elif Turut"))
        canvas.drawRightString(w - 15 * mm, 8 * mm, _t(f"Sayfa {doc.page}"))
        canvas.restoreState()

    return draw


def _signature_block(st: Dict[str, ParagraphStyle], width: float, expert_name: str, reg_no: str) -> Table:
    """Boş imza ve tarih alanları (elle doldurulur)."""
    def cell(title: str, name: str = "", sub: str = "") -> List[Any]:
        return [_p(f"<b>{_e(title)}</b>", st["cell"]), Spacer(1, 22), _p(_e(name) if name else "&nbsp;", st["cell"]),
                _p(_e(sub), st["small"])]

    half = width / 2
    t = Table([[cell("Ekspertiz uzmanı", expert_name, f"Sicil / ruhsat no: {reg_no or '____________'}"), cell("Şirket yetkilisi onayı", "", "Ad Soyad / Unvan: ______________")],
               [_p(_e("İmza: ____________________   Tarih: ___ / ___ / ______"), st["cell"]), _p(_e("İmza: ____________________   Tarih: ___ / ___ / ______"), st["cell"])]],
              colWidths=[half, half])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, C_LINE), ("LINEAFTER", (0, 0), (0, -1), 0.6, C_LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 7), ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    return t


def _report_id(run: wf.AnalysisRun, meta: ReportMeta, when: dt.datetime) -> str:
    h = hashlib.sha256(repr((run.request.kind, run.request.region.ring, run.request.event_date, meta.parcel_id, when.isoformat())).encode()).hexdigest()[:6].upper()
    return f"OF-{when:%Y%m%d}-{h}"


def _validate(meta: ReportMeta) -> ReportMeta:
    if meta.report_type not in REPORT_TYPES:
        raise ReportError(f"Rapor türü {REPORT_TYPES} içinden olmalı.")
    ph = meta.field_photo
    if ph is not None:
        if not (math.isfinite(ph.lat) and math.isfinite(ph.lon) and -90.0 <= ph.lat <= 90.0 and -180.0 <= ph.lon <= 180.0):
            raise ReportError("Saha fotoğrafı konumu geçersiz (enlem −90..90, boylam −180..180).")
    ys = meta.yield_summary
    if ys is not None and not all(math.isfinite(v) for v in (ys.value, ys.lo, ys.hi)):
        raise ReportError("Rekolte özeti sayısal olmalı.")
    return meta


# --------------------------------------------------------------------------- #
# Ana işlev
# --------------------------------------------------------------------------- #
def build_pdf(run: wf.AnalysisRun, meta: ReportMeta) -> bytes:
    """Analiz çıktısından PDF rapor üretir.

    Args:
        run: ``workflow.run_analysis`` çıktısı.
        meta: Kullanıcı girdileri (parsel/poliçe, uzman, fotoğraf, rekolte özeti).

    Returns:
        PDF baytları.

    Raises:
        ReportError: Girdi geçersizse.
    """
    meta = _validate(meta)
    _Fonts.setup()
    st = _styles()
    when = meta.generated_at or dt.datetime.now(dt.timezone.utc)
    rid = _report_id(run, meta, when)
    res, req = run.result, run.request
    region = req.region
    insurance = meta.report_type == "sigorta"
    title = "Hasar değerlendirme raporu" if insurance else "Tarımsal gelişim raporu"

    buf = io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=21 * mm, bottomMargin=17 * mm, title=_t(f"ORBITAL FIELD — {title}"),
                          author="Elif Turut", subject=_t(f"{res.kind.label} · {region.name}"), creator="ORBITAL FIELD")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=_page_deco(rid, run.is_demo, _t(title)))])
    W = doc.width
    story: List[Any] = []

    story.append(_p(_e("UYDU TABANLI HASAR DEĞERLENDİRME RAPORU" if insurance else "UYDU TABANLI TARIMSAL GELİŞİM RAPORU"), st["h1"]))
    story.append(_p(_e(f"{res.kind.label} · {region.name}"), st["sub"]))
    if run.is_demo:
        story.append(_p(_e("Bu rapor SENTETİK demo veriyle üretildi; gerçek bir olayı, parseli ya da hasarı temsil etmez. Yalnızca yöntem gösterimidir."), st["banner"]))

    # 1. Rapor bilgileri
    story.append(_p("1. Rapor bilgileri", st["h2"]))
    info = [("Rapor no", rid), ("Oluşturma zamanı", f"{when:%d.%m.%Y %H:%M} UTC"), ("Analiz türü", res.kind.label), ("Veri kaynağı", run.source_name),
            ("Olay / referans günü", f"{req.event_date:%d.%m.%Y}"), ("Tarih aralığı", f"{req.start:%d.%m.%Y} – {req.end:%d.%m.%Y}"),
            ("Parsel kimliği", _clean(meta.parcel_id, MAX_TEXT) or "—")]
    if insurance:
        info.append(("Poliçe no", _clean(meta.policy_no, MAX_TEXT) or "—"))
    if meta.insured_name.strip():
        info.append(("Sigortalı / üretici" if insurance else "Üretici", _clean(meta.insured_name, MAX_TEXT)))
    story.append(_kv_table(info, st, (0.28 * W, 0.72 * W)))

    # 2. Parsel koordinatları
    story.append(_p("2. Parsel koordinatları", st["h2"]))
    w_, s_, e_, n_ = region.bbox
    lat0, lon0 = region.centre
    story.append(_kv_table([("Alan", f"{_fmt(region.area_ha, 2)} ha ({_fmt(region.area_ha * 10, 0)} dekar)"), ("Merkez (enlem, boylam)", f"{lat0:.6f}°, {lon0:.6f}°"),
                            ("Sınır kutusu", f"{s_:.5f}–{n_:.5f}° K · {w_:.5f}–{e_:.5f}° D"), ("Köşe sayısı", str(len(region.ring))),
                            ("Koordinat sistemi", "WGS84 (EPSG:4326)")], st, (0.28 * W, 0.72 * W)))
    story.append(Spacer(1, 4))
    ring = list(region.ring)
    shown = ring[:30]
    cols_n = 3
    per = int(math.ceil(len(shown) / cols_n))
    head = ["#", "Enlem °", "Boylam °"] * cols_n
    rows: List[List[str]] = []
    for r_i in range(per):
        row: List[str] = []
        for c_i in range(cols_n):
            k = r_i + c_i * per
            row += [str(k + 1), f"{shown[k][1]:.6f}", f"{shown[k][0]:.6f}"] if k < len(shown) else ["", "", ""]
        rows.append(row)
    cw = [0.04 * W, 0.15 * W, 0.15 * W] * cols_n
    story.append(_grid_table(head, rows, st, cw, right=(1, 2, 4, 5, 7, 8)))
    if len(ring) > len(shown):
        story.append(_p(_e(f"Parselin {len(ring)} köşesinden ilk {len(shown)} tanesi gösterildi."), st["small"]))

    # 3. Uydu verisi
    story.append(_p("3. Kullanılan uydu görüntüleri", st["h2"]))
    pre, post = res.pre_meta, res.post_meta

    def cloud_txt(m: an.SceneMeta) -> str:
        if not math.isfinite(m.cloud_pct):
            return "bulut bilinmiyor"
        flag = f" — {an.cloud_level(m.cloud_pct)[0]}" if m.cloud_pct > an.CLOUD_WARN_PCT else ""
        return f"karo bulutu %{m.cloud_pct:.0f}{flag}"

    story.append(_grid_table(["", "Tarih", "Uydu", "Bulut", "Sahne kimliği"],
                             [["ÖNCE", f"{pre.date:%d.%m.%Y}", pre.platform, cloud_txt(pre), pre.scene_id], ["SONRA", f"{post.date:%d.%m.%Y}", post.platform, cloud_txt(post), post.scene_id]],
                             st, [0.1 * W, 0.14 * W, 0.17 * W, 0.26 * W, 0.33 * W]))
    story.append(_p(_e(f"Katalogda {len(run.catalog)} sahne bulundu; karo bulutu %{req.max_cloud:.0f} üstü sahneler elendi. "
                       f"Bulutlu/gölgeli pikseller (SCL sınıfları) analizden ayrıca çıkarıldı."), st["small"]))

    # 4. Sonuç
    story.append(_p("4. Sonuç", st["h2"]))
    ins, agri = run.insurance, run.agri
    if insurance:
        rows_r = [("Otomatik değerlendirme", ins.decision), ("Hasar oranı (alan ağırlıklı)", f"%{_fmt(100 * ins.damage_ratio, 1)}" if math.isfinite(ins.damage_ratio) else "—"),
                  ("Gösterge tazminat", f"₺{_fmt(ins.indicative_claim_tl, 0)}"), ("Sigorta bedeli", f"₺{_fmt(req.insured_tl, 0)}"),
                  ("Etkilenen alan", f"{_fmt(res.affected_ha, 1)} ha (%{_fmt(res.affected_pct, 1)} / ölçülebilir)"), ("Ölçülebilir alan", f"{_fmt(res.valid_ha, 1)} ha (%{_fmt(res.valid_pct, 0)})")]
        if math.isfinite(ins.mean_dnbr):
            rows_r.insert(2, ("Ortalama dNBR", _fmt(ins.mean_dnbr, 3)))
        story.append(_kv_table(rows_r, st, (0.34 * W, 0.66 * W)))
        story.append(_p(_e(ins.severity_text + ". Bu değerlendirme gösterge niteliğindedir; ekspertiz kararının yerine geçmez."), st["small"]))
        for n in ins.notes:
            story.append(_p(_e("• " + n), st["small"]))
    else:
        story.append(_kv_table([("Bitki sağlığı", agri.health), ("Ortalama NDVI (önce → sonra)", f"{_fmt(agri.mean_pre, 2)} → {_fmt(agri.mean_post, 2)}"),
                                ("Bitkili alan", f"{_fmt(agri.veg_ha, 1)} ha"),
                                ("Stresli alan (NDVI ≥ %25 düştü)", f"{_fmt(agri.stressed_ha, 1)} ha (%{_fmt(agri.stressed_pct, 1)})"),
                                ("Gelişen alan (ΔNDVI ≥ +0,05)", f"{_fmt(agri.growth_ha, 1)} ha"), ("Ölçülebilir alan", f"{_fmt(res.valid_ha, 1)} ha (%{_fmt(res.valid_pct, 0)})")],
                               st, (0.34 * W, 0.66 * W)))
        for n in agri.notes:
            story.append(_p(_e("• " + n), st["small"]))

    # Görüntüler
    img_w = (W - 8 * mm) / 3

    def img_cell(png: Optional[bytes], cap: str) -> List[Any]:
        if png is None:
            return [_p(_e("Görüntü yok"), st["small"]), _p(_e(cap), st["small"])]
        im = Image.open(io.BytesIO(png))
        h_ = img_w * im.height / im.width
        return [RLImage(io.BytesIO(png), width=img_w, height=h_), _p(_e(cap), st["small"])]

    pre_png = _png(run.rgb_pre) if run.rgb_pre is not None else None
    post_png = _png(run.rgb_post) if run.rgb_post is not None else None
    chg_png = _change_map_png(res, run.rgb_post)
    tbl = Table([[img_cell(pre_png, f"ÖNCE · {pre.date:%d.%m.%Y}"), img_cell(post_png, f"SONRA · {post.date:%d.%m.%Y}"), img_cell(chg_png, f"DEĞİŞİM · {res.kind.change_label}")]],
                colWidths=[img_w + 4 * mm] * 3)
    tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    story.append(Spacer(1, 5))
    story.append(KeepTogether([tbl]))

    # Sınıf tablosu
    ct = res.class_table()
    rows_c = [[r.sinif, _fmt(r.alan_ha, 1), f"%{_fmt(r.oran_pct, 1)}"] for r in ct.itertuples()]
    t_cls = _grid_table(["Sınıf", "Alan (ha)", "Oran (ölçülebilir alana)"], rows_c, st, [0.4 * W, 0.2 * W, 0.4 * W], right=(1, 2))
    sw = [("BACKGROUND", (0, i + 1), (0, i + 1), colors.HexColor(c[1]).clone(alpha=0.28)) for i, c in enumerate(res.kind.classes)]
    t_cls.setStyle(TableStyle(sw))
    story.append(Spacer(1, 4))
    story.append(t_cls)

    # Zaman serisi
    story.append(_p("5. İndeks zaman serisi (bölge içi, bulut/gölge dışı pikseller)", st["h2"]))
    story.append(_series_chart(run.series, res.kind.index.upper(), req.event_date, pre.date, post.date, W, 150))
    story.append(_p(_e("Çizgiler yalnızca bulut/gölge dışı piksellerden hesaplanır; her nokta bir uydu geçişidir. Seçilen önce/sonra sahneleri daire içindedir."), st["small"]))

    # Güvenilirlik
    story.append(_p("6. Güvenilirlik göstergeleri", st["h2"]))
    rel = res.reliability
    labels = {"berraklik": "Berraklık (kullanılabilir piksel)", "bulut_karo": "Karo bulutu", "zamanlama": "Olaya göre zamanlama", "ornek": "Örnek büyüklüğü"}
    rows_rel = [("Güvenilirlik puanı", f"{rel.score} / 100 — {rel.level}")] + [(labels.get(k, k), f"%{rel.parts.get(k, 0.0) * 100:.0f}") for k in labels]
    story.append(_kv_table(rows_rel, st, (0.34 * W, 0.66 * W)))
    if rel.level == "Düşük":
        story.append(_p(_e("GÜVENİLİRLİK DÜŞÜK: bulut/gölge ya da zamanlama nedeniyle sonuç dikkatle okunmalı; ek sahne ya da saha doğrulaması önerilir."), st["warn"]))
    for n in rel.notes:
        story.append(_p(_e("• " + n), st["small"]))
    story.append(_p(_e("Bu puan bir veri kalitesi göstergesidir; istatistiksel güven aralığı değildir. Ağırlıklar sezgiseldir ve gerçek hasar verisiyle kalibre edilmemiştir."), st["small"]))

    # Hava ve yangın
    story.append(_p("7. Hava durumu doğrulaması", st["h2"]))
    if run.weather_rows:
        story.append(_grid_table(["Gösterge", "Değer", "Durum", "Kural"],
                                 [[r.label, r.value, {True: "destekliyor", False: "desteklemiyor", None: "bilgi"}[r.flag], r.rule] for r in run.weather_rows], st,
                                 [0.33 * W, 0.15 * W, 0.17 * W, 0.35 * W]))
    if run.weather_error:
        story.append(_p(_e(f"Hava verisi alınamadı: {run.weather_error}"), st["warn"]))
    story.append(_p(_e("Kaynak: " + ("SENTETİK demo verisi." if run.is_demo else "Open-Meteo arşivi (ERA5 tabanlı yeniden-analiz; istasyon ölçümü değildir).")
                       + " Eşikler ayarlanabilir başlangıç değerleridir, resmî sınır değildir."), st["small"]))
    if req.kind == "yangin":
        story.append(_p("8. Yangın noktaları (NASA FIRMS)" + (" — SENTETİK" if run.is_demo else ""), st["h2"]))
        hs = run.hotspots
        if run.hotspots_error:
            story.append(_p(_e(f"Yangın noktaları alınamadı: {run.hotspots_error}"), st["warn"]))
        elif hs is None or hs.empty:
            story.append(_p(_e("Bölgeyle ve tarih aralığıyla eşleşen yangın noktası yok."), st["base"]))
        else:
            story.append(_p(_e(f"{len(hs)} nokta eşleşti (çokgen içi {int(hs['icinde'].sum())}); toplam FRP {_fmt(float(pd.to_numeric(hs['frp_mw'], errors='coerce').sum()), 0)} MW."), st["base"]))
            rows_h = [[f"{r.tarih_utc:%d.%m.%Y %H:%M}", f"{r.enlem:.4f}", f"{r.boylam:.4f}", _fmt(r.frp_mw, 1), "içinde" if r.icinde else f"{r.uzaklik_m:.0f} m dışında"]
                      for r in hs.head(10).itertuples()]
            story.append(_grid_table(["Zaman (UTC)", "Enlem", "Boylam", "FRP MW", "Konum"], rows_h, st, [0.27 * W, 0.15 * W, 0.15 * W, 0.15 * W, 0.28 * W], right=(1, 2, 3)))

    # Rekolte
    ys = meta.yield_summary
    if ys is not None:
        story.append(_p("Rekolte tahmini", st["h2"]))
        story.append(_kv_table([("Ürün / evre", f"{ys.crop} · {ys.stage_label}"), ("Tahmin", f"{_fmt(ys.value, 2)} t/ha"),
                                (f"%{int(round(ys.level * 100))} aralık", f"{_fmt(ys.lo, 2)} – {_fmt(ys.hi, 2)} t/ha"), ("Geçmiş verim girdisi", f"{_fmt(ys.prior_yield, 2)} t/ha"),
                                ("Sezon-dışı hata (LOSO)", f"RMSE {_fmt(ys.loso_rmse, 2)} t/ha · MAPE %{_fmt(ys.loso_mape, 1)}")], st, (0.34 * W, 0.66 * W)))
        if ys.synthetic:
            story.append(_p(_e("Model SENTETİK geçmiş veriyle eğitildi; bu değerler yalnızca yöntemi gösterir, gerçek rekolte tahmini değildir."), st["banner"]))
        for w_msg in ys.warnings:
            story.append(_p(_e("• " + w_msg), st["small"]))

    # Yöntem ve sınırlar
    story.append(_p("Yöntem, eşikler ve sınırlar", st["h2"]))
    story.append(_p(_e(res.kind.method), st["base"]))
    story.append(_p(_e(res.kind.caveat), st["small"]))
    ev = th.evidence_table(req.rules)
    keys = {"yangin": "Yangın", "sel": "Sel", "kuraklik": "Kuraklık", "tarim": "Tarım"}
    rel_rows = ev[ev["alan"].str.startswith(keys[req.kind]) | ev["alan"].isin(["Bulut", "Güvenilirlik puanı"])]
    story.append(_grid_table(["Eşik", "Değer", "Dayanak", "Doğrulama"], [[f"{r.alan}: {r.esik}", _fmt(r.deger, 2), r.dayanak, r.dogrulama] for r in rel_rows.itertuples()], st,
                             [0.27 * W, 0.08 * W, 0.4 * W, 0.25 * W], right=(1,)))
    story.append(_p(_e("Sınıflandırma eşikleri literatürdeki sınırlara hizalı politika seçimleridir; Sentinel-2 ve tarım arazisi için bu projede yerel olarak doğrulanmamıştır. "
                       "Doğrulama için uygulamadaki saha-doğrulama paneli (yanlış pozitif/negatif oranları, ROC) kendi etiketli parsel verinizle kullanılmalıdır."), st["small"]))

    # Uzman
    story.append(PageBreak())
    story.append(_p("Uzman değerlendirmesi ve onay", st["h2"]))
    decision = _clean(meta.expert_decision, MAX_TEXT) or "Beklemede (uzman kararı girilmedi)"
    story.append(_kv_table([("Otomatik değerlendirme", ins.decision if insurance else agri.health), ("Uzman kararı", decision),
                            ("Uzman", _clean(meta.expert_name, MAX_TEXT) or "—"), ("Sicil / ruhsat no", _clean(meta.expert_reg_no, MAX_TEXT) or "—")], st, (0.28 * W, 0.72 * W)))
    note = _clean(meta.expert_note, MAX_NOTE)
    if note:
        story.append(Spacer(1, 4))
        story.append(_p("<b>Uzman notu</b>", st["cell"]))
        story.append(_p(_e(note).replace("\n", "<br/>"), st["base"]))
    story.append(Spacer(1, 12))
    story.append(_signature_block(st, W, _clean(meta.expert_name, MAX_TEXT), _clean(meta.expert_reg_no, MAX_TEXT)))
    story.append(Spacer(1, 6))
    story.append(_p(_e("Bu belge, yukarıdaki imza ve tarih alanları ilgili kişilerce doldurulup imzalanana kadar bir karar belgesi değildir. "
                       "Uydu tabanlı değerlendirme destekleyici bir göstergedir; hasar tespiti ve tazminat kararı sigorta şirketinin yetkili ekspertiz sürecine aittir."), st["small"]))

    # Saha fotoğrafı eki
    ph = meta.field_photo
    if ph is not None:
        jpg = prepare_photo(ph.data)
        story.append(_p("Ek: Saha fotoğrafı", st["h2"]))
        im = Image.open(io.BytesIO(jpg))
        max_w, max_h = W * 0.8, 95 * mm
        k = min(max_w / im.width, max_h / im.height)
        story.append(RLImage(io.BytesIO(jpg), width=im.width * k, height=im.height * k))
        inside = an.point_in_region(region, ph.lon, ph.lat)
        d_c = haversine_m(ph.lat, ph.lon, lat0, lon0)
        story.append(Spacer(1, 4))
        rows_p = [("Konum (elle girildi)", f"{ph.lat:.6f}°, {ph.lon:.6f}°"), ("Parsele göre", "çokgenin içinde" if inside else f"çokgenin dışında (merkeze {_fmt(d_c, 0)} m)"),
                  ("Çekim zamanı", _clean(ph.taken, MAX_TEXT) or "—"), ("Açıklama", _clean(ph.caption, MAX_TEXT) or "—")]
        story.append(_kv_table(rows_p, st, (0.28 * W, 0.72 * W)))
        story.append(_p(_e("Konum, uzmanın girdiği değerdir; fotoğraf dosyasındaki EXIF/GPS verisi okunmaz ve rapora aktarılmaz. Fotoğraf yeniden kodlanmış ve küçültülmüştür."), st["small"]))

    doc.build(story)
    return buf.getvalue()
