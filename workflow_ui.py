"""
workflow_ui.py — "ANALİZ İŞ AKIŞI" sekmesinin görsel bileşenleri (saf HTML / SVG üreticileri)
============================================================================================

Bu modül Streamlit'e bağlı değildir: her işlev bir HTML/SVG dizgesi döndürür. Böylece bileşenler tarayıcıda gerçekten
çizilip (ekran görüntüsüyle) doğrulanabilir ve birim-test edilebilir. Streamlit'e bağlayan kısım ``render_workflow_tab`` (dosya sonu);
uygulamanın yardımcıları ona ``Kit`` içinde verilir (döngüsel içe aktarma yok).

Bileşenler (kullanıcının örnek ekran görüntüsündeki yerleşimi izler): adım izleyici, sahne listesi (bulut rozetli),
indeks zaman serisi (ortalama / medyan / min / maks + bulut çubukları), önce/sonra kaydırmalı karşılaştırma, sınıf tablosu,
güvenilirlik göstergesi, hava durumu doğrulaması, yangın noktası eşleştirmesi, sigorta ve tarım görünümleri, rekolte tahmini,
eşik dayanağı ve ROC.

Geliştirici: Elif Turut
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import io
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import analysis as an
import yield_model as ym

__all__ = ["fmt", "png_data_uri", "build_steps_html", "build_scene_list_html", "build_series_svg", "build_class_table_html", "build_reliability_html",
           "build_weather_html", "build_hotspots_html", "build_compare_html", "build_sector_views_html", "build_yield_html",
           "build_evidence_html", "build_document", "alert_rows", "build_alert_html", "build_support_html", "build_roc_svg", "build_points_html", "build_diag_html", "build_region_card_html", "region_to_pixels",
           "DECISION_COLORS", "STEP_LABELS"]

STEP_LABELS: Tuple[str, ...] = ("BÖLGE", "ANALİZ TÜRÜ", "KAYNAK VE TARİH", "İŞLE", "SONUÇ", "RAPOR")
C_ACCENT, C_AMBER, C_DANGER, C_OK, C_MUTED, C_DIM = "#6ea8fe", "#f2b84b", "#fc3d21", "#4cc38a", "#8b93a1", "#69717f"
C_LINE, C_SOFT, C_TEXT = "rgba(255,255,255,.09)", "#cfd4dc", "#e6e8ec"
DECISION_COLORS: Dict[str, str] = {"ONAY": C_OK, "KISMİ ONAY": C_ACCENT, "EKSPERTİZ": C_AMBER, "RET": C_DANGER}
HEALTH_COLORS: Dict[str, str] = {"Sağlıklı": C_OK, "İzlemede": "#e3c35a", "Stresli": "#ec9a4d", "Kritik": C_DANGER, "Ölçülemedi": C_MUTED}
LEVEL_COLORS: Dict[str, str] = {"Yüksek": C_OK, "Orta": C_AMBER, "Düşük": C_DANGER}

_CSS = f"""
.wf{{font-family:Inter,'Helvetica Neue',Arial,sans-serif;color:{C_TEXT};}}
.wf .muted{{color:{C_MUTED}}} .wf .dim{{color:{C_DIM}}}
.wf-card{{border:1px solid {C_LINE};background:#10131a;border-radius:12px;padding:14px 16px;}}
.wf-h{{font:600 .66rem Inter,sans-serif;letter-spacing:.16em;color:{C_MUTED};margin:0 0 10px;text-transform:none}}
.wf-steps{{display:flex;gap:6px;flex-wrap:wrap;margin:4px 0 18px}}
.wf-step{{flex:1 1 120px;min-width:112px;display:flex;align-items:center;gap:9px;padding:9px 12px;border-radius:10px;border:1px solid {C_LINE};background:#10131a;color:{C_MUTED};font:600 .62rem Inter,sans-serif;letter-spacing:.12em}}
.wf-step i{{font-style:normal;display:grid;place-items:center;width:22px;height:22px;border-radius:50%;border:1px solid {C_LINE};font:600 .68rem Inter,sans-serif;letter-spacing:0}}
.wf-step.done{{color:{C_SOFT};border-color:rgba(76,195,138,.35)}} .wf-step.done i{{background:{C_OK};border-color:{C_OK};color:#06140d}}
.wf-step.now{{color:#fff;border-color:{C_ACCENT};background:rgba(110,168,254,.08)}} .wf-step.now i{{border-color:{C_ACCENT};color:{C_ACCENT}}}
.wf-scenes{{display:flex;flex-direction:column;gap:6px;max-height:430px;overflow:auto;padding-right:4px}}
.wf-scene{{display:grid;grid-template-columns:1fr auto;gap:2px 10px;align-items:center;padding:8px 10px;border-radius:9px;border:1px solid {C_LINE};background:#0d1016;font-size:.78rem}}
.wf-scene.sel{{border-color:{C_ACCENT};background:rgba(110,168,254,.08)}} .wf-scene.out{{opacity:.5}}
.wf-scene b{{font-weight:600;color:#fff}} .wf-scene small{{color:{C_MUTED};font-size:.68rem;grid-column:1 / span 2}}
.wf-cloud{{font:600 .68rem ui-monospace,Menlo,monospace;padding:2px 8px;border-radius:99px;border:1px solid;white-space:nowrap}}
.wf-tag{{display:inline-block;font:600 .56rem Inter,sans-serif;letter-spacing:.1em;padding:2px 7px;border-radius:99px;margin-left:6px;vertical-align:1px}}
.wf-pill{{display:inline-block;font:700 .72rem Inter,sans-serif;letter-spacing:.1em;padding:5px 12px;border-radius:99px;border:1px solid}}
.wf table{{width:100%;border-collapse:collapse;font-size:.8rem}} .wf th{{font:600 .62rem Inter,sans-serif;letter-spacing:.1em;color:{C_MUTED};text-align:left;padding:6px 8px;border-bottom:1px solid {C_LINE};text-transform:uppercase}}
.wf td{{padding:7px 8px;border-bottom:1px solid rgba(255,255,255,.05);vertical-align:middle}} .wf td.num{{text-align:right;font-variant-numeric:tabular-nums}}
.wf-bar{{height:6px;border-radius:99px;background:rgba(255,255,255,.07);overflow:hidden;min-width:60px}} .wf-bar>span{{display:block;height:100%;border-radius:99px}}
.wf-sw{{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:8px;vertical-align:-1px}}
.wf-two{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:14px}}
.wf-note{{font-size:.76rem;color:{C_MUTED};line-height:1.55;margin:6px 0 0}} .wf-warn{{color:{C_AMBER}}} .wf-bad{{color:{C_DANGER}}} .wf-ok{{color:{C_OK}}}
.wf-kv{{display:grid;grid-template-columns:auto 1fr;gap:5px 14px;font-size:.8rem;margin:8px 0}} .wf-kv span:nth-child(odd){{color:{C_MUTED}}} .wf-kv span:nth-child(even){{text-align:right;font-variant-numeric:tabular-nums}}
.wf-big{{font:300 2.1rem/1.1 Inter,sans-serif;letter-spacing:-.02em;color:#fff}} .wf-big small{{font-size:.9rem;color:{C_MUTED};font-weight:400;letter-spacing:0}}
.wf ul{{margin:6px 0 0;padding-left:18px;font-size:.78rem;color:{C_SOFT};line-height:1.6}}
"""


def esc(value: object) -> str:
    """HTML için kaçışlar."""
    return html.escape(str(value), quote=True)


def fmt(x: float, d: int = 1) -> str:
    """Türkçe sayı biçimi (binlik nokta, ondalık virgül); NaN → '—'."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def png_data_uri(arr: np.ndarray) -> str:
    """``uint8`` (r, c, 3|4) diziyi PNG data-URI'ye çevirir.

    Raises:
        ValueError: Dizi biçimi desteklenmiyorsa.
    """
    from PIL import Image  # Pillow Streamlit'in bağımlılığıdır

    a = np.asarray(arr)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError("PNG için uint8 (r, c, 3|4) dizi gerekir.")
    buf = io.BytesIO()
    Image.fromarray(a, "RGBA" if a.shape[2] == 4 else "RGB").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _wrap(inner: str) -> str:
    return f'<div class="wf"><style>{_CSS}</style>{inner}</div>'


# --------------------------------------------------------------------------- #
# Adım izleyici, bölge kartı
# --------------------------------------------------------------------------- #
def build_steps_html(done: Sequence[int], current: int) -> str:
    """Altı adımlı izleyici.

    Args:
        done: Tamamlanan adım numaraları (1–6).
        current: Şu anki adım (1–6).
    """
    parts = []
    for i, label in enumerate(STEP_LABELS, start=1):
        cls = "done" if i in done else ("now" if i == current else "")
        mark = "✓" if i in done else str(i)
        parts.append(f'<div class="wf-step {cls}"><i>{mark}</i>{esc(label)}</div>')
    return _wrap(f'<div class="wf-steps">{"".join(parts)}</div>')


def build_region_card_html(region: an.Region, source_note: str = "") -> str:
    """Seçili bölgenin özet kartı (alan, sınır kutusu, köşe sayısı)."""
    w, s, e, n = region.bbox
    lat, lon = region.centre
    rows = [("Alan", f"{fmt(region.area_ha, 1)} ha ({fmt(region.area_ha * 10, 0)} dekar)"), ("Merkez", f"{lat:.5f}°K · {lon:.5f}°D"),
            ("Sınır kutusu", f"{s:.4f}–{n:.4f}°K · {w:.4f}–{e:.4f}°D"), ("Köşe sayısı", str(len(region.ring)))]
    kv = "".join(f"<span>{esc(k)}</span><span>{esc(v)}</span>" for k, v in rows)
    note = f'<p class="wf-note">{esc(source_note)}</p>' if source_note else ""
    return _wrap(f'<div class="wf-card"><p class="wf-h">SEÇİLİ BÖLGE · {esc(region.name)}</p><div class="wf-kv">{kv}</div>{note}</div>')


# --------------------------------------------------------------------------- #
# Sahne listesi
# --------------------------------------------------------------------------- #
def _cloud_style(c: float) -> Tuple[str, str]:
    label, level = an.cloud_level(c)
    color = {"ok": C_OK, "mid": C_AMBER, "low": C_DANGER, "unknown": C_MUTED}[level]
    return color, label


def build_scene_list_html(catalog: pd.DataFrame, limit: int = 18) -> str:
    """Sahne listesi: tarih, uydu, bulut rozeti (renkli) ve durum; seçili önce/sonra sahneleri vurgulu.

    Args:
        catalog: ``AnalysisRun.catalog``.
        limit: Gösterilecek en çok satır (seçili olanlar her zaman dahil, kalanı en yeni önce).
    """
    if catalog.empty:
        return _wrap('<div class="wf-card"><p class="wf-h">SAHNELER</p><p class="wf-note">Sahne yok.</p></div>')
    sel = catalog[catalog["durum"].str.contains("seçildi")]
    rest = catalog[~catalog.index.isin(sel.index)].sort_values("tarih", ascending=False).head(max(limit - len(sel), 0))
    show = pd.concat([sel, rest]).sort_values("tarih", ascending=False)
    rows = []
    for _, r in show.iterrows():
        color, label = _cloud_style(float(r["bulut_karo"]))
        cloud_txt = "—" if not math.isfinite(float(r["bulut_karo"])) else f"{float(r['bulut_karo']):.0f}%"
        state = str(r["durum"])
        selected = "seçildi" in state
        out = state.startswith("elendi") or state.startswith("okunamadı")
        tag = ""
        if selected:
            tag = f'<span class="wf-tag" style="background:{C_ACCENT};color:#06101f">{esc(state.split(" ")[0])}</span>'
        elif out:
            tag = f'<span class="wf-tag" style="border:1px solid {C_LINE};color:{C_MUTED}">{esc(state)}</span>'
        rows.append(f'<div class="wf-scene {"sel" if selected else ""} {"out" if out else ""}">'
                    f'<span><b>{r["tarih"]:%d.%m.%Y}</b>{tag}</span>'
                    f'<span class="wf-cloud" style="color:{color};border-color:{color}" title="Karo bulut oranı · güvenilirlik: {esc(label)}">☁ {cloud_txt}</span>'
                    f'<small>{esc(r["uydu"])}</small></div>')
    n_hidden = len(catalog) - len(show)
    more = f'<p class="wf-note">+ {n_hidden} sahne daha (liste kısaltıldı)</p>' if n_hidden > 0 else ""
    return _wrap(f'<div class="wf-card"><p class="wf-h">SAHNELER · {len(catalog)} KAYIT</p><div class="wf-scenes">{"".join(rows)}</div>{more}'
                 f'<p class="wf-note">☁ karo bulut oranı. <span class="wf-ok">≤ %10 yüksek</span> · <span class="wf-warn">≤ %20 orta</span> · '
                 f'<span class="wf-bad">&gt; %20 düşük güvenilirlik</span></p></div>')


# --------------------------------------------------------------------------- #
# Zaman serisi (SVG)
# --------------------------------------------------------------------------- #
def _nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = math.floor(lo / step) * step
    out, v = [], start
    while v <= hi + step * 0.5:
        out.append(round(v, 10))
        v += step
    return out


def build_series_svg(series: pd.DataFrame, index_label: str, event_date: Optional[dt.date] = None, pre_date: Optional[dt.date] = None,
                     post_date: Optional[dt.date] = None, width: int = 820, height: int = 340) -> str:
    """İndeks zaman serisi: ortalama, medyan, %2 ve %98 dilimleri (çizgi) + yerel bulut oranı (alt çubuklar).

    Args:
        series: ``analysis.build_series`` çıktısı.
        index_label: Eksen başlığı (örn. ``NDVI``).
        event_date: Olay günü (kesikli kırmızı çizgi).
        pre_date: Seçilen önce sahnesi (işaret).
        post_date: Seçilen sonra sahnesi (işaret).
        width: Çizim genişliği (viewBox).
        height: Çizim yüksekliği (viewBox).
    """
    s = series.dropna(subset=["ortalama"]).copy()
    if s.empty:
        return _wrap('<div class="wf-card"><p class="wf-h">İNDEKS ZAMAN SERİSİ</p><p class="wf-note">Ölçülebilir sahne yok.</p></div>')
    L, R, T, B = 52, 46, 18, 92
    pw, ph = width - L - R, height - T - B
    t0, t1 = s["tarih"].min(), s["tarih"].max()
    for d in (event_date, pre_date, post_date):
        if d is not None:
            t0, t1 = min(t0, pd.Timestamp(d)), max(t1, pd.Timestamp(d))
    span = max((t1 - t0).days, 1)
    vals = pd.concat([s["minimum"], s["maksimum"], s["ortalama"], s["medyan"]]).dropna()
    lo, hi = float(vals.min()), float(vals.max())
    pad = max((hi - lo) * 0.08, 0.02)
    lo, hi = lo - pad, hi + pad
    ticks = _nice_ticks(lo, hi)
    lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])

    def X(d: Any) -> float:
        return L + pw * ((pd.Timestamp(d) - t0).days / span)

    def Y(v: float) -> float:
        return T + ph * (1 - (v - lo) / (hi - lo))

    g: List[str] = []
    for tk in ticks:
        y = Y(tk)
        g.append(f'<line x1="{L}" x2="{L + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="rgba(255,255,255,.07)"/>'
                 f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" fill="{C_MUTED}" font-size="11">{fmt(tk, 2)}</text>')
    n_x = min(7, span // 7 + 1)
    for i in range(n_x + 1):
        d = t0 + pd.Timedelta(days=round(span * i / max(n_x, 1)))
        g.append(f'<line x1="{X(d):.1f}" x2="{X(d):.1f}" y1="{T + ph}" y2="{T + ph + 4}" stroke="{C_DIM}"/>'
                 f'<text x="{X(d):.1f}" y="{T + ph + 17}" text-anchor="middle" fill="{C_MUTED}" font-size="11">{d:%d.%m}</text>')
        if i == 0 or i == n_x:
            g.append(f'<text x="{X(d):.1f}" y="{T + ph + 29}" text-anchor="middle" fill="{C_DIM}" font-size="10">{d:%Y}</text>')

    # bulut çubukları (alt şerit)
    base = T + ph + 40
    bar_h = 28
    g.append(f'<text x="{L - 8}" y="{base + bar_h / 2 + 4}" text-anchor="end" fill="{C_DIM}" font-size="10">bulut</text>')
    bw = max(2.0, min(10.0, pw / max(len(s), 1) * 0.55))
    for _, r in s.iterrows():
        c = r["bulut_yerel"] if math.isfinite(r["bulut_yerel"]) else r["bulut_karo"]
        if not math.isfinite(c):
            continue
        h = max(1.0, bar_h * min(c, 100.0) / 100.0)
        col = C_DANGER if c > an.CLOUD_WARN_PCT else (C_AMBER if c > 10 else "#5b6577")
        g.append(f'<rect x="{X(r["tarih"]) - bw / 2:.1f}" y="{base + bar_h - h:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="1.5" fill="{col}" opacity=".85">'
                 f'<title>{r["tarih"]:%d.%m.%Y} · bulut %{c:.0f}</title></rect>')

    if event_date is not None:
        x = X(event_date)
        g.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{T + ph}" stroke="{C_DANGER}" stroke-dasharray="5 4" stroke-width="1.3"/>'
                 f'<text x="{x + 5:.1f}" y="{T + 11}" fill="{C_DANGER}" font-size="10" font-weight="600">OLAY {event_date:%d.%m}</text>')

    def line(col: str, key: str, sw: float, dash: str = "", op: float = 1.0) -> None:
        pts = [(X(r["tarih"]), Y(r[key])) for _, r in s.iterrows() if math.isfinite(r[key])]
        if len(pts) >= 2:
            g.append(f'<polyline fill="none" stroke="{col}" stroke-width="{sw}" {"stroke-dasharray=" + chr(34) + dash + chr(34) if dash else ""} '
                     f'opacity="{op}" stroke-linejoin="round" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>')

    line("#7c8798", "maksimum", 1.1, "3 3", .9)
    line("#7c8798", "minimum", 1.1, "3 3", .9)
    line(C_OK, "medyan", 1.6)
    line(C_ACCENT, "ortalama", 2.2)
    for _, r in s.iterrows():
        g.append(f'<circle cx="{X(r["tarih"]):.1f}" cy="{Y(r["ortalama"]):.1f}" r="3" fill="{C_ACCENT}"><title>{r["tarih"]:%d.%m.%Y} · ort {fmt(r["ortalama"], 3)} · '
                 f'medyan {fmt(r["medyan"], 3)} · min {fmt(r["minimum"], 3)} · maks {fmt(r["maksimum"], 3)} · geçerli %{r["gecerli_oran"]:.0f}</title></circle>')
    for d, lab, col in ((pre_date, "ÖNCE", C_AMBER), (post_date, "SONRA", C_OK)):
        if d is None:
            continue
        row = s[s["tarih"] == pd.Timestamp(d)]
        if row.empty:
            continue
        x, y = X(d), Y(float(row["ortalama"].iloc[0]))
        g.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="none" stroke="{col}" stroke-width="2"/>'
                 f'<text x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle" fill="{col}" font-size="10" font-weight="700">{lab}</text>')
    legend = [(C_ACCENT, "Ortalama"), (C_OK, "Medyan"), ("#7c8798", "Min / maks (%2–%98)")]
    lx = L
    lg = []
    for col, name in legend:
        lg.append(f'<rect x="{lx}" y="{height - 11}" width="14" height="3" rx="1.5" fill="{col}"/><text x="{lx + 20}" y="{height - 6}" fill="{C_MUTED}" font-size="11">{name}</text>')
        lx += 20 + len(name) * 6.6 + 16
    svg = (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(index_label)} zaman serisi" style="width:100%;height:auto;display:block;font-family:Inter,sans-serif">'
           f'<text x="{L}" y="12" fill="{C_MUTED}" font-size="11" font-weight="600" letter-spacing="1">{esc(index_label)}</text>{"".join(g)}{"".join(lg)}</svg>')
    return _wrap(f'<div class="wf-card"><p class="wf-h">İNDEKS ZAMAN SERİSİ · BÖLGE İÇİ İSTATİSTİK</p>{svg}'
                 f'<p class="wf-note">Çizgiler yalnızca bulut/gölge dışı piksellerden hesaplanır. Alttaki çubuklar sahnenin bölge içi (yoksa karo) bulut oranıdır; '
                 f'kırmızı = %{an.CLOUD_WARN_PCT:g} üstü.</p></div>')


# --------------------------------------------------------------------------- #
# Sınıf tablosu, güvenilirlik
# --------------------------------------------------------------------------- #
def build_class_table_html(result: an.ChangeResult) -> str:
    """Değişim sınıfları: renk, alan (ha), oran (kullanılabilir alana göre) ve çubuk; altta toplamlar."""
    t = result.class_table()
    rows = []
    mx = max(float(t["oran_pct"].max()), 1.0) if len(t) else 1.0
    for _, r in t.iterrows():
        w = 0 if not math.isfinite(r["oran_pct"]) else 100 * r["oran_pct"] / mx
        rows.append(f'<tr><td><i class="wf-sw" style="background:{esc(r["renk"])}"></i>{esc(r["sinif"])}</td><td class="num">{fmt(r["alan_ha"], 1)}</td>'
                    f'<td class="num">%{fmt(r["oran_pct"], 1)}</td><td style="width:34%"><div class="wf-bar"><span style="width:{w:.0f}%;background:{esc(r["renk"])}"></span></div></td></tr>')
    foot = (f'<p class="wf-note">Bölge {fmt(result.region_ha, 1)} ha · ölçülebilir {fmt(result.valid_ha, 1)} ha (%{fmt(result.valid_pct, 0)}) · '
            f'etkilenen {fmt(result.affected_ha, 1)} ha (%{fmt(result.affected_pct, 1)}). Bulut/gölge altındaki alan sınıflandırılmaz.</p>')
    return _wrap(f'<div class="wf-card"><p class="wf-h">DEĞİŞİM SINIFLARI · {esc(result.kind.change_label.upper())}</p>'
                 f'<table><thead><tr><th>Sınıf</th><th class="num" style="text-align:right">ha</th><th style="text-align:right">oran</th><th></th></tr></thead>'
                 f'<tbody>{"".join(rows)}</tbody></table>{foot}</div>')


_PART_LABELS = {"berraklik": ("Berraklık (kullanılabilir piksel)", 0.45), "bulut_karo": ("Karo bulutu", 0.15), "zamanlama": ("Olaya göre zamanlama", 0.25),
                "ornek": ("Örnek büyüklüğü (piksel)", 0.15)}


def build_reliability_html(rel: an.Reliability) -> str:
    """Güvenilirlik göstergesi: halka puan, düzey, bileşen çubukları ve uyarılar (istatistiksel güven aralığı DEĞİLDİR)."""
    col = LEVEL_COLORS.get(rel.level, C_MUTED)
    circ = 2 * math.pi * 44
    dash = circ * rel.score / 100.0
    ring = (f'<svg viewBox="0 0 110 110" style="width:112px;height:112px;flex:none"><circle cx="55" cy="55" r="44" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="9"/>'
            f'<circle cx="55" cy="55" r="44" fill="none" stroke="{col}" stroke-width="9" stroke-linecap="round" stroke-dasharray="{dash:.1f} {circ:.1f}" transform="rotate(-90 55 55)"/>'
            f'<text x="55" y="53" text-anchor="middle" fill="#fff" font-size="26" font-weight="300" font-family="Inter,sans-serif">{rel.score}</text>'
            f'<text x="55" y="70" text-anchor="middle" fill="{C_MUTED}" font-size="9" font-family="Inter,sans-serif" letter-spacing="1">/ 100</text></svg>')
    bars = []
    for key, (lab, w) in _PART_LABELS.items():
        v = rel.parts.get(key, 0.0)
        bars.append(f'<div style="margin:7px 0"><div style="display:flex;justify-content:space-between;font-size:.76rem"><span>{esc(lab)} <span class="dim">· ağırlık {int(w * 100)}%</span></span>'
                    f'<span>{int(round(v * 100))}%</span></div><div class="wf-bar"><span style="width:{v * 100:.0f}%;background:{col if v >= .5 else C_AMBER}"></span></div></div>')
    badge = (f'<span class="wf-pill" style="color:{C_DANGER};border-color:{C_DANGER};margin-left:10px">GÜVENİLİRLİK DÜŞÜK</span>' if rel.level == "Düşük" else "")
    notes = "".join(f"<li>{esc(n)}</li>" for n in rel.notes)
    return _wrap(f'<div class="wf-card"><p class="wf-h">GÜVENİLİRLİK GÖSTERGELERİ</p><div style="display:flex;gap:18px;align-items:center;flex-wrap:wrap">{ring}'
                 f'<div style="flex:1;min-width:230px"><div><span class="wf-pill" style="color:{col};border-color:{col}">{esc(rel.level.upper())}</span>{badge}</div>{"".join(bars)}</div></div>'
                 f'<ul>{notes}</ul><p class="wf-note">Bu puan bir <b>veri kalitesi göstergesidir</b> (bulut/gölge, zamanlama, örnek büyüklüğü); istatistiksel güven aralığı değildir. '
                 f'Formül ve ağırlıklar sezgiseldir ve gerçek hasar verisiyle kalibre edilmemiştir.</p></div>')


# --------------------------------------------------------------------------- #
# Hava, yangın noktası
# --------------------------------------------------------------------------- #
def _flag_icon(flag: Optional[bool]) -> str:
    if flag is True:
        return '<span class="wf-ok" title="Olayı destekliyor">✓ destekliyor</span>'
    if flag is False:
        return '<span class="wf-bad" title="Olayı desteklemiyor">✗ desteklemiyor</span>'
    return '<span class="muted" title="Karar için kullanılmaz / veri yok">— bilgi</span>'


def build_weather_html(rows: Sequence[an.EvidenceRow], weather: Optional[pd.DataFrame], event_date: Optional[dt.date], error: Optional[str],
                       source_note: str, width: int = 640, height: int = 190) -> str:
    """Hava durumu doğrulaması: koşul satırları + olay çevresinde günlük yağış (çubuk) ve en yüksek sıcaklık (çizgi).

    Args:
        rows: ``analysis.weather_evidence`` çıktısı.
        weather: Günlük hava tablosu.
        event_date: Olay günü.
        error: Hava verisi alınamadıysa ileti.
        source_note: Kaynak notu (örn. Open-Meteo/ERA5 ya da SENTETİK).
        width: viewBox genişliği.
        height: viewBox yüksekliği.
    """
    trs = "".join(f'<tr><td>{esc(r.label)}</td><td class="num">{esc(r.value)}</td><td>{_flag_icon(r.flag)}</td><td class="dim" style="font-size:.7rem">{esc(r.rule)}</td></tr>' for r in rows)
    chart = ""
    if weather is not None and not weather.empty and event_date is not None:
        d = weather.copy()
        d["tarih"] = pd.to_datetime(d["tarih"])
        ev = pd.Timestamp(event_date)
        d = d[(d["tarih"] >= ev - pd.Timedelta(days=14)) & (d["tarih"] <= ev + pd.Timedelta(days=14))]
        if len(d) >= 3:
            L, R, T, B = 40, 40, 14, 28
            pw, ph = width - L - R, height - T - B
            t0 = d["tarih"].min()
            span = max((d["tarih"].max() - t0).days, 1)
            pmax = max(float(d["yagis_mm"].max()), 5.0)
            tmin, tmax = float(d["tmax_c"].min()) - 2, float(d["tmax_c"].max()) + 2
            def X(t: pd.Timestamp) -> float:
                return L + pw * ((t - t0).days / span)

            g = [f'<line x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}" stroke="{C_DIM}"/>']
            bw = max(3.0, pw / len(d) * 0.6)
            for _, r in d.iterrows():
                if not math.isfinite(r["yagis_mm"]):
                    continue
                h = ph * r["yagis_mm"] / pmax
                g.append(f'<rect x="{X(r["tarih"]) - bw / 2:.1f}" y="{T + ph - h:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{C_ACCENT}" opacity=".75"><title>{r["tarih"]:%d.%m} · yağış {r["yagis_mm"]:.1f} mm</title></rect>')
            pts = " ".join(f"{X(r['tarih']):.1f},{T + ph * (1 - (r['tmax_c'] - tmin) / (tmax - tmin)):.1f}" for _, r in d.iterrows() if math.isfinite(r["tmax_c"]))
            g.append(f'<polyline fill="none" stroke="{C_AMBER}" stroke-width="1.8" points="{pts}"/>')
            xe = X(ev)
            g.append(f'<line x1="{xe:.1f}" x2="{xe:.1f}" y1="{T}" y2="{T + ph}" stroke="{C_DANGER}" stroke-dasharray="4 3"/>')
            for i in range(0, 5):
                t = t0 + pd.Timedelta(days=round(span * i / 4))
                g.append(f'<text x="{X(t):.1f}" y="{height - 8}" text-anchor="middle" fill="{C_MUTED}" font-size="10">{t:%d.%m}</text>')
            g.append(f'<text x="{L - 6}" y="{T + 8}" text-anchor="end" fill="{C_ACCENT}" font-size="10">{pmax:.0f} mm</text>'
                     f'<text x="{L + pw + 6}" y="{T + 8}" fill="{C_AMBER}" font-size="10">{tmax:.0f}°C</text>')
            chart = (f'<svg viewBox="0 0 {width} {height}" style="width:100%;max-width:760px;height:auto;display:block;margin-top:10px;font-family:Inter,sans-serif" role="img" aria-label="Olay çevresinde yağış ve sıcaklık">'
                     f'{"".join(g)}</svg><p class="wf-note"><span style="color:{C_ACCENT}">■</span> günlük yağış (mm) · <span style="color:{C_AMBER}">━</span> en yüksek sıcaklık (°C) · '
                     f'<span class="wf-bad">┆</span> olay günü (±14 gün)</p>')
    err = f'<p class="wf-note wf-warn">Hava verisi alınamadı: {esc(error)}</p>' if error else ""
    note_txt = source_note.rstrip(". ")
    return _wrap(f'<div class="wf-card"><p class="wf-h">HAVA DURUMU DOĞRULAMASI</p>{err}<table><thead><tr><th>Gösterge</th><th style="text-align:right">Değer</th><th>Durum</th><th>Kural</th></tr></thead>'
                 f'<tbody>{trs}</tbody></table>{chart}<p class="wf-note">{esc(note_txt)}. Eşikler ayarlanabilir başlangıç değerleridir, resmî sınır değildir.</p></div>')


def build_hotspots_html(matched: Optional[pd.DataFrame], error: Optional[str], window: Tuple[dt.date, dt.date], is_demo: bool) -> str:
    """FIRMS yangın noktası eşleştirmesi: sayı, çokgen içi/tampon, zaman aralığı, FRP ve ilk satırlar."""
    title = "YANGIN NOKTALARI · NASA FIRMS" + (" (SENTETİK)" if is_demo else "")
    if error:
        return _wrap(f'<div class="wf-card"><p class="wf-h">{title}</p><p class="wf-note wf-warn">Yangın noktaları alınamadı: {esc(error)}</p></div>')
    if matched is None or matched.empty:
        return _wrap(f'<div class="wf-card"><p class="wf-h">{title}</p><p class="wf-note">{window[0]:%d.%m.%Y}–{window[1]:%d.%m.%Y} aralığında bölgeyle eşleşen yangın noktası yok.</p></div>')
    inside = int(matched["icinde"].sum())
    frp = float(pd.to_numeric(matched["frp_mw"], errors="coerce").sum())
    def where(r: pd.Series) -> str:
        return "içinde" if r["icinde"] else "%.0f m dışında" % r["uzaklik_m"]

    rows = "".join(f'<tr><td>{r["tarih_utc"]:%d.%m %H:%M}</td><td class="num">{r["enlem"]:.4f}</td><td class="num">{r["boylam"]:.4f}</td><td class="num">{fmt(r["frp_mw"], 1)}</td>'
                   f'<td>{where(r)}</td></tr>' for _, r in matched.head(8).iterrows())
    return _wrap(f'<div class="wf-card"><p class="wf-h">{title}</p><div class="wf-kv"><span>Eşleşen nokta</span><span>{len(matched)} (çokgen içi {inside})</span>'
                 f'<span>Zaman aralığı</span><span>{matched["tarih_utc"].min():%d.%m %H:%M} → {matched["tarih_utc"].max():%d.%m %H:%M} UTC</span><span>Toplam FRP</span><span>{fmt(frp, 0)} MW</span></div>'
                 f'<table><thead><tr><th>Zaman (UTC)</th><th style="text-align:right">Enlem</th><th style="text-align:right">Boylam</th><th style="text-align:right">FRP MW</th><th>Konum</th></tr></thead><tbody>{rows}</tbody></table>'
                 f'<p class="wf-note">Eşleşme kuralı: çokgen içinde ya da kenarına 1 km içinde ve {window[0]:%d.%m}–{window[1]:%d.%m} tarih aralığında. '
                 f'VIIRS 375 m / MODIS 1 km çözünürlüğündedir; uydu geçişi sürekli gözlem değildir.</p></div>')


# --------------------------------------------------------------------------- #
# Önce/sonra karşılaştırma (iframe belgesi)
# --------------------------------------------------------------------------- #
def region_to_pixels(region: an.Region, bbox: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
    """Bölge köşelerini sınır kutusuna göre yüzde (x, y) koordinatlarına çevirir (y yukarıdan aşağı)."""
    w, s, e, n = bbox
    return [(100.0 * (lon - w) / (e - w), 100.0 * (n - lat) / (n - s)) for lon, lat in region.ring]


def build_compare_html(pre_rgb: np.ndarray, post_rgb: np.ndarray, overlay_rgba: Optional[np.ndarray], outline: Sequence[Tuple[float, float]],
                       pre_label: str, post_label: str, overlay_label: str, legend: Sequence[Tuple[str, str]], note: str = "",
                       categorical: bool = True) -> str:
    """Kaydırmalı önce/sonra karşılaştırma + değişim katmanı anahtarı (kendi içinde çalışan tam HTML belgesi; iframe için).

    Args:
        pre_rgb: Önce gerçek renk görüntüsü.
        post_rgb: Sonra gerçek renk görüntüsü.
        overlay_rgba: Değişim katmanı (RGBA; ``None`` ise anahtar gizlenir).
        outline: Bölge çokgeni, yüzde koordinatları (``region_to_pixels``).
        pre_label: Sol etiket.
        post_label: Sağ etiket.
        overlay_label: Katman anahtarı etiketi.
        legend: ``(renk, ad)`` çiftleri.
        note: Alt not.
        categorical: Katman sınıflıysa piksel görünümü korunur (yumuşatma kapalı).
    """
    h, w = pre_rgb.shape[:2]
    pre_u, post_u = png_data_uri(pre_rgb), png_data_uri(post_rgb)
    ov_u = png_data_uri(overlay_rgba) if overlay_rgba is not None else ""
    poly = " ".join(f"{x:.2f},{y:.2f}" for x, y in outline)
    leg = "".join(f'<span><i style="background:{esc(c)}"></i>{esc(n)}</span>' for c, n in legend)
    smooth = "image-rendering:pixelated;" if categorical else ""
    toggle = (f'<label class="tg"><input type="checkbox" id="ov" checked> {esc(overlay_label)}</label>' if ov_u else "")
    ov_img = f'<img id="ovimg" class="ov" src="{ov_u}" alt="" style="{smooth}">' if ov_u else ""
    return f"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{{box-sizing:border-box}} html,body{{margin:0;background:#0b0c10;color:#e6e8ec;font-family:Inter,'Helvetica Neue',Arial,sans-serif}}
.wrap{{padding:10px}} .hd{{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px;font-size:.7rem;letter-spacing:.12em;color:#8b93a1;font-weight:600}}
.tg{{font-size:.76rem;letter-spacing:0;color:#cfd4dc;font-weight:500;display:flex;gap:6px;align-items:center;cursor:pointer}}
.stage{{position:relative;width:min(100%,{max(int(560 * w / h), 200)}px);aspect-ratio:{w}/{h};margin:0 auto;border-radius:10px;overflow:hidden;border:1px solid rgba(255,255,255,.1);background:#000;touch-action:none}}
.stage img.base,.stage .clip,.stage svg.pl{{position:absolute;inset:0;width:100%;height:100%}} img.base{{object-fit:fill}}
.clip{{clip-path:inset(0 0 0 var(--p,50%))}} .clip img{{position:absolute;inset:0;width:100%;height:100%;object-fit:fill}}
.ov{{opacity:.78}} .bar{{position:absolute;top:0;bottom:0;left:var(--p,50%);width:2px;margin-left:-1px;background:#fff;box-shadow:0 0 0 1px rgba(0,0,0,.4);pointer-events:none}}
.bar::after{{content:"⇆";position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:34px;height:34px;border-radius:50%;background:#fff;color:#111;display:grid;place-items:center;font-size:16px}}
input[type=range]{{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0}}
.lb{{position:absolute;top:10px;padding:4px 10px;border-radius:99px;background:rgba(0,0,0,.62);font-size:.68rem;font-weight:600;letter-spacing:.06em;pointer-events:none}}
.lb.l{{left:10px}} .lb.r{{right:10px}}
.lg{{display:flex;gap:6px 14px;flex-wrap:wrap;margin-top:9px;font-size:.72rem;color:#cfd4dc}} .lg i{{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}}
.nt{{margin-top:6px;font-size:.7rem;color:#8b93a1;line-height:1.5}}
</style></head><body><div class="wrap" id="wrap-root">
<div class="hd"><span>ÖNCE / SONRA · SÜRÜKLEYİN</span>{toggle}</div>
<div class="stage" id="stage"><img class="base" src="{pre_u}" alt="Önce görüntüsü">
<div class="clip"><img src="{post_u}" alt="Sonra görüntüsü">{ov_img}</div>
<svg class="pl" viewBox="0 0 100 100" preserveAspectRatio="none"><polygon points="{poly}" fill="none" stroke="#6ea8fe" stroke-width="0.7" vector-effect="non-scaling-stroke" style="stroke-width:2px"/></svg>
<div class="bar"></div><span class="lb l">{esc(pre_label)}</span><span class="lb r">{esc(post_label)}</span>
<input type="range" id="sl" min="0" max="100" value="50" aria-label="Önce ve sonra görüntüsü karşılaştırma kaydırıcısı"></div>
<div class="lg">{leg}</div><div class="nt">{esc(note)}</div></div>
<script>
const st=document.getElementById('stage'),sl=document.getElementById('sl');
function set(v){{st.style.setProperty('--p',v+'%')}} sl.addEventListener('input',e=>set(e.target.value)); set(sl.value);
const ov=document.getElementById('ov'),oi=document.getElementById('ovimg'); if(ov&&oi){{ov.addEventListener('change',()=>{{oi.style.display=ov.checked?'block':'none'}})}}
{_FIT_JS.replace("wf-root", "wrap-root")}
</script></body></html>"""


# --------------------------------------------------------------------------- #
# Sektör görünümleri
# --------------------------------------------------------------------------- #
def build_sector_views_html(result: an.ChangeResult, ins: an.InsuranceView, agri: an.AgriView, insured_tl: float) -> str:
    """Aynı sahne çiftinden iki sektör görünümü yan yana: sigorta (hasar/karar/tazminat) ve tarım (gelişim/stres)."""
    dcol = DECISION_COLORS.get(ins.decision, C_MUTED)
    hcol = HEALTH_COLORS.get(agri.health, C_MUTED)
    pair = f"{result.pre_meta.date:%d.%m.%Y} → {result.post_meta.date:%d.%m.%Y}"
    ins_notes = "".join(f"<li>{esc(n)}</li>" for n in ins.notes)
    agri_notes = "".join(f"<li>{esc(n)}</li>" for n in agri.notes)
    ratio = "—" if not math.isfinite(ins.damage_ratio) else f"%{fmt(100 * ins.damage_ratio, 1)}"
    dnbr = f'<span>Ortalama dNBR</span><span>{fmt(ins.mean_dnbr, 3)}</span>' if math.isfinite(ins.mean_dnbr) else ""
    sig = (f'<div class="wf-card"><p class="wf-h">SİGORTA GÖRÜNÜMÜ · AFET SONRASI HASAR DEĞERLENDİRMESİ</p>'
           f'<div style="margin:2px 0 10px"><span class="wf-pill" style="color:{dcol};border-color:{dcol}">{esc(ins.decision)}</span></div>'
           f'<div class="wf-big">{esc(ratio)} <small>hasar oranı (alan ağırlıklı)</small></div>'
           f'<div class="wf-kv"><span>Gösterge tazminat</span><span>₺{fmt(ins.indicative_claim_tl, 0)}</span><span>Sigorta bedeli</span><span>₺{fmt(insured_tl, 0)}</span>{dnbr}'
           f'<span>Özet</span><span>{esc(ins.severity_text)}</span></div><ul>{ins_notes}</ul>'
           f'<p class="wf-note">Karar, uygulamanın tek karar kuralıyla (Kalibrasyon sekmesindeki aynı eşikler) verilir. Gösterge niteliğindedir; ekspertizin yerine geçmez.</p></div>')
    veg_pct = "—" if not math.isfinite(agri.stressed_pct) else f"%{fmt(agri.stressed_pct, 1)}"
    tar = (f'<div class="wf-card"><p class="wf-h">TARIM GÖRÜNÜMÜ · BİTKİ GELİŞİMİ VE STRES</p>'
           f'<div style="margin:2px 0 10px"><span class="wf-pill" style="color:{hcol};border-color:{hcol}">{esc(agri.health.upper())}</span></div>'
           f'<div class="wf-big">{fmt(agri.mean_pre, 2)} → {fmt(agri.mean_post, 2)} <small>ortalama NDVI</small></div>'
           f'<div class="wf-kv"><span>Bitkili alan</span><span>{fmt(agri.veg_ha, 1)} ha</span><span>Stresli alan (NDVI ≥ %25 düştü)</span><span>{fmt(agri.stressed_ha, 1)} ha · {veg_pct}</span>'
           f'<span>Gelişen alan (ΔNDVI ≥ +0,05)</span><span>{fmt(agri.growth_ha, 1)} ha</span></div><ul>{agri_notes}</ul></div>')
    return _wrap(f'<p class="wf-note" style="margin:0 0 10px">Aynı parsel, aynı sahne çifti (<b>{esc(pair)}</b>), aynı çekirdek hesap — iki sektör için iki görünüm.</p><div class="wf-two">{sig}{tar}</div>')


# --------------------------------------------------------------------------- #
# Rekolte
# --------------------------------------------------------------------------- #
def build_yield_html(est: ym.YieldEstimate, model: ym.YieldModel, compare: Optional[pd.DataFrame], holdout: Optional[ym.Metrics], prior_yield: float) -> str:
    """Rekolte tahmini: değer + aralık çubuğu, özellik katkıları, doğruluk ölçütleri ve model karşılaştırması."""
    lo_ax = max(0.0, est.lo * 0.7)
    hi_ax = est.hi * 1.15 if est.hi > 0 else 1.0
    def pos(v: float) -> float:
        return 100.0 * (v - lo_ax) / max(hi_ax - lo_ax, 1e-9)

    rng = (f'<div style="position:relative;height:30px;margin:10px 0 4px"><div style="position:absolute;left:0;right:0;top:13px;height:4px;border-radius:4px;background:rgba(255,255,255,.08)"></div>'
           f'<div style="position:absolute;left:{pos(est.lo):.1f}%;width:{pos(est.hi) - pos(est.lo):.1f}%;top:9px;height:12px;border-radius:6px;background:rgba(110,168,254,.35);border:1px solid {C_ACCENT}"></div>'
           f'<div style="position:absolute;left:{pos(est.value):.1f}%;top:3px;width:3px;height:24px;margin-left:-1px;background:#fff;border-radius:2px"></div></div>')
    mx = max(float(est.contributions["katki_t_ha"].abs().max()), 1e-9)
    contrib = "".join(
        f'<tr><td>{esc(r.ozellik)}</td><td class="num">{fmt(r.deger, 2)}</td><td style="width:44%"><div style="display:flex;align-items:center;height:14px">'
        f'<div style="flex:1;display:flex;justify-content:flex-end"><span style="display:block;height:8px;border-radius:4px 0 0 4px;width:{max(0, -r.katki_t_ha) / mx * 100:.0f}%;background:{C_DANGER}"></span></div>'
        f'<div style="width:1px;height:14px;background:{C_DIM}"></div>'
        f'<div style="flex:1"><span style="display:block;height:8px;border-radius:0 4px 4px 0;width:{max(0, r.katki_t_ha) / mx * 100:.0f}%;background:{C_OK}"></span></div></div></td>'
        f'<td class="num">{"+" if r.katki_t_ha >= 0 else "−"}{fmt(abs(r.katki_t_ha), 2)}</td></tr>' for r in est.contributions.itertuples())
    m = model.loso
    met = [("LOSO RMSE", f"{fmt(m.rmse, 2)} t/ha"), ("LOSO MAPE", f"%{fmt(m.mape, 1)}"), ("LOSO R²", fmt(m.r2, 2)),
           (f"Aralık kapsaması (hedef %{int(est.level * 100)})", f"%{fmt(m.coverage, 0)}"), ("Eğitim", f"{model.n_train} gözlem · {model.n_seasons} sezon")]
    if holdout is not None:
        met.append(("Zaman-bazlı sınama (son sezonlar)", f"RMSE {fmt(holdout.rmse, 2)} · kapsama %{fmt(holdout.coverage, 0)}"))
    kv = "".join(f"<span>{esc(k)}</span><span>{esc(v)}</span>" for k, v in met)
    cmp_html = ""
    if compare is not None and not compare.empty:
        best = float(compare["rmse"].min())
        body = "".join(f'<tr><td>{esc(r.model)}</td><td class="num">{fmt(r.rmse, 3)}</td><td class="num">{fmt(r.mape, 1)}</td><td class="num">{fmt(r.r2, 2)}</td>'
                       f'<td>{"<span class=wf-ok>en iyi</span>" if abs(r.rmse - best) < 1e-12 else ""}</td></tr>' for r in compare.itertuples())
        cmp_html = (f'<p class="wf-h" style="margin-top:16px">MODEL KARŞILAŞTIRMASI · SEZON-DIŞI (LOSO) HATA</p><table><thead><tr><th>Model</th><th style="text-align:right">RMSE t/ha</th>'
                    f'<th style="text-align:right">MAPE %</th><th style="text-align:right">R²</th><th></th></tr></thead><tbody>{body}</tbody></table>')
    warns = "".join(f'<li class="{"wf-warn" if "ekstrapolasyon" in w or "≤ 0" in w else ""}">{esc(w)}</li>' for w in est.warnings)
    return _wrap(f'<div class="wf-card"><p class="wf-h">REKOLTE TAHMİNİ · {esc(model.crop.upper())} · {esc(ym.STAGE_LABELS[est.stage].upper())}</p>'
                 f'<div class="wf-big">{fmt(est.value, 2)} <small>t/ha · %{int(est.level * 100)} aralık {fmt(est.lo, 2)} – {fmt(est.hi, 2)}</small></div>{rng}'
                 f'<p class="wf-note">Geçmiş verim girdisi: {fmt(prior_yield, 2)} t/ha. Aralık, sezon-dışı (LOSO) hatalardan conformal yöntemle hesaplanır.</p>'
                 f'<table style="margin-top:8px"><thead><tr><th>Özellik</th><th style="text-align:right">Değer</th><th style="text-align:center">Katkı (t/ha)</th><th></th></tr></thead><tbody>{contrib}</tbody></table>'
                 f'<div class="wf-kv" style="margin-top:12px">{kv}</div>{cmp_html}<ul>{warns}</ul>'
                 f'<p class="wf-note">Rekolte yalnızca NDVI\'dan çıkarılmaz: geçmiş verim, sezon hava özetleri (yağış, GDD, sıcak gün, su dengesi), fenolojik evre ve NDVI birlikte kullanılır.</p></div>')


# --------------------------------------------------------------------------- #
# Eşik dayanağı ve ROC
# --------------------------------------------------------------------------- #
def build_evidence_html(evidence: pd.DataFrame) -> str:
    """Eşik dayanak tablosu (alan, eşik, değer, dayanak, rol, doğrulama durumu)."""
    def colr(s: str) -> str:
        if s.startswith("Literatür"):
            return C_OK
        return C_AMBER if s.startswith(("Sezgisel", "Yaygın", "Politika")) else C_MUTED

    rows = "".join(f'<tr><td>{esc(r.alan)}</td><td>{esc(r.esik)}</td><td class="num">{fmt(r.deger, 2)}</td><td style="font-size:.74rem">{esc(r.dayanak)}</td>'
                   f'<td style="font-size:.74rem">{esc(r.rol)}</td><td style="color:{colr(r.dogrulama)};font-size:.74rem">{esc(r.dogrulama)}</td></tr>' for r in evidence.itertuples())
    return _wrap(f'<div class="wf-card"><p class="wf-h">EŞİKLER NEDEN BU DEĞERLER?</p><table><thead><tr><th>Alan</th><th>Eşik</th><th style="text-align:right">Değer</th><th>Dayanak</th><th>Rol</th><th>Doğrulama</th></tr></thead>'
                 f'<tbody>{rows}</tbody></table><p class="wf-note">Yangın eşikleri USGS FIREMON dNBR sınıflarıdır (ABD ormanı, Landsat, CBI saha plotları). Sentinel-2\'ye ve tarım arazisine aktarımı '
                 f'bu projede <b>yerel olarak doğrulanmamıştır</b>; karar eşikleri bu sınırlara hizalı <b>politika seçimleridir</b>. Aşağıdaki doğrulama paneli kendi saha verinizle bunu sınamanızı sağlar.</p></div>')


def build_roc_svg(roc: pd.DataFrame, points: pd.DataFrame, auc: float, size: int = 300) -> str:
    """ROC eğrisi ve uygulamadaki eşiklerin işletim noktaları (yanlış pozitif oranı x, duyarlılık y)."""
    pad = 46
    s = size - 2 * pad
    def X(v: float) -> float:
        return pad + s * v

    def Y(v: float) -> float:
        return pad + s * (1 - v)

    g = [f'<rect x="{pad}" y="{pad}" width="{s}" height="{s}" fill="none" stroke="{C_LINE}"/>', f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}" stroke="{C_DIM}" stroke-dasharray="4 4"/>']
    for t in (0, 0.25, 0.5, 0.75, 1):
        g.append(f'<text x="{X(t)}" y="{size - pad + 16}" text-anchor="middle" fill="{C_MUTED}" font-size="10">{fmt(t, 2)}</text>'
                 f'<text x="{pad - 6}" y="{Y(t) + 3}" text-anchor="end" fill="{C_MUTED}" font-size="10">{fmt(t, 2)}</text>')
    pts = " ".join(f"{X(r.fpr):.1f},{Y(r.tpr):.1f}" for r in roc.itertuples())
    g.append(f'<polyline fill="none" stroke="{C_ACCENT}" stroke-width="2" points="{pts}"/>')
    for r in points.itertuples():
        if not (math.isfinite(r.fpr) and math.isfinite(r.sensitivity)):
            continue
        g.append(f'<circle cx="{X(r.fpr):.1f}" cy="{Y(r.sensitivity):.1f}" r="5" fill="{C_AMBER}" stroke="#000"><title>eşik {r.threshold:.2f}: FPR {r.fpr:.2f}, duyarlılık {r.sensitivity:.2f}</title></circle>'
                 f'<text x="{X(r.fpr) + 8:.1f}" y="{Y(r.sensitivity) + 12:.1f}" fill="{C_AMBER}" font-size="10" font-weight="600">{r.threshold:.2f}</text>')
    g.append(f'<text x="{size / 2}" y="{size - 6}" text-anchor="middle" fill="{C_MUTED}" font-size="10">yanlış pozitif oranı</text>'
             f'<text x="11" y="{size / 2}" fill="{C_MUTED}" font-size="10" transform="rotate(-90 11 {size / 2})" text-anchor="middle">duyarlılık</text>'
             f'<text x="{X(0.98)}" y="{Y(0.06)}" text-anchor="end" fill="#fff" font-size="12">AUC {fmt(auc, 3)}</text>')
    return _wrap(f'<svg viewBox="0 0 {size} {size}" style="width:100%;max-width:{size + 60}px;height:auto;font-family:Inter,sans-serif" role="img" aria-label="ROC eğrisi">{"".join(g)}</svg>')


def build_points_html(points: pd.DataFrame) -> str:
    """Eşik başına yanlış pozitif / yanlış negatif oranları (Wilson %95 aralığıyla) ve karışıklık sayıları."""
    def rate(v: float, lo: float, hi: float) -> str:
        return "—" if not math.isfinite(v) else f"%{fmt(100 * v, 1)} <span class='dim'>(%{fmt(100 * lo, 0)}–%{fmt(100 * hi, 0)})</span>"

    rows = "".join(f'<tr><td><b>{r.threshold:.2f}</b></td><td class="num">{r.tp}</td><td class="num">{r.fp}</td><td class="num">{r.tn}</td><td class="num">{r.fn}</td>'
                   f'<td class="num">{rate(r.fpr, r.fpr_alt, r.fpr_ust)}</td><td class="num">{rate(r.fnr, r.fnr_alt, r.fnr_ust)}</td><td class="num">{fmt(r.youden, 2)}</td></tr>' for r in points.itertuples())
    return _wrap(f'<div class="wf-card"><p class="wf-h">EŞİK BAŞINA HATA ORANLARI</p><table><thead><tr><th>Eşik (dNBR ≥)</th><th style="text-align:right">DP</th><th style="text-align:right">YP</th><th style="text-align:right">DN</th><th style="text-align:right">YN</th>'
                 f'<th style="text-align:right">Yanlış pozitif oranı</th><th style="text-align:right">Yanlış negatif oranı</th><th style="text-align:right">Youden J</th></tr></thead><tbody>{rows}</tbody></table>'
                 f'<p class="wf-note">DP/YP/DN/YN: doğru/yanlış pozitif/negatif. Yanlış pozitif = hasarsız parsele hasar denmesi; yanlış negatif = hasarlı parselin kaçırılması. Parantez: Wilson %95 aralığı.</p></div>')


_FIT_JS = """(function(){var r=document.getElementById('wf-root');if(!r){return;}function fit(){var h=Math.ceil(r.getBoundingClientRect().height)+10;try{var f=window.frameElement;if(f){f.style.height=h+'px';f.setAttribute('height',h);}}catch(e){}}
window.addEventListener('load',fit);window.addEventListener('resize',fit);if(window.ResizeObserver){new ResizeObserver(fit).observe(r);}setTimeout(fit,120);setTimeout(fit,700);})();"""


def build_document(inner: str) -> str:
    """Bileşen HTML'ini kendi içinde çalışan tam belgeye sarar (``components.html`` çerçevesi için).

    Belge, çerçeve yüksekliğini içeriğe uydurmaya çalışır (aynı kökenli çerçevede ``frameElement``); bu mümkün değilse
    çağıranın verdiği sabit yükseklik geçerli kalır.

    Args:
        inner: ``build_*_html`` çıktılarının birleşimi.
    """
    return ('<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            "<style>html,body{margin:0;background:transparent}#wf-root{padding:2px 1px 6px}"
            ".wf-split{display:grid;grid-template-columns:minmax(260px,1fr) minmax(0,2fr);gap:14px;align-items:start}"
            "@media(max-width:760px){.wf-split{grid-template-columns:1fr}}.wf-stack>*+*{margin-top:14px}</style></head>"
            f'<body><div id="wf-root" class="wf-stack">{inner}</div><script>{_FIT_JS}</script></body></html>')


def alert_rows(kind: str, ins: an.InsuranceView, agri: an.AgriView, result: an.ChangeResult, approve_dnbr: float, partial_dnbr: float) -> List[Tuple[str, str, str]]:
    """Otomatik uyarı koşulları: ``(düzey, başlık, ileti)`` satırları (boş liste = tetiklenen uyarı yok).

    Koşullar uygulamadaki karar eşikleriyle aynıdır (yangında dNBR ≥ tam onay eşiği → KRİTİK).
    """
    out: List[Tuple[str, str, str]] = []
    area = f"{fmt(result.region_ha, 1)} ha bölgenin {fmt(result.affected_ha, 1)} ha'ı etkilendi (%{fmt(result.affected_pct, 1)})"
    if kind == "yangin" and math.isfinite(ins.mean_dnbr):
        if ins.mean_dnbr >= approve_dnbr:
            out.append(("KRİTİK", "Yüksek şiddetli yangın izi", f"Ortalama dNBR {fmt(ins.mean_dnbr, 3)} ≥ {fmt(approve_dnbr, 2)}. {area}."))
        elif ins.mean_dnbr >= partial_dnbr:
            out.append(("UYARI", "Orta şiddette yangın izi", f"Ortalama dNBR {fmt(ins.mean_dnbr, 3)} ≥ {fmt(partial_dnbr, 2)}. {area}."))
    if kind == "sel" and math.isfinite(result.affected_pct) and result.affected_pct >= 20.0:
        out.append(("UYARI", "Geniş su altında kalma", f"{area}."))
    if kind == "kuraklik" and agri.health in ("Stresli", "Kritik"):
        out.append(("UYARI", "Kuraklık stresi", f"Bitki sağlığı: {agri.health}; stresli alan %{fmt(agri.stressed_pct, 1)}."))
    if kind == "tarim" and math.isfinite(agri.stressed_pct) and agri.stressed_pct >= 25.0:
        out.append(("UYARI", "Bitki stresi yaygın", f"Stresli alan %{fmt(agri.stressed_pct, 1)} (NDVI ≥ %25 düştü)."))
    if result.reliability.level == "Düşük":
        out.append(("VERİ KALİTESİ", "Güvenilirlik düşük", "Bulut/gölge ya da zamanlama nedeniyle sonuç dikkatle okunmalı; ek sahne ya da saha doğrulaması önerilir."))
    return out


def build_alert_html(rows: Sequence[Tuple[str, str, str]], region_name: str, is_demo: bool) -> str:
    """Bildirim önizlemesi (SİMÜLASYON): tetiklenen uyarılar ve gönderilecek metin; gerçek SMS/e-posta gönderilmez."""
    cols = {"KRİTİK": C_DANGER, "UYARI": C_AMBER, "VERİ KALİTESİ": C_ACCENT}
    if not rows:
        body = '<p class="wf-note">Bu analizde otomatik uyarı koşulu tetiklenmedi.</p>'
    else:
        items = "".join(
            f'<div style="border:1px solid {cols.get(lv, C_MUTED)};border-left-width:4px;border-radius:9px;padding:9px 12px;margin:8px 0;background:#0d1016">'
            f'<span class="wf-pill" style="color:{cols.get(lv, C_MUTED)};border-color:{cols.get(lv, C_MUTED)};font-size:.6rem;padding:2px 9px">{esc(lv)}</span> '
            f'<b style="margin-left:6px">{esc(t)}</b><div class="wf-note" style="margin-top:5px">{esc(m)}</div></div>' for lv, t, m in rows)
        body = items
    demo = " Veri sentetiktir." if is_demo else ""
    return _wrap(f'<div class="wf-card"><p class="wf-h">OTOMATİK UYARI · BİLDİRİM ÖNİZLEMESİ (SİMÜLASYON)</p>{body}'
                 f'<p class="wf-note">Bölge: {esc(region_name)}. Bu bir önizlemedir: gerçek SMS / e-posta / webhook gönderilmez.{demo}</p></div>')


def build_support_html(sv: an.SupportView, valid_ha: float) -> str:
    """Kuraklık destek ödemesi simülasyonu kartı (parametrik; varsayımsal kural)."""
    col = C_OK if sv.eligible and not sv.needs_review else (C_AMBER if sv.needs_review else C_MUTED)
    status = "DOĞRULAMA GEREKLİ" if sv.needs_review else ("UYGUN" if sv.eligible else "UYGUN DEĞİL")
    ratio = "—" if not math.isfinite(sv.damage_ratio) else f"%{fmt(100 * sv.damage_ratio, 1)}"
    notes = "".join(f"<li>{esc(n)}</li>" for n in sv.notes)
    return _wrap(f'<div class="wf-card"><p class="wf-h">KURAKLIK DESTEK ÖDEMESİ · PARAMETRİK SİMÜLASYON</p>'
                 f'<div style="margin:2px 0 10px"><span class="wf-pill" style="color:{col};border-color:{col}">{status}</span></div>'
                 f'<div class="wf-big">₺{fmt(sv.payment_tl, 0)} <small>simüle edilen ödeme</small></div>'
                 f'<div class="wf-kv"><span>Hasar oranı</span><span>{ratio} (eşik %{fmt(100 * sv.min_ratio, 0)})</span><span>Ölçülebilir alan</span><span>{fmt(valid_ha, 1)} ha</span>'
                 f'<span>Ödemeye esas alan</span><span>{fmt(sv.payable_ha, 1)} ha</span><span>Birim destek (varsayımsal)</span><span>₺{fmt(sv.unit_tl_ha, 0)}/ha</span></div>'
                 f'<ul>{notes}</ul></div>')


def build_diag_html(rows: Sequence[Any]) -> str:
    """Canlı bağlantı tanılama tablosu (``live_sources.DiagnosticRow`` listesi)."""
    trs = "".join(f'<tr><td>{esc(r.step)}</td><td>{"<span class=wf-ok>✓ çalışıyor</span>" if r.ok else "<span class=wf-bad>✗ hata</span>"}</td><td style="font-size:.76rem">{esc(r.detail)}</td>'
                  f'<td class="num">{r.ms} ms</td></tr>' for r in rows)
    return _wrap(f'<div class="wf-card"><p class="wf-h">CANLI BAĞLANTI SINAMASI</p><table><thead><tr><th>Adım</th><th>Durum</th><th>Ayrıntı</th><th style="text-align:right">Süre</th></tr></thead><tbody>{trs}</tbody></table>'
                 f'<p class="wf-note">Başarısız adım varsa analiz canlı kaynakla çalışmaz; sessizce demoya geçilmez.</p></div>')
