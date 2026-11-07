"""
workflow_page.py — "ANALİZ İŞ AKIŞI" sekmesinin Streamlit sayfası
================================================================

Altı adım, tek sayfada:

1. Haritadan bölge ya da parsel seç (demo senaryo · haritadan çiz · koordinat · GeoJSON)
2. Analiz türünü seç (sel · yangın · kuraklık · tarımsal gelişim)
3. Uydu veri kaynağını ve tarih aralığını belirle (demo ya da canlı Sentinel-2; bulut süzgeci)
4. Gerçek uydu görüntülerini işle
5. Değişim haritası, istatistik ve güvenilirlik göstergelerini oluştur (+ aynı parselin sigorta ve tarım görünümleri, rekolte, doğrulama)
6. Sonucu sigorta ya da tarım raporuna (PDF) dönüştür

Uygulamanın yardımcıları ``Kit`` içinde verilir (döngüsel içe aktarma olmaması için); Streamlit işlev içinde içe aktarılır, böylece
saf yardımcılar (GeoJSON ayrıştırma, varsayılan tarihler, uyarı koşulları) Streamlit'siz sınanabilir.

Dürüstlük: canlı kaynak hata verirse sessizce demoya geçilmez; demo çıktılar her yerde SENTETİK diye etiketlenir.

Geliştirici: Elif Turut
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import importlib.util
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

import analysis as an
import data_engine as de
import data_sources as ds
import demo_imagery as di
import live_sources as ls
import thresholds as th
import workflow as wf
import workflow_ui as ui
import yield_model as ym

__all__ = ["Kit", "render_workflow_tab", "parse_geojson_text", "default_dates", "scenario_for", "fingerprint", "yield_for_run", "KIND_ORDER", "MODES"]

LOGGER = logging.getLogger(__name__)

KIND_ORDER: Tuple[str, ...] = ("yangin", "sel", "kuraklik", "tarim")
KIND_HELP: Dict[str, str] = {"yangin": "Yanık alan: NBR farkı (dNBR)", "sel": "Su altında kalan alan: MNDWI", "kuraklik": "Bitki stresi: NDVI, geçen yılın aynı dönemine göre",
                             "tarim": "Bitki gelişimi: NDVI farkı, rekolte tahmini"}
MODE_DEMO, MODE_MAP, MODE_COORD, MODE_GEOJSON = "Demo senaryo", "Haritadan çiz", "Koordinat", "GeoJSON"
MODES: Tuple[str, ...] = (MODE_DEMO, MODE_MAP, MODE_COORD, MODE_GEOJSON)
SRC_DEMO, SRC_LIVE = "Demo (sentetik)", "Canlı — Sentinel-2 (Earth Search)"
MAX_GEOJSON_BYTES = 2_000_000
MAX_HISTORY_BYTES = 5_000_000
MAX_VALIDATION_BYTES = 5_000_000
EXPERT_DECISIONS: Tuple[str, ...] = ("Beklemede", "Onay", "Kısmi onay", "Ekspertiz / saha incelemesi gerekli", "Ret")
_AMBER = "rgba(242,184,75,.5)"


@dataclass(frozen=True)
class Kit:
    """Uygulamadan gelen yardımcılar (``app.py`` doldurur).

    Attributes:
        render_html: HTML parçasını sayfaya basar.
        show_frame: Tam HTML belgesini yalıtılmış çerçevede gösterir (``(markup, height)``).
        panel: Kenarlıklı kap.
        hero: ``build_hero_html``.
        section_title: ``build_section_title``.
        kpi_grid: ``build_kpi_grid``.
        kpi: ``Kpi`` sınıfı.
        rules: Geçerli sigorta karar kuralları.
        draw_map: ``(bölge | None, anahtar) → çizilen GeoJSON geometrisi | None`` (folium Draw).
        read_secret: Ortam/gizli ayar okuyucu (``ad → değer``; yoksa boş).
        csv_bytes: DataFrame → CSV baytları.
        show_table: DataFrame tablosu.
        html_escape: HTML kaçışı.
    """

    render_html: Callable[[str], None]
    show_frame: Callable[[str, int], None]
    panel: Callable[[], Any]
    hero: Callable[..., str]
    section_title: Callable[..., str]
    kpi_grid: Callable[..., str]
    kpi: Callable[..., Any]
    rules: Callable[[], Any]
    draw_map: Callable[[Optional[an.Region], str], Optional[Mapping[str, Any]]]
    read_secret: Callable[[str], str]
    csv_bytes: Callable[[pd.DataFrame], bytes]
    show_table: Callable[..., None]
    html_escape: Callable[[object], str]


# --------------------------------------------------------------------------- #
# Saf yardımcılar (Streamlit'siz sınanabilir)
# --------------------------------------------------------------------------- #
def parse_geojson_text(text: str, name: str = "Yüklenen bölge") -> an.Region:
    """GeoJSON metninden bölge kurar: ``Polygon``, ``MultiPolygon`` (ilk çokgen), ``Feature`` ve ``FeatureCollection`` (ilk çokgen özellik).

    Args:
        text: GeoJSON metni.
        name: Bölge adı.

    Raises:
        an.RegionError: Metin geçersiz, çok büyük ya da çokgen içermiyorsa; bölge kısıtları sağlanmıyorsa.
    """
    raw = (text or "").strip()
    if not raw:
        raise an.RegionError("GeoJSON boş.")
    if len(raw.encode("utf-8")) > MAX_GEOJSON_BYTES:
        raise an.RegionError("GeoJSON dosyası çok büyük (en çok 2 MB).")
    try:
        obj = json.loads(raw)
    except (ValueError, RecursionError):
        raise an.RegionError("GeoJSON okunamadı (geçerli JSON değil).") from None

    def polygon_of(geom: Any) -> Optional[Mapping[str, Any]]:
        if not isinstance(geom, Mapping):
            return None
        if geom.get("type") == "Polygon":
            return geom
        if geom.get("type") == "MultiPolygon":
            polys = geom.get("coordinates")
            if isinstance(polys, list) and polys:
                return {"type": "Polygon", "coordinates": polys[0]}
        return None

    if isinstance(obj, Mapping) and obj.get("type") == "FeatureCollection":
        feats = obj.get("features")
        found = None
        for f in feats if isinstance(feats, list) else []:
            found = polygon_of(f.get("geometry") if isinstance(f, Mapping) else None)
            if found:
                break
        geom = found
    elif isinstance(obj, Mapping) and obj.get("type") == "Feature":
        geom = polygon_of(obj.get("geometry"))
    else:
        geom = polygon_of(obj)
    if geom is None:
        raise an.RegionError("GeoJSON içinde çokgen (Polygon / MultiPolygon) bulunamadı.")
    return an.region_from_geojson(geom, name)


def default_dates(kind: str, custom: bool, today: Optional[dt.date] = None) -> Tuple[dt.date, dt.date, dt.date]:
    """Varsayılan ``(başlangıç, bitiş, olay)`` tarihleri.

    Demo senaryoda hazır pencere; özel bölgede bugünden geriye doğru makul bir pencere (kullanıcı olay gününü kendi olayına göre değiştirmelidir).
    """
    if not custom:
        scn = di.preset(kind)
        s, e = di.default_window(scn)
        return (s, e, scn.event_date)
    end = (today or dt.datetime.now(dt.timezone.utc).date()) - dt.timedelta(days=3)
    if kind == "kuraklik":
        return (end - dt.timedelta(days=150), end, end - dt.timedelta(days=45))
    return (end - dt.timedelta(days=150), end, end - dt.timedelta(days=75))


def scenario_for(kind: str, region: an.Region, start: dt.date, end: dt.date, event: dt.date) -> di.DemoScenario:
    """Kullanıcının bölge ve tarihleriyle özelleştirilmiş sentetik senaryo (kuraklıkta katalog bir yıl geriye uzatılır)."""
    base = di.preset(kind)
    # kuraklıkta "geçen yıl aynı dönem" için iş akışı kataloğu geriye uzatır; sentetik katalog da o kadar geriye üretilmeli
    first = start - dt.timedelta(days=wf.YOY_EXTENSION_DAYS) if kind == "kuraklik" else start
    return dataclasses.replace(base, region=region, start=first, end=end, event_date=event, title=f"Sentetik — {region.name}")


def fingerprint(req: wf.AnalysisRequest, live: bool, crop: str, prior: float) -> str:
    """Girdilerin kısa özeti (sonucun eskidiğini anlamak için)."""
    payload = repr((req.kind, req.region.ring, req.start, req.end, req.event_date, req.max_cloud, req.insured_tl, live, crop, round(float(prior), 3),
                    dataclasses.astuple(req.rules) if dataclasses.is_dataclass(req.rules) else repr(req.rules)))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def parcel_id_for(region: an.Region) -> str:
    """Bölge köşelerinden türetilen kararlı varsayılan parsel kimliği."""
    return "PRS-" + hashlib.sha256(repr(region.ring).encode()).hexdigest()[:6].upper()


@dataclass(frozen=True)
class YieldBundle:
    """Rekolte paneli için hesaplanmış çıktılar."""

    estimate: ym.YieldEstimate
    model: ym.YieldModel
    compare: Optional[pd.DataFrame]
    holdout: Optional[ym.Metrics]
    synthetic: bool


def yield_for_run(run: wf.AnalysisRun, crop: str, prior_yield: float, history: Optional[pd.DataFrame] = None) -> YieldBundle:
    """Çalıştırmanın NDVI serisi ve havasından rekolte tahmini üretir.

    Args:
        run: Analiz çıktısı (tarım).
        crop: Ürün adı.
        prior_yield: Parselin geçmiş verimi (t/ha).
        history: Kullanıcının geçmiş verim tablosu; ``None`` ise sentetik demo havuzu kullanılır (etiketlenir).

    Raises:
        ym.YieldError: Veri/model yetersizse.
    """
    if run.weather is None or run.weather.empty:
        raise ym.YieldError("Hava verisi yok; rekolte tahmini için sezon hava özeti gerekir.")
    if history is None:
        df, pool = ym.demo_model_pool(crop)
        synthetic = True
    else:
        df, synthetic = history, False
        pool = {}
        last_err: Optional[ym.YieldError] = None
        for stage in ym.STAGES:
            try:
                pool[stage] = ym.fit_yield_model(df, crop, stage)
            except ym.YieldError as exc:
                last_err = exc
        if not pool:
            raise last_err or ym.YieldError("Geçmiş tablodan model eğitilemedi.")
    est, _feats = ym.estimate_from_series(pool, run.series, run.weather, crop, prior_yield)
    model = pool[est.stage]
    try:
        cmp_ = ym.compare_models(df, crop, est.stage)
    except ym.YieldError:
        cmp_ = None
    try:
        hold: Optional[ym.Metrics] = ym.evaluate_holdout(df, crop, est.stage)
    except ym.YieldError:
        hold = None
    return YieldBundle(est, model, cmp_, hold, synthetic)


# --------------------------------------------------------------------------- #
# Sayfa
# --------------------------------------------------------------------------- #
def _pct0(x: float) -> str:
    return "?" if x is None or not np.isfinite(x) else f"%{x:.0f}"


def _h(kit: Kit, text: object) -> str:
    return kit.html_escape(text)


def _note(kit: Kit, text: str, level: str = "info") -> None:
    if level == "warn":
        kit.render_html(f'<div class="note info" style="border-color:{_AMBER};color:#f2b84b">{text}</div>')
    else:
        kit.render_html(f'<div class="note {level}">{text}</div>')


def _frame(kit: Kit, inner: str, height: int) -> None:
    kit.show_frame(ui.build_document(inner), height)


def _has_rasterio() -> bool:
    return importlib.util.find_spec("rasterio") is not None


def _region_section(kit: Kit, st: Any, kind: str, mode: str) -> Tuple[Optional[an.Region], str, bool]:
    """Adım 1 gövdesi: seçilen kipe göre bölge kurar. Dönüş: ``(bölge | None, ileti, ileti_hata_mı)``."""
    if mode == MODE_DEMO:
        return di.preset(kind).region, "", False
    if mode == MODE_MAP:
        current = st.session_state.get("wf_drawn")
        geom = kit.draw_map(current, f"wf_map_{int(st.session_state.get('wf_map_n', 0))}")
        if geom:
            try:
                reg = an.region_from_geojson(geom, "Haritadan çizilen bölge")
            except an.RegionError as exc:
                return current, str(exc), True
            st.session_state["wf_drawn"] = reg
            current = reg
        if st.button("Çizimi temizle", key="wf_clear_draw"):
            st.session_state.pop("wf_drawn", None)
            st.session_state["wf_map_n"] = int(st.session_state.get("wf_map_n", 0)) + 1  # harita anahtarı değişince çizimler sıfırlanır
            current = None
        if current is None:
            return None, "Haritada çokgen ya da dikdörtgen çizin (sol üstteki çizim araçları).", False
        return current, "", False
    if mode == MODE_COORD:
        c1, c2, c3 = st.columns(3)
        lat = c1.number_input("Merkez enlem (°)", min_value=-85.0, max_value=85.0, value=39.0, step=0.001, format="%.5f", key="wf_lat")
        lon = c2.number_input("Merkez boylam (°)", min_value=-180.0, max_value=180.0, value=35.0, step=0.001, format="%.5f", key="wf_lon")
        half = c3.number_input("Yarı kenar (km)", min_value=0.05, max_value=12.0, value=1.0, step=0.1, key="wf_half")
        try:
            return an.region_from_center("Koordinatla seçilen bölge", float(lat), float(lon), float(half)), "", False
        except an.RegionError as exc:
            return None, str(exc), True
    text = st.text_area("GeoJSON (Polygon / MultiPolygon / Feature / FeatureCollection)", value="", height=110, key="wf_geojson_text")
    up = st.file_uploader("ya da .geojson / .json dosyası yükleyin", type=["geojson", "json"], key="wf_geojson_file")
    if up is not None:
        try:
            text = up.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            return None, "Dosya UTF-8 metin olmalı.", True
    if not (text or "").strip():
        return None, "GeoJSON yapıştırın ya da dosya yükleyin.", False
    try:
        return parse_geojson_text(text, "GeoJSON bölgesi"), "", False
    except an.RegionError as exc:
        return None, str(exc), True


def _make_source(kit: Kit, live: bool, scn: di.DemoScenario) -> Any:
    if not live:
        return di.DemoSource(scn)
    creds = ds.load_credentials({"FIRMS_MAP_KEY": kit.read_secret("FIRMS_MAP_KEY")})
    return ls.LiveSource(creds=creds)


def _execute(kit: Kit, st: Any, req: wf.AnalysisRequest, live: bool, scn: di.DemoScenario) -> Optional[wf.AnalysisRun]:
    """Analizi ilerleme çubuğuyla çalıştırır; hata iletisini gösterir ve ``None`` döndürür."""
    bar = st.progress(0.0, text="Başlıyor…")

    def cb(frac: float, msg: str) -> None:
        bar.progress(float(min(max(frac, 0.0), 1.0)), text=str(msg))

    try:
        return wf.run_analysis(_make_source(kit, live, scn), req, cb)
    except (wf.WorkflowError, ds.DataSourceError, an.AnalysisError) as exc:
        extra = " Canlı kaynak seçili olduğu için sentetik veriye geçilmedi." if live else ""
        _note(kit, f"<b>Analiz yapılamadı.</b> {_h(kit, exc)}{extra}", "err")
    except Exception as exc:  # noqa: BLE001 — arayüz sınırı: ayrıntı günlüğe, kullanıcıya tür adı
        LOGGER.exception("Analiz iş akışı beklenmeyen hatayla durdu")
        _note(kit, f"<b>Beklenmeyen hata</b> ({_h(kit, type(exc).__name__)}); ayrıntı günlüğe yazıldı.", "err")
    finally:
        bar.empty()
    return None


def _results(kit: Kit, st: Any, run: wf.AnalysisRun, fp: str, crop: str, prior: float, support: Tuple[float, float]) -> None:
    """Adım 5: sonuç panoları."""
    res, req = run.result, run.request
    rel, ins, agri = res.reliability, run.insurance, run.agri
    rules = req.rules
    lvl_col = {"Yüksek": "#4cc38a", "Orta": "#f2b84b", "Düşük": "#fc3d21"}.get(rel.level, "")
    K = kit.kpi
    kit.render_html(kit.kpi_grid([
        K("Bölge", f"{ui.fmt(res.region_ha, 1)} ha", f"ölçülebilir <b>%{ui.fmt(res.valid_pct, 0)}</b>"),
        K("Etkilenen alan", f"{ui.fmt(res.affected_ha, 1)} ha", f"ölçülebilir alanın <b>%{ui.fmt(res.affected_pct, 1)}</b>'i", "#f2b84b"),
        K("Güvenilirlik", f"{rel.score}/100", f"<b>{kit.html_escape(rel.level)}</b> · veri kalitesi göstergesi", lvl_col, alert=rel.level == "Düşük"),
        K("Sahne çifti", f"{res.pre_meta.date:%d.%m} → {res.post_meta.date:%d.%m}", f"bulut {_pct0(res.pre_meta.cloud_pct)} / {_pct0(res.post_meta.cloud_pct)}"),
        K("Sigorta görünümü", ins.decision, f"hasar oranı <b>%{ui.fmt(100 * ins.damage_ratio, 1)}</b>"),
        K("Tarım görünümü", agri.health, f"NDVI <b>{ui.fmt(agri.mean_pre, 2)} → {ui.fmt(agri.mean_post, 2)}</b>"),
    ], columns=3))
    if rel.level == "Düşük":
        _note(kit, "<b>Güvenilirlik düşük.</b> Bulut/gölge ya da zamanlama nedeniyle sonuç dikkatle okunmalı; daha berrak sahneler seçin ya da saha doğrulaması yapın.", "warn")
    for w in run.warnings:
        _note(kit, _h(kit, w), "warn")

    cache = st.session_state.setdefault("wf_docs", {})
    if cache.get("fp") != fp:
        cache.clear()
        cache["fp"] = fp

    # önce / sonra kaydırıcı
    kit.render_html(kit.section_title("ÖNCE / SONRA KARŞILAŞTIRMA", "kaydırıcıyı sürükleyin · değişim katmanını açıp kapatın"))
    if "compare" not in cache:
        shape = res.cls.shape + (3,)
        gray = np.full(shape, 190, dtype=np.uint8)
        pre_rgb = run.rgb_pre if run.rgb_pre is not None else gray
        post_rgb = run.rgb_post if run.rgb_post is not None else gray
        classes = [c[1] for c in res.kind.classes]
        overlay = an.colorize(res.delta, 0.0, 1.0, mask=res.region_mask, classes=classes, class_idx=res.cls, alpha=205)
        note = ("Gerçek renkli önizleme üretilemedi; gri zemin kullanıldı. " if run.rgb_pre is None or run.rgb_post is None else "") + \
               ("SENTETİK demo görüntü — gerçek uydu verisi değildir." if run.is_demo else "Sentinel-2 L2A, gerçek renk (B04-B03-B02).")
        cache["compare"] = ui.build_compare_html(pre_rgb, post_rgb, overlay, ui.region_to_pixels(req.region, res.grid.bbox), f"ÖNCE · {res.pre_meta.date:%d.%m.%Y}",
                                                 f"SONRA · {res.post_meta.date:%d.%m.%Y}", "Değişim sınıfları", [(c[1], c[0]) for c in res.kind.classes], note)
    kit.show_frame(cache["compare"], 760)

    kit.render_html(kit.section_title("SAHNELER VE İNDEKS ZAMAN SERİSİ", "bulut rozetli sahne listesi · ortalama / medyan / min / maks"))
    _frame(kit, f'<div class="wf-split">{ui.build_scene_list_html(run.catalog)}'
                f'{ui.build_series_svg(run.series, res.kind.index.upper(), req.event_date, res.pre_meta.date, res.post_meta.date)}</div>', 640)

    kit.render_html(kit.section_title("DEĞİŞİM İSTATİSTİĞİ VE GÜVENİLİRLİK", "sınıf alanları · veri kalitesi bileşenleri"))
    _frame(kit, f'<div class="wf-two">{ui.build_class_table_html(res)}{ui.build_reliability_html(rel)}</div>', 520)

    kit.render_html(kit.section_title("HAVA VE YANGIN KANITI", "olay koşullarını bağımsız veriyle doğrulama"))
    wx_note = "Kaynak: SENTETİK demo verisi" if run.is_demo else "Kaynak: Open-Meteo arşivi (ERA5 tabanlı yeniden-analiz; istasyon ölçümü değil)"
    evidence = ui.build_weather_html(run.weather_rows, run.weather, req.event_date, run.weather_error, wx_note)
    if req.kind == "yangin":
        evidence += ui.build_hotspots_html(run.hotspots, run.hotspots_error, (req.start, req.end), run.is_demo)
    _frame(kit, evidence, 760 if req.kind == "yangin" else 480)

    kit.render_html(kit.section_title("AYNI PARSEL · İKİ SEKTÖR", "ortak analiz motoru: sigorta hasar değerlendirmesi ve tarım gelişimi"))
    alerts = ui.alert_rows(req.kind, ins, agri, res, rules.approve_dnbr, rules.partial_dnbr)
    sector = ui.build_sector_views_html(res, ins, agri, req.insured_tl) + ui.build_alert_html(alerts, req.region.name, run.is_demo)
    if req.kind == "kuraklik":
        try:
            sector += ui.build_support_html(an.drought_support(res, ins, support[0], support[1]), res.valid_ha)
        except an.AnalysisError as exc:
            _note(kit, _h(kit, exc), "err")
    _frame(kit, sector, 900 if req.kind == "kuraklik" else 700)

    if req.kind == "tarim":
        _yield_panel(kit, st, run, crop, prior)
    else:
        _note(kit, "Rekolte tahmini yalnızca <b>Tarımsal gelişim</b> analizinde üretilir (ürün türü, geçmiş verim ve fenolojik evre gerektirir).")


def _yield_panel(kit: Kit, st: Any, run: wf.AnalysisRun, crop: str, prior: float) -> None:
    kit.render_html(kit.section_title("REKOLTE TAHMİNİ", "geçmiş verim · meteoroloji · ürün türü · fenolojik evre · NDVI — güven aralığı ve doğruluk ölçütleriyle"))
    up = st.file_uploader("Kendi geçmiş verim tablonuz (CSV; boşsa SENTETİK demo model kullanılır)", type=["csv"], key="wf_history")
    st.download_button("↓  ŞABLON (CSV)", data=ym.history_template_csv().encode("utf-8"), file_name="gecmis_verim_sablonu.csv", mime="text/csv", key="wf_history_tpl")
    hist: Optional[pd.DataFrame] = None
    if up is not None:
        raw = up.getvalue()
        if len(raw) > MAX_HISTORY_BYTES:
            _note(kit, "Geçmiş verim dosyası çok büyük (en çok 5 MB).", "err")
            return
        try:
            hist = ym.load_history_csv(raw.decode("utf-8-sig"))
        except (ym.YieldError, UnicodeDecodeError) as exc:
            _note(kit, f"Geçmiş verim tablosu okunamadı: {_h(kit, exc)}", "err")
            return
    try:
        bundle = yield_for_run(run, crop, prior, hist)
    except ym.YieldError as exc:
        _note(kit, f"<b>Rekolte tahmini üretilemedi.</b> {_h(kit, exc)}", "warn")
        return
    if bundle.synthetic:
        _note(kit, _h(kit, ym.SYNTHETIC_NOTE), "warn")
    _frame(kit, ui.build_yield_html(bundle.estimate, bundle.model, bundle.compare, bundle.holdout, prior), 1150)
    st.session_state["wf_yield"] = (bundle, crop, prior)


def _threshold_panel(kit: Kit, st: Any) -> None:
    """Eşiklerin dayanağı ve saha doğrulaması (analizden bağımsız)."""
    kit.render_html(kit.section_title("BİLİMSEL DOĞRULAMA · EŞİKLER", "hangi indeks, neden bu eşikler, yanlış pozitif / negatif oranları"))
    _frame(kit, ui.build_evidence_html(th.evidence_table(kit.rules())), 640)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Yöntem sınaması — bilinen yanma alanlı SENTETİK sahneler")
        if st.button("SENTETİK DOĞRULAMAYI ÇALIŞTIR", key="wf_synth"):
            with st.spinner("Sentetik sahneler üretiliyor…"):
                df = th.synthetic_validation()
                st.session_state["wf_synth_rep"] = (th.evaluate_thresholds(df, n_boot=200), th.synthetic_sweep(base=df))
    with c2:
        st.caption("Gerçek doğrulama — kendi saha etiketleriniz (parsel başına dNBR ve hasarlı/hasarsız)")
        up = st.file_uploader("Saha doğrulama CSV'si", type=["csv"], key="wf_truth")
        st.download_button("↓  ŞABLON (CSV)", data=th.ground_truth_template_csv().encode("utf-8"), file_name="saha_dogrulama_sablonu.csv", mime="text/csv", key="wf_truth_tpl")
    if up is not None:
        raw = up.getvalue()
        if len(raw) > MAX_VALIDATION_BYTES:
            _note(kit, "Doğrulama dosyası çok büyük (en çok 5 MB).", "err")
        else:
            try:
                rep = th.evaluate_thresholds(th.load_ground_truth_csv(raw.decode("utf-8-sig")), n_boot=300)
                _threshold_report(kit, rep, f"SAHA DOĞRULAMASI · {rep.n} etiketli parsel ({rep.n_pos} hasarlı, {rep.n_neg} hasarsız)", None)
            except (th.ThresholdError, UnicodeDecodeError) as exc:
                _note(kit, f"Doğrulama yapılamadı: {_h(kit, exc)}", "err")
    synth = st.session_state.get("wf_synth_rep")
    if synth:
        rep, sweep = synth
        _note(kit, _h(kit, th.SYNTHETIC_VALIDATION_NOTE), "warn")
        _threshold_report(kit, rep, f"SENTETİK DOĞRULAMA · {rep.n} blok ({rep.n_pos} hasarlı, {rep.n_neg} hasarsız)", sweep)


def _threshold_report(kit: Kit, rep: th.ThresholdReport, title: str, sweep: Optional[pd.DataFrame]) -> None:
    kit.render_html(f'<div class="note info"><b>{_h(kit, title)}</b><br>AUC <b>{ui.fmt(rep.auc, 3)}</b> (%95 aralık {ui.fmt(rep.auc_ci[0], 2)}–{ui.fmt(rep.auc_ci[1], 2)}) · '
                    f'Youden J en iyi eşik <b>{ui.fmt(rep.best, 2)}</b> (%95 aralık {ui.fmt(rep.best_ci[0], 2)}–{ui.fmt(rep.best_ci[1], 2)}).</div>')
    _frame(kit, f'<div class="wf-split">{ui.build_roc_svg(rep.roc, rep.points, rep.auc)}{ui.build_points_html(rep.points)}</div>', 420)
    for n in rep.notes:
        _note(kit, _h(kit, n), "warn")
    if sweep is not None and not sweep.empty:
        st_cfg = {"hasarli_tanimi": "Hasarlı tanımı (gerçek şiddet ≥)", "en_iyi_esik": "En iyi eşik", "fpr_027": "YP oranı @0,27", "fnr_027": "YN oranı @0,27",
                  "fpr_044": "YP oranı @0,44", "fnr_044": "YN oranı @0,44"}
        kit.show_table(sweep.rename(columns=st_cfg).round(3), None, 170)
        kit.render_html('<div class="note info">En iyi eşik, "hasarlı"nın nasıl tanımlandığına göre kayar: tek bir sayı evrensel değildir. '
                        "Bu yüzden eşikler kendi saha etiketlerinizle sınanmalıdır.</div>")


def _report_section(kit: Kit, st: Any, run: wf.AnalysisRun, fp: str) -> None:
    """Adım 6: PDF rapor formu."""
    req = run.request
    kit.render_html(kit.section_title("ADIM 6 · RAPORA DÖNÜŞTÜR", "sigorta hasar raporu ya da tarım gelişim raporu (PDF) — uzman kararı, imza ve tarih alanlarıyla"))
    rtype_default = 1 if req.kind == "tarim" else 0
    with kit.panel():
        rtype_label = st.radio("Rapor türü", ["Sigorta hasar raporu", "Tarım gelişim raporu"], index=rtype_default, horizontal=True, key=f"wf_rtype_{req.kind}")
        rtype = "tarim" if rtype_label.startswith("Tarım") else "sigorta"
        c1, c2, c3 = st.columns(3)
        parcel = c1.text_input("Parsel kimliği", value=parcel_id_for(req.region), max_chars=60, key=f"wf_parcel_{fp}")
        policy = c2.text_input("Poliçe no", value="", max_chars=60, key="wf_policy", disabled=rtype != "sigorta")
        insured = c3.text_input("Sigortalı / üretici (isteğe bağlı)", value="", max_chars=80, key="wf_insured")
        e1, e2, e3 = st.columns(3)
        expert = e1.text_input("Ekspertiz uzmanı", value="", max_chars=80, key="wf_expert")
        reg_no = e2.text_input("Sicil / ruhsat no", value="", max_chars=40, key="wf_reg")
        decision = e3.selectbox("Uzman kararı", list(EXPERT_DECISIONS), key="wf_decision")
        note = st.text_area("Uzman notu", value="", height=90, max_chars=1500, key="wf_expert_note")
        st.caption("Saha fotoğrafı ve konum notu (isteğe bağlı) — konum elle girilir; dosyadaki EXIF/GPS verisi okunmaz.")
        p1, p2, p3 = st.columns([2, 1, 1])
        photo_up = p1.file_uploader("Saha fotoğrafı (JPEG/PNG, en çok 10 MB)", type=["jpg", "jpeg", "png"], key="wf_photo")
        lat0, lon0 = req.region.centre
        plat = p2.number_input("Fotoğraf enlemi (°)", min_value=-90.0, max_value=90.0, value=float(round(lat0, 6)), format="%.6f", key=f"wf_plat_{fp}")
        plon = p3.number_input("Fotoğraf boylamı (°)", min_value=-180.0, max_value=180.0, value=float(round(lon0, 6)), format="%.6f", key=f"wf_plon_{fp}")
        pcap = st.text_input("Fotoğraf açıklaması / çekim zamanı", value="", max_chars=160, key="wf_pcap")
        go = st.button("PDF RAPORU OLUŞTUR", key="wf_make_pdf", type="primary")
    if go:
        try:
            import workflow_report as wr

            photo = wr.FieldPhoto(photo_up.getvalue(), float(plat), float(plon), pcap, "") if photo_up is not None else None
            ysum = None
            bundle = st.session_state.get("wf_yield")
            if rtype == "tarim" and bundle and req.kind == "tarim":
                b, crop_name, prior_y = bundle
                ysum = wr.YieldSummary(crop_name, ym.STAGE_LABELS[b.estimate.stage], b.estimate.value, b.estimate.lo, b.estimate.hi, b.estimate.level, float(prior_y),
                                       b.model.loso.rmse, b.model.loso.mape, b.synthetic, b.estimate.warnings)
            meta = wr.ReportMeta(rtype, parcel, policy, insured, expert, reg_no, "" if decision == "Beklemede" else decision, note, photo, ysum)
            pdf = wr.build_pdf(run, meta)
            st.session_state["wf_pdf"] = (fp, pdf, f"orbital-field-{rtype}-raporu-{req.event_date:%Y%m%d}.pdf")
        except ImportError:
            _note(kit, "PDF için <b>reportlab</b> paketi gerekir (requirements.txt).", "err")
        except Exception as exc:  # noqa: BLE001 — ReportError ve beklenmeyen hatalar
            LOGGER.exception("PDF rapor üretilemedi")
            _note(kit, f"<b>Rapor üretilemedi.</b> {_h(kit, exc) if exc.__class__.__name__ == 'ReportError' else _h(kit, type(exc).__name__)}", "err")
    saved = st.session_state.get("wf_pdf")
    if saved and saved[0] == fp:
        _note(kit, f"Rapor hazır ({len(saved[1]) // 1024} KB). Belge, imza ve tarih alanları doldurulana kadar bir karar belgesi değildir.", "ok")
        st.download_button("↓  PDF RAPORU İNDİR", data=saved[1], file_name=saved[2], mime="application/pdf", key="wf_pdf_dl")


def render_workflow_tab(kit: Kit) -> None:
    """ANALİZ İŞ AKIŞI sekmesini çizer (yetki denetimi çağıranda yapılır)."""
    import streamlit as st

    kit.render_html(kit.hero(
        "ÇEKİRDEK ÖZELLİK · ANALİZ İŞ AKIŞI", "BÖLGE SEÇ, UYDUYU İŞLE,", "RAPORA DÖNÜŞTÜR",
        "Haritadan parsel seçin, sel · yangın · kuraklık · tarımsal gelişim analizini çalıştırın; değişim haritası, istatistik ve güvenilirlik göstergeleri üretin; "
        "aynı sonuçtan sigorta ya da tarım raporu alın.",
        ["SENTINEL-2 L2A · BULUT SÜZGECİ", "ÖNCE / SONRA", "NASA FIRMS · OPEN-METEO", "PDF RAPOR"]))

    tracker, box1, box2, box3, box4 = st.container(), st.container(), st.container(), st.container(), st.container()

    # ---- adım 2 (önce okunur: demo bölge türe bağlıdır)
    with box2:
        kit.render_html(kit.section_title("ADIM 2 · ANALİZ TÜRÜ", "sel · yangın · kuraklık · tarımsal gelişim"))
        kind = st.radio("Analiz türü", list(KIND_ORDER), format_func=lambda k: an.KINDS[k].label, horizontal=True, key="wf_kind", label_visibility="collapsed")
        spec = an.KINDS[kind]
        st.caption(f"{KIND_HELP[kind]} · {spec.method}")

    # ---- adım 1
    with box1:
        kit.render_html(kit.section_title("ADIM 1 · BÖLGE VEYA PARSEL", "haritadan çizin, koordinat girin, GeoJSON yükleyin ya da demo senaryo seçin"))
        mode = st.radio("Bölge kaynağı", list(MODES), horizontal=True, key="wf_region_mode")
        region, region_msg, region_is_err = _region_section(kit, st, kind, mode)
        if region_msg:
            _note(kit, _h(kit, region_msg), "err" if region_is_err else "info")
        if region is not None:
            kit.render_html(ui.build_region_card_html(region, "Kurgusal demo bölge." if mode == MODE_DEMO else ""))

    custom = mode != MODE_DEMO
    # ---- adım 3
    with box3:
        kit.render_html(kit.section_title("ADIM 3 · VERİ KAYNAĞI VE TARİH ARALIĞI", "bulut süzgeci, olay günü, sigorta bedeli"))
        if custom:
            src = st.radio("Uydu veri kaynağı", [SRC_DEMO, SRC_LIVE], horizontal=True, key="wf_source")
            live = src == SRC_LIVE
        else:
            live = False
            st.caption("Demo senaryoda veri kaynağı SENTETİKTİR; gerçek veriyle çalışmak için bölgeyi haritadan çizin / koordinat girin ve canlı kaynağı seçin.")
        d_start, d_end, d_event = default_dates(kind, custom)
        tag = f"{kind}_{'c' if custom else 'd'}_{'live' if live else 'demo'}"
        today = max(dt.datetime.now(dt.timezone.utc).date(), d_end)
        c1, c2 = st.columns([3, 2])
        picked = c1.date_input("Tarih aralığı", value=(d_start, d_end), min_value=dt.date(2017, 1, 1), max_value=today, format="DD.MM.YYYY", key=f"wf_dates_{tag}")
        event = c2.date_input("Olay / referans günü", value=d_event, min_value=dt.date(2017, 1, 1), max_value=today, format="DD.MM.YYYY", key=f"wf_event_{tag}")
        s1, s2, s3 = st.columns(3)
        max_cloud = s1.slider("Karo bulut üst sınırı (%)", 0, 100, 40, 5, key="wf_cloud")
        insured_tl = s2.number_input("Sigorta bedeli (₺)", min_value=0.0, value=1_000_000.0, step=50_000.0, key="wf_insured_tl")
        max_px = s3.slider("Izgara (piksel, uzun kenar)", 48, 220, 140, 4, key="wf_px", help="Büyüdükçe hesap ve okuma süresi artar.")
        crop, prior = "Buğday", 4.8
        if kind == "tarim":
            t1, t2 = st.columns(2)
            crop = t1.selectbox("Ürün", [c.name for c in de.CROPS], key="wf_crop")
            prior = float(t2.number_input("Parselin geçmiş verimi (t/ha)", min_value=0.1, max_value=40.0, value=4.8, step=0.1, key="wf_prior"))
        support_unit, support_min = 1000.0, 0.30
        if kind == "kuraklik":
            q1, q2 = st.columns(2)
            support_unit = float(q1.number_input("Destek simülasyonu: birim destek (₺/ha, VARSAYIMSAL)", min_value=0.0, value=1000.0, step=100.0, key="wf_support_unit"))
            support_min = float(q2.slider("Destek simülasyonu: uygunluk eşiği (hasar oranı %, VARSAYIMSAL)", 5, 80, 30, 5, key="wf_support_min")) / 100.0
        if live:
            have_key = bool(kit.read_secret("FIRMS_MAP_KEY"))
            if not _has_rasterio():
                _note(kit, "<b>rasterio</b> kurulu değil: canlı Sentinel-2 okuması çalışmaz (requirements.txt).", "warn")
            if kind == "yangin" and not have_key:
                _note(kit, "<b>FIRMS_MAP_KEY</b> tanımlı değil: yangın noktaları alınamaz (analiz sürer, bu kanıt eksik kalır).", "warn")
            if st.button("CANLI BAĞLANTIYI SINA", key="wf_diag"):
                with st.spinner("Katalog, pencere okuma, hava ve FIRMS sınanıyor…"):
                    probe = region.centre if region is not None else (39.0, 35.0)
                    st.session_state["wf_diag_rows"] = ls.run_diagnostics(_make_source(kit, True, di.preset(kind)), probe[0], probe[1])
            rows = st.session_state.get("wf_diag_rows")
            if rows:
                _frame(kit, ui.build_diag_html(rows), 60 + 44 * len(rows))

    # ---- adım 4
    req: Optional[wf.AnalysisRequest] = None
    with box4:
        kit.render_html(kit.section_title("ADIM 4 · GÖRÜNTÜLERİ İŞLE", "sahneler bulunur, bulut süzülür, indeksler hesaplanır, önce/sonra çifti seçilir"))
        problem = ""
        if region is None:
            problem = "Önce bir bölge seçin (adım 1)."
        elif not (isinstance(picked, (tuple, list)) and len(picked) == 2):
            problem = "Tarih aralığının her iki ucunu da seçin."
        else:
            try:
                req = wf.AnalysisRequest(kind=kind, region=region, start=picked[0], end=picked[1], event_date=event, max_cloud=float(max_cloud), max_px=int(max_px),
                                         insured_tl=float(insured_tl), rules=kit.rules())
                req.validate()
            except wf.WorkflowError as exc:
                problem, req = str(exc), None
        if problem:
            _note(kit, _h(kit, problem), "info")
        fp = fingerprint(req, live, crop, prior) if req is not None else ""
        run_clicked = st.button("ANALİZİ ÇALIŞTIR", key="wf_run", type="primary", disabled=req is None)
        if run_clicked and req is not None:
            scn = scenario_for(kind, req.region, req.start, req.end, req.event_date)
            run = _execute(kit, st, req, live, scn)
            if run is not None:
                st.session_state["wf_run"] = (fp, run, crop, prior)
                st.session_state.pop("wf_pdf", None)
                st.session_state.pop("wf_yield", None)

    saved = st.session_state.get("wf_run")
    fresh = bool(saved and req is not None and saved[0] == fp)
    done: List[int] = [2]
    if region is not None:
        done.append(1)
    if req is not None:
        done.append(3)
    if fresh:
        done += [4, 5]
        pdf_saved = st.session_state.get("wf_pdf")
        if pdf_saved and pdf_saved[0] == fp:
            done.append(6)
    current = next((i for i in range(1, 7) if i not in done), 6)
    with tracker:
        kit.render_html(ui.build_steps_html(done, current))

    if not saved:
        _note(kit, "Henüz analiz çalıştırılmadı. Adımları doldurup <b>ANALİZİ ÇALIŞTIR</b>'a basın.")
    else:
        run_fp, run, run_crop, run_prior = saved
        if not fresh:
            _note(kit, "Girdiler değişti: aşağıdaki sonuçlar <b>önceki çalıştırmaya</b> aittir. Yeniden çalıştırın.", "warn")
        kit.render_html(kit.section_title("ADIM 5 · SONUÇ", f"{run.source_name} · {run.request.region.name}"))
        _results(kit, st, run, run_fp, run_crop, run_prior, (support_unit, support_min))
        _report_section(kit, st, run, run_fp)
    _threshold_panel(kit, st)

