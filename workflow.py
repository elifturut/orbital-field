"""
workflow.py — Analiz iş akışının orkestrasyonu (Streamlit'ten bağımsız)
======================================================================

Altı adımı tek çağrıda yürütür; ``DemoSource`` (sentetik) ve ``LiveSource`` (gerçek) aynı arayüzle çalışır:

1. Bölge (``analysis.Region``) → 2. analiz türü → 3. veri kaynağı ve tarih aralığı →
4. katalog, bulut süzgeci, sahne okuma ve indeks zaman serisi → 5. önce/sonra seçimi, değişim haritası, istatistik ve
güvenilirlik, hava ve yangın noktası kanıtı → 6. aynı sonuçtan sigorta ve tarım görünümleri.

Hata politikası: canlı kaynak hata verirse **sessizce demoya düşülmez**; ``WorkflowError`` yükselir, arayüz gösterir.
Hava ve yangın noktası gibi *yardımcı kanıtlar* ise başarısız olursa analiz sürer, hata ``AnalysisRun`` içinde bildirilir.

Geliştirici: Elif Turut
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

import analysis as an
import data_engine as de
import data_sources as dsrc

__all__ = ["WorkflowError", "AnalysisRequest", "AnalysisRun", "SceneSource", "run_analysis", "YOY_EXTENSION_DAYS"]

#: Kuraklık analizinde "geçen yıl aynı dönem" için kataloğun geriye uzatıldığı gün sayısı.
YOY_EXTENSION_DAYS = 380
#: Zaman serisi için en çok işlenecek sahne (olaya yakın olanlar önceliklidir).
MAX_SERIES_SCENES = 40
SERIES_PX = 64


class WorkflowError(RuntimeError):
    """İş akışı bir adımda ilerleyemedi (ileti kullanıcıya gösterilebilir, kimlik bilgisi içermez)."""


class SceneSource(Protocol):
    """Sahne kaynağı arayüzü (``DemoSource`` ve ``LiveSource``)."""

    name: str
    is_demo: bool

    def catalog(self, region: an.Region, start: dt.date, end: dt.date) -> List[an.SceneMeta]:
        """Aralıktaki sahneler."""

    def load(self, meta: an.SceneMeta, region: an.Region, grid: an.Grid, bands: Sequence[str]) -> an.SceneData:
        """Sahneyi ızgaraya okur."""

    def weather(self, region: an.Region, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Günlük hava tablosu."""

    def hotspots(self, region: an.Region, start: dt.date, end: dt.date, event_day: Optional[dt.date] = None,
                 grid: Optional[an.Grid] = None) -> pd.DataFrame:
        """Aktif yangın noktaları."""


@dataclass(frozen=True)
class AnalysisRequest:
    """İş akışı girdisi.

    Attributes:
        kind: ``analysis.KINDS`` anahtarı.
        region: Bölge.
        start: Tarih aralığı başı.
        end: Tarih aralığı sonu.
        event_date: Olay (hasar/başlangıç/referans) günü.
        max_cloud: Karo bulut üst sınırı (%); üstündeki sahneler elenir.
        max_px: Son analiz ızgarasında uzun kenar piksel sayısı.
        pre_id: Elle seçilen önce sahnesi (``None`` = otomatik).
        post_id: Elle seçilen sonra sahnesi (``None`` = otomatik).
        insured_tl: Sigorta bedeli (TL).
        rules: Sigorta karar kuralları.
    """

    kind: str
    region: an.Region
    start: dt.date
    end: dt.date
    event_date: dt.date
    max_cloud: float = 40.0
    max_px: int = 140
    pre_id: Optional[str] = None
    post_id: Optional[str] = None
    insured_tl: float = 1_000_000.0
    rules: de.ClaimRules = field(default_factory=de.ClaimRules)

    def validate(self) -> None:
        """Girdiyi doğrular.

        Raises:
            WorkflowError: Geçersiz girdide.
        """
        if self.kind not in an.KINDS:
            raise WorkflowError(f"Bilinmeyen analiz türü: {self.kind!r}")
        if self.end < self.start:
            raise WorkflowError("Bitiş tarihi başlangıçtan önce olamaz.")
        if (self.end - self.start).days > 400:
            raise WorkflowError("Tarih aralığı en çok 400 gün olabilir.")
        if not self.start <= self.event_date <= self.end:
            raise WorkflowError("Olay tarihi seçilen tarih aralığının içinde olmalı.")
        if not 0.0 <= float(self.max_cloud) <= 100.0:
            raise WorkflowError("Bulut eşiği 0 ile 100 arasında olmalı.")
        if not 16 <= int(self.max_px) <= 400:
            raise WorkflowError("Izgara boyutu 16 ile 400 piksel arasında olmalı.")
        if not (math.isfinite(float(self.insured_tl)) and self.insured_tl >= 0):
            raise WorkflowError("Sigorta bedeli geçerli bir sayı olmalı.")


@dataclass
class AnalysisRun:
    """İş akışı çıktısı (arayüz ve rapor bunu okur).

    Attributes:
        request: Girdi.
        source_name: Kaynak adı.
        is_demo: Sentetik mi.
        catalog: Tüm sahneler + durum sütunu.
        series: İndeks zaman serisi (``analysis.build_series``).
        pair: Seçilen çift.
        result: Değişim sonucu.
        rgb_pre: Önce gerçek renkli önizleme (olabilir ``None``).
        rgb_post: Sonra gerçek renkli önizleme.
        weather: Günlük hava (olabilir ``None``).
        weather_rows: Koşul doğrulaması satırları.
        weather_error: Hava alınamadıysa ileti.
        hotspots: Bölgeyle eşleşen yangın noktaları (yalnızca yangın).
        hotspots_error: Yangın noktası alınamadıysa ileti.
        insurance: Sigorta görünümü.
        agri: Tarım görünümü.
        warnings: Kullanıcıya gösterilecek uyarılar.
        notes: Kaynağın bilgi notları (ör. SCL yok).
    """

    request: AnalysisRequest
    source_name: str
    is_demo: bool
    catalog: pd.DataFrame
    series: pd.DataFrame
    pair: an.PairChoice
    result: an.ChangeResult
    rgb_pre: Optional[np.ndarray]
    rgb_post: Optional[np.ndarray]
    weather: Optional[pd.DataFrame]
    weather_rows: List[an.EvidenceRow]
    weather_error: Optional[str]
    hotspots: Optional[pd.DataFrame]
    hotspots_error: Optional[str]
    insurance: an.InsuranceView
    agri: an.AgriView
    warnings: List[str]
    notes: List[str]


Progress = Optional[Callable[[float, str], None]]


def _pick_for_series(metas: Sequence[an.SceneMeta], event: dt.date, cap: int, yoy: bool = False) -> List[an.SceneMeta]:
    """Sahne sayısı sınırı aşılırsa olaya (yoy'da geçen yılın aynı gününe de) en yakın ``cap`` sahneyi tutar (tarihe göre sıralı)."""
    if len(metas) <= cap:
        return list(metas)
    last_year = event - dt.timedelta(days=365)

    def dist(m: an.SceneMeta) -> int:
        d = abs((m.date - event).days)
        return min(d, abs((m.date - last_year).days)) if yoy else d

    keep = sorted(metas, key=dist)[:cap]
    return sorted(keep, key=lambda m: (m.date, m.scene_id))


def run_analysis(source: SceneSource, req: AnalysisRequest, progress: Progress = None) -> AnalysisRun:
    """Altı adımlı analizi çalıştırır.

    Args:
        source: Sahne kaynağı.
        req: Girdi.
        progress: ``(oran 0–1, ileti)`` geri çağrısı (isteğe bağlı).

    Returns:
        ``AnalysisRun``.

    Raises:
        WorkflowError: Katalog boşsa, sahneler okunamazsa ya da uygun önce/sonra çifti bulunamazsa.
    """
    req.validate()

    def tick(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    kind = an.KINDS[req.kind]
    yoy = req.kind == "kuraklik"
    warnings: List[str] = []
    cat_start = req.start - dt.timedelta(days=YOY_EXTENSION_DAYS) if yoy else req.start

    tick(0.05, "Katalog aranıyor")
    try:
        metas = source.catalog(req.region, cat_start, req.end)
    except dsrc.DataSourceError as exc:
        raise WorkflowError(f"Uydu kataloğuna ulaşılamadı: {exc}") from None
    if not metas:
        raise WorkflowError("Bu bölge ve tarih aralığında sahne bulunamadı. Tarih aralığını genişletin.")

    unknown = [m for m in metas if not math.isfinite(m.cloud_pct)]
    kept = an.filter_scenes(metas, req.max_cloud, cat_start, req.end)
    if unknown:
        warnings.append(f"{len(unknown)} sahnenin bulut bilgisi yok; belirsizlik sessizce kabul edilmediği için elendi.")
    if not kept:
        raise WorkflowError(f"Bulut eşiği (%{req.max_cloud:g}) altında sahne yok. Eşiği yükseltin ya da tarih aralığını genişletin.")

    series_scenes = _pick_for_series(kept, req.event_date, MAX_SERIES_SCENES, yoy)
    bands = an.bands_for(req.kind, with_rgb=False)
    sgrid = an.make_grid(req.region.bbox, SERIES_PX)
    smask = an.polygon_mask(sgrid, req.region.ring)

    stats: List[an.SceneStats] = []
    status: Dict[str, str] = {}
    first_err: Optional[str] = None
    for i, meta in enumerate(series_scenes):
        tick(0.10 + 0.50 * i / max(len(series_scenes), 1), f"Sahne okunuyor ({i + 1}/{len(series_scenes)}): {meta.date:%d.%m.%Y}")
        try:
            data = source.load(meta, req.region, sgrid, bands)
            st, _ = an.scene_stats(data, kind.index, smask)
            stats.append(st)
        except (dsrc.DataSourceError, an.AnalysisError) as exc:
            status[meta.scene_id] = f"okunamadı: {exc}"
            first_err = first_err or str(exc)
    if not stats:
        raise WorkflowError(f"Hiçbir sahne okunamadı. İlk hata: {first_err}")
    if status:
        warnings.append(f"{len(status)} sahne okunamadı ve seri dışında kaldı.")
    series = an.build_series(stats)

    tick(0.62, "Önce/sonra sahneleri seçiliyor")
    if req.pre_id or req.post_id:
        ids = set(series["sahne"])
        pair = an.PairChoice(req.pre_id if req.pre_id in ids else None, req.post_id if req.post_id in ids else None,
                             ("Sahneler elle seçildi.",))
        if pair.pre_id is None or pair.post_id is None:
            raise WorkflowError("Elle seçilen sahne(ler) seride yok (elenmiş ya da okunamamış olabilir).")
    else:
        pair = an.pick_pair(series, req.event_date, mode="yoy" if yoy else "sequential")
    if pair.pre_id is None or pair.post_id is None:
        raise WorkflowError("Uygun önce/sonra sahne çifti bulunamadı: " + " ".join(pair.notes) +
                            " Bulut eşiğini yükseltin ya da tarih aralığını genişletin.")

    by_id = {m.scene_id: m for m in series_scenes}
    grid = an.make_grid(req.region.bbox, req.max_px)
    full_bands = an.bands_for(req.kind, with_rgb=True)
    tick(0.68, "Seçilen sahneler tam çözünürlükte okunuyor")
    try:
        pre_data = source.load(by_id[pair.pre_id], req.region, grid, full_bands)
        post_data = source.load(by_id[pair.post_id], req.region, grid, full_bands)
    except (dsrc.DataSourceError, an.AnalysisError) as exc:
        raise WorkflowError(f"Seçilen sahneler okunamadı: {exc}") from None
    if pre_data.meta.date >= post_data.meta.date:
        raise WorkflowError("Önce sahnesi sonra sahnesinden önce olmalı.")

    tick(0.80, "Değişim analizi")
    try:
        result = an.analyse_change(req.kind, pre_data, post_data, req.region, grid, req.event_date)
    except an.AnalysisError as exc:
        raise WorkflowError(str(exc)) from None

    tick(0.88, "Hava durumu doğrulaması")
    weather: Optional[pd.DataFrame] = None
    weather_error: Optional[str] = None
    try:
        weather = source.weather(req.region, cat_start, req.end)
    except (dsrc.DataSourceError, an.AnalysisError, ValueError) as exc:
        weather_error = str(exc)
    weather_rows = an.weather_evidence(req.kind, weather, req.event_date)

    hotspots: Optional[pd.DataFrame] = None
    hotspots_error: Optional[str] = None
    if req.kind == "yangin":
        tick(0.93, "Yangın noktaları eşleştiriliyor")
        h_start = req.event_date - dt.timedelta(days=3)
        h_end = min(req.end, req.event_date + dt.timedelta(days=10))
        try:
            raw = source.hotspots(req.region, h_start, h_end, req.event_date, grid)
            hotspots = an.match_hotspots(raw, req.region, h_start, h_end)
            if hotspots.empty:
                warnings.append("Olay tarihi çevresinde bölgeyle eşleşen yangın noktası bulunamadı; tarihi ya da bölgeyi kontrol edin.")
        except (dsrc.DataSourceError, an.AnalysisError, ValueError) as exc:
            hotspots_error = str(exc)

    ins = an.insurance_view(result, req.rules, req.insured_tl)
    agri = an.agri_view(result)

    rel = result.reliability
    if rel.level == "Düşük":
        warnings.append(f"Güvenilirlik düşük (puan {rel.score}/100): sonucu saha teyidi olmadan kullanmayın.")
    for m in (result.pre_meta, result.post_meta):
        if math.isfinite(m.cloud_pct) and m.cloud_pct > an.CLOUD_WARN_PCT:
            warnings.append(f"{m.date:%d.%m.%Y} sahnesinin karo bulutu %{m.cloud_pct:.0f} (> %{an.CLOUD_WARN_PCT:g}).")
    if source.is_demo:
        warnings.append("Bu sonuç SENTETİK demo verisinden üretildi; gerçek uydu görüntüsü değildir.")

    cat_rows = []
    kept_ids = {m.scene_id for m in kept}
    series_ids = {m.scene_id for m in series_scenes}
    for m in sorted(metas, key=lambda x: (x.date, x.scene_id)):
        if m.scene_id == pair.pre_id:
            state = "ÖNCE (seçildi)"
        elif m.scene_id == pair.post_id:
            state = "SONRA (seçildi)"
        elif m.scene_id in status:
            state = status[m.scene_id]
        elif m.scene_id not in kept_ids:
            state = "elendi: bulut" if math.isfinite(m.cloud_pct) else "elendi: bulut bilgisi yok"
        elif m.scene_id not in series_ids:
            state = "seri dışı (sahne sınırı)"
        else:
            state = "seride"
        cat_rows.append((pd.Timestamp(m.date), m.scene_id, m.platform, m.cloud_pct, state))
    catalog = pd.DataFrame(cat_rows, columns=["tarih", "sahne", "uydu", "bulut_karo", "durum"])

    tick(1.0, "Tamam")
    return AnalysisRun(req, source.name, bool(source.is_demo), catalog, series, pair, result, an.true_color(pre_data),
                       an.true_color(post_data), weather, weather_rows, weather_error, hotspots, hotspots_error, ins, agri,
                       warnings, list(getattr(source, "notes", [])))
