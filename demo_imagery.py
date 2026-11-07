"""
demo_imagery.py — Çevrimdışı **sentetik** Sentinel-2 benzeri sahne üretici (demo kaynağı)
=========================================================================================

Gerçek uydu görüntüsü **değildir**. Canlı Sentinel-2 bağlantısı kurulamadığında (ya da kullanıcı demo seçtiğinde) analiz
iş akışının tamamını uçtan uca çalıştırmak için kullanılır. Üretilen her şey arayüzde "SENTETİK" etiketiyle gösterilir.

Dünya modeli (tohuma bağlı, tekrarlanabilir)
--------------------------------------------
* Voronoi hücreleriyle orman / buğday-benzeri kış ürünü / yaz ürünü / bahçe / çıplak arazi örtüsü.
* Her örtünün fenolojik eğrisi (çift-lojistik) → bitki örtüsü oranı ``fv(gün)``.
* Bantlar; bitki, toprak, yanmış alan ve su uç-üyelerinin doğrusal karışımıdır (yaklaşık yansıtım değerleri).
* Olay etkisi: yangın izi (dNBR), sel (kalıcı nehir + yükselip çekilen taşkın), kuraklık (kademeli bitki stresi) ya da hafif
  tarımsal stres.
* Bulut (SCL 8/9), ince sirrus (10) ve bulut gölgesi (3) sahne başına rastgele; karo bulutu yerel orandan sapar.

``truth`` (gerçek durum) alanları da döndürülür; böylece ``thresholds.py`` sınıflandırmayı **bilinen gerçeğe** karşı sınayabilir.
Bu bir *yöntem doğrulamasıdır*, sahada doğrulama yerine geçmez.

Geliştirici: Elif Turut
"""

from __future__ import annotations

import datetime as dt
import functools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import analysis as an

__all__ = ["default_window", "DemoScenario", "DemoWorld", "DemoSource", "preset", "preset_keys", "world", "catalog", "load_scene", "demo_weather",
           "demo_hotspots", "truth_masks", "DEMO_LABEL", "ENDMEMBERS"]

DEMO_LABEL = "SENTETİK DEMO — gerçek uydu görüntüsü değildir"

#: Yaklaşık yansıtım uç-üyeleri (mavi, yeşil, kırmızı, YKI, KDKI-1, KDKI-2).
ENDMEMBERS: Dict[str, Tuple[float, ...]] = {
    "veg": (0.030, 0.070, 0.040, 0.450, 0.200, 0.100),
    "soil": (0.100, 0.130, 0.170, 0.220, 0.300, 0.280),
    "burn": (0.040, 0.050, 0.070, 0.100, 0.170, 0.200),
    "water": (0.060, 0.070, 0.040, 0.020, 0.010, 0.010),
}
_COVER = ("orman", "kis_urunu", "yaz_urunu", "bahce", "cıplak")
_COVER_P: Dict[str, Tuple[float, ...]] = {
    "yangin": (0.60, 0.12, 0.10, 0.08, 0.10), "sel": (0.12, 0.40, 0.18, 0.15, 0.15),
    "kuraklik": (0.05, 0.45, 0.25, 0.15, 0.10), "tarim": (0.0, 0.92, 0.04, 0.03, 0.01),
}


@dataclass(frozen=True)
class DemoScenario:
    """Sentetik senaryo tanımı.

    Attributes:
        kind: ``analysis.KINDS`` anahtarı.
        region: Analiz bölgesi.
        event_date: Olay (yangın/sel/kuraklık başlangıcı ya da tarımda referans) günü.
        start: Katalog başlangıcı.
        end: Katalog sonu.
        seed: Rastgelelik tohumu.
        title: Görünen ad.
    """

    kind: str
    region: an.Region
    event_date: dt.date
    start: dt.date
    end: dt.date
    seed: int = 7
    title: str = ""


def _poly(name: str, centre: Tuple[float, float], half_km: float, shape: Tuple[Tuple[float, float], ...]) -> an.Region:
    """Merkez ve yarı kenardan, kutu içine oransal yerleşen düzensiz çokgen bölge kurar."""
    box = an.region_from_center(name, centre[0], centre[1], half_km)
    w, s, e, n = box.bbox
    ring = tuple((w + fx * (e - w), s + fy * (n - s)) for fx, fy in shape)
    return an.Region(name, ring)


_SHAPE = ((0.18, 0.22), (0.62, 0.12), (0.88, 0.38), (0.80, 0.80), (0.42, 0.90), (0.14, 0.64))


def default_window(scn: DemoScenario) -> Tuple[dt.date, dt.date]:
    """Arayüzde varsayılan analiz tarih aralığı (kuraklıkta katalogdaki geçen yıl kısmı hariç; onu iş akışı ekler)."""
    if scn.kind == "kuraklik":
        return (scn.start + dt.timedelta(days=365), scn.end)
    return (scn.start, scn.end)


def preset_keys() -> Tuple[str, ...]:
    """Hazır senaryo anahtarları (analiz türleriyle aynı)."""
    return tuple(an.KINDS)


def preset(kind: str) -> DemoScenario:
    """Hazır sentetik senaryo.

    Konumlar yalnızca gerçekçi bir coğrafi bağlam içindir; anlatılan olaylar **kurgusaldır**, gerçek bir olayı temsil etmez.

    Args:
        kind: ``yangin``/``sel``/``kuraklik``/``tarim``.
    """
    if kind == "yangin":
        return DemoScenario(kind, _poly("Demo orman parseli (yangın)", (36.85, 28.27), 1.6, _SHAPE), dt.date(2026, 7, 22),
                            dt.date(2026, 6, 1), dt.date(2026, 9, 30), 11, "Orman yangını — Akdeniz kıyısı (kurgusal)")
    if kind == "sel":
        return DemoScenario(kind, _poly("Demo taşkın bölgesi", (41.02, 40.52), 1.6, _SHAPE), dt.date(2026, 7, 14),
                            dt.date(2026, 6, 5), dt.date(2026, 9, 10), 23, "Sel / taşkın — Karadeniz kıyısı (kurgusal)")
    if kind == "kuraklik":
        return DemoScenario(kind, _poly("Demo kuraklık parseli", (37.87, 32.48), 1.6, _SHAPE), dt.date(2026, 6, 15),
                            dt.date(2025, 4, 20), dt.date(2026, 8, 31), 37, "Kuraklık — İç Anadolu (kurgusal)")
    if kind == "tarim":
        return DemoScenario(kind, _poly("Demo tarım parseli", (37.05, 35.40), 1.6, _SHAPE), dt.date(2026, 4, 25),
                            dt.date(2026, 2, 20), dt.date(2026, 6, 30), 41, "Tarımsal gelişim — Çukurova (kurgusal)")
    raise an.AnalysisError(f"Bilinmeyen senaryo: {kind!r}")


def _smooth(rng: np.random.Generator, shape: Tuple[int, int], corr: float) -> np.ndarray:
    """Birim varyanslı, uzamsal ilintili gürültü (FFT Gauss süzgeci)."""
    noise = rng.standard_normal(shape)
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.fftfreq(shape[1])[None, :]
    kernel = np.exp(-2.0 * (np.pi * max(corr, 1e-6)) ** 2 * (kx ** 2 + ky ** 2))
    out = np.real(np.fft.ifft2(np.fft.fft2(noise) * kernel))
    sd = float(out.std())
    return (out - out.mean()) / sd if sd > 0 else np.zeros(shape)


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _voronoi(rng: np.random.Generator, shape: Tuple[int, int], n: int) -> np.ndarray:
    """``n`` rastgele tohumla en yakın-tohum hücre kimliği."""
    ys = rng.uniform(0, shape[0], n)
    xs = rng.uniform(0, shape[1], n)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    d = (yy[..., None] - ys) ** 2 + (xx[..., None] - xs) ** 2
    return np.argmin(d, axis=-1)


@dataclass
class DemoWorld:
    """Bir senaryonun zamandan bağımsız alanları.

    Attributes:
        scn: Senaryo.
        grid: Izgara.
        cover: Hücre başına örtü indeksi (``_COVER``).
        cell: Piksel → hücre kimliği.
        shift: Hücre başına fenoloji kayması (gün).
        tex: Piksel dokusu (±).
        burn: Yangın şiddeti (0–1).
        elev: Göreli yükseklik (nehre uzaklık + gürültü; küçük = alçak).
        river: Kalıcı nehir maskesi.
        stress: Kuraklık/stres alanı (0–1).
    """

    scn: DemoScenario
    grid: an.Grid
    cover: np.ndarray
    cell: np.ndarray
    shift: np.ndarray
    tex: np.ndarray
    burn: np.ndarray
    elev: np.ndarray
    river: np.ndarray
    stress: np.ndarray

    # ---- fenoloji ----
    def fv(self, date: dt.date) -> np.ndarray:
        """Verilen gündeki bitki örtüsü oranı (0–1)."""
        doy = float(date.timetuple().tm_yday)
        d = doy - self.shift[self.cell]
        cov = self.cover[self.cell]
        out = np.full(self.cell.shape, 0.06)
        forest = 0.80 + 0.05 * math.cos(2 * math.pi * (doy - 200) / 365.0)
        winter = 0.12 + 0.72 * (_sig((d - 85) / 14.0) - _sig((d - 160) / 9.0))
        summer = 0.10 + 0.74 * (_sig((d - 150) / 16.0) - _sig((d - 250) / 12.0))
        orchard = 0.55 + 0.15 * math.sin(2 * math.pi * (doy - 110) / 365.0)
        out = np.where(cov == 0, forest, out)
        out = np.where(cov == 1, winter, out)
        out = np.where(cov == 2, summer, out)
        out = np.where(cov == 3, orchard, out)
        return np.clip(out * (1.0 + 0.10 * self.tex), 0.02, 0.97)

    # ---- olay alanları ----
    def flood_level(self, date: dt.date) -> float:
        """Sel su seviyesi (``elev`` ile karşılaştırılır; olaydan önce 0, sonra üstel çekilir)."""
        if self.scn.kind != "sel":
            return 0.0
        days = (date - self.scn.event_date).days
        return 0.0 if days < 0 else 0.32 * math.exp(-days / 7.0)

    def water(self, date: dt.date) -> np.ndarray:
        """Su oranı (0–1): kalıcı nehir + o günkü taşkın."""
        w = self.river.astype(np.float32)
        lvl = self.flood_level(date)
        if lvl > 0.0:
            w = np.maximum(w, _sig((lvl - self.elev) * 40.0).astype(np.float32))
        return w

    def stress_at(self, date: dt.date) -> np.ndarray:
        """Bitki stresi (0–1) o gün için."""
        if self.scn.kind not in ("kuraklik", "tarim"):
            return np.zeros(self.cell.shape, dtype=np.float32)
        ramp = float(np.clip((date - self.scn.event_date).days / 50.0, 0.0, 1.0))
        amp = 1.0 if self.scn.kind == "kuraklik" else 0.35
        return (self.stress * ramp * amp).astype(np.float32)

    def burn_at(self, date: dt.date) -> np.ndarray:
        """Yanmış alan oranı (0–1) o gün için."""
        if self.scn.kind != "yangin" or date < self.scn.event_date:
            return np.zeros(self.cell.shape, dtype=np.float32)
        return self.burn.astype(np.float32)

    def flood_damage_at(self, date: dt.date) -> np.ndarray:
        """Selden sonra bitki örtüsü kaybı çarpanı (1 = hasar yok)."""
        if self.scn.kind != "sel" or date <= self.scn.event_date:
            return np.ones(self.cell.shape, dtype=np.float32)
        hit = self.elev < 0.32
        return np.where(hit, 0.65, 1.0).astype(np.float32)

    # ---- sahne ----
    def reflectance(self, date: dt.date) -> Dict[str, np.ndarray]:
        """Bulutsuz yansıtım bantları (gürültüsüz)."""
        fv = self.fv(date) * self.flood_damage_at(date)
        st = self.stress_at(date)
        fv = fv * (1.0 - 0.5 * st)
        veg = np.array(ENDMEMBERS["veg"], dtype=np.float32)
        soil = np.array(ENDMEMBERS["soil"], dtype=np.float32)
        burn_e = np.array(ENDMEMBERS["burn"], dtype=np.float32)
        water_e = np.array(ENDMEMBERS["water"], dtype=np.float32)
        out = np.zeros(self.cell.shape + (6,), dtype=np.float32)
        for i in range(6):
            v = np.full(self.cell.shape, veg[i], dtype=np.float32)
            if an.BANDS[i] == "nir":
                v = v - 0.12 * st
            elif an.BANDS[i] == "swir16":
                v = v + 0.07 * st
            elif an.BANDS[i] == "red":
                v = v + 0.02 * st
            band = fv * v + (1.0 - fv) * soil[i]
            b = self.burn_at(date) * np.clip(0.2 + fv / 0.8, 0.0, 1.0)
            band = (1.0 - b) * band + b * burn_e[i]
            w = self.water(date)
            out[..., i] = (1.0 - w) * band + w * water_e[i]
        return {name: out[..., i] for i, name in enumerate(an.BANDS)}

    def render(self, meta: an.SceneMeta) -> an.SceneData:
        """Sahneyi bulut/gölge/gürültüyle birlikte üretir.

        Args:
            meta: ``catalog`` çıktısı; ``cloud_pct`` yerel bulut hedefini belirler (karo bulutu ondan sapabilir).
        """
        rng = np.random.default_rng([self.scn.seed, meta.date.toordinal(), 17])
        bands = self.reflectance(meta.date)
        shape = self.cell.shape
        fv = self.fv(meta.date)
        water = self.water(meta.date)
        scl = np.where(water > 0.5, 6, np.where(fv > 0.3, 4, 5)).astype(np.uint8)
        target = float(_plan(self.scn)[1].get(meta.scene_id, 0.0))
        for k in bands:
            bands[k] = bands[k] + rng.normal(0.0, 0.008, shape).astype(np.float32)
        if target > 0.0:
            n = _smooth(rng, shape, 9.0)
            thr = float(np.quantile(n, 1.0 - min(target, 99.5) / 100.0))
            core = n > thr
            dy, dx = 4, 6
            shadow = np.roll(np.roll(core, dy, axis=0), dx, axis=1) & ~core
            cirrus = (n > thr - 0.22) & ~core
            thick = n > thr + 0.45
            bright = np.clip(0.55 + 0.10 * rng.standard_normal(shape), 0.35, 0.9).astype(np.float32)
            for k in bands:
                bands[k] = np.where(shadow, bands[k] * 0.45, bands[k])
                bands[k] = np.where(cirrus, 0.7 * bands[k] + 0.3 * bright, bands[k])
                bands[k] = np.where(core, bright, bands[k])
            scl = np.where(shadow, 3, scl)
            scl = np.where(cirrus, 10, scl)
            scl = np.where(core, np.where(thick, 9, 8), scl).astype(np.uint8)
        bands = {k: np.clip(v, 0.001, 1.0).astype(np.float32) for k, v in bands.items()}
        return an.SceneData(meta, bands, scl)


def _build_world(scn: DemoScenario, grid: an.Grid) -> DemoWorld:
    rng = np.random.default_rng([scn.seed, grid.rows, grid.cols])
    shape = (grid.rows, grid.cols)
    n_cells = 18
    cell = _voronoi(rng, shape, n_cells)
    cover = rng.choice(len(_COVER), size=n_cells, p=_COVER_P[scn.kind])
    shift = rng.normal(0.0, 7.0, n_cells)
    tex = _smooth(rng, shape, 2.0) * 0.6 + rng.standard_normal(shape) * 0.4
    # yangın izi: bölge merkezine yakın düzensiz bir leke, çekirdekte şiddetli
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    cy, cx = shape[0] * (0.5 + rng.uniform(-0.06, 0.06)), shape[1] * (0.5 + rng.uniform(-0.06, 0.06))
    r = np.hypot((yy - cy) / (shape[0] * 0.46), (xx - cx) / (shape[1] * 0.50))
    wob = _smooth(rng, shape, 6.0) * 0.18
    burn = np.clip(1.25 - (r + wob) * 1.05, 0.0, 1.0) ** 0.8
    burn = np.where(burn < 0.04, 0.0, burn)
    # nehir: bölgeyi kat eden kıvrımlı hat; yükseklik = nehre uzaklık + gürültü
    cols = np.arange(shape[1])
    meander = shape[0] * (0.55 + 0.12 * np.sin(cols / shape[1] * 2 * np.pi * 1.3 + rng.uniform(0, 6.28)))
    dist = np.abs(yy - meander[None, :]) / shape[0]
    elev = np.clip(dist * 2.2 + _smooth(rng, shape, 5.0) * 0.07, 0.0, 3.0)
    river = np.abs(yy - meander[None, :]) <= max(1.0, shape[0] * 0.012)
    stress = np.clip(0.5 + 0.35 * _smooth(rng, shape, 12.0), 0.0, 1.0)
    return DemoWorld(scn, grid, cover, cell, shift, tex.astype(np.float32), burn.astype(np.float32), elev.astype(np.float32),
                     river, stress.astype(np.float32))


@functools.lru_cache(maxsize=8)
def _cached_world(scn: DemoScenario, grid: an.Grid) -> DemoWorld:
    return _build_world(scn, grid)


def world(scn: DemoScenario, grid: an.Grid) -> DemoWorld:
    """Senaryo ve ızgara için (önbellekli) dünya modeli."""
    return _cached_world(scn, grid)


@functools.lru_cache(maxsize=16)
def _plan(scn: DemoScenario) -> Tuple[Tuple[an.SceneMeta, ...], Dict[str, float]]:
    """Katalog ve sahne başına yerel bulut hedefi (ızgaradan bağımsız, tohuma bağlı).

    5 günde bir sahne (S2A/S2B dönüşümlü); olay çevresinde en az birer berrak sahne garanti edilir, bir sonraki sahne ise
    bilerek bulutludur (süzgeç ve maskenin çalıştığını göstermek için). Karo bulutu yüzdesi yerel hedeften sapar
    (gerçek karo-yerel farkını göstermek için).
    """
    rng = np.random.default_rng([scn.seed, 99])
    days = (scn.end - scn.start).days
    dates = [scn.start + dt.timedelta(days=5 * i) for i in range(days // 5 + 1)]
    local = []
    for _ in dates:
        u = rng.random()
        local.append(float(rng.uniform(0, 8) if u < 0.35 else (rng.uniform(8, 25) if u < 0.65 else rng.uniform(25, 95))))
    pre_i = max((i for i, d in enumerate(dates) if d <= scn.event_date - dt.timedelta(days=3)), default=None)
    post_i = min((i for i, d in enumerate(dates) if d >= scn.event_date + dt.timedelta(days=7)), default=None)
    if pre_i is not None:
        local[pre_i] = float(rng.uniform(0.0, 5.0))
    if post_i is not None:
        local[post_i] = float(rng.uniform(0.0, 6.0))
        if post_i + 1 < len(dates):
            local[post_i + 1] = 72.0
    metas: List[an.SceneMeta] = []
    targets: Dict[str, float] = {}
    for i, (d, c) in enumerate(zip(dates, local)):
        sid = f"DEMO_{scn.kind}_{d:%Y%m%d}"
        targets[sid] = c
        tile = float(np.clip(c + rng.normal(0.0, 6.0), 0.0, 100.0))
        metas.append(an.SceneMeta(sid, d, "Sentinel-2A (demo)" if i % 2 == 0 else "Sentinel-2B (demo)", round(tile, 1), "demo"))
    return tuple(metas), targets


def catalog(scn: DemoScenario) -> List[an.SceneMeta]:
    """Sentetik sahne kataloğu (bkz. ``_plan``)."""
    return list(_plan(scn)[0])


def load_scene(scn: DemoScenario, grid: an.Grid, meta: an.SceneMeta) -> an.SceneData:
    """Katalogdaki bir sahneyi üretir (bulut, gölge ve gürültü dahil).

    Args:
        scn: Senaryo.
        grid: Izgara.
        meta: ``catalog`` çıktısındaki kayıt.
    """
    return world(scn, grid).render(meta)


class DemoSource:
    """``live_sources.LiveSource`` ile aynı arayüzü sunan sentetik kaynak."""

    is_demo = True
    name = DEMO_LABEL

    def __init__(self, scn: DemoScenario) -> None:
        """Kaynağı kurar.

        Args:
            scn: Senaryo (kullanıcının bölge ve tarih seçimleriyle ``dataclasses.replace`` edilmiş olabilir).
        """
        self.scn = scn
        self.notes: List[str] = []

    def catalog(self, region: an.Region, start: dt.date, end: dt.date) -> List[an.SceneMeta]:
        """Aralıktaki sentetik sahneler."""
        return [m for m in catalog(self.scn) if start <= m.date <= end]

    def load(self, meta: an.SceneMeta, region: an.Region, grid: an.Grid, bands: Sequence[str]) -> an.SceneData:
        """Sentetik sahne (tüm bantlar üretilir; ``bands`` yok sayılır)."""
        return load_scene(self.scn, grid, meta)

    def weather(self, region: an.Region, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Sentetik günlük hava (aralıkla kesilir)."""
        df = demo_weather(self.scn)
        return df[(df["tarih"] >= pd.Timestamp(start)) & (df["tarih"] <= pd.Timestamp(end))].reset_index(drop=True)

    def hotspots(self, region: an.Region, start: dt.date, end: dt.date, event_day: Optional[dt.date] = None,
                 grid: Optional[an.Grid] = None) -> pd.DataFrame:
        """Sentetik yangın noktaları (görüntülerle aynı ızgaradan; ``grid`` verilmezse 160 pikselli ızgara)."""
        return demo_hotspots(self.scn, grid or an.make_grid(region.bbox, 160))


def truth_masks(scn: DemoScenario, grid: an.Grid, date: dt.date) -> Dict[str, np.ndarray]:
    """Bilinen gerçek durum (yalnızca yöntem doğrulaması için).

    Returns:
        ``burn`` (0–1 şiddet), ``flooded`` (o gün su altı), ``stress`` (0–1) alanları.
    """
    w = world(scn, grid)
    return {"burn": w.burn_at(date), "flooded": (w.water(date) > 0.5), "stress": w.stress_at(date)}


def demo_weather(scn: DemoScenario) -> pd.DataFrame:
    """Senaryoyla tutarlı sentetik günlük hava tablosu (``analysis.weather_evidence`` şeması).

    Gerçek ölçüm değildir; olay günlerine uygun aşırılıklar eklenir (sel: yağış, yangın: sıcak-kuru-rüzgârlı, kuraklık: kurak).
    """
    rng = np.random.default_rng([scn.seed, 5])
    dates = pd.date_range(scn.start, scn.end, freq="D")
    doy = dates.dayofyear.to_numpy().astype(float)
    tmax = 19.0 + 11.0 * np.sin(2 * np.pi * (doy - 105) / 365.0) + rng.normal(0, 2.2, len(dates))
    tmin = tmax - 9.0 + rng.normal(0, 1.2, len(dates))
    wet_p = 0.28 - 0.14 * np.sin(2 * np.pi * (doy - 105) / 365.0)
    rain = np.where(rng.random(len(dates)) < wet_p, rng.gamma(1.4, 4.5, len(dates)), 0.0)
    wind = 12.0 + np.abs(rng.normal(0, 6.0, len(dates)))
    ev = pd.Timestamp(scn.event_date)
    idx = {d: i for i, d in enumerate(dates)}
    if scn.kind == "sel":
        for off, mm in ((-2, 34.0), (-1, 58.0), (0, 46.0)):
            if ev + pd.Timedelta(days=off) in idx:
                rain[idx[ev + pd.Timedelta(days=off)]] = mm
    elif scn.kind == "yangin":
        for i, d in enumerate(dates):
            if ev - pd.Timedelta(days=10) <= d <= ev + pd.Timedelta(days=3):
                tmax[i] += 5.0
                tmin[i] += 3.0
                rain[i] = 0.0
        if ev in idx:
            tmax[idx[ev]] = max(tmax[idx[ev]], 38.0)
            wind[idx[ev]] = 46.0
    elif scn.kind == "kuraklik":
        for i, d in enumerate(dates):
            if ev - pd.Timedelta(days=45) <= d <= ev + pd.Timedelta(days=35):
                rain[i] *= 0.1
                tmax[i] += 4.0
    et0 = np.clip(1.0 + 0.19 * tmax, 0.3, None) * (1.0 + 0.1 * rng.standard_normal(len(dates)))
    return pd.DataFrame({"tarih": dates, "yagis_mm": np.round(rain, 1), "tmax_c": np.round(tmax, 1), "tmin_c": np.round(tmin, 1),
                         "ruzgar_kmh": np.round(wind, 0), "et0_mm": np.round(np.clip(et0, 0.2, None), 1)})


def demo_hotspots(scn: DemoScenario, grid: an.Grid, n: int = 28) -> pd.DataFrame:
    """Yangın izi içine düşen sentetik "aktif yangın" noktaları (``data_sources.HOTSPOT_COLUMNS`` şeması).

    Yalnızca yangın senaryosunda dolu döner; ``kaynak``/``uydu`` sütunları açıkça "demo" yazar.
    """
    cols = ["enlem", "boylam", "frp_mw", "parlaklik_k", "kaynak", "guven", "tarih_utc", "gece_gunduz", "uydu"]
    if scn.kind != "yangin":
        return pd.DataFrame(columns=cols)
    w = world(scn, grid)
    rng = np.random.default_rng([scn.seed, 71])
    rr, cc = np.nonzero(w.burn > 0.35)
    if rr.size == 0:
        return pd.DataFrame(columns=cols)
    pick = rng.choice(rr.size, size=min(n, rr.size), replace=False)
    lon, lat = grid.centres()
    hours = rng.integers(0, 96, pick.size)
    stamps = pd.Timestamp(scn.event_date, tz="UTC") + pd.to_timedelta(hours, unit="h")
    # nokta kümeleri 375 m'lik piksel belirsizliği taklit eder
    jitter_lon = rng.normal(0, 0.0009, pick.size)
    jitter_lat = rng.normal(0, 0.0009, pick.size)
    return pd.DataFrame({
        "enlem": lat[rr[pick]] + jitter_lat, "boylam": lon[cc[pick]] + jitter_lon,
        "frp_mw": np.round(rng.lognormal(1.6, 0.7, pick.size), 1), "parlaklik_k": np.round(rng.uniform(330, 395, pick.size), 1),
        "kaynak": "demo", "guven": np.round(rng.uniform(0.6, 1.0, pick.size), 2), "tarih_utc": stamps,
        "gece_gunduz": np.where((stamps.hour >= 6) & (stamps.hour < 18), "D", "N"), "uydu": "demo",
    }).sort_values("tarih_utc").reset_index(drop=True)
