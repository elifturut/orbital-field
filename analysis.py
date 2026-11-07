"""
analysis.py — Analiz iş akışının çekirdeği (bölge → analiz türü → sahneler → indeks → değişim → güvenilirlik → sektör görünümü)
=================================================================================================================

Streamlit, ağ ve harita kütüphanesinden bağımsızdır (yalnızca ``numpy`` ve ``pandas``); bu yüzden çevrimdışı test edilir.
Sahne verisi ister canlı (``live_sources.py``) ister sentetik (``demo_imagery.py``) kaynaktan gelsin, buradaki işlevler
aynı ``SceneData`` nesnelerini işler. Aynı sonuç iki sektör görünümü üretir:

* **Sigorta**: hasar oranı, otomatik karar (``data_engine.InsuranceSimulator.decide_codes`` — uygulamadaki tek karar kuralı) ve
  gösterge niteliğinde tazminat.
* **Tarım**: bitki gelişimi sınıfları ve stresli alan (rekolte tahmini ``yield_model.py``'dedir).

Önemli sınırlar
---------------
* Güvenilirlik puanı **istatistiksel bir güven aralığı değildir**; bulut/geçersiz piksel, zamanlama ve örnek büyüklüğünden
  türetilen, formülü açık bir *veri kalitesi* göstergesidir.
* Sınıf eşikleri literatürden başlangıç değerleridir (bkz. ``thresholds.py``); yerel doğrulama yapılmadan kesin hüküm sayılmaz.
* Hava durumu "koşul doğrulaması" göstergeleri resmî eşik değildir; ayarlanabilir varsayılanlardır.

Geliştirici: Elif Turut
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import data_engine as de

__all__ = [
    "BANDS", "INDEX_BANDS", "INDEX_LABELS", "INVALID_SCL", "CLOUD_SCL", "KINDS", "AnalysisKind", "AnalysisError",
    "Region", "RegionError", "Grid", "SceneMeta", "SceneData", "SceneStats", "ChangeResult", "Reliability", "PairChoice",
    "InsuranceView", "AgriView", "EvidenceRow", "region_from_bbox", "region_from_center", "region_from_geojson",
    "make_grid", "polygon_mask", "true_color", "bands_for", "distance_to_ring_m", "compute_index", "valid_mask", "cloud_level", "filter_scenes",
    "scene_stats", "build_series", "pick_pair", "classify_change", "analyse_change", "assess_reliability",
    "insurance_view", "agri_view", "weather_evidence", "match_hotspots", "growing_degree_days", "colorize",
    "CLOUD_WARN_PCT", "MAX_BBOX_KM", "point_in_region", "SupportView", "drought_support",
]

#: Ham bantlar (Sentinel-2 L2A): mavi, yeşil, kırmızı, yakın kızılötesi, kısa dalga kızılötesi 1 ve 2.
BANDS: Tuple[str, ...] = ("blue", "green", "red", "nir", "swir16", "swir22")
#: İndeks → gereken bantlar.
INDEX_BANDS: Dict[str, Tuple[str, ...]] = {
    "ndvi": ("red", "nir"), "evi": ("blue", "red", "nir"), "nbr": ("nir", "swir22"),
    "ndmi": ("nir", "swir16"), "mndwi": ("green", "swir16"),
}
INDEX_LABELS: Dict[str, str] = {
    "ndvi": "NDVI (bitki yeşilliği)", "evi": "EVI (iyileştirilmiş bitki indeksi)", "nbr": "NBR (yanma oranı)",
    "ndmi": "NDMI (bitki nemi)", "mndwi": "MNDWI (su)",
}
#: Sentinel-2 SCL sınıfları: 0 veri yok, 1 doygun, 3 bulut gölgesi, 8/9 bulut (orta/yüksek olasılık), 10 ince sirrus, 11 kar/buz.
INVALID_SCL = frozenset({0, 1, 3, 8, 9, 10, 11})
#: Bulut sayılan SCL sınıfları (yerel bulut oranı için): gölge, orta/yüksek bulut, sirrus.
CLOUD_SCL = frozenset({3, 8, 9, 10})
#: Bu bulut yüzdesinin üstü "güvenilirlik düşük" uyarısı alır (kullanıcı önerisi: %20).
CLOUD_WARN_PCT = 20.0
#: Canlı analiz için sınır kutusunun en uzun kenarı (km).
MAX_BBOX_KM = 25.0
MIN_AREA_HA = 0.5
M_PER_DEG_LAT = 111_320.0


class AnalysisError(ValueError):
    """Analiz girdisi ya da verisi geçersiz."""


class RegionError(AnalysisError):
    """Bölge (çokgen/sınır kutusu) geçersiz."""


# --------------------------------------------------------------------------- #
# Bölge ve ızgara
# --------------------------------------------------------------------------- #
def _m_per_deg_lon(lat: float) -> float:
    """Verilen enlemde 1° boylamın metre karşılığı."""
    return M_PER_DEG_LAT * max(math.cos(math.radians(lat)), 1e-6)


@dataclass(frozen=True)
class Region:
    """Analiz bölgesi: (boylam, enlem) köşeli kapalı olmayan halka.

    Attributes:
        name: Görünen ad.
        ring: ``(boylam, enlem)`` köşeleri (ilk köşe sonda tekrarlanmaz).
    """

    name: str
    ring: Tuple[Tuple[float, float], ...]

    def __post_init__(self) -> None:
        ring = self.ring
        if len(ring) > 1 and ring[0] == ring[-1]:
            object.__setattr__(self, "ring", tuple(ring[:-1]))
            ring = self.ring
        if not 3 <= len(ring) <= 500:
            raise RegionError("Bölge en az 3, en çok 500 köşeli olmalı.")
        try:
            ring = tuple((float(lon), float(lat)) for lon, lat in ring)
        except (TypeError, ValueError):
            raise RegionError("Köşe koordinatları sayısal olmalı.") from None
        object.__setattr__(self, "ring", ring)
        for lon, lat in ring:
            if not (math.isfinite(lon) and math.isfinite(lat)):
                raise RegionError("Köşe koordinatları sayısal olmalı.")
            if not (-180.0 <= lon <= 180.0 and -85.0 <= lat <= 85.0):
                raise RegionError("Koordinat aralık dışı (boylam −180..180, enlem −85..85).")
        if len(set(ring)) < 3:
            raise RegionError("Bölge en az 3 farklı köşe içermeli.")
        w, s, e, n = self.bbox
        if max((e - w) * _m_per_deg_lon((s + n) / 2), (n - s) * M_PER_DEG_LAT) / 1000.0 > MAX_BBOX_KM:
            raise RegionError(f"Bölge çok büyük: sınır kutusunun en uzun kenarı en çok {MAX_BBOX_KM:g} km olabilir.")
        if self.area_ha < MIN_AREA_HA:
            raise RegionError(f"Bölge çok küçük (en az {MIN_AREA_HA:g} ha).")

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """``(batı, güney, doğu, kuzey)``."""
        lons = [p[0] for p in self.ring]
        lats = [p[1] for p in self.ring]
        return (min(lons), min(lats), max(lons), max(lats))

    @property
    def centre(self) -> Tuple[float, float]:
        """``(enlem, boylam)`` sınır kutusu merkezi."""
        w, s, e, n = self.bbox
        return ((s + n) / 2.0, (w + e) / 2.0)

    @property
    def area_ha(self) -> float:
        """Çokgenin yaklaşık alanı (ha; yerel eşdikdörtgen izdüşümü + ayakkabı bağı formülü)."""
        lat0 = self.centre[0]
        kx = _m_per_deg_lon(lat0)
        xs = np.array([p[0] * kx for p in self.ring])
        ys = np.array([p[1] * M_PER_DEG_LAT for p in self.ring])
        area_m2 = 0.5 * abs(float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))
        return area_m2 / 10_000.0

    def geojson(self) -> Dict[str, Any]:
        """GeoJSON ``Polygon`` geometrisi."""
        ring = [list(p) for p in self.ring] + [list(self.ring[0])]
        return {"type": "Polygon", "coordinates": [ring]}


def region_from_bbox(name: str, west: float, south: float, east: float, north: float) -> Region:
    """Dikdörtgen bölge kurar.

    Args:
        name: Görünen ad.
        west: Batı boylamı.
        south: Güney enlemi.
        east: Doğu boylamı.
        north: Kuzey enlemi.
    """
    if not (west < east and south < north):
        raise RegionError("Sınır kutusu geçersiz: batı < doğu ve güney < kuzey olmalı.")
    return Region(name, ((west, south), (east, south), (east, north), (west, north)))


def region_from_center(name: str, lat: float, lon: float, half_km: float) -> Region:
    """Merkez ve yarı kenar uzunluğundan (km) kare bölge kurar.

    Args:
        name: Görünen ad.
        lat: Merkez enlemi.
        lon: Merkez boylamı.
        half_km: Karenin yarı kenarı (km).
    """
    if not (isinstance(half_km, (int, float)) and math.isfinite(half_km) and 0.05 <= half_km <= MAX_BBOX_KM / 2):
        raise RegionError(f"Yarı kenar 0,05 ile {MAX_BBOX_KM / 2:g} km arasında olmalı.")
    if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and math.isfinite(lat) and math.isfinite(lon)):
        raise RegionError("Merkez koordinatı sayısal olmalı.")
    dlat = half_km * 1000.0 / M_PER_DEG_LAT
    dlon = half_km * 1000.0 / _m_per_deg_lon(lat)
    return region_from_bbox(name, lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def region_from_geojson(geometry: Any, name: str = "Çizilen bölge") -> Region:
    """GeoJSON ``Polygon`` (ya da onu taşıyan ``Feature``) geometrisinden bölge kurar; ilk halka kullanılır.

    Raises:
        RegionError: Geometri çokgen değilse ya da geçersizse.
    """
    geom = geometry
    if isinstance(geom, Mapping) and geom.get("type") == "Feature":
        geom = geom.get("geometry")
    if not isinstance(geom, Mapping) or geom.get("type") != "Polygon":
        raise RegionError("Yalnızca çokgen (Polygon) desteklenir.")
    try:
        ring = tuple((float(p[0]), float(p[1])) for p in geom["coordinates"][0])
    except (KeyError, IndexError, TypeError, ValueError):
        raise RegionError("Çokgen koordinatları okunamadı.") from None
    return Region(name, ring)


@dataclass(frozen=True)
class Grid:
    """Sınır kutusu üzerinde düzenli piksel ızgarası (satır 0 = kuzey).

    Attributes:
        bbox: ``(batı, güney, doğu, kuzey)``.
        rows: Satır sayısı.
        cols: Sütun sayısı.
    """

    bbox: Tuple[float, float, float, float]
    rows: int
    cols: int

    @property
    def pixel_m(self) -> Tuple[float, float]:
        """``(yükseklik_m, genişlik_m)`` piksel boyutu."""
        w, s, e, n = self.bbox
        return ((n - s) * M_PER_DEG_LAT / self.rows, (e - w) * _m_per_deg_lon((s + n) / 2.0) / self.cols)

    @property
    def pixel_area_ha(self) -> float:
        """Bir pikselin alanı (ha)."""
        dy, dx = self.pixel_m
        return dy * dx / 10_000.0

    def centres(self) -> Tuple[np.ndarray, np.ndarray]:
        """Piksel merkezlerinin ``(boylam[cols], enlem[rows])`` dizileri (enlem kuzeyden güneye)."""
        w, s, e, n = self.bbox
        lon = w + (np.arange(self.cols) + 0.5) * (e - w) / self.cols
        lat = n - (np.arange(self.rows) + 0.5) * (n - s) / self.rows
        return lon, lat


def make_grid(bbox: Tuple[float, float, float, float], max_px: int = 160, min_res_m: float = 10.0) -> Grid:
    """Sınır kutusuna uygun ızgara kurar: uzun kenar ``max_px`` pikseli aşmaz, piksel ``min_res_m``'den ince olmaz.

    Args:
        bbox: ``(batı, güney, doğu, kuzey)``.
        max_px: Uzun kenardaki en çok piksel.
        min_res_m: En ince piksel boyutu (Sentinel-2 için 10 m).
    """
    if not 8 <= int(max_px) <= 1024:
        raise AnalysisError("max_px 8 ile 1024 arasında olmalı.")
    w, s, e, n = bbox
    width_m = (e - w) * _m_per_deg_lon((s + n) / 2.0)
    height_m = (n - s) * M_PER_DEG_LAT
    if width_m <= 0 or height_m <= 0:
        raise AnalysisError("Sınır kutusu geçersiz.")
    long_px = int(min(max_px, max(8, math.ceil(max(width_m, height_m) / min_res_m))))
    scale = long_px / max(width_m, height_m)
    return Grid(bbox, max(2, int(round(height_m * scale))), max(2, int(round(width_m * scale))))


def polygon_mask(grid: Grid, ring: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Çokgenin içinde kalan piksellerin maskesi (tek-çift kuralı; köşe sayısı kadar vektörel adım).

    Args:
        grid: Izgara.
        ring: ``(boylam, enlem)`` köşeleri.

    Returns:
        ``(rows, cols)`` boolean dizi.
    """
    lon, lat = grid.centres()
    px, py = np.meshgrid(lon, lat)
    inside = np.zeros(px.shape, dtype=bool)
    pts = list(ring)
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        if y1 == y2:
            continue
        crosses = (y1 > py) != (y2 > py)
        x_int = (x2 - x1) * (py - y1) / (y2 - y1) + x1
        inside ^= crosses & (px < x_int)
    return inside


def distance_to_ring_m(lon: np.ndarray, lat: np.ndarray, ring: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Noktaların çokgen kenarlarına en kısa uzaklığı (m; yerel eşdikdörtgen izdüşümü).

    Args:
        lon: Nokta boylamları.
        lat: Nokta enlemleri.
        ring: Çokgen köşeleri.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lat0 = float(np.mean([p[1] for p in ring]))
    kx = _m_per_deg_lon(lat0)
    pts = [(p[0] * kx, p[1] * M_PER_DEG_LAT) for p in ring]
    x, y = lon * kx, lat * M_PER_DEG_LAT
    best = np.full(x.shape, np.inf)
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        dx, dy = x2 - x1, y2 - y1
        seg2 = dx * dx + dy * dy
        t = np.clip(((x - x1) * dx + (y - y1) * dy) / seg2, 0.0, 1.0) if seg2 > 0 else np.zeros_like(x)
        best = np.minimum(best, np.hypot(x - (x1 + t * dx), y - (y1 + t * dy)))
    return best


# --------------------------------------------------------------------------- #
# İndeksler ve maskeler
# --------------------------------------------------------------------------- #
def _norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(a − b) / (a + b)``; payda ~0 olan yerlerde ``nan``."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    den = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / den
    return np.where(np.abs(den) < 1e-6, np.nan, out).astype(np.float32)


def compute_index(name: str, bands: Mapping[str, np.ndarray]) -> np.ndarray:
    """Bant sözlüğünden indeks hesaplar (yansıtım 0–1 ölçeğinde olmalı).

    Args:
        name: ``INDEX_BANDS`` anahtarı (ndvi, evi, nbr, ndmi, mndwi).
        bands: Bant adı → 2B dizi.

    Raises:
        AnalysisError: Bilinmeyen indeks ya da eksik bant.
    """
    if name not in INDEX_BANDS:
        raise AnalysisError(f"Bilinmeyen indeks: {name!r}")
    missing = [b for b in INDEX_BANDS[name] if b not in bands]
    if missing:
        raise AnalysisError(f"{name} için eksik bant: {missing}")
    if name == "ndvi":
        return _norm_diff(bands["nir"], bands["red"])
    if name == "nbr":
        return _norm_diff(bands["nir"], bands["swir22"])
    if name == "ndmi":
        return _norm_diff(bands["nir"], bands["swir16"])
    if name == "mndwi":
        return _norm_diff(bands["green"], bands["swir16"])
    nir, red, blue = (np.asarray(bands[k], dtype=np.float32) for k in ("nir", "red", "blue"))
    den = nir + 6.0 * red - 7.5 * blue + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        evi = 2.5 * (nir - red) / den
    return np.clip(np.where(np.abs(den) < 1e-6, np.nan, evi), -1.0, 1.5).astype(np.float32)


def bands_for(kind_key: str, with_rgb: bool = False) -> Tuple[str, ...]:
    """Bir analiz türü için okunması gereken bantlar (ana indeks + NDVI; isteğe bağlı gerçek renk).

    Args:
        kind_key: Analiz türü.
        with_rgb: Gerçek renkli önizleme için mavi/yeşil/kırmızı da eklensin mi.
    """
    if kind_key not in KINDS:
        raise AnalysisError(f"Bilinmeyen analiz türü: {kind_key!r}")
    need = set(INDEX_BANDS[KINDS[kind_key].index]) | set(INDEX_BANDS["ndvi"])
    if with_rgb:
        need |= {"blue", "green", "red"}
    return tuple(b for b in BANDS if b in need)


def true_color(data: "SceneData", gain: float = 3.2, gamma: float = 1.0 / 1.8) -> Optional[np.ndarray]:
    """Gerçek renkli (kırmızı-yeşil-mavi) önizleme; bulut/gölge pikselleri olduğu gibi gösterilir.

    Args:
        data: Sahne verisi (mavi, yeşil, kırmızı bantları gerekir).
        gain: Parlaklık kazancı (yansıtım 0,3 civarı beyaza yaklaşır).
        gamma: Gama düzeltmesi.

    Returns:
        ``(satır, sütun, 3)`` ``uint8`` dizi; bant eksikse ``None``.
    """
    if not all(b in data.bands for b in ("red", "green", "blue")):
        return None
    rgb = np.stack([data.bands["red"], data.bands["green"], data.bands["blue"]], axis=-1)
    rgb = np.clip(np.nan_to_num(rgb, nan=0.0) * gain, 0.0, 1.0) ** gamma
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def valid_mask(scl: Optional[np.ndarray], bands: Mapping[str, np.ndarray]) -> np.ndarray:
    """Kullanılabilir piksel maskesi: SCL'ye göre bulut/gölge/kar/veri-yok dışı ve bantlar sonlu.

    Args:
        scl: Sahne sınıflandırma haritası (``None`` olabilir).
        bands: Bant sözlüğü.
    """
    first = next(iter(bands.values()))
    ok = np.ones(first.shape, dtype=bool)
    for arr in bands.values():
        ok &= np.isfinite(arr)
    if scl is not None:
        ok &= ~np.isin(scl, list(INVALID_SCL))
    return ok


# --------------------------------------------------------------------------- #
# Sahneler
# --------------------------------------------------------------------------- #
@dataclass
class SceneMeta:
    """Bir uydu sahnesinin kataloğ kaydı.

    Attributes:
        scene_id: Benzersiz kimlik.
        date: Çekim günü (UTC).
        platform: Örn. ``Sentinel-2A``.
        cloud_pct: Karo düzeyinde bulut yüzdesi (``nan`` = bilinmiyor).
        source: ``earth-search`` ya da ``demo``.
        assets: Bant adı → adres (canlı kaynakta).
        band_scale: Bant adı → ``(ölçek, kayma)``; yansıtım = ham × ölçek + kayma.
        bbox: Sahnenin (karo) sınır kutusu ``(batı, güney, doğu, kuzey)``; bilinmiyorsa ``None``.
    """

    scene_id: str
    date: dt.date
    platform: str
    cloud_pct: float
    source: str
    assets: Dict[str, str] = field(default_factory=dict)
    band_scale: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class SceneData:
    """Bir sahnenin ızgaraya oturtulmuş bantları.

    Attributes:
        meta: Kataloğ kaydı.
        bands: Bant adı → 2B ``float32`` yansıtım (0–1).
        scl: Sahne sınıflandırma haritası (``uint8``) ya da ``None``.
    """

    meta: SceneMeta
    bands: Dict[str, np.ndarray]
    scl: Optional[np.ndarray] = None


def cloud_level(cloud_pct: float) -> Tuple[str, str]:
    """Bulut yüzdesinden güvenilirlik etiketi ve düzeyi.

    Returns:
        ``(etiket, düzey)``; düzey ``ok``/``mid``/``low``/``unknown``. %10'a kadar yüksek, %20'ye kadar orta, üstü düşük.
    """
    if cloud_pct is None or not math.isfinite(cloud_pct):
        return ("Bilinmiyor", "unknown")
    if cloud_pct <= 10.0:
        return ("Yüksek", "ok")
    if cloud_pct <= CLOUD_WARN_PCT:
        return ("Orta", "mid")
    return ("Düşük — bulutlu", "low")


def filter_scenes(scenes: Sequence[SceneMeta], max_cloud: float, start: Optional[dt.date] = None,
                  end: Optional[dt.date] = None) -> List[SceneMeta]:
    """Bulut ve tarih aralığına göre sahneleri süzer (tarihe göre artan sıralı).

    Args:
        scenes: Sahne kayıtları.
        max_cloud: En çok karo bulut yüzdesi; bilinmeyen bulut süzgeçten **geçmez** (belirsizlik sessizce kabul edilmez).
        start: İlk gün (dahil).
        end: Son gün (dahil).
    """
    out = []
    for s in scenes:
        if start and s.date < start or end and s.date > end:
            continue
        if not math.isfinite(s.cloud_pct) or s.cloud_pct > max_cloud:
            continue
        out.append(s)
    return sorted(out, key=lambda m: (m.date, m.scene_id))


@dataclass(frozen=True)
class SceneStats:
    """Bölge içindeki bir sahnenin indeks istatistiği.

    Attributes:
        scene_id: Sahne kimliği.
        date: Çekim günü.
        platform: Uydu.
        cloud_tile: Karo bulut yüzdesi.
        cloud_local: Bölge içi bulut/gölge oranı (%).
        valid_pct: Bölgede kullanılabilir piksel oranı (%).
        mean: Ortalama.
        median: Medyan.
        minimum: %2'lik dilim (uç değerlere dayanıklı alt sınır).
        maximum: %98'lik dilim.
        std: Standart sapma.
        n_valid: Kullanılabilir piksel sayısı.
    """

    scene_id: str
    date: dt.date
    platform: str
    cloud_tile: float
    cloud_local: float
    valid_pct: float
    mean: float
    median: float
    minimum: float
    maximum: float
    std: float
    n_valid: int


def scene_stats(data: SceneData, index: str, region_mask: np.ndarray) -> Tuple[SceneStats, np.ndarray]:
    """Sahnenin bölge içi indeks istatistiğini ve indeks rasterini üretir.

    Args:
        data: Sahne verisi.
        index: İndeks adı.
        region_mask: Bölge maskesi.

    Returns:
        ``(istatistik, indeks_raster)``; raster bulut/gölge pikselleri ``nan``.
    """
    idx = compute_index(index, data.bands)
    ok = valid_mask(data.scl, {k: v for k, v in data.bands.items() if k in INDEX_BANDS[index]}) & np.isfinite(idx)
    idx = np.where(ok, idx, np.nan).astype(np.float32)
    n_region = int(region_mask.sum())
    if n_region == 0:
        raise AnalysisError("Bölge maskesi boş.")
    if data.scl is not None:
        in_region = data.scl[region_mask]
        local_cloud = 100.0 * float(np.isin(in_region, list(CLOUD_SCL)).sum()) / n_region
    else:
        local_cloud = math.nan
    vals = idx[region_mask & ok]
    valid_pct = 100.0 * vals.size / n_region
    if vals.size:
        mean, med = float(vals.mean()), float(np.median(vals))
        lo, hi = (float(x) for x in np.percentile(vals, [2, 98]))
        std = float(vals.std())
    else:
        mean = med = lo = hi = std = math.nan
    stats = SceneStats(data.meta.scene_id, data.meta.date, data.meta.platform, float(data.meta.cloud_pct), local_cloud,
                       valid_pct, mean, med, lo, hi, std, int(vals.size))
    return stats, idx


def build_series(stats: Sequence[SceneStats]) -> pd.DataFrame:
    """İstatistikleri zaman serisi tablosuna çevirir (tarihe göre sıralı).

    Columns: ``tarih, sahne, uydu, bulut_karo, bulut_yerel, gecerli_oran, ortalama, medyan, minimum, maksimum, std, n``.
    """
    cols = ["tarih", "sahne", "uydu", "bulut_karo", "bulut_yerel", "gecerli_oran", "ortalama", "medyan", "minimum",
            "maksimum", "std", "n"]
    rows = [(pd.Timestamp(s.date), s.scene_id, s.platform, s.cloud_tile, s.cloud_local, s.valid_pct, s.mean, s.median,
             s.minimum, s.maximum, s.std, s.n_valid) for s in stats]
    return pd.DataFrame(rows, columns=cols).sort_values("tarih").reset_index(drop=True)


@dataclass(frozen=True)
class PairChoice:
    """Önce/sonra sahne çifti önerisi.

    Attributes:
        pre_id: Olaydan önceki sahne.
        post_id: Olaydan sonraki sahne.
        notes: Seçim açıklamaları.
    """

    pre_id: Optional[str]
    post_id: Optional[str]
    notes: Tuple[str, ...]


def pick_pair(series: pd.DataFrame, event_date: dt.date, min_valid_pct: float = 60.0, max_gap_days: int = 60,
              mode: str = "sequential") -> PairChoice:
    """Önce/sonra sahnesi seçer.

    ``sequential`` (yangın, sel, tarım): olaydan önce ve sonra; maliyet = ``(100 − kullanılabilir %) + 0,5 × olaya uzaklık (gün)``.

    ``yoy`` (kuraklık): "sonra" = olaydan 5 hafta sonrasına yakın en berrak sahne; "önce" = **geçen yıl aynı dönemden**
    (±15 gün) en berrak sahne; maliyet = ``(100 − kullanılabilir %) + 1,0 × gün sapması``. Bu, mevsimsel değişimi stresten ayırır.

    Args:
        series: ``build_series`` çıktısı.
        event_date: Olay (hasar/başlangıç) tarihi.
        min_valid_pct: Bunun altındaki sahneler aday olmaz.
        max_gap_days: Olaya en çok uzaklık (gün).
        mode: ``sequential`` ya da ``yoy``.
    """
    if mode not in ("sequential", "yoy"):
        raise AnalysisError(f"Bilinmeyen eşleştirme kipi: {mode!r}")
    notes: List[str] = []
    ev = pd.Timestamp(event_date)
    if mode == "yoy":
        post_gap = (series["tarih"] - ev).dt.days
        post_c = series[(post_gap > 0) & (post_gap <= max_gap_days) & (series["gecerli_oran"] >= min_valid_pct)].copy()
        if post_c.empty:
            return PairChoice(None, None, (f"Olaydan sonra {max_gap_days} gün içinde en az %{min_valid_pct:g} kullanılabilir pikselli sahne yok.",))
        post_c["maliyet"] = (100.0 - post_c["gecerli_oran"]) + 0.5 * (post_gap.loc[post_c.index] - 35).abs()
        post = post_c.loc[post_c["maliyet"].idxmin()]
        dev = (series["tarih"] - (post["tarih"] - pd.Timedelta(days=365))).dt.days.abs()
        pre_c = series[(dev <= 15) & (series["gecerli_oran"] >= min_valid_pct)].copy()
        if pre_c.empty:
            return PairChoice(None, str(post["sahne"]), ("Geçen yıl aynı dönemden (±15 gün) kullanılabilir sahne yok; tarih aralığını bir yıl geriye uzatın.",))
        pre_c["maliyet"] = (100.0 - pre_c["gecerli_oran"]) + 1.0 * dev.loc[pre_c.index]
        pre = pre_c.loc[pre_c["maliyet"].idxmin()]
        return PairChoice(str(pre["sahne"]), str(post["sahne"]), ("Kıyas: geçen yılın aynı dönemi (mevsimsel fark giderilir).",))
    picked: Dict[str, Optional[str]] = {}
    for side, label in (("pre", "önce"), ("post", "sonra")):
        gap = (ev - series["tarih"]).dt.days if side == "pre" else (series["tarih"] - ev).dt.days
        cand = series[(gap > 0) & (gap <= max_gap_days) & (series["gecerli_oran"] >= min_valid_pct)].copy()
        if cand.empty:
            picked[side] = None
            notes.append(f"Olaydan {label} {max_gap_days} gün içinde en az %{min_valid_pct:g} kullanılabilir pikselli sahne yok.")
            continue
        cand["maliyet"] = (100.0 - cand["gecerli_oran"]) + 0.5 * gap.loc[cand.index]
        best = cand.loc[cand["maliyet"].idxmin()]
        picked[side] = str(best["sahne"])
    return PairChoice(picked["pre"], picked["post"], tuple(notes))


# --------------------------------------------------------------------------- #
# Analiz türleri ve değişim sınıflandırması
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnalysisKind:
    """Bir analiz türünün tanımı.

    Attributes:
        key: Kimlik.
        label: Türkçe ad.
        index: Ana indeks.
        extra_index: Zaman serisinde ek gösterilebilecek indeks.
        classes: ``(etiket, renk)`` sınıf listesi.
        affected: Etkilenmiş sayılan sınıf indeksleri.
        weights: Sınıf başına gösterge hasar ağırlığı (0–1; **varsayım**, sigorta görünümünde kullanılır).
        change_label: Değişim ölçüsünün adı.
        method: Yöntem özeti (arayüzde gösterilir).
        caveat: Bu türe özgü sınır.
    """

    key: str
    label: str
    index: str
    extra_index: str
    classes: Tuple[Tuple[str, str], ...]
    affected: Tuple[int, ...]
    weights: Tuple[float, ...]
    change_label: str
    method: str
    caveat: str


KINDS: Dict[str, AnalysisKind] = {
    "yangin": AnalysisKind(
        "yangin", "Yangın hasarı", "nbr", "ndvi",
        (("Artan yeşillik", "#2f9e5b"), ("Yanmamış", "#8bd17c"), ("Düşük şiddet", "#f1e05a"), ("Orta-düşük", "#f4a640"),
         ("Orta-yüksek", "#e5483a"), ("Yüksek şiddet", "#7a2316")),
        (2, 3, 4, 5), (0.0, 0.0, 0.2, 0.5, 0.8, 1.0), "dNBR (önce − sonra NBR)",
        "NBR = (YKI − KDKI-2)/(YKI + KDKI-2); dNBR = önce − sonra; sınıflar USGS FIREMON (Key & Benson 2006) eşikleriyle.",
        "Sınıflar orman yangınları için geliştirilmiştir; tarım arazisinde (anız, ekin) ayrıca yerel doğrulama gerekir."),
    "sel": AnalysisKind(
        "sel", "Sel / taşkın", "mndwi", "ndvi",
        (("Su yok", "#2b2f3a"), ("Kalıcı su", "#3b7ddd"), ("Yeni su (sel)", "#43d1e8"), ("Çekilen su", "#a78bfa")),
        (2,), (0.0, 0.0, 1.0, 0.0), "MNDWI farkı (sonra − önce)",
        "MNDWI = (YEŞİL − KDKI-1)/(YEŞİL + KDKI-1); MNDWI > 0 su sayılır; yeni su = sonra su ∧ önce su değil.",
        "Optik görüntü bulutlu havada sel alanını göremez; Sentinel-1 (radar) bu iş için daha güvenilirdir ve henüz bağlı değil."),
    "kuraklik": AnalysisKind(
        "kuraklik", "Kuraklık / bitki stresi", "ndvi", "ndmi",
        (("Değişim yok / artış", "#4cc38a"), ("Hafif düşüş", "#f1e05a"), ("Orta düşüş", "#f4a640"),
         ("Şiddetli düşüş", "#e5483a"), ("Bitkisiz (analiz dışı)", "#2b2f3a")),
        (1, 2, 3), (0.0, 0.15, 0.4, 0.75, 0.0), "NDVI bağıl değişimi",
        "Bağıl değişim = (bu yıl − geçen yıl aynı dönem)/geçen yıl NDVI; geçen yıl NDVI < 0,2 olan pikseller bitkisiz sayılıp dışarıda bırakılır. "
        "Aynı mevsimin kıyaslanması, olgunlaşma/hasat gibi doğal mevsimsel değişimin stres sanılmasını azaltır.",
        "Sınıf sınırları (−%10/−%25/−%45) kalibre edilmemiş başlangıç değerleridir; geçen yıl da olağandışı (kurak/ıslak) olduysa kıyas yanıltır."),
    "tarim": AnalysisKind(
        "tarim", "Tarımsal gelişim", "ndvi", "evi",
        (("Belirgin düşüş", "#e5483a"), ("Düşüş", "#f4a640"), ("Değişmedi", "#9aa3b2"), ("Artış", "#8bd17c"),
         ("Belirgin artış", "#2f9e5b")),
        (0, 1), (1.0, 0.5, 0.0, 0.0, 0.0), "NDVI farkı (sonra − önce)",
        "ΔNDVI = sonra − önce; ±0,05 değişmedi, ±0,15 belirgin.",
        "Büyüme mevsiminde düşüş olgunlaşmanın doğal sonucu olabilir; sezon aşamasını (fenoloji) birlikte okuyun."),
}


def classify_change(kind_key: str, pre: np.ndarray, post: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Önce/sonra indeks rasterlerinden değişim ölçüsünü ve sınıf indeksini üretir.

    Args:
        kind_key: ``KINDS`` anahtarı.
        pre: Önce indeksi (``nan`` = geçersiz).
        post: Sonra indeksi.

    Returns:
        ``(delta, sınıf)``; sınıf ``int8`` (geçersiz piksel ``-1``).
    """
    if kind_key not in KINDS:
        raise AnalysisError(f"Bilinmeyen analiz türü: {kind_key!r}")
    pre = np.asarray(pre, dtype=np.float64)
    post = np.asarray(post, dtype=np.float64)
    ok = np.isfinite(pre) & np.isfinite(post)
    # 6 ondalığa yuvarlama: float32 gösterim gürültüsü sınıf sınırındaki değerleri yanlış tarafa atmasın.
    if kind_key == "yangin":
        delta = np.round(pre - post, 6)
        cls = np.digitize(delta, (-0.10,) + de.DNBR_EDGES)
    elif kind_key == "sel":
        delta = np.round(post - pre, 6)
        pre_w, post_w = pre > 0.0, post > 0.0
        cls = np.select([pre_w & post_w, post_w & ~pre_w, pre_w & ~post_w], [1, 2, 3], default=0)
    elif kind_key == "kuraklik":
        delta = np.round(post - pre, 6)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.round(delta / np.maximum(pre, 0.2), 6)
        cls = np.select([pre < 0.2, rel >= -0.10, rel >= -0.25, rel >= -0.45], [4, 0, 1, 2], default=3)
    else:
        delta = np.round(post - pre, 6)
        cls = np.digitize(delta, (-0.15, -0.05, 0.05, 0.15))
    return delta.astype(np.float32), np.where(ok, cls, -1).astype(np.int8)


@dataclass(frozen=True)
class Reliability:
    """Veri kalitesi göstergesi (istatistiksel güven aralığı değildir).

    Attributes:
        score: 0–100 puan.
        level: ``Yüksek``/``Orta``/``Düşük``.
        parts: Bileşen → 0–1 değer.
        notes: Açıklayıcı uyarılar.
    """

    score: int
    level: str
    parts: Dict[str, float]
    notes: Tuple[str, ...]


#: Güvenilirlik bileşen ağırlıkları (toplam 1).
RELIABILITY_WEIGHTS: Dict[str, float] = {"berraklik": 0.45, "bulut_karo": 0.15, "zamanlama": 0.25, "ornek": 0.15}


def _ramp(x: float, full: float, zero: float) -> float:
    """``x = full`` iken 1, ``x = zero`` iken 0 olan doğrusal rampa (0–1 kırpılı)."""
    if full == zero:
        return 1.0
    return float(min(1.0, max(0.0, (x - zero) / (full - zero))))


def assess_reliability(kind_key: str, pre: SceneStats, post: SceneStats, region_pixels: int,
                       event_date: Optional[dt.date]) -> Reliability:
    """Önce/sonra çiftinin veri kalitesini puanlar.

    Bileşenler (ağırlık): berraklık = iki sahnenin en düşük kullanılabilir piksel oranı (0,45); karo bulutu ≤ %20 tam puan,
    %80'de sıfır (0,15); zamanlama = olaydan önceki ≤ 15 gün / sonraki ≤ 30 gün tam puan, 60/90 günde 0,3 (0,25);
    örnek = bölgedeki piksel sayısı ≥ 400 tam, 30'da sıfır (0,15).

    Args:
        kind_key: Analiz türü.
        pre: Önce sahnesi istatistiği.
        post: Sonra sahnesi istatistiği.
        region_pixels: Bölgedeki piksel sayısı.
        event_date: Olay tarihi (``None`` ise zamanlama nötr 0,7 sayılır).
    """
    notes: List[str] = []
    clarity = min(pre.valid_pct, post.valid_pct) / 100.0
    if min(pre.valid_pct, post.valid_pct) < 60.0:
        notes.append(f"Kullanılabilir piksel düşük (önce %{pre.valid_pct:.0f}, sonra %{post.valid_pct:.0f}); bulut/gölge maskelendi.")
    tile = [c for c in (pre.cloud_tile, post.cloud_tile) if math.isfinite(c)]
    cloud_part = _ramp(max(tile), 20.0, 80.0) if tile else 0.5
    if tile and max(tile) > CLOUD_WARN_PCT:
        notes.append(f"Karo bulutu %{max(tile):.0f} (> %{CLOUD_WARN_PCT:g}): güvenilirlik düşük uyarısı.")
    if event_date is None:
        timing = 0.7
        notes.append("Olay tarihi verilmedi; zamanlama nötr sayıldı.")
    elif kind_key == "kuraklik":
        post_gap = (post.date - event_date).days
        misalign = abs((post.date - pre.date).days - 365)
        pre_t = _ramp(float(misalign), 5.0, 25.0)
        post_t = 0.0 if post_gap <= 0 else 0.3 + 0.7 * _ramp(post_gap, 30.0, 90.0)
        timing = (pre_t + post_t) / 2.0
        if misalign > 15:
            notes.append(f"Kıyas sahnesi geçen yılın aynı gününden {misalign} gün sapıyor; mevsim evresi farklı olabilir.")
        if post_gap <= 0:
            notes.append("Sonra görüntüsü olaydan sonra değil.")
    else:
        pre_gap = (event_date - pre.date).days
        post_gap = (post.date - event_date).days
        pre_t = 0.0 if pre_gap <= 0 else 0.3 + 0.7 * _ramp(pre_gap, 15.0, 60.0)
        post_t = 0.0 if post_gap <= 0 else 0.3 + 0.7 * _ramp(post_gap, 30.0, 90.0)
        timing = (pre_t + post_t) / 2.0
        if pre_gap <= 0:
            notes.append("Önce görüntüsü olaydan önce değil.")
        if post_gap <= 0:
            notes.append("Sonra görüntüsü olaydan sonra değil.")
        if pre_gap > 30:
            notes.append(f"Önce görüntüsü olaydan {pre_gap} gün önce; arada başka değişim olmuş olabilir.")
        if post_gap > 45:
            notes.append(f"Sonra görüntüsü olaydan {post_gap} gün sonra; toparlanma ya da hasat etkisi karışabilir.")
        if kind_key == "yangin" and 0 < post_gap < 7:
            notes.append("Yangın sonrası ilk günlerde dNBR eksik kalabilir (is/duman); birkaç hafta sonrası önerilir.")
    sample = _ramp(float(region_pixels), 400.0, 30.0)
    if region_pixels < 100:
        notes.append(f"Bölgede yalnızca {region_pixels} piksel var; istatistik kararsız olabilir.")
    parts = {"berraklik": clarity, "bulut_karo": cloud_part, "zamanlama": timing, "ornek": sample}
    score = int(round(100.0 * sum(RELIABILITY_WEIGHTS[k] * v for k, v in parts.items())))
    level = "Yüksek" if score >= 75 else ("Orta" if score >= 50 else "Düşük")
    notes.append(KINDS[kind_key].caveat)
    return Reliability(score, level, parts, tuple(notes))


@dataclass
class ChangeResult:
    """Önce/sonra değişim analizi sonucu.

    Attributes:
        kind: Analiz türü.
        grid: Izgara.
        pre_meta: Önce sahnesi.
        post_meta: Sonra sahnesi.
        pre_index: Önce indeksi.
        post_index: Sonra indeksi.
        delta: Değişim ölçüsü.
        cls: Sınıf indeksi (``-1`` = geçersiz).
        region_mask: Bölge maskesi.
        class_ha: Sınıf başına alan (ha).
        region_ha: Bölge alanı (ha; piksel sayımından).
        valid_ha: Kullanılabilir alan (ha).
        affected_ha: Etkilenen alan (ha).
        reliability: Veri kalitesi.
        ndvi_pre: Önce NDVI rasteri (tarım görünümü için; her analiz türünde hesaplanır).
        ndvi_post: Sonra NDVI rasteri.
        event_date: Olay tarihi.
    """

    kind: AnalysisKind
    grid: Grid
    pre_meta: SceneMeta
    post_meta: SceneMeta
    pre_index: np.ndarray
    post_index: np.ndarray
    delta: np.ndarray
    cls: np.ndarray
    region_mask: np.ndarray
    class_ha: np.ndarray
    region_ha: float
    valid_ha: float
    affected_ha: float
    reliability: Reliability
    ndvi_pre: np.ndarray
    ndvi_post: np.ndarray
    event_date: Optional[dt.date] = None

    @property
    def affected_pct(self) -> float:
        """Etkilenen alanın kullanılabilir alana oranı (%)."""
        return 100.0 * self.affected_ha / self.valid_ha if self.valid_ha > 0 else math.nan

    @property
    def valid_pct(self) -> float:
        """Bölgede kullanılabilir alan oranı (%)."""
        return 100.0 * self.valid_ha / self.region_ha if self.region_ha > 0 else 0.0

    @property
    def mean_delta(self) -> float:
        """Kullanılabilir piksellerde ortalama değişim ölçüsü."""
        sel = self.region_mask & (self.cls >= 0)
        return float(np.nanmean(self.delta[sel])) if sel.any() else math.nan

    def class_table(self) -> pd.DataFrame:
        """Sınıf tablosu: ``sinif, alan_ha, oran_pct`` (oran kullanılabilir alana göre)."""
        pct = 100.0 * self.class_ha / self.valid_ha if self.valid_ha > 0 else np.full(self.class_ha.shape, np.nan)
        return pd.DataFrame({"sinif": [c[0] for c in self.kind.classes], "renk": [c[1] for c in self.kind.classes],
                             "alan_ha": self.class_ha.round(2), "oran_pct": np.round(pct, 1)})


def analyse_change(kind_key: str, pre: SceneData, post: SceneData, region: Region, grid: Grid,
                   event_date: Optional[dt.date] = None) -> ChangeResult:
    """Önce/sonra sahnelerinden değişim haritası, alan istatistiği ve güvenilirlik üretir.

    Args:
        kind_key: Analiz türü.
        pre: Önce sahnesi (olaydan önce olmalı).
        post: Sonra sahnesi.
        region: Bölge.
        grid: Her iki sahnenin ortak ızgarası.
        event_date: Olay tarihi.

    Raises:
        AnalysisError: Sahneler aynı gün, ters sırada ya da ızgara uyumsuzsa.
    """
    kind = KINDS.get(kind_key)
    if kind is None:
        raise AnalysisError(f"Bilinmeyen analiz türü: {kind_key!r}")
    if pre.meta.date >= post.meta.date:
        raise AnalysisError("Önce sahnesi sonra sahnesinden önce olmalı.")
    mask = polygon_mask(grid, region.ring)
    pre_stats, pre_idx = scene_stats(pre, kind.index, mask)
    post_stats, post_idx = scene_stats(post, kind.index, mask)
    if pre_idx.shape != post_idx.shape or pre_idx.shape != (grid.rows, grid.cols):
        raise AnalysisError("Sahne ızgaraları uyumsuz.")
    delta, cls = classify_change(kind_key, pre_idx, post_idx)
    cls = np.where(mask, cls, -1).astype(np.int8)
    px = grid.pixel_area_ha
    counts = np.array([int(((cls == i) & mask).sum()) for i in range(len(kind.classes))], dtype=float)
    class_ha = counts * px
    region_ha = float(mask.sum()) * px
    valid_ha = float((cls >= 0).sum()) * px
    affected = float(class_ha[list(kind.affected)].sum())
    rel = assess_reliability(kind_key, pre_stats, post_stats, int(mask.sum()), event_date)
    if kind.index == "ndvi":
        ndvi_pre, ndvi_post = pre_idx, post_idx
    else:
        ndvi_pre = scene_stats(pre, "ndvi", mask)[1]
        ndvi_post = scene_stats(post, "ndvi", mask)[1]
    return ChangeResult(kind, grid, pre.meta, post.meta, pre_idx, post_idx, delta, cls, mask, class_ha, region_ha, valid_ha,
                        affected, rel, ndvi_pre, ndvi_post, event_date)


# --------------------------------------------------------------------------- #
# Sektör görünümleri (ortak çekirdeğin iki yüzü)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InsuranceView:
    """Sigorta görünümü (gösterge; ekspertizin yerine geçmez).

    Attributes:
        damage_ratio: Alan ağırlıklı hasar oranı (0–1).
        decision: ``RET``/``EKSPERTİZ``/``KISMİ ONAY``/``ONAY``.
        indicative_claim_tl: Gösterge tazminat (muafiyet ve teminat uygulanmış).
        severity_text: Kısa özet.
        notes: Uyarılar.
        mean_dnbr: Yangın analizinde bölge ortalama dNBR (diğerlerinde ``nan``).
    """

    damage_ratio: float
    decision: str
    indicative_claim_tl: float
    severity_text: str
    notes: Tuple[str, ...]
    mean_dnbr: float = math.nan


def insurance_view(result: ChangeResult, rules: de.ClaimRules, insured_tl: float) -> InsuranceView:
    """Değişim sonucundan sigorta görünümü üretir.

    Yangında karar, uygulamanın tek karar kuralıyla (``decide_codes``) verilir: bölge ortalama dNBR ve gözlem güveni
    (kullanılabilir piksel oranı × karo bulutu bileşeni). Diğer türlerde hasar oranı sınıf ağırlıklarından (varsayım) gelir ve
    karar yalnızca "ekspertiz" ya da "ret" olur — otomatik onay önerilmez.

    Args:
        result: ``analyse_change`` çıktısı.
        rules: Karar kuralları.
        insured_tl: Sigorta bedeli (TL).
    """
    if not (isinstance(insured_tl, (int, float)) and math.isfinite(insured_tl) and insured_tl >= 0):
        raise AnalysisError("Sigorta bedeli geçerli bir sayı olmalı.")
    kind = result.kind
    sel = result.region_mask & (result.cls >= 0)
    notes: List[str] = []
    if not sel.any():
        return InsuranceView(math.nan, "EKSPERTİZ", 0.0, "Kullanılabilir piksel yok", ("Bulut/gölge nedeniyle ölçüm yapılamadı.",))
    mean_dnbr = math.nan
    if kind.key == "yangin":
        pix_ratio = de.damage_ratio(result.delta[sel], rules)
        ratio = float(pix_ratio.mean())
        mean_dnbr = float(result.delta[sel].mean())
        conf = float(result.reliability.parts["berraklik"] * (0.5 + 0.5 * result.reliability.parts["bulut_karo"]))
        code = int(de.InsuranceSimulator.decide_codes(np.array([mean_dnbr]), np.array([conf]), np.array([True]), rules)[0])
        decision = de.DECISION_LABELS[code]
        if conf < rules.min_confidence:
            notes.append(f"Gözlem güveni %{100 * conf:.0f} < eşik %{100 * rules.min_confidence:.0f}: otomatik onay verilmez, ekspertiz önerilir.")
        sev = int(de.classify_dnbr(np.array([mean_dnbr]))[0])
        severity = f"Ortalama dNBR {mean_dnbr:.2f} · {de.SEVERITY_LABELS[sev]} şiddet"
        gross = insured_tl * (rules.coverage_pct / 100.0) * ratio
        claim = max(0.0, gross - insured_tl * (rules.deductible_pct / 100.0))
    else:
        w = np.array(kind.weights, dtype=float)
        ratio = float(w[result.cls[sel]].mean())
        decision = "RET" if ratio < 0.02 else "EKSPERTİZ"
        severity = f"Etkilenen alan %{result.affected_pct:.1f} · ağırlıklı hasar oranı %{100 * ratio:.1f}"
        claim = max(0.0, insured_tl * (rules.coverage_pct / 100.0) * ratio - insured_tl * (rules.deductible_pct / 100.0))
        notes.append("Sınıf ağırlıkları varsayımdır; bu tür için otomatik onay önerilmez, ekspertiz gerekir.")
    if result.reliability.level == "Düşük":
        notes.append("Veri güvenilirliği düşük: sonucu ekspertiz teyidi olmadan kullanmayın.")
    return InsuranceView(ratio, decision, float(claim), severity, tuple(notes), mean_dnbr)


@dataclass(frozen=True)
class AgriView:
    """Tarım görünümü (her analiz türünde aynı parselin NDVI'sından hesaplanır).

    Attributes:
        veg_ha: Bitkili (önce NDVI ≥ 0,2) ve ölçülebilir alan (ha).
        stressed_ha: Bağıl NDVI düşüşü ≥ %25 olan alan (ha).
        stressed_pct: Bitkili alana oranı (%).
        growth_ha: NDVI'sı ≥ 0,05 artan alan (ha).
        mean_pre: Önce ortalama NDVI (bitkili piksellerde).
        mean_post: Sonra ortalama NDVI.
        health: ``Sağlıklı``/``İzlemede``/``Stresli``/``Kritik`` (ortalama bağıl değişime göre; ``Ölçülemedi`` olabilir).
        notes: Uyarılar.
    """

    veg_ha: float
    stressed_ha: float
    stressed_pct: float
    growth_ha: float
    mean_pre: float
    mean_post: float
    health: str
    notes: Tuple[str, ...]


def agri_view(result: ChangeResult) -> AgriView:
    """Aynı değişim sonucundan tarım görünümü üretir: bitki gelişimi ve stresli alan (NDVI tabanlı).

    Sigorta görünümüyle **aynı sahne çiftini** kullanır; yangın/sel gibi olaylarda NDVI düşüşü olayın etkisini de içerir.
    """
    notes: List[str] = []
    pre, post = result.ndvi_pre, result.ndvi_post
    ok = result.region_mask & np.isfinite(pre) & np.isfinite(post)
    veg = ok & (pre >= 0.2)
    if not veg.any():
        return AgriView(0.0, 0.0, math.nan, 0.0, math.nan, math.nan, "Ölçülemedi", ("Bitkili ve ölçülebilir piksel yok.",))
    px = result.grid.pixel_area_ha
    rel = (post - pre) / np.maximum(pre, 0.2)
    pre_m, post_m = float(pre[veg].mean()), float(post[veg].mean())
    mean_rel = (post_m - pre_m) / max(pre_m, 0.2)
    health = "Sağlıklı" if mean_rel >= -0.10 else ("İzlemede" if mean_rel >= -0.25 else ("Stresli" if mean_rel >= -0.45 else "Kritik"))
    stressed = veg & (rel < -0.25)
    growth = veg & ((post - pre) >= 0.05)
    if result.kind.key in ("sel", "yangin"):
        notes.append("NDVI düşüşü bu olayın doğrudan etkisini (su altında kalma/yanma) de içerir; bitki stresiyle aynı şey değildir.")
    if result.kind.key == "kuraklik":
        notes.append("Kıyas geçen yılın aynı dönemidir; geçen yıl da olağandışıysa sonuç yanıltır.")
    else:
        notes.append("Büyüme döneminde düşüş olgunlaşmanın doğal sonucu olabilir; fenolojik evreyi birlikte okuyun.")
    notes.append("Rekolte tahmini ayrı, çok değişkenli bir modelle yapılır; NDVI tek başına yeterli değildir.")
    return AgriView(float(veg.sum()) * px, float(stressed.sum()) * px, 100.0 * float(stressed.sum()) / float(veg.sum()),
                    float(growth.sum()) * px, pre_m, post_m, health, tuple(notes))


# --------------------------------------------------------------------------- #
# Hava durumu ve yangın noktası kanıtı
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvidenceRow:
    """Koşul doğrulaması satırı.

    Attributes:
        label: Gösterge adı.
        value: Değer metni.
        flag: ``True`` = olayı destekler, ``False`` = desteklemez, ``None`` = veri yok.
        rule: Kullanılan kural (varsayılan, resmî eşik değil).
    """

    label: str
    value: str
    flag: Optional[bool]
    rule: str


def growing_degree_days(tmax: Sequence[float], tmin: Sequence[float], base_c: float = 5.0, cap_c: float = 30.0) -> float:
    """Büyüme derece-günü toplamı (basit ortalama yöntemi; sıcaklıklar ``[base, cap]`` aralığına kırpılır).

    Args:
        tmax: Günlük en yüksek sıcaklıklar (°C).
        tmin: Günlük en düşük sıcaklıklar (°C).
        base_c: Taban sıcaklık.
        cap_c: Üst sınır.
    """
    hi = np.clip(np.asarray(tmax, dtype=float), base_c, cap_c)
    lo = np.clip(np.asarray(tmin, dtype=float), base_c, cap_c)
    return float(np.nansum((hi + lo) / 2.0 - base_c))


def weather_evidence(kind_key: str, daily: Optional[pd.DataFrame], event_date: Optional[dt.date]) -> List[EvidenceRow]:
    """Olay tarihindeki hava koşullarını türe göre özetler (varsayılan, ayarlanabilir göstergeler).

    Args:
        kind_key: Analiz türü.
        daily: Günlük tablo: ``tarih, yagis_mm, tmax_c, tmin_c, ruzgar_kmh, et0_mm`` (eksik sütun/veri ``None`` bayrağı verir).
        event_date: Olay tarihi.
    """
    if daily is None or daily.empty or event_date is None:
        return [EvidenceRow("Hava durumu", "veri yok", None, "Hava verisi alınamadı ya da olay tarihi seçilmedi.")]
    d = daily.copy()
    d["tarih"] = pd.to_datetime(d["tarih"])
    ev = pd.Timestamp(event_date)

    def window(days_before: int, days_after: int = 0) -> pd.DataFrame:
        return d[(d["tarih"] >= ev - pd.Timedelta(days=days_before)) & (d["tarih"] <= ev + pd.Timedelta(days=days_after))]

    def col(frame: pd.DataFrame, name: str) -> Optional[pd.Series]:
        return frame[name] if name in frame.columns and frame[name].notna().any() else None

    rows: List[EvidenceRow] = []
    if kind_key == "sel":
        p3 = col(window(2), "yagis_mm")
        total = float(p3.sum()) if p3 is not None else math.nan
        rows.append(EvidenceRow("3 günlük yağış toplamı (olay dahil)", f"{total:.0f} mm" if math.isfinite(total) else "yok",
                                None if not math.isfinite(total) else total >= 40.0, "≥ 40 mm → aşırı yağış göstergesi"))
        p1 = col(window(1), "yagis_mm")
        mx = float(p1.max()) if p1 is not None else math.nan
        rows.append(EvidenceRow("Olaydan önceki 2 günde en yağışlı gün", f"{mx:.0f} mm" if math.isfinite(mx) else "yok",
                                None if not math.isfinite(mx) else mx >= 25.0, "≥ 25 mm/gün → kuvvetli yağış göstergesi"))
    elif kind_key == "kuraklik":
        p30 = col(window(30), "yagis_mm")
        pall = col(d, "yagis_mm")
        if p30 is not None and pall is not None and len(d) >= 60:
            expected = float(pall.mean()) * len(window(30))
            total = float(p30.sum())
            ratio = total / expected if expected > 0 else math.nan
            rows.append(EvidenceRow("Olaydan önceki 30 gün yağış / dönem ortalaması",
                                    f"{total:.0f} mm (oran {ratio:.2f})" if math.isfinite(ratio) else "yok",
                                    None if not math.isfinite(ratio) else ratio < 0.5, "oran < 0,5 → kurak dönem göstergesi"))
        else:
            rows.append(EvidenceRow("Yağış anomalisi", "veri yok (en az 60 günlük pencere gerekir)", None, "oran < 0,5"))
        t14 = col(window(14), "tmax_c")
        hot = float(t14.mean()) if t14 is not None else math.nan
        rows.append(EvidenceRow("Son 14 gün ortalama en yüksek sıcaklık", f"{hot:.1f} °C" if math.isfinite(hot) else "yok",
                                None if not math.isfinite(hot) else hot >= 32.0, "≥ 32 °C → sıcak stres göstergesi"))
    elif kind_key == "yangin":
        tmax = col(window(0), "tmax_c")
        tv = float(tmax.iloc[-1]) if tmax is not None and len(tmax) else math.nan
        rows.append(EvidenceRow("Olay günü en yüksek sıcaklık", f"{tv:.1f} °C" if math.isfinite(tv) else "yok",
                                None if not math.isfinite(tv) else tv >= 30.0, "≥ 30 °C → yangın koşulu göstergesi"))
        wind = col(window(0), "ruzgar_kmh")
        wv = float(wind.iloc[-1]) if wind is not None and len(wind) else math.nan
        rows.append(EvidenceRow("Olay günü en yüksek rüzgâr", f"{wv:.0f} km/sa" if math.isfinite(wv) else "yok",
                                None if not math.isfinite(wv) else wv >= 30.0, "≥ 30 km/sa → yayılmayı hızlandırıcı rüzgâr"))
        p7 = col(window(7), "yagis_mm")
        ps = float(p7.sum()) if p7 is not None else math.nan
        rows.append(EvidenceRow("Olaydan önceki 7 gün yağış", f"{ps:.1f} mm" if math.isfinite(ps) else "yok",
                                None if not math.isfinite(ps) else ps < 2.0, "< 2 mm → kuru koşul göstergesi"))
    else:
        tmax, tmin = col(d, "tmax_c"), col(d, "tmin_c")
        if tmax is not None and tmin is not None:
            rows.append(EvidenceRow("Büyüme derece-günü (taban 5 °C)", f"{growing_degree_days(d['tmax_c'], d['tmin_c']):.0f} °C·gün",
                                    None, "bilgi amaçlı; ürüne göre yorumlanır"))
            rows.append(EvidenceRow("Sıcak gün sayısı (Tmax ≥ 32 °C)", f"{int((d['tmax_c'] >= 32.0).sum())} gün",
                                    None, "bilgi amaçlı"))
        pp = col(d, "yagis_mm")
        if pp is not None:
            rows.append(EvidenceRow("Dönem toplam yağış", f"{float(pp.sum()):.0f} mm", None, "bilgi amaçlı"))
    return rows or [EvidenceRow("Hava durumu", "uygun veri yok", None, "")]


def match_hotspots(hotspots: Optional[pd.DataFrame], region: Region, start: dt.date, end: dt.date,
                   buffer_km: float = 1.0) -> pd.DataFrame:
    """FIRMS yangın noktalarını bölge ve tarih aralığıyla eşleştirir.

    Bir nokta, çokgenin içindeyse ya da kenarına ``buffer_km`` içindeyse ve tarihi ``[start, end]`` aralığındaysa eşleşir
    (VIIRS 375 m, MODIS 1 km çözünürlüklü olduğundan konum belirsizliği tamponla karşılanır).

    Args:
        hotspots: ``data_sources.HOTSPOT_COLUMNS`` şemalı tablo.
        region: Bölge.
        start: İlk gün.
        end: Son gün.
        buffer_km: Kenar tamponu (km).

    Returns:
        Eşleşen noktalar; ek sütunlar: ``icinde`` (çokgen içi), ``uzaklik_m``, ``gun`` (tarih).
    """
    cols = ["enlem", "boylam", "frp_mw", "kaynak", "tarih_utc", "icinde", "uzaklik_m", "gun"]
    if hotspots is None or hotspots.empty:
        return pd.DataFrame(columns=cols)
    h = hotspots.copy()
    h["tarih_utc"] = pd.to_datetime(h["tarih_utc"], utc=True, errors="coerce")
    h = h.dropna(subset=["enlem", "boylam", "tarih_utc"])
    h["gun"] = h["tarih_utc"].dt.date
    h = h[(h["gun"] >= start) & (h["gun"] <= end)]
    if h.empty:
        return pd.DataFrame(columns=cols)
    dist = distance_to_ring_m(h["boylam"].to_numpy(), h["enlem"].to_numpy(), region.ring)
    w, s, e, n = region.bbox
    inside = np.zeros(len(h), dtype=bool)
    box = (h["boylam"].to_numpy() >= w) & (h["boylam"].to_numpy() <= e) & (h["enlem"].to_numpy() >= s) & (h["enlem"].to_numpy() <= n)
    for i in np.flatnonzero(box):
        inside[i] = _point_in_ring(float(h["boylam"].iloc[i]), float(h["enlem"].iloc[i]), region.ring)
    h["icinde"] = inside
    h["uzaklik_m"] = np.where(inside, 0.0, dist)
    out = h[(h["icinde"]) | (h["uzaklik_m"] <= buffer_km * 1000.0)]
    return out[cols].sort_values("tarih_utc").reset_index(drop=True)


def _point_in_ring(lon: float, lat: float, ring: Sequence[Tuple[float, float]]) -> bool:
    """Tek nokta için tek-çift kuralı çokgen içi sınaması."""
    inside = False
    pts = list(ring)
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


@dataclass(frozen=True)
class SupportView:
    """Kuraklık destek ödemesi simülasyonu (parametrik; gerçek bir destek programının kuralı DEĞİLDİR).

    Attributes:
        eligible: Hasar oranı uygunluk eşiğine ulaştı mı.
        damage_ratio: Alan ağırlıklı kuraklık hasar oranı (0–1).
        payable_ha: Ödemeye esas alan (ha) = ölçülebilir alan × hasar oranı (uygunsa).
        payment_tl: Simüle edilen ödeme (TL).
        unit_tl_ha: Varsayımsal birim destek (TL/ha).
        min_ratio: Varsayımsal uygunluk eşiği (0–1).
        needs_review: Güvenilirlik düşük olduğundan elle doğrulama gerekir.
        notes: Açıklamalar.
    """

    eligible: bool
    damage_ratio: float
    payable_ha: float
    payment_tl: float
    unit_tl_ha: float
    min_ratio: float
    needs_review: bool
    notes: Tuple[str, ...]


def drought_support(result: ChangeResult, ins: InsuranceView, unit_tl_per_ha: float, min_ratio: float = 0.30) -> SupportView:
    """Kuraklık sonucundan parametrik destek ödemesi simülasyonu üretir.

    Kural (tamamen varsayımsal, kullanıcı ayarlı): hasar oranı ``min_ratio`` ve üstüyse ödeme = ölçülebilir alan × hasar oranı × birim destek.
    Güvenilirlik "Düşük" ise ödeme önerilmez; elle doğrulama istenir.

    Args:
        result: Kuraklık analizi sonucu.
        ins: Aynı sonucun sigorta görünümü (hasar oranı kaynağı).
        unit_tl_per_ha: Birim destek (TL/ha, ≥ 0).
        min_ratio: Uygunluk eşiği (0–1).

    Raises:
        AnalysisError: Analiz türü kuraklık değilse ya da parametreler geçersizse.
    """
    if result.kind.key != "kuraklik":
        raise AnalysisError("Destek ödemesi simülasyonu yalnızca kuraklık analizi içindir.")
    if not (math.isfinite(unit_tl_per_ha) and unit_tl_per_ha >= 0 and 0.0 < min_ratio <= 1.0):
        raise AnalysisError("Birim destek ≥ 0 ve eşik (0, 1] aralığında olmalı.")
    ratio = ins.damage_ratio
    review = result.reliability.level == "Düşük"
    eligible = bool(math.isfinite(ratio) and ratio >= min_ratio)
    payable = result.valid_ha * ratio if eligible else 0.0
    pay = 0.0 if review else payable * unit_tl_per_ha
    notes = ["Birim destek ve uygunluk eşiği VARSAYIMSALDIR; gerçek bir destek programının kuralı ya da tutarı değildir."]
    if review:
        notes.append("Güvenilirlik düşük: otomatik ödeme önerilmez, saha doğrulaması gerekir.")
    if not eligible and math.isfinite(ratio):
        notes.append(f"Hasar oranı %{100 * ratio:.1f}, eşik %{100 * min_ratio:.0f} altında: ödeme yok.")
    return SupportView(eligible, ratio, payable, pay, float(unit_tl_per_ha), float(min_ratio), review, tuple(notes))


def point_in_region(region: Region, lon: float, lat: float) -> bool:
    """Nokta bölge çokgeninin içinde mi (sınır üzerindeki noktalar için sonuç belirsizdir)."""
    w, s, e, n = region.bbox
    return w <= lon <= e and s <= lat <= n and _point_in_ring(lon, lat, region.ring)


# --------------------------------------------------------------------------- #
# Görselleştirme
# --------------------------------------------------------------------------- #
def _hex_rgb(color: str) -> Tuple[int, int, int]:
    c = color.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))


#: NDVI/NBR benzeri sürekli renk rampası (kırmızı → sarı → yeşil), görseldeki gibi.
RAMP_STOPS: Tuple[Tuple[float, str], ...] = (
    (0.0, "#d7191c"), (0.25, "#fdae61"), (0.5, "#ffff33"), (0.75, "#a6d96a"), (1.0, "#1a9641"),
)


def colorize(values: np.ndarray, lo: float, hi: float, mask: Optional[np.ndarray] = None, alpha: int = 235,
             classes: Optional[Sequence[str]] = None, class_idx: Optional[np.ndarray] = None) -> np.ndarray:
    """Rasteri RGBA'ya boyar (``uint8``; geçersiz/maskesiz pikseller saydam).

    Args:
        values: Sürekli değerler (``classes`` verilmediğinde kullanılır).
        lo: Rampa alt sınırı.
        hi: Rampa üst sınırı.
        mask: ``True`` olan pikseller çizilir.
        alpha: Çizilen piksellerin opaklığı (0–255).
        classes: Sınıf renkleri (hex); verilirse ``class_idx`` boyanır.
        class_idx: Sınıf indeksleri (``-1`` = çizme).
    """
    shape = (class_idx if classes is not None else values).shape
    out = np.zeros(shape + (4,), dtype=np.uint8)
    if classes is not None:
        assert class_idx is not None
        palette = np.array([_hex_rgb(c) for c in classes], dtype=np.uint8)
        ok = class_idx >= 0
        if mask is not None:
            ok &= mask
        out[ok, :3] = palette[class_idx[ok]]
        out[ok, 3] = alpha
        return out
    t = np.clip((np.asarray(values, dtype=np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    xs = np.array([p[0] for p in RAMP_STOPS])
    cols = np.array([_hex_rgb(p[1]) for p in RAMP_STOPS], dtype=np.float32)
    ok = np.isfinite(values) if mask is None else (np.isfinite(values) & mask)
    tt = np.where(ok, t, 0.0)
    for ch in range(3):
        out[..., ch] = np.interp(tt, xs, cols[:, ch]).astype(np.uint8)
    out[..., 3] = np.where(ok, alpha, 0)
    return out
