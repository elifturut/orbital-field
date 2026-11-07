"""
live_sources.py — Canlı veri: Sentinel-2 (Earth Search STAC + COG), Open-Meteo hava durumu, NASA FIRMS
=======================================================================================================

Analiz iş akışının gerçek veri ayağı. Üç kaynak, ortak bir ``LiveSource`` arkasında toplanır:

* **Sentinel-2 L2A**: AWS üzerindeki *Earth Search* STAC kataloğu (anahtarsız) sahneleri bulur; bölgeye karşılık gelen küçük
  pencere Cloud-Optimized GeoTIFF'ten ``rasterio`` ile **yalnızca o pencere** okunur (tüm sahne indirilmez). Yansıtım =
  ham × ölçek + kayma (``raster:bands`` üstverisi); SCL ile bulut/gölge maskelenir (``analysis``).
* **Open-Meteo**: ERA5 tabanlı arşiv + tahmin uç noktaları (anahtarsız): günlük yağış, sıcaklık, rüzgâr, ET0.
  Bu **yeniden analiz verisidir (≈ 9–25 km), istasyon ölçümü değildir**.
* **NASA FIRMS**: aktif yangın noktaları; tarih aralığı en çok 5 günlük parçalara bölünür, kaynak olayın yaşına göre seçilir.

Dürüstlük notu
--------------
Bu modül ağ erişimi olmayan bir ortamda yazıldı. İstek/yanıt biçimleri belgelere göre kuruldu ve **sahte oturum/okuyucu ile
çevrimdışı test edildi**; gerçek sunuculara ve gerçek ``rasterio`` okumasına karşı henüz çalıştırılmadı. Bu yüzden arayüzde
"Canlı bağlantıyı sına" düğmesi (``run_diagnostics``) vardır: her adımı ayrı ayrı sınar ve hatayı açık gösterir. Hata olursa
iş akışı sessizce sentetik veriye **düşmez**; kullanıcıya bildirir.

Geliştirici: Elif Turut
"""

from __future__ import annotations

import datetime as dt
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import analysis as an
import data_sources as ds

__all__ = ["EARTH_SEARCH_URL", "COLLECTIONS", "OPEN_METEO_FORECAST_URL", "OPEN_METEO_ARCHIVE_URL", "StacClient", "OpenMeteoClient",
           "LiveSource", "LiveSourceError", "DiagnosticRow", "rasterio_reader", "parse_stac_items", "best_per_date",
           "firms_chunks", "pick_firms_source", "run_diagnostics", "BandReader"]

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
#: Önce Collection-1 (güncel), olmazsa eski koleksiyon denenir.
COLLECTIONS: Tuple[str, ...] = ("sentinel-2-c1-l2a", "sentinel-2-l2a")
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_DAILY_VARS = "precipitation_sum,temperature_2m_max,temperature_2m_min,wind_speed_10m_max,et0_fao_evapotranspiration"
_PLATFORMS = {"sentinel-2a": "Sentinel-2A", "sentinel-2b": "Sentinel-2B", "sentinel-2c": "Sentinel-2C"}
#: Earth Search varlık adı → bizim bant adı.
_ASSET_KEYS: Dict[str, str] = {"blue": "blue", "green": "green", "red": "red", "nir": "nir", "swir16": "swir16",
                               "swir22": "swir22", "scl": "scl"}
#: Okuyucu imzası: (adres, sınır kutusu, ızgara, kategorik mi) → 2B ham dizi (``uint16``/``uint8``).
BandReader = Callable[[str, Tuple[float, float, float, float], an.Grid, bool], np.ndarray]


class LiveSourceError(ds.DataSourceError):
    """Canlı kaynaktan veri alınamadı (ileti kimlik bilgisi içermez)."""


# --------------------------------------------------------------------------- #
# STAC — Sentinel-2 sahne kataloğu
# --------------------------------------------------------------------------- #
def parse_stac_items(payload: Mapping[str, Any], source: str = "earth-search") -> List[an.SceneMeta]:
    """STAC ``FeatureCollection`` yanıtını ``SceneMeta`` listesine çevirir (bozuk kayıtlar atlanır).

    Args:
        payload: ``/search`` yanıtı.
        source: Kaynak etiketi.
    """
    out: List[an.SceneMeta] = []
    for feat in payload.get("features", []) or []:
        try:
            props = feat["properties"]
            when = pd.Timestamp(props["datetime"])
            day = when.tz_convert("UTC").date() if when.tzinfo else when.date()
            assets = feat.get("assets", {}) or {}
            urls: Dict[str, str] = {}
            scale: Dict[str, Tuple[float, float]] = {}
            for key, band in _ASSET_KEYS.items():
                a = assets.get(key)
                if not a or not a.get("href"):
                    continue
                urls[band] = str(a["href"])
                rb = (a.get("raster:bands") or [{}])[0]
                if band != "scl":
                    scale[band] = (float(rb.get("scale", 0.0001)), float(rb.get("offset", 0.0)))
            if "red" not in urls or "nir" not in urls:
                continue
            cloud = props.get("eo:cloud_cover")
            platform = _PLATFORMS.get(str(props.get("platform", "")).lower(), str(props.get("platform", "Sentinel-2")))
            bb = feat.get("bbox")
            bbox = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])) if bb and len(bb) >= 4 else None
            out.append(an.SceneMeta(str(feat["id"]), day, platform, float(cloud) if cloud is not None else math.nan, source,
                                    urls, scale, bbox))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _covers(bbox: Optional[Tuple[float, float, float, float]], target: Tuple[float, float, float, float]) -> bool:
    """Sahne sınır kutusu hedef kutuyu tamamen kapsıyor mu (bilinmiyorsa ``True`` sayılır)."""
    if bbox is None:
        return True
    return bbox[0] <= target[0] and bbox[1] <= target[1] and bbox[2] >= target[2] and bbox[3] >= target[3]


def best_per_date(metas: Sequence[an.SceneMeta], target: Tuple[float, float, float, float]) -> List[an.SceneMeta]:
    """Aynı güne düşen kayıtlardan birini seçer: bölgeyi tam kapsayan, sonra en az bulutlu, sonra en küçük kimlik.

    Bölge iki karonun kesiştiği yerdeyse tam kapsayan karo yoksa o günün kaydı **yine de** tutulur (kısmi kapsama okuma
    sırasında sıfır/boş piksellerle görünür ve kullanılabilir piksel oranını düşürür).
    """
    groups: Dict[dt.date, List[an.SceneMeta]] = {}
    for m in metas:
        groups.setdefault(m.date, []).append(m)
    out = []
    for day in sorted(groups):
        out.append(min(groups[day], key=lambda m: (not _covers(m.bbox, target), m.cloud_pct if math.isfinite(m.cloud_pct) else 101.0,
                                                  m.scene_id)))
    return out


class StacClient:
    """Earth Search STAC istemcisi (anahtarsız)."""

    def __init__(self, session: Optional[ds.HttpSession] = None, base_url: str = EARTH_SEARCH_URL,
                 collections: Sequence[str] = COLLECTIONS) -> None:
        """İstemciyi kurar.

        Args:
            session: HTTP oturumu (``None`` ise ``requests.Session``).
            base_url: STAC kök adresi.
            collections: Sırayla denenecek koleksiyon kimlikleri.
        """
        self.session = session or ds._default_session()
        self.base_url = base_url.rstrip("/")
        self.collections = tuple(collections)
        self.collection_used: Optional[str] = None

    def search(self, bbox: Tuple[float, float, float, float], start: dt.date, end: dt.date, max_pages: int = 5,
               page_size: int = 100) -> List[an.SceneMeta]:
        """Bölge ve tarih aralığındaki sahneleri getirir (bulut süzgeci istemcide uygulanır).

        Args:
            bbox: ``(batı, güney, doğu, kuzey)``.
            start: İlk gün (dahil).
            end: Son gün (dahil).
            max_pages: En çok sayfa.
            page_size: Sayfa başına kayıt.

        Raises:
            LiveSourceError: Hiçbir koleksiyondan yanıt alınamazsa.
        """
        if end < start:
            raise LiveSourceError("Bitiş tarihi başlangıçtan önce olamaz.")
        errors: List[str] = []
        for coll in self.collections:
            try:
                metas = self._search_collection(coll, bbox, start, end, max_pages, page_size)
            except ds.DataSourceError as exc:
                errors.append(f"{coll}: {exc}")
                continue
            self.collection_used = coll
            return best_per_date(metas, bbox)
        raise LiveSourceError("Sentinel-2 kataloğuna ulaşılamadı — " + " | ".join(errors))

    def _search_collection(self, coll: str, bbox: Tuple[float, float, float, float], start: dt.date, end: dt.date,
                           max_pages: int, page_size: int) -> List[an.SceneMeta]:
        body: Dict[str, Any] = {
            "collections": [coll], "bbox": list(bbox), "limit": page_size,
            "datetime": f"{start:%Y-%m-%d}T00:00:00Z/{end:%Y-%m-%d}T23:59:59Z",
            "sortby": [{"field": "properties.datetime", "direction": "asc"}],
        }
        url = f"{self.base_url}/search"
        out: List[an.SceneMeta] = []
        for _ in range(max_pages):
            resp = ds._call(self.session.post, url, "Sentinel-2 kataloğu", (), json=body, timeout=60)
            payload = ds._json(resp, "Sentinel-2 kataloğu")
            out.extend(parse_stac_items(payload))
            nxt = next((lk for lk in payload.get("links", []) or [] if lk.get("rel") == "next"), None)
            if not nxt:
                break
            url = str(nxt.get("href", url))
            if nxt.get("method", "GET").upper() == "POST" and isinstance(nxt.get("body"), Mapping):
                body = {**body, **nxt["body"]} if nxt.get("merge") else dict(nxt["body"])
            else:
                break
        return out


# --------------------------------------------------------------------------- #
# COG okuma (rasterio) — yalnızca bölge penceresi
# --------------------------------------------------------------------------- #
def rasterio_reader(href: str, bbox: Tuple[float, float, float, float], grid: an.Grid, categorical: bool) -> np.ndarray:
    """COG'dan bölge penceresini ``grid`` boyutuna örnekleyerek okur (``rasterio`` gerekir).

    Sınır kutusu kaynak CBS'ye (UTM) çevrilir, pencere yalnızca o alanı kapsar; çıktı doğrudan ``(rows, cols)`` boyutuna
    yeniden örneklenir (kategorik bantta en yakın komşu, diğerlerinde çift doğrusal). Pencere dışına taşan kısım 0 ile
    doldurulur (``nodata``). Sınır kutusunun UTM'ye dönüşümü küçük bir eğrilik farkı bırakır (≤ 25 km kutuda birkaç on metre).

    Args:
        href: COG adresi (genelde ``https://...s3.../*.tif``).
        bbox: ``(batı, güney, doğu, kuzey)``.
        grid: Çıktı ızgarası.
        categorical: SCL gibi kategorik bant mı.

    Raises:
        LiveSourceError: ``rasterio`` yoksa ya da okuma başarısızsa.
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds
    except ImportError:
        raise LiveSourceError("Canlı görüntü için 'rasterio' paketi gerekli: pip install rasterio") from None
    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="1",
                          CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff", AWS_NO_SIGN_REQUEST="YES"):
            with rasterio.open(href) as src:
                bounds = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
                window = from_bounds(*bounds, transform=src.transform)
                arr = src.read(1, window=window, out_shape=(grid.rows, grid.cols), boundless=True, fill_value=0,
                               resampling=Resampling.nearest if categorical else Resampling.bilinear)
        return np.asarray(arr)
    except LiveSourceError:
        raise
    except Exception as exc:  # noqa: BLE001 — GDAL/rasterio hataları çok türlü; yalnızca tür bildirilir
        raise LiveSourceError(f"Görüntü penceresi okunamadı ({type(exc).__name__})") from None


# --------------------------------------------------------------------------- #
# Open-Meteo
# --------------------------------------------------------------------------- #
class OpenMeteoClient:
    """Open-Meteo günlük hava istemcisi (ERA5 arşivi + tahmin; anahtarsız)."""

    def __init__(self, session: Optional[ds.HttpSession] = None, today: Optional[dt.date] = None) -> None:
        """İstemciyi kurar.

        Args:
            session: HTTP oturumu.
            today: "Bugün" (testlerde sabitlemek için).
        """
        self.session = session or ds._default_session()
        self._today = today

    def _today_utc(self) -> dt.date:
        return self._today or dt.datetime.now(dt.timezone.utc).date()

    def daily(self, lat: float, lon: float, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Günlük hava tablosu: ``tarih, yagis_mm, tmax_c, tmin_c, ruzgar_kmh, et0_mm``.

        Son ~6 gün arşivde henüz yoktur; bu bölüm tahmin uç noktasından (``start_date``/``end_date``) alınır.

        Args:
            lat: Enlem.
            lon: Boylam.
            start: İlk gün.
            end: Son gün.

        Raises:
            LiveSourceError: İstek ya da yanıt hatasında.
        """
        if end < start:
            raise LiveSourceError("Bitiş tarihi başlangıçtan önce olamaz.")
        today = self._today_utc()
        end = min(end, today + dt.timedelta(days=15))
        cutoff = today - dt.timedelta(days=6)
        frames: List[pd.DataFrame] = []
        if start <= cutoff:
            frames.append(self._fetch(OPEN_METEO_ARCHIVE_URL, lat, lon, start, min(end, cutoff)))
        if end > cutoff:
            frames.append(self._fetch(OPEN_METEO_FORECAST_URL, lat, lon, max(start, cutoff + dt.timedelta(days=1)), end))
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return out.drop_duplicates("tarih").sort_values("tarih").reset_index(drop=True)

    def _fetch(self, url: str, lat: float, lon: float, start: dt.date, end: dt.date) -> pd.DataFrame:
        params = {"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "start_date": f"{start:%Y-%m-%d}", "end_date": f"{end:%Y-%m-%d}",
                  "daily": _DAILY_VARS, "timezone": "UTC"}
        resp = ds._call(self.session.get, url, "Open-Meteo", (), params=params, timeout=60)
        data = ds._json(resp, "Open-Meteo")
        daily = data.get("daily")
        if not isinstance(daily, Mapping) or "time" not in daily:
            raise LiveSourceError("Open-Meteo: yanıtta 'daily' bölümü yok" + (f" ({data.get('reason')})" if data.get("reason") else ""))

        def col(name: str) -> pd.Series:
            vals = daily.get(name)
            return pd.to_numeric(pd.Series(vals if vals is not None else [None] * len(daily["time"])), errors="coerce")

        return pd.DataFrame({"tarih": pd.to_datetime(pd.Series(daily["time"])), "yagis_mm": col("precipitation_sum"),
                             "tmax_c": col("temperature_2m_max"), "tmin_c": col("temperature_2m_min"),
                             "ruzgar_kmh": col("wind_speed_10m_max"), "et0_mm": col("et0_fao_evapotranspiration")})


# --------------------------------------------------------------------------- #
# FIRMS
# --------------------------------------------------------------------------- #
def firms_chunks(start: dt.date, end: dt.date, size: int = 5, max_chunks: int = 12) -> List[Tuple[dt.date, int]]:
    """``[start, end]`` aralığını ``(başlangıç, gün sayısı ≤ size)`` parçalarına böler.

    Raises:
        LiveSourceError: Aralık ``max_chunks × size`` günden uzunsa (FIRMS işlem kotasını korumak için).
    """
    if end < start:
        raise LiveSourceError("Bitiş tarihi başlangıçtan önce olamaz.")
    total = (end - start).days + 1
    if total > size * max_chunks:
        raise LiveSourceError(f"Yangın noktası sorgusu en çok {size * max_chunks} gün olabilir (istenen: {total}).")
    out = []
    cur = start
    while cur <= end:
        n = min(size, (end - cur).days + 1)
        out.append((cur, n))
        cur += dt.timedelta(days=n)
    return out


def pick_firms_source(event_day: dt.date, today: Optional[dt.date] = None) -> str:
    """Olayın yaşına göre FIRMS kaynağı: ~60 günden yeniyse NRT, eskiyse standart işlenmiş (SP) veri.

    NRT (yakın gerçek zamanlı) veri yaklaşık son 2 ayı kapsar; daha eski olaylar standart ürüne taşınmıştır (VIIRS S-NPP).
    Sınır tam bilinmediği için sonuç boşsa arayüz diğer kaynağı da denemelidir.
    """
    now = today or dt.datetime.now(dt.timezone.utc).date()
    return "VIIRS_SNPP_NRT" if (now - event_day).days <= 60 else "VIIRS_SNPP_SP"


# --------------------------------------------------------------------------- #
# Birleşik canlı kaynak
# --------------------------------------------------------------------------- #
class LiveSource:
    """Canlı kaynak: katalog + pencere okuma + hava + yangın noktası."""

    name = "Canlı (Sentinel-2 · Open-Meteo · FIRMS)"
    is_demo = False

    def __init__(self, session: Optional[ds.HttpSession] = None, creds: Optional[ds.Credentials] = None,
                 reader: Optional[BandReader] = None, today: Optional[dt.date] = None, max_workers: int = 6) -> None:
        """Kaynağı kurar.

        Args:
            session: HTTP oturumu (STAC, Open-Meteo, FIRMS ortak).
            creds: Kimlik bilgileri (FIRMS için ``firms_map_key``).
            reader: COG okuyucu (``None`` ise ``rasterio_reader``).
            today: "Bugün" (testlerde).
            max_workers: Eşzamanlı bant okuma sayısı.
        """
        self.session = session or ds._default_session()
        self.creds = creds or ds.load_credentials()
        self.reader: BandReader = reader or rasterio_reader
        self.stac = StacClient(self.session)
        self.meteo = OpenMeteoClient(self.session, today)
        self._today = today
        self._workers = max(1, int(max_workers))
        self.notes: List[str] = []

    def catalog(self, region: an.Region, start: dt.date, end: dt.date) -> List[an.SceneMeta]:
        """Sahne kataloğu (gün başına bir kayıt)."""
        return self.stac.search(region.bbox, start, end)

    def load(self, meta: an.SceneMeta, region: an.Region, grid: an.Grid, bands: Sequence[str]) -> an.SceneData:
        """Sahnenin istenen bantlarını ve SCL'yi ızgaraya okuyup yansıtıma çevirir.

        Raises:
            LiveSourceError: Adres yoksa ya da okuma başarısızsa.
        """
        if not meta.assets:
            raise LiveSourceError("Sahne kaydında bant adresi yok.")
        want = list(dict.fromkeys(bands))
        missing = [b for b in want + ["scl"] if b not in meta.assets]
        if missing:
            if "scl" in missing:
                self.notes.append(f"{meta.scene_id}: SCL yok; bulut maskesi uygulanamadı.")
                missing.remove("scl")
            if missing:
                raise LiveSourceError(f"Sahnede eksik bant: {missing}")
        names = want + (["scl"] if "scl" in meta.assets else [])

        def read(name: str) -> Tuple[str, np.ndarray]:
            arr = self.reader(meta.assets[name], region.bbox, grid, name == "scl")
            if arr.shape != (grid.rows, grid.cols):
                raise LiveSourceError(f"{name}: okunan boyut {arr.shape}, beklenen {(grid.rows, grid.cols)}")
            return name, arr

        with ThreadPoolExecutor(max_workers=min(self._workers, len(names))) as pool:
            raw = dict(pool.map(read, names))
        refl: Dict[str, np.ndarray] = {}
        for b in want:
            scale, offset = meta.band_scale.get(b, (0.0001, 0.0))
            dn = raw[b].astype(np.float32)
            val = dn * scale + offset
            nodata = dn == 0
            val = np.where(nodata, np.nan, np.clip(val, 0.0, 1.5)).astype(np.float32)
            refl[b] = val
        scl = raw["scl"].astype(np.uint8) if "scl" in raw else None
        if scl is not None:
            nodata_any = np.zeros(scl.shape, dtype=bool)
            for b in want:
                nodata_any |= raw[b] == 0
            scl = np.where(nodata_any, 0, scl).astype(np.uint8)
        return an.SceneData(meta, refl, scl)

    def weather(self, region: an.Region, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Günlük hava tablosu (bölge merkezi)."""
        lat, lon = region.centre
        return self.meteo.daily(lat, lon, start, end)

    def hotspots(self, region: an.Region, start: dt.date, end: dt.date, event_day: Optional[dt.date] = None,
                 grid: Optional[an.Grid] = None) -> pd.DataFrame:
        """FIRMS aktif yangın noktaları (5 günlük parçalarla, bölge sınır kutusunda; ``grid`` yok sayılır).

        Raises:
            LiveSourceError: ``FIRMS_MAP_KEY`` yoksa ya da istek başarısızsa.
        """
        if not self.creds.has_firms:
            raise LiveSourceError("FIRMS MAP_KEY tanımlı değil (FIRMS_MAP_KEY gizli anahtarı).")
        client = ds.FirmsClient(self.creds.firms_map_key, self.session)
        source = pick_firms_source(event_day or start, self._today)
        frames = []
        for day, n in firms_chunks(start, end):
            frames.append(client.hotspots(region.bbox, source, n, day))
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=list(ds.HOTSPOT_COLUMNS))
        out = pd.concat(frames, ignore_index=True).drop_duplicates(["enlem", "boylam", "tarih_utc"])
        return out.sort_values("tarih_utc").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Tanılama
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DiagnosticRow:
    """Bir tanılama adımının sonucu.

    Attributes:
        step: Adım adı.
        ok: Başarılı mı.
        detail: Kısa ayrıntı (kimlik bilgisi içermez).
        ms: Süre (ms).
    """

    step: str
    ok: bool
    detail: str
    ms: int


def run_diagnostics(source: LiveSource, probe_lat: float = 39.0, probe_lon: float = 35.0) -> List[DiagnosticRow]:
    """Canlı bağlantıyı adım adım sınar: katalog → pencere okuma → hava → FIRMS anahtarı.

    Her adım ayrı yakalanır; biri başarısız olsa diğerleri yine denenir.

    Args:
        source: Sınanacak kaynak.
        probe_lat: Sınama noktası enlemi (varsayılan: Türkiye ortası).
        probe_lon: Sınama noktası boylamı.
    """
    rows: List[DiagnosticRow] = []
    today = source._today or dt.datetime.now(dt.timezone.utc).date()
    region = an.region_from_center("Sınama", probe_lat, probe_lon, 0.5)

    def step(name: str, fn: Callable[[], str]) -> Optional[str]:
        t0 = time.perf_counter()
        try:
            detail = fn()
            rows.append(DiagnosticRow(name, True, detail, int((time.perf_counter() - t0) * 1000)))
            return detail
        except Exception as exc:  # noqa: BLE001 — tanılama her hatayı göstermek içindir
            msg = str(exc) if isinstance(exc, ds.DataSourceError) else f"{type(exc).__name__}"
            rows.append(DiagnosticRow(name, False, msg[:240], int((time.perf_counter() - t0) * 1000)))
            return None

    found: List[an.SceneMeta] = []

    def cat() -> str:
        metas = source.catalog(region, today - dt.timedelta(days=45), today)
        found.extend(m for m in metas if math.isfinite(m.cloud_pct))
        return f"{len(metas)} sahne (koleksiyon: {source.stac.collection_used})"

    step("Sentinel-2 kataloğu (Earth Search STAC)", cat)

    def win() -> str:
        if not found:
            raise LiveSourceError("Önceki adım sahne bulamadı; pencere okuma sınanamadı.")
        meta = min(found, key=lambda m: m.cloud_pct)
        grid = an.make_grid(region.bbox, 24)
        data = source.load(meta, region, grid, ("red", "nir"))
        red = data.bands["red"]
        ok = float(np.isfinite(red).mean())
        med = float(np.nanmedian(red)) if np.isfinite(red).any() else math.nan
        return f"{meta.scene_id}: {grid.rows}×{grid.cols} piksel okundu, geçerli %{100 * ok:.0f}, kırmızı medyan yansıtım {med:.3f}"

    step("COG pencere okuma (rasterio)", win)

    def wx() -> str:
        df = source.weather(region, today - dt.timedelta(days=20), today - dt.timedelta(days=10))
        return f"{len(df)} gün, toplam yağış {float(df['yagis_mm'].sum()):.0f} mm"

    step("Hava durumu (Open-Meteo / ERA5)", wx)

    def fm() -> str:
        if not source.creds.has_firms:
            raise LiveSourceError("FIRMS_MAP_KEY tanımlı değil (isteğe bağlı; yangın noktası için gerekir).")
        df = source.hotspots(region, today - dt.timedelta(days=3), today)
        return f"{len(df)} nokta (sınama kutusu)"

    step("Yangın noktaları (NASA FIRMS)", fm)
    return rows
