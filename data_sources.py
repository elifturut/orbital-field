"""
data_sources.py — Gerçek veri adaptörleri (Copernicus Sentinel-2 İstatistik API + NASA FIRMS)
============================================================================================

Prototipteki sentetik simülatörlerin (``data_engine.py``) yerine geçecek canlı veri hattının
ilk halkası. İki kaynak desteklenir:

* **Sentinel-2 L2A** — Copernicus Data Space Ecosystem (CDSE) Sentinel Hub *Statistical API*.
  Parsel çokgeni başına NDVI / NBR zaman serisi ve olay öncesi-sonrası dNBR üretir. Görüntü
  indirmez; istatistiği sunucuda hesaplatır, yalnızca özet sayıları getirir.
* **NASA FIRMS** — VIIRS 375 m ve MODIS 1 km aktif yangın noktaları (``area/csv`` uç noktası).
  Çıktı, ``data_engine`` içindeki termal anomali tablosuyla aynı sütun adlarını kullanır
  (``enlem``, ``boylam``, ``frp_mw``, ``parlaklik_k``, ``kaynak``, ``guven``).

Tasarım ilkeleri
----------------
* Ağ erişimi ``HttpSession`` arayüzü (``post``/``get``) arkasındadır; testlerde sahte oturum
  verilebilir, üretimde ``requests.Session`` kullanılır.
* Kimlik bilgileri ortam değişkenlerinden okunur (``CDSE_CLIENT_ID``, ``CDSE_CLIENT_SECRET``,
  ``FIRMS_MAP_KEY``), günlüğe ve hata iletilerine **yazılmaz**.
* Her hata ``DataSourceError`` olarak yükselir; arayüz bunu yakalayıp sentetik veriye dönebilir.

Önemli not: bu modül ağ erişimi olmayan bir ortamda yazıldı. İstek/yanıt biçimleri CDSE ve FIRMS
belgelerine göre kurulup sahte oturumla çevrimdışı test edildi; **canlı uç noktalara karşı henüz
çalıştırılmadı**. CDSE İstatistik API'sinin tam yolu (``CDSE_STATISTICS_URL``) belgelerden tek başına
doğrulanamadı; ilk canlı denemede 404 alınırsa yalnızca bu sabit değiştirilmelidir.

Örnek::

    from data_sources import CdseAuth, CdseStatisticsClient, load_credentials
    cred = load_credentials()
    client = CdseStatisticsClient(CdseAuth(cred.cdse_client_id, cred.cdse_client_secret))
    seri = client.series(parsel_geojson_geometry, dt.date(2026, 5, 1), dt.date(2026, 7, 31))

Geliştirici: Elif Turut
"""

from __future__ import annotations

import datetime as dt
import io
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "CDSE_STATISTICS_URL", "CDSE_TOKEN_URL", "FIRMS_BASE_URL", "FIRMS_SOURCES", "PIPELINE",
    "CdseAuth", "CdseStatisticsClient", "Credentials", "DataSourceError", "FirmsClient",
    "HttpSession", "PipelineSource", "build_firms_url", "build_statistics_payload",
    "load_credentials", "parse_firms_csv", "parse_statistics_response",
]

CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_STATISTICS_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov"
EPSG4326_URI = "http://www.opengis.net/def/crs/EPSG/0/4326"

#: FIRMS ``SOURCE`` değeri → tabloya yazılan kaynak etiketi (``data_engine`` ile aynı).
FIRMS_SOURCES: Dict[str, str] = {
    "VIIRS_SNPP_NRT": "VIIRS-375m", "VIIRS_NOAA20_NRT": "VIIRS-375m", "VIIRS_NOAA21_NRT": "VIIRS-375m",
    "VIIRS_SNPP_SP": "VIIRS-375m", "VIIRS_NOAA20_SP": "VIIRS-375m", "VIIRS_NOAA21_SP": "VIIRS-375m",
    "MODIS_NRT": "MODIS-1km", "MODIS_SP": "MODIS-1km",
}

#: VIIRS güven sınıfları (l/n/h) için kullanılan sayısal karşılıklar (varsayım; 0-1 ölçeği).
VIIRS_CONFIDENCE: Dict[str, float] = {"l": 0.30, "low": 0.30, "n": 0.70, "nominal": 0.70, "h": 0.95, "high": 0.95}

STATS_COLUMNS: Tuple[str, ...] = (
    "baslangic", "bitis", "ortalama", "minimum", "maksimum", "std",
    "ornek_sayisi", "bos_piksel_sayisi", "gecerli_oran",
)
HOTSPOT_COLUMNS: Tuple[str, ...] = (
    "enlem", "boylam", "frp_mw", "parlaklik_k", "kaynak", "guven", "tarih_utc", "gece_gunduz", "uydu",
)

# --------------------------------------------------------------------------- #
# Evalscript'ler (Sentinel Hub) — bulut/gölge/kar pikselleri SCL ile maskelenir
# --------------------------------------------------------------------------- #
NDVI_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "SCL", "dataMask"] }],
    output: [{ id: "ndvi", bands: 1 }, { id: "dataMask", bands: 1 }]
  };
}
function evaluatePixel(s) {
  var den = s.B08 + s.B04;
  var valid = (den == 0) ? 0 : 1;
  var clear = (s.SCL == 3 || s.SCL == 8 || s.SCL == 9 || s.SCL == 10 || s.SCL == 11) ? 0 : 1;
  return { ndvi: [valid ? (s.B08 - s.B04) / den : 0], dataMask: [s.dataMask * valid * clear] };
}
"""

NBR_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B08", "B12", "SCL", "dataMask"] }],
    output: [{ id: "nbr", bands: 1 }, { id: "dataMask", bands: 1 }]
  };
}
function evaluatePixel(s) {
  var den = s.B08 + s.B12;
  var valid = (den == 0) ? 0 : 1;
  var clear = (s.SCL == 3 || s.SCL == 8 || s.SCL == 9 || s.SCL == 10 || s.SCL == 11) ? 0 : 1;
  return { nbr: [valid ? (s.B08 - s.B12) / den : 0], dataMask: [s.dataMask * valid * clear] };
}
"""

INDEX_EVALSCRIPTS: Dict[str, str] = {"ndvi": NDVI_EVALSCRIPT, "nbr": NBR_EVALSCRIPT}


class DataSourceError(RuntimeError):
    """Bir veri kaynağına erişim ya da yanıtı ayrıştırma hatası (kimlik bilgisi içermez)."""


class HttpSession(Protocol):
    """``requests.Session`` ile uyumlu en küçük arayüz (testlerde sahte oturum verilebilir)."""

    def post(self, url: str, **kwargs: Any) -> Any:
        """POST isteği gönderir."""

    def get(self, url: str, **kwargs: Any) -> Any:
        """GET isteği gönderir."""


# --------------------------------------------------------------------------- #
# Ortak yardımcılar
# --------------------------------------------------------------------------- #
def _default_session() -> HttpSession:
    """``requests.Session`` döndürür; paket yoksa anlaşılır bir hata verir.

    Returns:
        Yeni HTTP oturumu.

    Raises:
        DataSourceError: ``requests`` kurulu değilse.
    """
    try:
        import requests
    except ImportError:  # pragma: no cover - ortama bağlı
        raise DataSourceError("Canlı veri için 'requests' paketi gerekli: pip install requests") from None
    return requests.Session()


def _redact(text: str, secrets: Sequence[str]) -> str:
    """Metindeki gizli değerleri maskeler.

    Args:
        text: Maskelenecek metin.
        secrets: Metinde geçmemesi gereken değerler.

    Returns:
        Gizli değerleri ``***`` ile değiştirilmiş metin.
    """
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


def _call(method: Callable[..., Any], url: str, what: str, secrets: Sequence[str], **kwargs: Any) -> Any:
    """HTTP çağrısı yapar, bağlantı/HTTP hatalarını kimlik bilgisi sızdırmadan ``DataSourceError`` yapar.

    Args:
        method: ``session.post`` ya da ``session.get``.
        url: Uç nokta adresi.
        what: Hata iletisinde görünecek işlem adı.
        secrets: İletilerden maskelenecek değerler.
        **kwargs: ``method``'a aktarılan istek argümanları.

    Returns:
        2xx durum kodlu yanıt nesnesi.

    Raises:
        DataSourceError: Bağlantı kurulamazsa ya da durum kodu 2xx değilse.
    """
    try:
        resp = method(url, **kwargs)
    except Exception as exc:  # noqa: BLE001 — istisna iletisi URL (ve anahtar) taşıyabilir; yalnızca türü bildirilir
        raise DataSourceError(f"{what}: bağlantı kurulamadı ({type(exc).__name__})") from None
    status = int(getattr(resp, "status_code", 0))
    if 200 <= status < 300:
        return resp
    if status == 429:
        raise DataSourceError(f"{what}: istek sınırı aşıldı (HTTP 429); bir süre bekleyip yeniden deneyin.")
    body = _redact(str(getattr(resp, "text", ""))[:200], secrets)
    raise DataSourceError(f"{what}: HTTP {status} — {body}")


def _json(resp: Any, what: str) -> Mapping[str, Any]:
    """Yanıt gövdesini JSON olarak okur.

    Args:
        resp: Yanıt nesnesi.
        what: Hata iletisinde görünecek işlem adı.

    Returns:
        Ayrıştırılmış JSON sözlüğü.

    Raises:
        DataSourceError: Gövde JSON değilse ya da sözlük değilse.
    """
    try:
        data = resp.json()
    except ValueError:
        raise DataSourceError(f"{what}: yanıt JSON değil") from None
    if not isinstance(data, Mapping):
        raise DataSourceError(f"{what}: beklenmeyen yanıt biçimi")
    return data


def _num(value: Any) -> float:
    """Sayı, sayı biçimli metin ya da ``"NaN"``/``None`` değerini ``float`` yapar.

    Args:
        value: Ham değer.

    Returns:
        Ondalık sayı; çevrilemezse ``nan``.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


@dataclass(frozen=True)
class Credentials:
    """Canlı veri kaynakları için kimlik bilgileri (``repr`` içinde gizli değer göstermez).

    Attributes:
        cdse_client_id: CDSE OAuth istemci kimliği.
        cdse_client_secret: CDSE OAuth istemci sırrı.
        firms_map_key: NASA FIRMS MAP_KEY.
    """

    cdse_client_id: str = ""
    cdse_client_secret: str = field(default="", repr=False)
    firms_map_key: str = field(default="", repr=False)

    @property
    def has_cdse(self) -> bool:
        """CDSE kimlik bilgileri tam mı?"""
        return bool(self.cdse_client_id and self.cdse_client_secret)

    @property
    def has_firms(self) -> bool:
        """FIRMS anahtarı var mı?"""
        return bool(self.firms_map_key)


def load_credentials(env: Optional[Mapping[str, str]] = None) -> Credentials:
    """Kimlik bilgilerini ortam değişkenlerinden okur.

    Args:
        env: Okunacak sözlük; ``None`` ise ``os.environ``.

    Returns:
        ``Credentials`` (eksik olanlar boş dizge).
    """
    src = os.environ if env is None else env
    return Credentials(
        cdse_client_id=str(src.get("CDSE_CLIENT_ID", "")).strip(),
        cdse_client_secret=str(src.get("CDSE_CLIENT_SECRET", "")).strip(),
        firms_map_key=str(src.get("FIRMS_MAP_KEY", "")).strip(),
    )


# --------------------------------------------------------------------------- #
# Sentinel-2 — CDSE Sentinel Hub İstatistik API
# --------------------------------------------------------------------------- #
@dataclass
class CdseAuth:
    """CDSE OAuth2 *client credentials* belirteci yöneticisi (belirteci önbelleğe alır).

    Belirteç süresi dolana kadar yeniden kullanılır; CDSE belirteç isteklerini hız sınırına tabi
    tuttuğu için her istekte yeni belirteç alınmaz.

    Attributes:
        client_id: OAuth istemci kimliği.
        client_secret: OAuth istemci sırrı.
        session: HTTP oturumu (``None`` ise ``requests.Session``).
        token_url: Belirteç uç noktası.
        clock: Saniye cinsinden tekdüze saat (testlerde sahtesi verilir).
    """

    client_id: str
    client_secret: str = field(repr=False)
    session: Optional[HttpSession] = None
    token_url: str = CDSE_TOKEN_URL
    clock: Callable[[], float] = time.monotonic
    _token: str = field(default="", init=False, repr=False)
    _expires_at: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Kimlik bilgilerini doğrular ve gerekirse oturum oluşturur."""
        if not self.client_id or not self.client_secret:
            raise DataSourceError("CDSE kimlik bilgileri eksik (CDSE_CLIENT_ID / CDSE_CLIENT_SECRET).")
        if self.session is None:
            self.session = _default_session()

    def token(self) -> str:
        """Geçerli erişim belirtecini döndürür; süresi dolmuşsa yenisini alır.

        Returns:
            ``Bearer`` belirteci.

        Raises:
            DataSourceError: Kimlik doğrulama başarısız olursa.
        """
        if self._token and self.clock() < self._expires_at - 30.0:
            return self._token
        assert self.session is not None  # __post_init__ garanti eder
        resp = _call(
            self.session.post, self.token_url, "CDSE kimlik doğrulama", (self.client_secret,),
            data={"grant_type": "client_credentials", "client_id": self.client_id,
                  "client_secret": self.client_secret},
            headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=30,
        )
        body = _json(resp, "CDSE kimlik doğrulama")
        token = str(body.get("access_token", ""))
        if not token:
            raise DataSourceError("CDSE kimlik doğrulama: yanıtta access_token yok")
        ttl = _num(body.get("expires_in", 300.0))
        self._token = token
        self._expires_at = self.clock() + (ttl if math.isfinite(ttl) and ttl > 0 else 300.0)
        return token


def _flatten_coords(node: Any) -> List[Tuple[float, float]]:
    """İç içe GeoJSON koordinat listesini ``(boylam, enlem)`` çiftlerine açar.

    Args:
        node: ``coordinates`` değeri.

    Returns:
        Tüm köşe noktaları.
    """
    if isinstance(node, (list, tuple)) and node and isinstance(node[0], (int, float)):
        return [(float(node[0]), float(node[1]))]
    out: List[Tuple[float, float]] = []
    if isinstance(node, (list, tuple)):
        for child in node:
            out.extend(_flatten_coords(child))
    return out


def _validate_geometry(geometry: Mapping[str, Any]) -> List[Tuple[float, float]]:
    """Çokgen geometrisini doğrular ve köşe noktalarını döndürür.

    Args:
        geometry: WGS84 (boylam, enlem) GeoJSON ``Polygon`` ya da ``MultiPolygon``.

    Returns:
        Köşe noktaları.

    Raises:
        DataSourceError: Geometri türü/koordinatlar geçersizse.
    """
    if geometry.get("type") not in ("Polygon", "MultiPolygon"):
        raise DataSourceError("Geometri Polygon ya da MultiPolygon olmalı.")
    pts = _flatten_coords(geometry.get("coordinates"))
    if len(pts) < 4:
        raise DataSourceError("Geometri en az 4 köşe noktası içermeli.")
    if any(not (-180.0 <= x <= 180.0 and -90.0 <= y <= 90.0) for x, y in pts):
        raise DataSourceError("Koordinatlar (boylam, enlem) derece cinsinden olmalı.")
    return pts


def build_statistics_payload(
    geometry: Mapping[str, Any], start: dt.date, end: dt.date, index: str = "ndvi",
    interval_days: int = 5, resolution_m: float = 10.0, max_cloud_pct: Optional[float] = 60.0,
) -> Dict[str, Any]:
    """Sentinel Hub İstatistik API istek gövdesini kurar.

    Zaman aralığı ``interval_days`` uzunluğunda **tam** dilimlere bölünür; sona kalan eksik dilim
    API tarafından atlanacağı için ``to`` tam dilim sınırına çekilir.

    Args:
        geometry: WGS84 GeoJSON çokgeni.
        start: İlk gün (dahil).
        end: Son gün (dahil).
        index: ``"ndvi"`` ya da ``"nbr"``.
        interval_days: Toplama dilimi uzunluğu (gün, ≥ 1).
        resolution_m: Piksel boyu (metre); derece cinsine çevrilip ``resx``/``resy`` olur.
        max_cloud_pct: Sahne bulut üst sınırı (%), ``None`` ise filtre yok.

    Returns:
        JSON'a çevrilebilir istek gövdesi.

    Raises:
        DataSourceError: Girdiler geçersizse.
    """
    if index not in INDEX_EVALSCRIPTS:
        raise DataSourceError(f"Bilinmeyen indeks: {index!r} (geçerli: {sorted(INDEX_EVALSCRIPTS)})")
    if end < start:
        raise DataSourceError("Bitiş tarihi başlangıçtan önce olamaz.")
    if interval_days < 1 or resolution_m <= 0:
        raise DataSourceError("interval_days ≥ 1 ve resolution_m > 0 olmalı.")
    pts = _validate_geometry(geometry)
    lat = float(np.mean([p[1] for p in pts]))
    total_days = (end - start).days + 1
    step = min(interval_days, total_days)
    n_slices = max(1, total_days // step)
    to = start + dt.timedelta(days=n_slices * step)
    data_filter: Dict[str, Any] = {"mosaickingOrder": "leastCC"}
    if max_cloud_pct is not None:
        data_filter["maxCloudCoverage"] = float(max_cloud_pct)
    return {
        "input": {
            "bounds": {"geometry": dict(geometry), "properties": {"crs": EPSG4326_URI}},
            "data": [{"type": "sentinel-2-l2a", "dataFilter": data_filter}],
        },
        "aggregation": {
            "timeRange": {"from": f"{start:%Y-%m-%d}T00:00:00Z", "to": f"{to:%Y-%m-%d}T00:00:00Z"},
            "aggregationInterval": {"of": f"P{step}D"},
            "evalscript": INDEX_EVALSCRIPTS[index],
            "resx": resolution_m / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6)),
            "resy": resolution_m / 110_574.0,
        },
    }


def parse_statistics_response(payload: Mapping[str, Any], output_id: str) -> pd.DataFrame:
    """İstatistik API yanıtını dilim başına bir satırlık tabloya çevirir.

    Veri olmayan dilimlerde API ``"NaN"`` metni döndürür; bunlar ``nan`` olur. ``error`` taşıyan
    dilimler ``nan`` satırı olarak korunur ki zaman ekseni bozulmasın.

    Args:
        payload: Yanıt JSON'ı.
        output_id: Evalscript çıktı kimliği (``"ndvi"`` / ``"nbr"``).

    Returns:
        ``STATS_COLUMNS`` sütunlu tablo (boş olabilir).
    """
    rows: List[Dict[str, Any]] = []
    for item in payload.get("data", []) or []:
        interval = item.get("interval", {}) or {}
        out = (item.get("outputs", {}) or {}).get(output_id) or {}
        bands = out.get("bands", {}) or {}
        band = bands.get("B0") or (next(iter(bands.values())) if bands else {}) or {}
        stats = band.get("stats", {}) or {}
        sample = _num(stats.get("sampleCount"))
        nodata = _num(stats.get("noDataCount"))
        valid = (1.0 - nodata / sample) if (math.isfinite(sample) and sample > 0 and math.isfinite(nodata)) else float("nan")
        rows.append({
            "baslangic": interval.get("from"), "bitis": interval.get("to"),
            "ortalama": _num(stats.get("mean")), "minimum": _num(stats.get("min")),
            "maksimum": _num(stats.get("max")), "std": _num(stats.get("stDev")),
            "ornek_sayisi": sample, "bos_piksel_sayisi": nodata, "gecerli_oran": valid,
        })
    frame = pd.DataFrame(rows, columns=list(STATS_COLUMNS))
    for col in ("baslangic", "bitis"):
        frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
    return frame


class CdseStatisticsClient:
    """Parsel çokgeni başına Sentinel-2 indeks istatistiklerini CDSE'den alan istemci."""

    def __init__(self, auth: CdseAuth, statistics_url: str = CDSE_STATISTICS_URL,
                 session: Optional[HttpSession] = None) -> None:
        """İstemciyi kurar.

        Args:
            auth: Belirteç yöneticisi.
            statistics_url: İstatistik API adresi.
            session: HTTP oturumu; ``None`` ise ``auth.session`` kullanılır.
        """
        self.auth = auth
        self.statistics_url = statistics_url
        self.session: HttpSession = session or auth.session or _default_session()

    def series(self, geometry: Mapping[str, Any], start: dt.date, end: dt.date, index: str = "ndvi",
               interval_days: int = 5, resolution_m: float = 10.0,
               max_cloud_pct: Optional[float] = 60.0) -> pd.DataFrame:
        """Bir parsel için indeks zaman serisini getirir.

        Args:
            geometry: WGS84 GeoJSON çokgeni.
            start: İlk gün (dahil).
            end: Son gün (dahil).
            index: ``"ndvi"`` ya da ``"nbr"``.
            interval_days: Dilim uzunluğu (gün).
            resolution_m: Piksel boyu (metre).
            max_cloud_pct: Sahne bulut üst sınırı (%).

        Returns:
            ``STATS_COLUMNS`` + ``indeks`` sütunlu tablo.

        Raises:
            DataSourceError: Erişim ya da ayrıştırma hatasında.
        """
        body = build_statistics_payload(geometry, start, end, index, interval_days, resolution_m, max_cloud_pct)
        resp = _call(
            self.session.post, self.statistics_url, "CDSE istatistik isteği", (self.auth.client_secret,),
            json=body, timeout=120,
            headers={"Authorization": f"Bearer {self.auth.token()}", "Accept": "application/json"},
        )
        frame = parse_statistics_response(_json(resp, "CDSE istatistik isteği"), index)
        frame["indeks"] = index
        return frame

    def dnbr(self, geometry: Mapping[str, Any], pre: Tuple[dt.date, dt.date],
             post: Tuple[dt.date, dt.date]) -> float:
        """Olay öncesi ve sonrası pencerelerin parsel-ortalama NBR farkını (dNBR) hesaplar.

        Her pencere tek dilim olarak toplanır; ``leastCC`` mozaiği bulutsuz pikselleri seçer.
        Değer, piksel bazlı dNBR'nin değil **parsel ortalaması farkının** yaklaşığıdır.

        Args:
            geometry: WGS84 GeoJSON çokgeni.
            pre: Olay öncesi ``(ilk, son)`` gün.
            post: Olay sonrası ``(ilk, son)`` gün.

        Returns:
            ``NBR_önce − NBR_sonra``.

        Raises:
            DataSourceError: Pencerelerden birinde bulutsuz gözlem yoksa ya da erişim hatasında.
        """
        means: List[float] = []
        for label, (a, b) in (("öncesi", pre), ("sonrası", post)):
            window = (b - a).days + 1
            frame = self.series(geometry, a, b, "nbr", interval_days=max(window, 1))
            value = float(frame["ortalama"].dropna().iloc[0]) if frame["ortalama"].notna().any() else float("nan")
            if not math.isfinite(value):
                raise DataSourceError(f"dNBR: olay {label} penceresinde bulutsuz gözlem bulunamadı.")
            means.append(value)
        return means[0] - means[1]


# --------------------------------------------------------------------------- #
# NASA FIRMS — aktif yangın noktaları
# --------------------------------------------------------------------------- #
def _fmt(x: float) -> str:
    """Koordinatı gereksiz sıfırlar olmadan biçimler.

    Args:
        x: Derece değeri.

    Returns:
        Metin.
    """
    return f"{x:.5f}".rstrip("0").rstrip(".")


def build_firms_url(map_key: str, source: str, bbox: Tuple[float, float, float, float], day_range: int = 1,
                    date: Optional[dt.date] = None, base_url: str = FIRMS_BASE_URL) -> str:
    """FIRMS ``area/csv`` isteği için adres kurar.

    Args:
        map_key: FIRMS MAP_KEY (ücretsiz).
        source: ``FIRMS_SOURCES`` anahtarlarından biri.
        bbox: ``(batı, güney, doğu, kuzey)`` derece.
        day_range: Gün sayısı (1–5).
        date: Başlangıç günü; ``None`` ise en güncel veri.
        base_url: Sunucu adresi.

    Returns:
        Tam adres (MAP_KEY içerir — günlüğe yazmayın).

    Raises:
        DataSourceError: Girdiler geçersizse.
    """
    if not map_key:
        raise DataSourceError("FIRMS MAP_KEY eksik (FIRMS_MAP_KEY).")
    if source not in FIRMS_SOURCES:
        raise DataSourceError(f"Bilinmeyen FIRMS kaynağı: {source!r} (geçerli: {sorted(FIRMS_SOURCES)})")
    if not 1 <= int(day_range) <= 5:
        raise DataSourceError("day_range 1 ile 5 arasında olmalı.")
    w, s, e, n = bbox
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise DataSourceError("bbox (batı, güney, doğu, kuzey) geçersiz.")
    url = f"{base_url}/api/area/csv/{map_key}/{source}/{_fmt(w)},{_fmt(s)},{_fmt(e)},{_fmt(n)}/{int(day_range)}"
    return f"{url}/{date:%Y-%m-%d}" if date else url


def _confidence(value: Any) -> float:
    """FIRMS güven değerini 0-1 ölçeğine çevirir.

    Args:
        value: VIIRS için ``l/n/h``; MODIS için 0-100 sayı.

    Returns:
        0-1 aralığında güven; bilinmiyorsa 0,5.
    """
    text = str(value).strip().lower()
    if text in VIIRS_CONFIDENCE:
        return VIIRS_CONFIDENCE[text]
    num = _num(text)
    return float(np.clip(num / 100.0, 0.0, 1.0)) if math.isfinite(num) else 0.5


def parse_firms_csv(text: str, source: str) -> pd.DataFrame:
    """FIRMS CSV yanıtını ``data_engine`` termal anomali şemasına çevirir.

    Args:
        text: CSV metni.
        source: ``FIRMS_SOURCES`` anahtarı (kaynak etiketini belirler).

    Returns:
        ``HOTSPOT_COLUMNS`` sütunlu tablo, zamana göre sıralı (boş olabilir).

    Raises:
        DataSourceError: Yanıt CSV değilse (ör. geçersiz anahtar iletisi) ya da zorunlu sütun yoksa.
    """
    label = FIRMS_SOURCES.get(source)
    if label is None:
        raise DataSourceError(f"Bilinmeyen FIRMS kaynağı: {source!r}")
    if not text.strip():
        return pd.DataFrame(columns=list(HOTSPOT_COLUMNS))
    if "latitude" not in text.lstrip().splitlines()[0].lower():
        raise DataSourceError(f"FIRMS yanıtı CSV değil: {text.strip()[:120]!r}")
    raw = pd.read_csv(io.StringIO(text))
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing = [c for c in ("latitude", "longitude", "acq_date", "acq_time") if c not in raw.columns]
    if missing:
        raise DataSourceError(f"FIRMS CSV zorunlu sütunları eksik: {missing}")
    hhmm = raw["acq_time"].fillna(0).astype(int).astype(str).str.zfill(4)
    stamp = pd.to_datetime(raw["acq_date"].astype(str) + " " + hhmm.str[:2] + ":" + hhmm.str[2:], utc=True, errors="coerce")
    bright_col = "bright_ti4" if "bright_ti4" in raw.columns else ("brightness" if "brightness" in raw.columns else None)
    out = pd.DataFrame({
        "enlem": pd.to_numeric(raw["latitude"], errors="coerce"),
        "boylam": pd.to_numeric(raw["longitude"], errors="coerce"),
        "frp_mw": pd.to_numeric(raw["frp"], errors="coerce") if "frp" in raw.columns else np.nan,
        "parlaklik_k": pd.to_numeric(raw[bright_col], errors="coerce") if bright_col else np.nan,
        "kaynak": label,
        "guven": raw["confidence"].map(_confidence) if "confidence" in raw.columns else 0.5,
        "tarih_utc": stamp,
        "gece_gunduz": raw["daynight"].astype(str).str.upper() if "daynight" in raw.columns else "",
        "uydu": raw["satellite"].astype(str) if "satellite" in raw.columns else "",
    })
    out = out.dropna(subset=["enlem", "boylam"]).sort_values("tarih_utc").reset_index(drop=True)
    return out[list(HOTSPOT_COLUMNS)]


class FirmsClient:
    """NASA FIRMS aktif yangın noktası istemcisi (``area/csv`` uç noktası)."""

    def __init__(self, map_key: str, session: Optional[HttpSession] = None, base_url: str = FIRMS_BASE_URL) -> None:
        """İstemciyi kurar.

        Args:
            map_key: FIRMS MAP_KEY.
            session: HTTP oturumu (``None`` ise ``requests.Session``).
            base_url: Sunucu adresi.

        Raises:
            DataSourceError: ``map_key`` boşsa.
        """
        if not map_key:
            raise DataSourceError("FIRMS MAP_KEY eksik (FIRMS_MAP_KEY).")
        self._map_key = map_key
        self.base_url = base_url
        self.session: HttpSession = session or _default_session()

    def hotspots(self, bbox: Tuple[float, float, float, float], source: str = "VIIRS_SNPP_NRT",
                 day_range: int = 1, date: Optional[dt.date] = None) -> pd.DataFrame:
        """Sınır kutusundaki aktif yangın noktalarını getirir.

        FIRMS, 10 dakikalık pencerede en fazla 5000 işleme izin verir; çok günlü sorgular birden
        fazla işlem sayılabilir. Bu yüzden sınır kutusunu ve gün aralığını dar tutun.

        Args:
            bbox: ``(batı, güney, doğu, kuzey)`` derece.
            source: ``FIRMS_SOURCES`` anahtarı.
            day_range: Gün sayısı (1–5).
            date: Başlangıç günü; ``None`` ise en güncel.

        Returns:
            ``HOTSPOT_COLUMNS`` sütunlu tablo.

        Raises:
            DataSourceError: Erişim ya da ayrıştırma hatasında (iletiler MAP_KEY içermez).
        """
        url = build_firms_url(self._map_key, source, bbox, day_range, date, self.base_url)
        resp = _call(self.session.get, url, "FIRMS isteği", (self._map_key,), timeout=60)
        return parse_firms_csv(str(getattr(resp, "text", "")), source)


# --------------------------------------------------------------------------- #
# Veri hattı özeti (arayüzdeki "Genel Bakış" sekmesi bunu gösterir)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PipelineSource:
    """Veri hattındaki bir kaynağın durumunu tarif eder.

    Attributes:
        name: Kaynak adı.
        product: Ürün ve çözünürlük.
        use: Platformda neyi besler.
        prototype: Bu prototipteki durum.
        adapter: Canlı bağlantı adaptörünün durumu.
        access: Erişim koşulu.
    """

    name: str
    product: str
    use: str
    prototype: str
    adapter: str
    access: str


PIPELINE: Tuple[PipelineSource, ...] = (
    PipelineSource("Sentinel-2 L2A (Earth Search STAC + COG)", "10 m; B04, B08, B12, SCL; bulut filtresi", "Analiz iş akışı: önce/sonra görüntüsü, NDVI, NBR, dNBR, değişim haritası",
                   "Sentetik (varsayılan); canlı seçenek var", "Yazıldı, sahte yanıtlarla çevrimdışı test edildi; canlı test bekliyor (iş akışında 'Canlı bağlantıyı sına')", "Anahtarsız genel API; rasterio gerekir"),
    PipelineSource("Sentinel-2 L2A (Copernicus CDSE)", "10 m; B04, B08, B12, SCL", "NDVI, NBR, dNBR; parsel hasar ve bitki sağlığı (istatistik API)",
                   "Sentetik", "Yazıldı, çevrimdışı test edildi; canlı test bekliyor", "Hesap + OAuth istemcisi; genel kullanıcı kotası ayda 10.000 istek"),
    PipelineSource("NASA FIRMS", "VIIRS 375 m, MODIS 1 km aktif yangın", "Termal anomali, FRP; olay tarihine ve koordinata göre eşleşen yangın noktaları",
                   "Sentetik", "Yazıldı, çevrimdışı test edildi; canlı test bekliyor", "Ücretsiz MAP_KEY"),
    PipelineSource("Parsel sınırları", "Haritadan çizim, koordinat, GeoJSON metni", "Sigorta ve tarım parselleri",
                   "Sentetik ızgara", "Analiz iş akışında kullanılabilir; toplu dosya yükleme planlı", "Müşteri çizer ya da yapıştırır"),
    PipelineSource("Poliçe ve hasar kayıtları", "Müşterinin CSV dosyası", "Karar tablosu, kural testi",
                   "Sentetik poliçeler", "Planlı", "Müşteri yükler"),
    PipelineSource("Meteoroloji (Open-Meteo, ERA5 tabanlı arşiv)", "Günlük yağış, sıcaklık, rüzgâr, ET0", "Sel/kuraklık/yangın olayını hava verisiyle doğrulama (istasyon ölçümü değildir)",
                   "Sentetik", "Yazıldı, sahte yanıtlarla çevrimdışı test edildi; canlı test bekliyor", "Anahtarsız genel API"),
)
