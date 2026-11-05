"""Harita altlıkları: koyu CARTO, Esri uydu, günlük NASA VIIRS (canlıya yakın) ve Sentinel-2 mozaik.

Bu modül ``streamlit`` ya da ``folium`` içe aktarmaz; adres kurma, tarih sınırlama ve etiket mantığı ağ
bağlantısı olmadan test edilebilir. Karoları tarayıcı kendisi çeker; sunucu hiçbir uydu isteği yapmaz.

"Canlı" ifadesi burada şu anlama gelir: NASA GIBS, VIIRS günlük bileşik görüntüsünü çekimden yaklaşık
üç saat sonra yayımlar. Bu bir video akışı ya da dakikalık görüntü değildir; çözünürlüğü yaklaşık 250 m'dir.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "Basemap", "BasemapChoice", "BASEMAPS", "LABELS", "DEFAULT_KEY", "GIBS_FIRST_DAY",
    "utc_today", "default_date", "coerce_date", "by_label", "by_key", "make_choice", "tile_url",
    "layer_name", "attribution", "sidebar_note",
    "FIRE_LABEL", "FIRE_LAYERS", "FIRE_NOTE", "fire_wms_options",
]

#: NASA GIBS WMTS (EPSG:3857). Şablon: .../{katman}/default/{tarih}/{matris}/{z}/{y}/{x}.{uzantı}
GIBS_WMTS = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best"
VIIRS_LAYER = "VIIRS_SNPP_CorrectedReflectance_TrueColor"
VIIRS_MATRIX = "GoogleMapsCompatible_Level9"
#: Seçilebilecek en eski gün; kaynağın gerçek başlangıcından bilerek daha temkinli tutuldu.
GIBS_FIRST_DAY = dt.date(2016, 1, 1)
DATE_TOKEN = "{date}"
DEFAULT_KEY = "dark"


@dataclass(frozen=True)
class Basemap:
    """Tek bir altlık katmanının tanımı.

    Attributes:
        key: Kararlı makine adı.
        label: Arayüzde görünen ad (radyo düğmesi ve harita katman denetimi).
        url: Karo adresi şablonu; tarihli katmanlarda ``{date}`` içerir.
        attribution: Harita köşesinde gösterilen kaynak notu.
        subdomains: Adreste ``{s}`` varsa kullanılacak alt alan adları.
        max_zoom: Haritanın izin verdiği en yüksek yakınlaştırma.
        max_native_zoom: Kaynağın gerçekten karo sunduğu en yüksek düzey; ötesinde karolar büyütülür.
        dated: ``True`` ise katman gün gün değişir.
        first_day: Tarihli katmanda seçilebilecek en eski gün.
        live: ``True`` ise yaklaşık gerçek zamanlı yayımlanan bir kaynaktır.
        note: Kenar çubuğunda gösterilen kısa açıklama.
    """

    key: str
    label: str
    url: str
    attribution: str
    subdomains: str = "abc"
    max_zoom: int = 19
    max_native_zoom: Optional[int] = None
    dated: bool = False
    first_day: Optional[dt.date] = None
    live: bool = False
    note: str = ""


BASEMAPS: Tuple[Basemap, ...] = (
    Basemap(
        key="dark",
        label="Koyu harita (CARTO)",
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attribution="© OpenStreetMap katkıda bulunanlar © CARTO",
        subdomains="abcd",
        note="Koyu vektör harita; uydu görüntüsü içermez.",
    ),
    Basemap(
        key="esri",
        label="Uydu altlığı (Esri)",
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attribution="Tiles © Esri",
        note="Yüksek çözünürlüklü sabit uydu mozaiği; canlı değildir.",
    ),
    Basemap(
        key="viirs",
        label="Canlı uydu · NASA VIIRS (günlük)",
        url=f"{GIBS_WMTS}/{VIIRS_LAYER}/default/{DATE_TOKEN}/{VIIRS_MATRIX}/{{z}}/{{y}}/{{x}}.jpg",
        attribution="NASA EOSDIS GIBS · VIIRS Suomi NPP",
        max_native_zoom=9,
        dated=True,
        first_day=GIBS_FIRST_DAY,
        live=True,
        note=("NASA GIBS günlük VIIRS görüntüsü; çekimden yaklaşık 3 saat sonra yayımlanır. "
              "Çözünürlük yaklaşık 250 m: il ve bölge ölçeği içindir, parsel ölçeğinde bulanık görünür. "
              "Bugünün görüntüsü eksik olabilir."),
    ),
    Basemap(
        key="eox",
        label="Sentinel-2 mozaik (EOX · yıllık)",
        url="https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{z}/{y}/{x}.jpg",
        attribution=("Sentinel-2 cloudless 2024 · s2maps.eu · EOX IT Services GmbH "
                     "(Copernicus Sentinel verisinin değiştirilmiş hali)"),
        max_native_zoom=14,
        note="10 m çözünürlüklü bulutsuz yıllık mozaik; canlı değildir. Ticari yayında EOX lisansı gerekir.",
    ),
)

LABELS: Tuple[str, ...] = tuple(b.label for b in BASEMAPS)
_BY_KEY: Dict[str, Basemap] = {b.key: b for b in BASEMAPS}
_BY_LABEL: Dict[str, Basemap] = {b.label: b for b in BASEMAPS}


def by_key(key: Any) -> Basemap:
    """Anahtara göre altlığı döndürür; bilinmeyen anahtarda koyu haritaya düşer."""
    return _BY_KEY.get(str(key), _BY_KEY[DEFAULT_KEY])


def by_label(label: Any) -> Basemap:
    """Arayüz etiketine göre altlığı döndürür; eski ya da bilinmeyen etikette koyu haritaya düşer."""
    return _BY_LABEL.get(str(label), _BY_KEY[DEFAULT_KEY])


def utc_today(now: Optional[dt.datetime] = None) -> dt.date:
    """UTC takvim gününü döndürür (GIBS günleri UTC'dir).

    Args:
        now: Test için verilebilir; ``None`` ise şimdiki an.

    Returns:
        UTC günü.
    """
    moment = now if now is not None else dt.datetime.now(dt.timezone.utc)
    if moment.tzinfo is not None:
        moment = moment.astimezone(dt.timezone.utc)
    return moment.date()


def default_date(today: Optional[dt.date] = None) -> dt.date:
    """Varsayılan görüntü günü: dün (UTC). Bugünün bileşiği çoğu zaman eksiktir."""
    return (today or utc_today()) - dt.timedelta(days=1)


def coerce_date(value: Any, today: Optional[dt.date] = None, first_day: dt.date = GIBS_FIRST_DAY) -> dt.date:
    """Herhangi bir tarih benzeri değeri geçerli aralığa oturtur.

    Args:
        value: ``date``, ``datetime``, ISO metni, tek elemanlı liste/demet ya da ``None``.
        today: Üst sınır; ``None`` ise UTC bugünü.
        first_day: Alt sınır.

    Returns:
        ``[first_day, today]`` aralığında bir gün. Okunamayan değer için :func:`default_date`.
    """
    top = today or utc_today()
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    day: Optional[dt.date] = None
    if isinstance(value, dt.datetime):
        day = value.date()
    elif isinstance(value, dt.date):
        day = value
    elif isinstance(value, str):
        try:
            day = dt.date.fromisoformat(value.strip()[:10])
        except ValueError:
            day = None
    if day is None:
        day = default_date(top)
    return min(max(day, first_day), top)


@dataclass(frozen=True)
class BasemapChoice:
    """Kullanıcının seçtiği altlık ve (tarihli katmanlar için) görüntü günü.

    Attributes:
        key: Seçili altlığın anahtarı.
        day: Tarihli katmanlarda kullanılan gün; tarihsiz katmanlarda yok sayılır.
    """

    key: str
    day: dt.date

    @property
    def spec(self) -> Basemap:
        """Seçili altlığın tanımı."""
        return by_key(self.key)

    @property
    def is_satellite(self) -> bool:
        """Koyu vektör harita dışındaki her seçenek uydu görüntüsüdür."""
        return self.key != DEFAULT_KEY


def make_choice(label: Any, day_value: Any = None, today: Optional[dt.date] = None) -> BasemapChoice:
    """Arayüz değerlerinden güvenli bir seçim kurar.

    Args:
        label: Radyo düğmesi etiketi (bilinmiyorsa koyu harita).
        day_value: Tarih alanının değeri (bozuksa dün).
        today: Test için UTC bugünü.

    Returns:
        ``BasemapChoice``.
    """
    spec = by_label(label)
    return BasemapChoice(key=spec.key, day=coerce_date(day_value, today, spec.first_day or GIBS_FIRST_DAY))


def tile_url(spec: Basemap, day: dt.date) -> str:
    """Karo adresini kurar; tarihli katmanda ``{date}`` yer tutucusunu ``YYYY-AA-GG`` ile doldurur.

    ``str.format`` kullanılmaz çünkü adreste ``{z}``, ``{y}``, ``{x}`` Leaflet'e bırakılan yer tutuculardır.

    Raises:
        ValueError: Sonuçta doldurulmamış ``{date}`` ya da eksik ``{z}/{y}/{x}`` kalırsa.
    """
    url = spec.url.replace(DATE_TOKEN, day.isoformat()) if spec.dated else spec.url
    if DATE_TOKEN in url:
        raise ValueError(f"{spec.key}: tarih yer tutucusu doldurulamadı")
    for token in ("{z}", "{x}", "{y}"):
        if token not in url:
            raise ValueError(f"{spec.key}: adreste {token} yok")
    return url


def layer_name(spec: Basemap, day: dt.date) -> str:
    """Harita katman denetiminde görünen ad; tarihli katmanda gün de yazılır."""
    return f"{spec.label} · {day.isoformat()}" if spec.dated else spec.label


def attribution(spec: Basemap, day: dt.date) -> str:
    """Harita köşesindeki kaynak notu; tarihli katmanda görüntü günü eklenir."""
    return f"{spec.attribution} · {day.isoformat()}" if spec.dated else spec.attribution


def sidebar_note(choice: BasemapChoice) -> str:
    """Kenar çubuğunda seçili altlığın altına yazılan kısa açıklama."""
    return choice.spec.note


# --------------------------------------------------------------------------- #
# Gerçek aktif yangın katmanı (NASA GIBS, WMS)
# --------------------------------------------------------------------------- #
#: GIBS WMS uç noktası (EPSG:3857). WMS seçildi çünkü karo matrisi adı gerektirmez ve ``time`` parametresini
#: doğrudan destekler.
GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi"
#: (GIBS katman kimliği, kısa ad). Kimlikler NASA FIRMS'in GIBS rehberindeki tabloda yer alır.
FIRE_LAYERS: Tuple[Tuple[str, str], ...] = (
    ("VIIRS_SNPP_Thermal_Anomalies_375m_All", "Suomi NPP"),
    ("VIIRS_NOAA20_Thermal_Anomalies_375m_All", "NOAA-20"),
)
FIRE_LABEL = "NASA aktif yangınlar (gerçek · VIIRS)"
FIRE_ATTRIBUTION = "NASA EOSDIS GIBS / FIRMS · VIIRS 375 m"
FIRE_NOTE = (
    "Gerçek NASA VIIRS aktif yangın tespitleri (Suomi NPP ve NOAA-20, 375 m). Haritadaki sentetik senaryo "
    "yangınıyla ilgisi yoktur. Seçili günde bu bölgede tespit yoksa katman boş görünür; boş görünüm "
    "yangın olmadığının kanıtı değildir (uydu geçişleri arasında boşluk, bulut ve yayın gecikmesi olur)."
)


def fire_wms_options(day: dt.date) -> Tuple[Dict[str, Any], ...]:
    """Aktif yangın katmanları için ``folium.WmsTileLayer`` argümanlarını kurar.

    Args:
        day: Görüntü günü (UTC); GIBS ``time`` parametresi olarak ``YYYY-AA-GG`` gider.

    Returns:
        Her uydu için bir argüman sözlüğü; ``**`` ile ``WmsTileLayer``'a verilebilir.
    """
    iso = day.isoformat()
    return tuple(
        {
            "url": GIBS_WMS, "layers": layer_id, "fmt": "image/png", "transparent": True, "version": "1.1.1",
            "attr": f"{FIRE_ATTRIBUTION} · {iso}", "name": f"{FIRE_LABEL} · {short} · {iso}", "time": iso,
        }
        for layer_id, short in FIRE_LAYERS
    )
