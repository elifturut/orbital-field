"""
data_engine.py — No-Code Çok Yönlü Uydu ve Saha Analiz Platformu / Sentetik Veri Motoru
=======================================================================================

Bu modül, platformun dört sektör sekmesi (Sigorta, Tarım, Belediye, Afet/Kurtarma)
için gereken tüm verileri **dış API'ye bağlanmadan**, ``numpy`` ile Gauss dağılımlı
ve mekânsal olarak ilintili (korelasyonlu) biçimde üretir.

Tasarım ilkeleri
----------------
* Streamlit / Folium / Plotly'ye bağımlılık yoktur; yalnızca ``numpy`` ve ``pandas``.
  Böylece modül arayüzden bağımsız olarak test edilebilir.
* Üretim deterministiktir: aynı ``seed`` her zaman aynı veriyi verir
  (``numpy.random.default_rng([seed, anahtar...])``).
* Tüm hesaplar vektörize edilmiştir (döngüler yalnızca küçük, kategorik listelerde).
* Fiziksel olarak anlamlı büyüklükler: NDVI/EVI [-1, 1], dNBR (USGS FIREMON sınıfları),
  sıcaklık (°C), parsel verimi (ton/ha), FRP (MW), parlaklık sıcaklığı (K).

Sınıflar
--------
``ParcelGrid``            : GeoJSON'a dönüşebilen parsel çokgenleri.
``InsuranceSimulator``    : Sigorta — dNBR hasar katmanı, risk skoru, otomatik tazminat kararı.
``AgricultureSimulator``  : Tarım — tarla sınırları, NDVI/EVI, fenoloji, su stresi, rekolte.
``MunicipalitySimulator`` : Belediye — T0/T1 arazi örtüsü, kaçak yapılaşma, mahalle istatistikleri.
``RescueSimulator``       : Afet — VIIRS/MODIS benzeri termal anomali, nüfus riski, tahliye rotaları.

Not: Tüm isimler, nüfuslar ve değerler **tamamen sentetiktir**; gerçek kişi, kurum ya da
mülkiyet verisi içermez.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Sabitler ve tip takma adları
# --------------------------------------------------------------------------- #
EARTH_RADIUS_KM: float = 6371.0088
KM_PER_DEG_LAT: float = 111.32
TR_TZ: dt.timezone = dt.timezone(dt.timedelta(hours=3))  # Türkiye: sabit UTC+3

LatLon = Tuple[float, float]

#: USGS FIREMON dNBR sınıf sınırları ve Türkçe etiketleri.
DNBR_EDGES: Tuple[float, ...] = (0.10, 0.27, 0.44, 0.66)
SEVERITY_LABELS: Tuple[str, ...] = (
    "Yanmamış / Çok Düşük",
    "Düşük",
    "Orta-Düşük",
    "Orta-Yüksek",
    "Yüksek",
)


# --------------------------------------------------------------------------- #
# Genel yardımcı fonksiyonlar
# --------------------------------------------------------------------------- #
def make_rng(seed: int, *keys: int) -> np.random.Generator:
    """Seed ve ek anahtarlardan bağımsız, tekrarlanabilir bir üreteç kurar.

    Args:
        seed: Ana tohum değeri.
        *keys: Alt akışları birbirinden ayıran tamsayı anahtarlar.

    Returns:
        ``numpy.random.Generator`` örneği.
    """
    parts = [int(seed) & 0x7FFFFFFF] + [int(k) & 0x7FFFFFFF for k in keys]
    return np.random.default_rng(parts)


def format_tr(value: float, decimals: int = 0) -> str:
    """Sayıyı Türkçe biçimde yazar (binlik ``.``, ondalık ``,``).

    Args:
        value: Biçimlendirilecek sayı.
        decimals: Ondalık basamak sayısı.

    Returns:
        Örn. ``1234567.8`` → ``"1.234.567,8"``; geçersiz değerde ``"—"``.
    """
    if value is None or not np.isfinite(value):
        return "—"
    text = f"{float(value):,.{decimals}f}"
    return text.replace(",", "§").replace(".", ",").replace("§", ".")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sayısal olarak güvenli lojistik fonksiyon.

    Args:
        x: Girdi dizisi veya skaler.

    Returns:
        ``1 / (1 + exp(-x))`` değerleri.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def offset_latlon(
    lat0: float, lon0: float, east_km: np.ndarray, north_km: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Bir merkez noktasından doğu/kuzey kaydırmayla enlem-boylam üretir.

    Args:
        lat0: Merkez enlemi (derece).
        lon0: Merkez boylamı (derece).
        east_km: Doğuya doğru kaydırma (km).
        north_km: Kuzeye doğru kaydırma (km).

    Returns:
        ``(enlem, boylam)`` dizileri.
    """
    east = np.asarray(east_km, dtype=float)
    north = np.asarray(north_km, dtype=float)
    lat = lat0 + north / KM_PER_DEG_LAT
    lon = lon0 + east / (KM_PER_DEG_LAT * math.cos(math.radians(lat0)))
    return lat, lon


def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Büyük daire (haversine) uzaklığını km cinsinden hesaplar (broadcast uyumlu).

    Args:
        lat1: Birinci nokta enlemleri.
        lon1: Birinci nokta boylamları.
        lat2: İkinci nokta enlemleri.
        lon2: İkinci nokta boylamları.

    Returns:
        Uzaklık dizisi (km).
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def gaussian_field(
    shape: Tuple[int, int], correlation_cells: float, rng: np.random.Generator
) -> np.ndarray:
    """Mekânsal olarak ilintili, sıfır ortalamalı, birim varyanslı Gauss alanı üretir.

    Beyaz Gauss gürültüsü frekans uzayında Gauss çekirdeğiyle süzülür (FFT);
    sonuç doğal görünümlü, yumuşak geçişli bir yüzey verir.

    Args:
        shape: ``(satır, sütun)``.
        correlation_cells: Gauss çekirdeğinin standart sapması (hücre).
        rng: Rastgele sayı üreteci.

    Returns:
        ``shape`` boyutunda, ortalaması 0 ve standart sapması 1 olan dizi.
    """
    h, w = shape
    noise = rng.standard_normal((h, w))
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    kernel = np.exp(-2.0 * (np.pi * max(correlation_cells, 1e-6)) ** 2 * (kx**2 + ky**2))
    smooth = np.real(np.fft.ifft2(np.fft.fft2(noise) * kernel))
    std = float(smooth.std())
    return (smooth - smooth.mean()) / std if std > 0 else np.zeros_like(smooth)


def classify_dnbr(dnbr: np.ndarray) -> np.ndarray:
    """dNBR değerlerini USGS FIREMON şiddet sınıfına (0–4) çevirir.

    Args:
        dnbr: dNBR dizisi.

    Returns:
        0 (yanmamış) – 4 (yüksek) arası tamsayı sınıf indeksleri.
    """
    return np.digitize(np.asarray(dnbr, dtype=float), DNBR_EDGES).astype(int)


# --------------------------------------------------------------------------- #
# Parsel geometrisi
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParcelGeometry:
    """Üretilmiş parsel çokgenlerinin vektörize geometrisi.

    Attributes:
        corners_lat: ``(P, 4)`` köşe enlemleri.
        corners_lon: ``(P, 4)`` köşe boylamları.
        centroid_lat: ``(P,)`` ağırlık merkezi enlemi.
        centroid_lon: ``(P,)`` ağırlık merkezi boylamı.
        x_km: ``(P,)`` merkeze göre doğu koordinatı (km).
        y_km: ``(P,)`` merkeze göre kuzey koordinatı (km).
        area_ha: ``(P,)`` parsel alanı (hektar).
        row: ``(P,)`` ızgara satır indeksi.
        col: ``(P,)`` ızgara sütun indeksi.
        rows: Izgara satır sayısı.
        cols: Izgara sütun sayısı.
    """

    corners_lat: np.ndarray
    corners_lon: np.ndarray
    centroid_lat: np.ndarray
    centroid_lon: np.ndarray
    x_km: np.ndarray
    y_km: np.ndarray
    area_ha: np.ndarray
    row: np.ndarray
    col: np.ndarray
    rows: int
    cols: int

    @property
    def count(self) -> int:
        """Parsel sayısını döndürür."""
        return int(self.centroid_lat.size)


@dataclass(frozen=True)
class ParcelGrid:
    """Hafifçe bozulmuş (jitter) dörtgenlerden oluşan parsel ızgarası üreticisi.

    Attributes:
        centre: ``(enlem, boylam)`` ızgara merkezi.
        rows: Satır sayısı.
        cols: Sütun sayısı.
        cell_km: Hücre kenar uzunluğu (km).
        jitter: Köşe düzensizliği (hücre boyutuna oranla Gauss sigması).
        gap: Komşu parseller arasındaki görsel boşluk oranı.
        seed: Tohum değeri.
    """

    centre: LatLon
    rows: int
    cols: int
    cell_km: float
    jitter: float = 0.10
    gap: float = 0.04
    seed: int = 0

    def build(self) -> ParcelGeometry:
        """Parsel köşelerini, merkezlerini ve alanlarını hesaplar.

        Returns:
            ``ParcelGeometry`` nesnesi.
        """
        rng = make_rng(self.seed, 11)
        ii, jj = np.meshgrid(
            np.arange(self.rows + 1), np.arange(self.cols + 1), indexing="ij"
        )
        lim = 0.35 * self.cell_km
        sigma = self.jitter * self.cell_km
        vx = (jj - self.cols / 2.0) * self.cell_km + np.clip(
            rng.normal(0.0, sigma, jj.shape), -lim, lim
        )
        vy = (ii - self.rows / 2.0) * self.cell_km + np.clip(
            rng.normal(0.0, sigma, ii.shape), -lim, lim
        )
        cxs = np.stack([vx[:-1, :-1], vx[:-1, 1:], vx[1:, 1:], vx[1:, :-1]], axis=-1)
        cys = np.stack([vy[:-1, :-1], vy[:-1, 1:], vy[1:, 1:], vy[1:, :-1]], axis=-1)
        cxs, cys = cxs.reshape(-1, 4), cys.reshape(-1, 4)
        mx, my = cxs.mean(axis=1), cys.mean(axis=1)
        shrink = 1.0 - self.gap
        gx = mx[:, None] + (cxs - mx[:, None]) * shrink
        gy = my[:, None] + (cys - my[:, None]) * shrink
        nx, ny = np.roll(gx, -1, axis=1), np.roll(gy, -1, axis=1)
        area_km2 = 0.5 * np.abs((gx * ny - nx * gy).sum(axis=1))
        lat0, lon0 = self.centre
        clat, clon = offset_latlon(lat0, lon0, gx, gy)
        mlat, mlon = offset_latlon(lat0, lon0, mx, my)
        idx = np.arange(self.rows * self.cols)
        return ParcelGeometry(
            corners_lat=clat,
            corners_lon=clon,
            centroid_lat=mlat,
            centroid_lon=mlon,
            x_km=mx,
            y_km=my,
            area_ha=area_km2 * 100.0,
            row=idx // self.cols,
            col=idx % self.cols,
            rows=self.rows,
            cols=self.cols,
        )


def parcels_to_geojson(geometry: ParcelGeometry, props: pd.DataFrame) -> dict:
    """Parsel çokgenlerini öznitelik tablosuyla birlikte GeoJSON'a çevirir.

    Args:
        geometry: ``ParcelGrid.build`` çıktısı.
        props: ``geometry.count`` satırlı öznitelik tablosu (sıra geometriyle aynı).

    Returns:
        GeoJSON ``FeatureCollection`` sözlüğü (koordinatlar ``[boylam, enlem]``).

    Raises:
        ValueError: Öznitelik satır sayısı parsel sayısına eşit değilse.
    """
    if len(props) != geometry.count:
        raise ValueError("Öznitelik tablosu satır sayısı parsel sayısıyla uyuşmuyor.")
    records = json.loads(props.to_json(orient="records", force_ascii=False))
    features: List[dict] = []
    for i, rec in enumerate(records):
        ring = [
            [float(lo), float(la)]
            for la, lo in zip(geometry.corners_lat[i], geometry.corners_lon[i])
        ]
        ring.append(ring[0])
        features.append(
            {
                "type": "Feature",
                "id": i,
                "properties": rec,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def neighbour_mean(values: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Izgara düzenindeki bir vektörün 3x3 komşuluk ortalamasını alır.

    Args:
        values: ``rows * cols`` uzunluklu vektör (satır-öncelikli).
        rows: Izgara satırı.
        cols: Izgara sütunu.

    Returns:
        Aynı uzunlukta, kenarlarda kopyalama (edge) dolgulu komşuluk ortalaması.
    """
    grid = np.asarray(values, dtype=float).reshape(rows, cols)
    pad = np.pad(grid, 1, mode="edge")
    acc = np.zeros_like(grid)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            acc += pad[dy : dy + rows, dx : dx + cols]
    return (acc / 9.0).ravel()


# =========================================================================== #
# 1) SİGORTA — Risk ve Hasar
# =========================================================================== #
ASSET_CATALOG: Dict[str, Tuple[float, float]] = {
    # varlık türü: (hektar başına sigorta bedeli TL, seçilme olasılığı)
    "Zeytinlik": (180_000.0, 0.28),
    "Narenciye Bahçesi": (210_000.0, 0.17),
    "Tahıl Tarlası": (60_000.0, 0.25),
    "Fındık / Meyve Bahçesi": (150_000.0, 0.15),
    "Sera": (420_000.0, 0.07),
    "Tarımsal Tesis": (300_000.0, 0.08),
}
_FIRST_INITIALS: Tuple[str, ...] = tuple("AEFHİMMOSYZ")
_SURNAMES: Tuple[str, ...] = (
    "Yılmaz", "Kaya", "Demir", "Çelik", "Şahin", "Yıldız", "Aydın", "Özdemir",
    "Arslan", "Doğan", "Kılıç", "Aslan", "Çetin", "Koç", "Kurt", "Özkan",
)


#: ``InsuranceSimulator.decide_codes`` çıktı sırası (indeks = kod).
DECISION_LABELS: Tuple[str, ...] = ("RET", "EKSPERTİZ", "KISMİ ONAY", "ONAY")


@dataclass(frozen=True)
class ClaimRules:
    """Otomatik tazminat karar kuralları (arayüzden ayarlanabilir).

    Attributes:
        approve_dnbr: Bu dNBR ve üzeri → tam onay (güven eşiği sağlanırsa).
        partial_dnbr: Bu dNBR ve üzeri (onay eşiğinin altı) → kısmi onay / avans.
        review_dnbr: Bu dNBR altı → ret (hasar eşiğin altında).
        min_confidence: Onay için gereken gözlem güveni (0–1; bulut/gölge etkisi).
        deductible_pct: Muafiyet (sigorta bedelinin yüzdesi).
        coverage_pct: Teminat oranı (%).
        advance_pct: Kısmi onayda hemen ödenen avans oranı (%).
    """

    approve_dnbr: float = 0.44
    partial_dnbr: float = 0.27
    review_dnbr: float = 0.10
    min_confidence: float = 0.80
    deductible_pct: float = 2.0
    coverage_pct: float = 90.0
    advance_pct: float = 70.0


@dataclass
class InsuranceDataset:
    """Sigorta sekmesi için üretilmiş veri paketi.

    Attributes:
        parcels: Parsel/poliçe tablosu (ham ölçümler + risk skoru).
        geometry: Parsel geometrisi.
        centre: Harita merkezi.
        event_label: Olay açıklaması (ör. orman yangını).
        event_date: Olay sonrası görüntü tarihi.
    """

    parcels: pd.DataFrame
    geometry: ParcelGeometry
    centre: LatLon
    event_label: str
    event_date: dt.date

    def geojson(self, frame: Optional[pd.DataFrame] = None) -> dict:
        """Parsel çokgenlerini GeoJSON olarak üretir.

        Args:
            frame: Öznitelik tablosu (varsayılan: ``self.parcels``); örn. karar sütunlu tablo.

        Returns:
            GeoJSON ``FeatureCollection``.
        """
        return parcels_to_geojson(self.geometry, self.parcels if frame is None else frame)


class InsuranceSimulator:
    """Yangın sonrası parsel bazlı dNBR hasar ve poliçe risk verisi üretir."""

    def __init__(
        self,
        seed: int = 2026,
        rows: int = 12,
        cols: int = 16,
        cell_km: float = 0.62,
        centre: LatLon = (37.0390, 28.3400),
        wind_bearing_deg: float = 70.0,
    ) -> None:
        """Simülatörü kurar.

        Args:
            seed: Tohum değeri.
            rows: Parsel ızgarası satır sayısı.
            cols: Parsel ızgarası sütun sayısı.
            cell_km: Parsel hücre boyutu (km).
            centre: Harita merkezi ``(enlem, boylam)``.
            wind_bearing_deg: Yangının ilerleme yönü (kuzeyden saat yönü, derece).
        """
        self.seed = seed
        self.rows = rows
        self.cols = cols
        self.cell_km = cell_km
        self.centre = centre
        self.wind_bearing_deg = wind_bearing_deg

    def simulate(self, event_date: Optional[dt.date] = None) -> InsuranceDataset:
        """Tüm parseller için ölçüm, hasar ve risk skorunu üretir.

        dNBR = NBR(öncesi) − NBR(sonrası). Hasar şiddeti, ateşin ilerleme yönünde
        uzamış bir Gauss yoğunluğu ile mekânsal ilintili Gauss alanının birleşimidir.

        Args:
            event_date: Olay sonrası görüntü tarihi.

        Returns:
            ``InsuranceDataset``.
        """
        rng = make_rng(self.seed, 21)
        geo = ParcelGrid(self.centre, self.rows, self.cols, self.cell_km, seed=self.seed).build()
        n = geo.count
        theta = math.radians(self.wind_bearing_deg)
        ux, uy = math.sin(theta), math.cos(theta)  # ilerleme ekseni
        along = (geo.x_km - (-1.8)) * ux + (geo.y_km - (-0.9)) * uy
        cross = -(geo.x_km - (-1.8)) * uy + (geo.y_km - (-0.9)) * ux
        envelope = np.exp(-0.5 * (((along - 2.6) / 2.9) ** 2 + (cross / 1.45) ** 2))
        texture = gaussian_field((self.rows, self.cols), 1.6, rng).ravel()
        intensity = np.clip(envelope * (1.0 + 0.30 * texture), 0.0, 1.35)

        nbr_pre = np.clip(rng.normal(0.52, 0.06, n), 0.25, 0.80)
        dnbr = np.clip(0.80 * intensity + rng.normal(0.0, 0.035, n), -0.08, 1.10)
        nbr_post = nbr_pre - dnbr
        ndvi_pre = np.clip(rng.normal(0.66, 0.08, n) + 0.10 * (nbr_pre - 0.52), 0.25, 0.92)
        ndvi_post = np.clip(ndvi_pre - 0.85 * np.clip(dnbr, 0.0, None) + rng.normal(0, 0.02, n), -0.10, 0.92)
        confidence = np.clip(rng.normal(0.89, 0.07, n), 0.45, 1.0)

        names = np.array(list(ASSET_CATALOG))
        probs = np.array([ASSET_CATALOG[k][1] for k in names])
        asset = rng.choice(names, size=n, p=probs / probs.sum())
        per_ha = np.array([ASSET_CATALOG[a][0] for a in asset])
        insured = per_ha * geo.area_ha * rng.lognormal(0.0, 0.18, n)
        holder = [
            f"{_FIRST_INITIALS[i]}. {_SURNAMES[j]}"
            for i, j in zip(rng.integers(0, len(_FIRST_INITIALS), n), rng.integers(0, len(_SURNAMES), n))
        ]

        dnbr_norm = np.clip(dnbr / 0.66, 0.0, 1.0)
        dryness = np.clip((0.72 - ndvi_pre) / 0.30, 0.0, 1.0)
        exposure = neighbour_mean(dnbr_norm, self.rows, self.cols)
        risk = 100.0 * np.clip(0.60 * dnbr_norm + 0.20 * dryness + 0.20 * exposure, 0.0, 1.0)
        risk_class = np.select(
            [risk < 25, risk < 50, risk < 75], ["Düşük", "Orta", "Yüksek"], default="Çok Yüksek"
        )
        sev_idx = classify_dnbr(dnbr)

        frame = pd.DataFrame(
            {
                "parsel_id": [f"P-{i + 1:03d}" for i in range(n)],
                "police_no": [f"POL-2026-{10_000 + i * 7:05d}" for i in range(n)],
                "sigortali": holder,
                "varlik_turu": asset,
                "alan_ha": geo.area_ha.round(2),
                "sigorta_bedeli_tl": insured.round(0),
                "police_aktif": rng.random(n) > 0.07,
                "ndvi_once": ndvi_pre.round(3),
                "ndvi_sonra": ndvi_post.round(3),
                "nbr_once": nbr_pre.round(3),
                "nbr_sonra": nbr_post.round(3),
                "dnbr": dnbr.round(3),
                "siddet": [SEVERITY_LABELS[k] for k in sev_idx],
                "siddet_no": sev_idx,
                "guven": confidence.round(2),
                "risk_skoru": risk.round(1),
                "risk_sinifi": risk_class,
                "enlem": geo.centroid_lat.round(5),
                "boylam": geo.centroid_lon.round(5),
            }
        )
        when = event_date or dt.date(2026, 8, 14)
        return InsuranceDataset(frame, geo, self.centre, "Orman yangını (simülasyon)", when)

    @staticmethod
    def decide_codes(dnbr: np.ndarray, conf: np.ndarray, active: np.ndarray, rules: ClaimRules) -> np.ndarray:
        """Karar kuralının tek kaynağı: her parsel için ``DECISION_LABELS`` indeksini üretir.

        ``evaluate_claims`` ve geriye dönük test (``calibration.py``) aynı işlevi kullanır;
        böylece sahada uygulanan kural ile test edilen kural ayrışamaz.

        Args:
            dnbr: dNBR değerleri.
            conf: Gözlem güveni (0–1).
            active: Poliçe aktif mi.
            rules: Karar kuralları.

        Returns:
            ``int8`` dizisi: 0=RET, 1=EKSPERTİZ, 2=KISMİ ONAY, 3=ONAY.
        """
        trusted = conf >= rules.min_confidence
        return np.select(
            [
                ~active,
                dnbr < rules.review_dnbr,
                (dnbr >= rules.approve_dnbr) & trusted,
                (dnbr >= rules.partial_dnbr) & trusted,
            ],
            [0, 0, 3, 2],
            default=1,
        ).astype(np.int8)

    @staticmethod
    def evaluate_claims(parcels: pd.DataFrame, rules: ClaimRules) -> pd.DataFrame:
        """Otomatik tazminat onay/ret kararlarını vektörize biçimde üretir.

        Karar sırası: RET (poliçe pasif ya da hasar eşiğin altında) → ONAY
        (dNBR ≥ onay eşiği ve güven yeterli) → KISMİ ONAY → EKSPERTİZ (kalan).

        Args:
            parcels: ``InsuranceDataset.parcels`` tablosu.
            rules: Karar kuralları.

        Returns:
            Karar, gerekçe, hasar yüzdesi, ödenecek ve bekleyen tutar sütunlu yeni tablo.
        """
        df = parcels.copy()
        dnbr = df["dnbr"].to_numpy(dtype=float)
        conf = df["guven"].to_numpy(dtype=float)
        active = df["police_aktif"].to_numpy(dtype=bool)
        trusted = conf >= rules.min_confidence

        decision = np.array(DECISION_LABELS, dtype=object)[
            InsuranceSimulator.decide_codes(dnbr, conf, active, rules)
        ]
        reason = np.select(
            [
                ~active,
                dnbr < rules.review_dnbr,
                decision == "ONAY",
                decision == "KISMİ ONAY",
                ~trusted,
            ],
            [
                "Poliçe dönem dışı / pasif",
                "dNBR hasar eşiğinin altında",
                "Yüksek şiddetli hasar, güvenilir gözlem",
                "Orta şiddetli hasar, avans ödemesi",
                "Gözlem güveni düşük (bulut/gölge) — yerinde keşif",
            ],
            default="Sınır hasar — eksper incelemesi",
        )
        span = max(0.66 - rules.review_dnbr, 1e-6)
        ratio = np.clip((dnbr - rules.review_dnbr) / span, 0.0, 1.0)
        insured = df["sigorta_bedeli_tl"].to_numpy(dtype=float)
        gross = insured * (rules.coverage_pct / 100.0) * ratio
        net = np.clip(gross - insured * (rules.deductible_pct / 100.0), 0.0, None)
        paid = np.select(
            [decision == "ONAY", decision == "KISMİ ONAY"],
            [net, net * (rules.advance_pct / 100.0)],
            default=0.0,
        )
        pending = np.select(
            [decision == "ONAY", decision == "KISMİ ONAY", decision == "EKSPERTİZ"],
            [0.0, net - paid, net],
            default=0.0,
        )
        df["karar"] = decision
        df["gerekce"] = reason
        df["hasar_yuzdesi"] = (ratio * 100.0).round(1)
        df["odenecek_tl"] = paid.round(0)
        df["bekleyen_tl"] = pending.round(0)
        return df

    @staticmethod
    def summarize_claims(claims: pd.DataFrame) -> Dict[str, float]:
        """Karar tablosundan sekme KPI'larını türetir.

        Args:
            claims: ``evaluate_claims`` çıktısı.

        Returns:
            Poliçe sayısı, karar adetleri, toplam ödeme/bekleyen, hasarlı alan ve otomasyon oranı.
        """
        counts = claims["karar"].value_counts()
        damaged = claims["dnbr"].to_numpy() >= DNBR_EDGES[0]
        total = max(len(claims), 1)
        auto = counts.get("ONAY", 0) + counts.get("KISMİ ONAY", 0) + counts.get("RET", 0)
        return {
            "policy_count": float(len(claims)),
            "approved": float(counts.get("ONAY", 0)),
            "partial": float(counts.get("KISMİ ONAY", 0)),
            "review": float(counts.get("EKSPERTİZ", 0)),
            "rejected": float(counts.get("RET", 0)),
            "paid_tl": float(claims["odenecek_tl"].sum()),
            "pending_tl": float(claims["bekleyen_tl"].sum()),
            "damaged_ha": float(claims.loc[damaged, "alan_ha"].sum()),
            "mean_dnbr": float(claims["dnbr"].mean()),
            "automation_pct": 100.0 * float(auto) / total,
        }


# =========================================================================== #
# 2) TARIM — Fenoloji ve Rekolte
# =========================================================================== #
@dataclass(frozen=True)
class CropProfile:
    """Bir bitki türünün fenolojik ve ekonomik parametreleri.

    Attributes:
        name: Türkçe ad.
        colour: Grafik rengi (hex).
        sos: Sezon başlangıcı (yılın günü; NDVI yükselişinin orta noktası).
        eos: Sezon sonu / olgunlaşma (yılın günü; NDVI düşüşünün orta noktası).
        peak_ndvi: Tepe NDVI.
        base_ndvi: Çıplak toprak / dinlenme NDVI.
        rise_k: Yükseliş eğimi (gün).
        fall_k: Düşüş eğimi (gün).
        base_yield: Beklenen verim (ton/ha).
        water_need: Su stresine duyarlılık (0–1).
        price_tl_t: Tahmini satış fiyatı (TL/ton).
    """

    name: str
    colour: str
    sos: float
    eos: float
    peak_ndvi: float
    base_ndvi: float
    rise_k: float
    fall_k: float
    base_yield: float
    water_need: float
    price_tl_t: float


CROPS: Tuple[CropProfile, ...] = (
    CropProfile("Buğday", "#e3c35a", 82, 158, 0.80, 0.16, 9.0, 8.0, 5.2, 0.55, 9_500.0),
    CropProfile("Arpa", "#ec9a4d", 78, 148, 0.76, 0.16, 8.5, 7.5, 4.6, 0.50, 8_200.0),
    CropProfile("Mısır", "#6ea8fe", 140, 232, 0.88, 0.15, 10.0, 11.0, 11.5, 0.90, 7_400.0),
    CropProfile("Ayçiçeği", "#a78bfa", 135, 218, 0.78, 0.15, 9.5, 9.0, 3.2, 0.60, 21_000.0),
    CropProfile("Pamuk", "#e5738f", 145, 262, 0.82, 0.15, 11.0, 13.0, 4.4, 0.85, 26_000.0),
    CropProfile("Şeker Pancarı", "#5fc98a", 118, 268, 0.85, 0.16, 12.0, 12.0, 62.0, 0.80, 2_300.0),
)
HEALTH_LABELS: Tuple[str, ...] = ("Kritik", "Stresli", "İzlemede", "Sağlıklı")


@dataclass(frozen=True)
class StressPeriod:
    """Ardışık gün bloğu olarak tespit edilen su stresi dönemi.

    Attributes:
        start: Başlangıç tarihi.
        end: Bitiş tarihi.
        days: Süre (gün).
        peak_index: Dönem içindeki en yüksek stres indeksi (0–1).
        mean_temp_c: Dönem ortalama sıcaklığı (°C).
    """

    start: dt.date
    end: dt.date
    days: int
    peak_index: float
    mean_temp_c: float


@dataclass
class AgricultureDataset:
    """Tarım sekmesi için üretilmiş veri paketi.

    Attributes:
        fields: Tarla tablosu (gözlem tarihindeki NDVI/EVI, sağlık, rekolte).
        geometry: Tarla geometrisi.
        centre: Harita merkezi.
        year: Sezon yılı.
        obs_date: Gözlem tarihi.
        crop_curves: 8 günlük kompozit NDVI (satır: tarih, sütun: bitki türü ortalaması).
        field_curves: 8 günlük kompozit NDVI (sütun: tarla kimliği).
        climate: Günlük sıcaklık ve su stresi indeksi.
        stress_periods: Tespit edilen su stresi dönemleri.
    """

    fields: pd.DataFrame
    geometry: ParcelGeometry
    centre: LatLon
    year: int
    obs_date: dt.date
    crop_curves: pd.DataFrame
    field_curves: pd.DataFrame
    climate: pd.DataFrame
    stress_periods: List[StressPeriod] = field(default_factory=list)

    def geojson(self) -> dict:
        """Tarla sınırlarını GeoJSON olarak döndürür.

        Returns:
            GeoJSON ``FeatureCollection``.
        """
        return parcels_to_geojson(self.geometry, self.fields)


def _runs(mask: np.ndarray, min_len: int = 1) -> List[Tuple[int, int]]:
    """Boolean dizideki ardışık ``True`` bloklarını ``(başlangıç, bitiş)`` olarak bulur.

    Args:
        mask: Boolean dizi.
        min_len: En kısa kabul edilen blok uzunluğu.

    Returns:
        Bitiş dahil indeks çiftleri listesi.
    """
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])
    edges = np.diff(padded)
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0] - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s + 1 >= min_len]


def _smooth(series: np.ndarray, width: int) -> np.ndarray:
    """Hareketli ortalama (kenar dolgulu) ile diziyi yumuşatır.

    Args:
        series: 1B dizi.
        width: Pencere genişliği.

    Returns:
        Aynı uzunlukta yumuşatılmış dizi.
    """
    kernel = np.ones(width) / width
    padded = np.pad(series, width // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: series.size]


def _double_logistic(
    doy: np.ndarray, base: np.ndarray, peak: np.ndarray, sos: np.ndarray,
    eos: np.ndarray, k1: np.ndarray, k2: np.ndarray,
) -> np.ndarray:
    """Çift lojistik fenoloji eğrisi (yükseliş − düşüş); parametreler yayınlanır.

    Args:
        doy: Yılın günü dizisi, şekil ``(T,)``.
        base: Taban NDVI, ``(F, 1)``.
        peak: Tepe NDVI, ``(F, 1)``.
        sos: Sezon başı, ``(F, 1)``.
        eos: Sezon sonu, ``(F, 1)``.
        k1: Yükseliş eğimi, ``(F, 1)``.
        k2: Düşüş eğimi, ``(F, 1)``.

    Returns:
        ``(F, T)`` NDVI dizisi.
    """
    return base + (peak - base) * (sigmoid((doy - sos) / k1) - sigmoid((doy - eos) / k2))


def _evi_from_ndvi(ndvi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """NDVI'ya tutarlı sentetik Kırmızı/NIR/Mavi bantlarından EVI hesaplar.

    EVI = 2.5·(NIR − Red) / (NIR + 6·Red − 7.5·Blue + 1)

    Args:
        ndvi: NDVI dizisi.
        rng: Rastgele sayı üreteci.

    Returns:
        Aynı şekilde EVI dizisi.
    """
    nd = np.clip(ndvi, -0.2, 0.93)
    total = 0.42 + 0.14 * nd + rng.normal(0.0, 0.012, nd.shape)  # NIR + Kırmızı yansıma toplamı
    red = total * (1.0 - nd) / 2.0
    nir = total * (1.0 + nd) / 2.0
    blue = red * np.clip(rng.normal(0.45, 0.03, nd.shape), 0.3, 0.7)
    evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0)
    return np.minimum(evi, 0.98 * nd)


class AgricultureSimulator:
    """Tarla sınırları, NDVI/EVI, fenoloji, su stresi ve rekolte verisi üretir."""

    def __init__(
        self,
        seed: int = 2026,
        year: int = 2026,
        centre: LatLon = (37.5850, 32.7700),
        rows: int = 3,
        cols: int = 4,
        cell_km: float = 1.15,
    ) -> None:
        """Simülatörü kurar.

        Args:
            seed: Tohum değeri.
            year: Sezon yılı.
            centre: Harita merkezi.
            rows: Tarla ızgarası satır sayısı.
            cols: Tarla ızgarası sütun sayısı.
            cell_km: Tarla hücre boyutu (km).
        """
        self.seed = seed
        self.year = year
        self.centre = centre
        self.rows = rows
        self.cols = cols
        self.cell_km = cell_km

    def _climate(self, rng: np.random.Generator) -> pd.DataFrame:
        """Günlük sıcaklık (°C) ve su stresi indeksini (0–1) üretir.

        Sıcaklık: yıllık sinüs + AR(1) Gauss gürültüsü. Stres: sıcaklığa bağlı taban
        bileşen + 2 rastgele kuraklık epizodunun Gauss tümsekleri.

        Args:
            rng: Rastgele sayı üreteci.

        Returns:
            ``tarih``, ``sicaklik_c``, ``su_stresi`` sütunlu tablo (365 gün).
        """
        doy = np.arange(1, 366)
        noise = np.zeros(365)
        eps = rng.normal(0.0, 1.6, 365)
        for i in range(1, 365):
            noise[i] = 0.72 * noise[i - 1] + eps[i]
        temp = 14.6 - 13.8 * np.cos(2.0 * np.pi * (doy - 24) / 365.0) + noise
        stress = 0.50 * sigmoid((temp - 27.0) / 2.0)
        for centre_doy, amp in zip(rng.uniform(176, 200, 1).tolist() + rng.uniform(222, 250, 1).tolist(),
                                   rng.uniform(0.55, 0.80, 2)):
            width = rng.uniform(8.0, 13.0)
            stress += amp * np.exp(-0.5 * ((doy - centre_doy) / width) ** 2)
        stress = np.clip(stress + rng.normal(0.0, 0.02, 365), 0.0, 1.0)
        start = dt.date(self.year, 1, 1)
        dates = [start + dt.timedelta(days=int(d) - 1) for d in doy]
        return pd.DataFrame({"tarih": pd.to_datetime(dates), "sicaklik_c": temp.round(1),
                             "su_stresi": stress.round(3)})

    def simulate(self, obs_doy: int = 170) -> AgricultureDataset:
        """Tüm tarlalar için gözlem tarihindeki durumu ve yıllık eğrileri üretir.

        Args:
            obs_doy: Gözlem tarihi (yılın günü, 1–365).

        Returns:
            ``AgricultureDataset``.
        """
        obs_doy = int(np.clip(obs_doy, 1, 365))
        rng = make_rng(self.seed, 31)
        geo = ParcelGrid(self.centre, self.rows, self.cols, self.cell_km, jitter=0.15, gap=0.05,
                         seed=self.seed + 5).build()
        n = geo.count
        climate = self._climate(rng)
        stress_eff = _smooth(climate["su_stresi"].to_numpy(), 7)
        doy = np.arange(1, 366, dtype=float)

        # Bitki ataması: her tür en az bir kez, kalanı rastgele.
        crop_idx = np.concatenate([np.arange(len(CROPS)), rng.integers(0, len(CROPS), n - len(CROPS))])[:n]
        rng.shuffle(crop_idx)
        col = lambda f: np.array([f(CROPS[i]) for i in crop_idx], dtype=float)[:, None]  # noqa: E731
        irrigation = np.clip(rng.normal(0.72, 0.18, n), 0.25, 1.0)
        sos = col(lambda c: c.sos) + rng.normal(0.0, 5.0, (n, 1))
        eos = col(lambda c: c.eos) + rng.normal(0.0, 6.0, (n, 1))
        peak = np.clip(col(lambda c: c.peak_ndvi) * rng.normal(1.0, 0.04, (n, 1)), 0.4, 0.93)
        base = col(lambda c: c.base_ndvi) + rng.normal(0.0, 0.01, (n, 1))
        k1, k2 = col(lambda c: c.rise_k), col(lambda c: c.fall_k)
        need = col(lambda c: c.water_need)

        nominal = _double_logistic(doy, base, peak, sos, eos, k1, k2)
        s_field = np.clip(stress_eff[None, :] * (1.30 - irrigation[:, None]), 0.0, 1.0)
        actual = base + (nominal - base) * (1.0 - 0.55 * need * s_field)
        actual = np.clip(actual + rng.normal(0.0, 0.012, actual.shape), -0.05, 0.93)

        # Gözlem günü değerleri
        o = obs_doy - 1
        ndvi_o, nom_o = actual[:, o], nominal[:, o]
        evi_o = _evi_from_ndvi(ndvi_o, make_rng(self.seed, 32, obs_doy))
        ratio = ndvi_o / np.clip(nom_o, 0.05, None)
        in_season = (nom_o - base[:, 0]) > 0.10
        ratio = np.where(in_season, ratio, 1.0)
        health_idx = np.digitize(ratio, [0.65, 0.80, 0.92])
        health = np.array(HEALTH_LABELS)[health_idx]

        # Rekolte: sezon boyunca (NDVI − taban) integrali, nominal integrale oranla.
        integ = (actual - base).clip(0.0, None).sum(axis=1)
        integ_nom = (nominal - base).clip(0.0, None).sum(axis=1)
        yield_ratio = np.clip((integ / np.clip(integ_nom, 1e-6, None)) ** 1.15, 0.2, 1.3)
        y_base = col(lambda c: c.base_yield)[:, 0]
        yield_t_ha = np.clip(y_base * yield_ratio * rng.normal(1.0, 0.035, n), 0.1, None)
        price = col(lambda c: c.price_tl_t)[:, 0]

        names = np.array([CROPS[i].name for i in crop_idx])
        fields = pd.DataFrame({
            "tarla_id": [f"T-{i + 1:02d}" for i in range(n)],
            "bitki": names,
            "alan_ha": geo.area_ha.round(1),
            "ndvi": ndvi_o.round(3),
            "evi": evi_o.round(3),
            "ndvi_beklenen": nom_o.round(3),
            "saglik": health,
            "su_stresi": np.where(in_season, s_field[:, o], 0.0).round(2),
            "sulama_kalitesi": irrigation.round(2),
            "rekolte_t_ha": yield_t_ha.round(2),
            "rekolte_toplam_t": (yield_t_ha * geo.area_ha).round(1),
            "gelir_tahmini_tl": (yield_t_ha * geo.area_ha * price).round(0),
            "ekim_gunu": sos[:, 0].round(0).astype(int),
            "enlem": geo.centroid_lat.round(5),
            "boylam": geo.centroid_lon.round(5),
        })

        # 8 günlük kompozitler
        comp = np.arange(0, 365, 8)
        dates8 = pd.to_datetime(climate["tarih"].to_numpy()[comp])
        field_curves = pd.DataFrame(actual[:, comp].T, index=dates8, columns=fields["tarla_id"]).round(3)
        crop_curves = pd.DataFrame(
            {c.name: field_curves.loc[:, (names == c.name)].mean(axis=1) for c in CROPS}
        ).round(3)

        periods: List[StressPeriod] = []
        temp_arr, st_arr = climate["sicaklik_c"].to_numpy(), climate["su_stresi"].to_numpy()
        for s, e in _runs(st_arr >= 0.45, min_len=5):
            periods.append(StressPeriod(
                dt.date(self.year, 1, 1) + dt.timedelta(days=s),
                dt.date(self.year, 1, 1) + dt.timedelta(days=e),
                e - s + 1, float(st_arr[s : e + 1].max()), float(temp_arr[s : e + 1].mean()),
            ))
        return AgricultureDataset(
            fields=fields, geometry=geo, centre=self.centre, year=self.year,
            obs_date=dt.date(self.year, 1, 1) + dt.timedelta(days=obs_doy - 1),
            crop_curves=crop_curves, field_curves=field_curves, climate=climate,
            stress_periods=periods,
        )

    @staticmethod
    def summarize(ds: AgricultureDataset) -> Dict[str, float]:
        """Tarım KPI'larını hesaplar.

        Args:
            ds: ``simulate`` çıktısı.

        Returns:
            Toplam alan, ortalama NDVI/EVI, stresli alan oranı, rekolte ve gelir tahmini.
        """
        f = ds.fields
        area = float(f["alan_ha"].sum())
        stressed = f["saglik"].isin(["Stresli", "Kritik"])
        return {
            "field_count": float(len(f)),
            "area_ha": area,
            "mean_ndvi": float(np.average(f["ndvi"], weights=f["alan_ha"])),
            "mean_evi": float(np.average(f["evi"], weights=f["alan_ha"])),
            "stressed_pct": 100.0 * float(f.loc[stressed, "alan_ha"].sum()) / max(area, 1e-9),
            "yield_total_t": float(f["rekolte_toplam_t"].sum()),
            "revenue_tl": float(f["gelir_tahmini_tl"].sum()),
            "stress_days": float(sum(p.days for p in ds.stress_periods)),
        }


# =========================================================================== #
# 3) BELEDİYE — Şehircilik ve İmar
# =========================================================================== #
LC_WATER, LC_GREEN, LC_BARE, LC_BUILT = 0, 1, 2, 3
LC_LABELS: Tuple[str, ...] = ("Su", "Yeşil alan", "Açık / boş arazi", "Yapılaşma")
NEIGHBOURHOODS: Tuple[str, ...] = (
    "Yeşiltepe", "Çınarlı", "Gülbahçe", "Akçaova",
    "Dere", "Yenikent", "Bağlıca", "Kuzey Sanayi",
)
_LC_PALETTE: np.ndarray = np.array(
    [[26, 64, 98], [46, 104, 54], [158, 136, 106], [178, 178, 184]], dtype=float
)


@dataclass
class MunicipalityDataset:
    """Belediye sekmesi için üretilmiş veri paketi.

    Attributes:
        t0: Başlangıç yılı.
        t1: Karşılaştırma yılı.
        centre: Harita merkezi.
        bounds: ``((güney, batı), (kuzey, doğu))`` görüntü sınırları.
        landcover_t0: T0 arazi örtüsü sınıfları (0=su, 1=yeşil, 2=boş, 3=yapı).
        landcover_t1: T1 arazi örtüsü sınıfları.
        rgba_t0: T0 uydu görüntüsü simülasyonu (``uint8`` RGBA).
        rgba_t1: T1 uydu görüntüsü simülasyonu (``uint8`` RGBA).
        sites: Tespit edilen kaçak yapılaşma noktaları.
        stats: Yıl × mahalle istatistikleri (uzun biçim).
        neighbourhood_rects: ``(ad, (güney, batı), (kuzey, doğu))`` mahalle kutuları.
    """

    t0: int
    t1: int
    centre: LatLon
    bounds: Tuple[LatLon, LatLon]
    landcover_t0: np.ndarray
    landcover_t1: np.ndarray
    rgba_t0: np.ndarray
    rgba_t1: np.ndarray
    sites: pd.DataFrame
    stats: pd.DataFrame
    neighbourhood_rects: List[Tuple[str, LatLon, LatLon]] = field(default_factory=list)


def render_landcover_rgba(
    landcover: np.ndarray, rng: np.random.Generator, scale: int = 3
) -> np.ndarray:
    """Arazi örtüsü sınıflarından uydu görüntüsü görünümlü RGBA doku üretir.

    Sınıf renkleri; Gauss parlaklık gürültüsü, düşük frekanslı doku ve yapı
    piksellerinde "çatı" benzeri blok deseniyle zenginleştirilir.

    Args:
        landcover: ``(H, W)`` sınıf matrisi.
        rng: Rastgele sayı üreteci.
        scale: Çıktı büyütme katsayısı (keskin kenar yerine ince doku için).

    Returns:
        ``(H*scale, W*scale, 4)`` ``uint8`` dizisi.
    """
    big = np.kron(landcover, np.ones((scale, scale), dtype=landcover.dtype))
    h, w = big.shape
    base = _LC_PALETTE[big]
    texture = gaussian_field((h, w), 2.2, rng)
    grain = rng.normal(0.0, 1.0, (h, w))
    factor = 1.0 + 0.05 * texture + 0.035 * grain
    roofs = rng.integers(0, 2, (h // 2 + 1, w // 2 + 1)).repeat(2, 0).repeat(2, 1)[:h, :w]
    factor = np.where(big == LC_BUILT, factor + 0.10 * (roofs - 0.5), factor)
    factor = np.where(big == LC_GREEN, factor + 0.05 * texture, factor)
    rgb = np.clip(base * factor[..., None], 0.0, 255.0)
    alpha = np.full((h, w, 1), 255.0)
    return np.concatenate([rgb, alpha], axis=-1).astype(np.uint8)


class MunicipalitySimulator:
    """Çok yıllı arazi örtüsü, kaçak yapılaşma ve mahalle istatistikleri üretir.

    Kentleşme, tek bir "kentleşme uygunluğu" Gauss alanının eşiği zamanla
    düşürülerek modellenir; böylece her yılın yapılaşma maskesi bir öncekini kapsar
    (yapılar yıkılmaz) ve T0–T1 karşılaştırması tutarlı olur.
    """

    YEAR_MIN: int = 2015
    YEAR_MAX: int = 2026

    def __init__(
        self,
        seed: int = 2026,
        size: int = 120,
        pixel_m: float = 30.0,
        centre: LatLon = (40.2000, 29.0600),
        block: int = 6,
    ) -> None:
        """Simülatörü kurar.

        Args:
            seed: Tohum değeri.
            size: Kare raster kenar uzunluğu (piksel); ``block`` ile tam bölünmeli.
            pixel_m: Piksel kenarı (m).
            centre: Harita merkezi.
            block: Kaçak yapı tespitinde kullanılan blok kenarı (piksel).

        Raises:
            ValueError: ``size`` blok boyutuna bölünmüyorsa.
        """
        if size % block != 0 or size % 4 != 0:
            raise ValueError("size, block ve 4'e tam bölünmelidir.")
        self.seed, self.size, self.pixel_m, self.centre, self.block = seed, size, pixel_m, centre, block
        rng = make_rng(seed, 41)
        yy, xx = np.mgrid[0:size, 0:size].astype(float)
        core = np.exp(-(((xx - 0.42 * size) ** 2 + (yy - 0.52 * size) ** 2) / (2 * (0.30 * size) ** 2)))
        self._urban = gaussian_field((size, size), 11.0, rng) + 1.5 * core
        self._veg = gaussian_field((size, size), 8.0, rng)
        self._zoning = gaussian_field((size, size), 10.0, rng)
        river = np.abs(yy - (0.62 * size + 0.08 * size * np.sin(xx / (0.11 * size)))) < 0.022 * size
        self._water = river
        self._urban = np.where(river, -np.inf, self._urban)
        self._nb = (yy // (size // 2)).astype(int) * 4 + (xx // (size // 4)).astype(int)

    def landcover(self, year: int) -> np.ndarray:
        """Verilen yıl için arazi örtüsü sınıf matrisini üretir.

        Args:
            year: ``YEAR_MIN`` – ``YEAR_MAX`` arası yıl.

        Returns:
            ``(size, size)`` ``int8`` sınıf matrisi.
        """
        k = (year - self.YEAR_MIN) / (self.YEAR_MAX - self.YEAR_MIN)
        built_frac = 0.15 + 0.25 * k
        finite = self._urban[np.isfinite(self._urban)]
        thr = np.quantile(finite, 1.0 - built_frac)
        built = self._urban > thr
        green = (self._veg > (-0.35 + 0.28 * k)) & ~built & ~self._water
        lc = np.full(self._urban.shape, LC_BARE, dtype=np.int8)
        lc[green] = LC_GREEN
        lc[built] = LC_BUILT
        lc[self._water] = LC_WATER
        return lc

    def _stats(self, years: Sequence[int]) -> pd.DataFrame:
        """Yıl × mahalle sınıf alanlarını ve altyapı yoğunluğunu hesaplar.

        Args:
            years: Yıl listesi.

        Returns:
            Uzun biçimli istatistik tablosu.
        """
        pix_ha = (self.pixel_m / 100.0) ** 2
        n_nb = len(NEIGHBOURHOODS)
        rng = make_rng(self.seed, 43)
        rows: List[dict] = []
        for y in years:
            lc = self.landcover(y)
            counts = np.bincount((self._nb * 4 + lc).ravel(), minlength=n_nb * 4).reshape(n_nb, 4)
            ha = counts * pix_ha
            total = ha.sum(axis=1)
            built_f, bare_f = ha[:, LC_BUILT] / total, ha[:, LC_BARE] / total
            infra = np.clip(2.2 + 21.0 * built_f + 0.8 * bare_f + rng.normal(0.0, 0.25, n_nb), 0.5, None)
            for i, name in enumerate(NEIGHBOURHOODS):
                rows.append({
                    "yil": int(y), "mahalle": name,
                    "yapili_ha": round(float(ha[i, LC_BUILT]), 1),
                    "yesil_ha": round(float(ha[i, LC_GREEN]), 1),
                    "acik_ha": round(float(ha[i, LC_BARE]), 1),
                    "su_ha": round(float(ha[i, LC_WATER]), 1),
                    "yapilasma_pct": round(100.0 * float(built_f[i]), 1),
                    "yesil_pct": round(100.0 * float(ha[i, LC_GREEN] / total[i]), 1),
                    "altyapi_yogunlugu": round(float(infra[i]), 2),
                    "toplam_ha": round(float(total[i]), 1),
                    "altyapi_km": round(float(infra[i] * total[i] / 100.0), 1),
                })
        return pd.DataFrame(rows)

    def simulate(self, t0: int = 2019, t1: int = 2026) -> MunicipalityDataset:
        """T0 / T1 görüntülerini, kaçak yapı noktalarını ve istatistikleri üretir.

        Kaçak yapı = T1'de yapılı, T0'da yapısız ve imar izin alanı dışında kalan piksel;
        ``block`` x ``block`` bloklarda toplanır, eşik üstü bloklar nokta olarak raporlanır.

        Args:
            t0: Başlangıç yılı.
            t1: Karşılaştırma yılı (``t0``'dan büyük olmalı).

        Returns:
            ``MunicipalityDataset``.

        Raises:
            ValueError: Yıllar aralık dışındaysa ya da ``t0 >= t1`` ise.
        """
        for y in (t0, t1):
            if not self.YEAR_MIN <= y <= self.YEAR_MAX:
                raise ValueError(f"Yıl {self.YEAR_MIN}-{self.YEAR_MAX} aralığında olmalıdır.")
        if t0 >= t1:
            raise ValueError("T0 yılı T1 yılından küçük olmalıdır.")
        rng = make_rng(self.seed, 42)
        lc0, lc1 = self.landcover(t0), self.landcover(t1)
        permitted = self._zoning > -0.40
        illegal = (lc1 == LC_BUILT) & (lc0 != LC_BUILT) & ~permitted

        b, s = self.block, self.size
        nb_ = s // b
        cnt = illegal.reshape(nb_, b, nb_, b).sum(axis=(1, 3))
        prev_green = (lc0 == LC_GREEN).reshape(nb_, b, nb_, b).sum(axis=(1, 3))
        rr, cc = np.mgrid[0:s, 0:s]
        sum_r = (illegal * rr).reshape(nb_, b, nb_, b).sum(axis=(1, 3))
        sum_c = (illegal * cc).reshape(nb_, b, nb_, b).sum(axis=(1, 3))
        br, bc = np.where(cnt >= max(int(0.25 * b * b), 1))
        # Nokta, bloğun merkezi değil, kaçak (izinsiz yeni) piksellerin ağırlık merkezidir.
        cen_r = (sum_r[br, bc] + 0.5 * cnt[br, bc]) / cnt[br, bc]
        cen_c = (sum_c[br, bc] + 0.5 * cnt[br, bc]) / cnt[br, bc]
        half_km = s * self.pixel_m / 2000.0
        north_km = half_km - cen_r * self.pixel_m / 1000.0
        east_km = cen_c * self.pixel_m / 1000.0 - half_km
        lat, lon = offset_latlon(self.centre[0], self.centre[1], east_km, north_km)
        area_m2 = cnt[br, bc] * self.pixel_m**2
        centre_r = np.clip(cen_r.astype(int), 0, s - 1)
        centre_c = np.clip(cen_c.astype(int), 0, s - 1)
        prior = np.where(prev_green[br, bc] >= 0.5 * cnt[br, bc], "Yeşil alan / tarım", "Boş arazi")
        sites = pd.DataFrame({
            "nokta_id": "",
            "mahalle": np.array(NEIGHBOURHOODS)[self._nb[centre_r, centre_c]],
            "enlem": lat.round(5), "boylam": lon.round(5),
            "kacak_alan_m2": area_m2.round(0),
            "onceki_arazi": prior,
            "guven": np.clip(rng.normal(0.87, 0.06, br.size), 0.55, 0.99).round(2),
        })
        px_area = self.pixel_m**2
        sites["oncelik"] = np.select(
            [sites["kacak_alan_m2"] >= 28 * px_area, sites["kacak_alan_m2"] >= 18 * px_area],
            ["Yüksek", "Orta"], default="Düşük")
        sites = sites.sort_values("kacak_alan_m2", ascending=False).reset_index(drop=True)
        sites["nokta_id"] = [f"KY-{i + 1:03d}" for i in range(len(sites))]

        sw, ne = offset_latlon(self.centre[0], self.centre[1], -half_km, -half_km), offset_latlon(
            self.centre[0], self.centre[1], half_km, half_km)
        rects: List[Tuple[str, LatLon, LatLon]] = []
        step = 2.0 * half_km / 4.0
        for i, name in enumerate(NEIGHBOURHOODS):
            r_i, c_i = divmod(i, 4)
            s_lat, w_lon = offset_latlon(self.centre[0], self.centre[1],
                                         -half_km + c_i * step, half_km - (r_i + 1) * half_km)
            n_lat, e_lon = offset_latlon(self.centre[0], self.centre[1],
                                         -half_km + (c_i + 1) * step, half_km - r_i * half_km)
            rects.append((name, (float(s_lat), float(w_lon)), (float(n_lat), float(e_lon))))
        years = list(range(self.YEAR_MIN, self.YEAR_MAX + 1))
        return MunicipalityDataset(
            t0=t0, t1=t1, centre=self.centre,
            bounds=((float(sw[0]), float(sw[1])), (float(ne[0]), float(ne[1]))),
            landcover_t0=lc0, landcover_t1=lc1,
            rgba_t0=render_landcover_rgba(lc0, make_rng(self.seed, 44, t0)),
            rgba_t1=render_landcover_rgba(lc1, make_rng(self.seed, 44, t1)),
            sites=sites, stats=self._stats(years), neighbourhood_rects=rects,
        )

    @staticmethod
    def summarize(ds: MunicipalityDataset) -> Dict[str, object]:
        """Belediye KPI'larını hesaplar.

        Args:
            ds: ``simulate`` çıktısı.

        Returns:
            Kaçak yapı sayısı/alanı, yapılaşma artışı, yeşil alan kaybı ve en riskli mahalle.
        """
        st = ds.stats
        a, b = st[st["yil"] == ds.t0], st[st["yil"] == ds.t1]
        green0, green1 = float(a["yesil_ha"].sum()), float(b["yesil_ha"].sum())
        sites = ds.sites
        worst = (sites.groupby("mahalle")["kacak_alan_m2"].sum().idxmax() if len(sites) else "—")
        return {
            "site_count": float(len(sites)),
            "illegal_m2": float(sites["kacak_alan_m2"].sum()),
            "high_priority": float((sites["oncelik"] == "Yüksek").sum()) if len(sites) else 0.0,
            "built_growth_ha": float(b["yapili_ha"].sum() - a["yapili_ha"].sum()),
            "green_loss_ha": green0 - green1,
            "green_loss_pct": 100.0 * (green0 - green1) / max(green0, 1e-9),
            "mean_infra": float(b["altyapi_yogunlugu"].mean()),
            "worst_neighbourhood": worst,
        }


# =========================================================================== #
# 4) AFET / KURTARMA — Kriz Yönetimi
# =========================================================================== #
PIXEL_AREA_HA: float = 14.06  # 375 m x 375 m (VIIRS I-band) piksel alanı
N_PATROL_LEGS: int = 5  # her saha ekibinin sırayla ziyaret ettiği yerleşim sayısı
_SETTLEMENT_NAMES: Tuple[str, ...] = (
    "Karaçam", "Çamlıdere", "Yeşilova", "Bozburun", "Kızılkaya",
    "Akyaka", "Delikyol", "Söğütlü", "Gökbel", "Taşlıca",
)
_ASSEMBLY_NAMES: Tuple[str, ...] = (
    "Toplanma A — Spor Kompleksi", "Toplanma B — Okul Bahçesi", "Toplanma C — Açık Otopark",
)
_TEAM_UNITS: Tuple[str, ...] = (
    "İtfaiye", "AFAD", "Orman Yangın", "112 Sağlık", "Jandarma",
    "İtfaiye", "Orman Yangın", "AFAD", "112 Sağlık",
)
RISK_LEVELS: Tuple[str, ...] = ("Düşük", "Orta", "Yüksek", "Kritik")
_AFFECTED_SHARE: Dict[str, float] = {"Düşük": 0.0, "Orta": 0.15, "Yüksek": 0.55, "Kritik": 1.0}


@dataclass
class RescueScenario:
    """Zamandan bağımsız (sabit) kriz senaryosu.

    Attributes:
        centre: Harita merkezi.
        start: Senaryo başlangıç zamanı (UTC+3).
        wind_bearing_deg: Rüzgârın estiği yön (kuzeyden saat yönü, derece).
        horizon_h: Simülasyon ufku (saat).
        hotspots: Tüm termal anomaliler (ateşlenme / sönme saatleri ile).
        settlements: Yerleşim tablosu.
        assembly: Toplanma alanları tablosu.
        teams: Saha ekipleri (üs, hedef, hareket parametreleri).
        route_paths: Yerleşim → ``(N, 2)`` rota köşe noktaları (enlem, boylam).
        route_assembly: Yerleşim → atanmış toplanma alanı adı.
    """

    centre: LatLon
    start: dt.datetime
    wind_bearing_deg: float
    horizon_h: int
    hotspots: pd.DataFrame
    settlements: pd.DataFrame
    assembly: pd.DataFrame
    teams: pd.DataFrame
    route_paths: Dict[str, np.ndarray]
    route_assembly: Dict[str, str]


@dataclass
class RescueSnapshot:
    """Belirli bir saatteki kriz durumu.

    Attributes:
        t_hours: Başlangıçtan geçen saat.
        timestamp: Anlık zaman damgası (UTC+3).
        active: Aktif termal anomaliler.
        heat_points: ``[enlem, boylam, ağırlık]`` ısı haritası noktaları.
        settlements: Risk sınıfı ve etkilenen nüfus eklenmiş yerleşim tablosu.
        routes: Tahliye rotaları (durum, uzunluk, ETA ile).
        teams: Anlık saha ekibi konumları ve durumları.
        kpis: Acil durum KPI'ları.
    """

    t_hours: float
    timestamp: dt.datetime
    active: pd.DataFrame
    heat_points: List[List[float]]
    settlements: pd.DataFrame
    routes: pd.DataFrame
    teams: pd.DataFrame
    kpis: Dict[str, float]


def _path_length_km(path: np.ndarray) -> float:
    """Köşe noktalarından oluşan yolun toplam uzunluğunu hesaplar.

    Args:
        path: ``(N, 2)`` enlem-boylam dizisi.

    Returns:
        Uzunluk (km).
    """
    return float(haversine_km(path[:-1, 0], path[:-1, 1], path[1:, 0], path[1:, 1]).sum())


def _min_distance_km(points: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Her nokta için ``(lat, lon)`` kümesine en yakın uzaklığı (km) bulur.

    Args:
        points: ``(N, 2)`` sorgu noktaları (enlem, boylam).
        lat: Hedef kümenin enlemleri.
        lon: Hedef kümenin boylamları.

    Returns:
        ``(N,)`` en küçük uzaklıklar; hedef küme boşsa ``inf``.
    """
    if lat.size == 0:
        return np.full(points.shape[0], np.inf)
    d = haversine_km(points[:, 0:1], points[:, 1:2], lat[None, :], lon[None, :])
    return d.min(axis=1)


class RescueSimulator:
    """Termal anomali (VIIRS/MODIS benzeri), nüfus riski, tahliye ve ekip simülasyonu."""

    def __init__(
        self,
        seed: int = 2026,
        centre: LatLon = (37.0200, 28.3100),
        wind_bearing_deg: float = 55.0,
        horizon_h: int = 72,
        n_hotspots: int = 1100,
    ) -> None:
        """Simülatörü kurar.

        Args:
            seed: Tohum değeri.
            centre: Harita merkezi.
            wind_bearing_deg: Rüzgârın estiği yön (derece).
            horizon_h: Simülasyon ufku (saat).
            n_hotspots: Toplam termal anomali piksel sayısı.
        """
        self.seed = seed
        self.centre = centre
        self.wind = wind_bearing_deg
        self.horizon_h = horizon_h
        self.n_hotspots = n_hotspots

    def _frame(self, along: np.ndarray, cross: np.ndarray, origin_km: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Rüzgâr eksenli (ilerleme, yanal) km koordinatlarını enlem-boylama çevirir.

        Args:
            along: Rüzgâr yönündeki mesafe (km).
            cross: Rüzgâra dik (sağa pozitif) mesafe (km).
            origin_km: Merkeze göre eksen orijini ``(doğu, kuzey)`` (km).

        Returns:
            ``(enlem, boylam)`` dizileri.
        """
        th = math.radians(self.wind)
        east = origin_km[0] + along * math.sin(th) + cross * math.cos(th)
        north = origin_km[1] + along * math.cos(th) - cross * math.sin(th)
        return offset_latlon(self.centre[0], self.centre[1], east, north)

    def build_scenario(self) -> RescueScenario:
        """Sabit senaryoyu (termal anomaliler, yerleşimler, rotalar, ekipler) üretir.

        Returns:
            ``RescueScenario``.
        """
        rng = make_rng(self.seed, 51)
        origin = (-3.2, -2.4)
        n_main = int(self.n_hotspots * 0.88)
        n_spot = self.n_hotspots - n_main

        u1 = rng.gamma(2.2, 2.0, n_main)
        v1 = rng.normal(0.0, 0.55 + 0.20 * u1)
        u2 = 7.5 + rng.gamma(2.0, 0.8, n_spot)
        v2 = 2.6 + rng.normal(0.0, 0.55 + 0.12 * (u2 - 7.5), n_spot)
        along, cross = np.concatenate([u1, u2]), np.concatenate([v1, v2])
        lat, lon = self._frame(along, cross, origin)
        ignite = np.concatenate([
            along[:n_main] / 0.24 + rng.normal(0.0, 1.5, n_main),
            14.0 + (along[n_main:] - 7.5) / 0.24 + rng.normal(0.0, 1.2, n_spot),
        ])
        ignite = np.clip(ignite, 0.0, None)
        burnout = ignite + rng.lognormal(math.log(9.0), 0.45, self.n_hotspots)
        is_modis = rng.random(self.n_hotspots) < 0.30
        frp = rng.lognormal(math.log(14.0), 0.70, self.n_hotspots) * np.where(is_modis, 1.8, 1.0)
        bright = np.clip(rng.normal(338.0, 14.0, self.n_hotspots) + 6.0 * np.log(frp / 14.0), 305.0, 400.0)
        conf = np.select([bright > 345.0, bright > 325.0], ["Yüksek", "Nominal"], default="Düşük")
        hot = pd.DataFrame({
            "enlem": lat.round(5), "boylam": lon.round(5), "frp_mw": frp.round(1),
            "parlaklik_k": bright.round(1), "kaynak": np.where(is_modis, "MODIS-1km", "VIIRS-375m"),
            "guven": conf, "ateslenme_saat": ignite.round(2), "sonme_saat": burnout.round(2),
        }).sort_values("ateslenme_saat").reset_index(drop=True)

        # Yerleşimler: merkez çevresinde 2.5–10 km halkada.
        ns = len(_SETTLEMENT_NAMES)
        ang = rng.uniform(0.0, 2.0 * np.pi, ns)
        rad = rng.uniform(2.5, 10.0, ns)
        s_lat, s_lon = offset_latlon(self.centre[0], self.centre[1], rad * np.cos(ang), rad * np.sin(ang))
        pop = np.clip(np.round(rng.lognormal(math.log(900.0), 0.85, ns), -1), 90, 9000).astype(int)
        settlements = pd.DataFrame({"yerlesim": _SETTLEMENT_NAMES, "enlem": s_lat.round(5),
                                    "boylam": s_lon.round(5), "nufus": pop})

        a_lat, a_lon = self._frame(np.array([-3.0, 3.0, -6.5]), np.array([8.0, -9.0, -2.0]), origin)
        assembly = pd.DataFrame({"ad": _ASSEMBLY_NAMES, "enlem": np.asarray(a_lat).round(5),
                                 "boylam": np.asarray(a_lon).round(5),
                                 "kapasite": np.array([4500, 3200, 2600])})

        # Rotalar: yerleşim → en uygun toplanma alanı (yangın ayak izine uzak olanlar tercih edilir).
        pts_s = settlements[["enlem", "boylam"]].to_numpy()
        pts_a = assembly[["enlem", "boylam"]].to_numpy()
        d_sa = haversine_km(pts_s[:, 0:1], pts_s[:, 1:2], pts_a[None, :, 0], pts_a[None, :, 1])
        a_fire = _min_distance_km(pts_a, hot["enlem"].to_numpy(), hot["boylam"].to_numpy())
        cost = d_sa + 0.8 * np.clip(6.0 - a_fire, 0.0, None)[None, :]
        pick = cost.argmin(axis=1)
        t = np.linspace(0.0, 1.0, 24)[:, None]
        paths: Dict[str, np.ndarray] = {}
        for i, name in enumerate(_SETTLEMENT_NAMES):
            p0, p1 = pts_s[i], pts_a[pick[i]]
            line = p0[None, :] + (p1 - p0)[None, :] * t
            d = p1 - p0
            normal = np.array([-d[1], d[0]]) / (np.hypot(d[0], d[1]) + 1e-12)
            amp = rng.normal(0.0, 0.004)
            wiggle = amp * np.sin(np.pi * t) * (1.0 + 0.4 * np.sin(3.0 * np.pi * t))
            paths[name] = line + wiggle * normal[None, :]

        # Ekipler: üssü toplanma alanları çevresinde, hedefi en çok tehdit altındaki yerleşimler.
        d_hot = _min_distance_km(pts_s, hot["enlem"].to_numpy(), hot["boylam"].to_numpy())
        order = np.argsort(d_hot)
        n_t = len(_TEAM_UNITS)
        n_s = len(order)
        seq = np.array([[order[(i + off) % n_s] for off in range(0, 2 * N_PATROL_LEGS, 2)] for i in range(n_t)])
        base_idx = rng.integers(0, len(assembly), n_t)
        b_lat = pts_a[base_idx, 0] + rng.normal(0.0, 0.006, n_t)
        b_lon = pts_a[base_idx, 1] + rng.normal(0.0, 0.006, n_t)
        names_arr = settlements["yerlesim"].to_numpy()
        teams = pd.DataFrame({
            "ekip": [f"EKP-{i + 1:02d}" for i in range(n_t)],
            "birim": _TEAM_UNITS,
            "uss_enlem": b_lat.round(5), "uss_boylam": b_lon.round(5),
            "cikis_saat": np.sort(rng.uniform(0.0, 6.0, n_t)).round(1),
            "hiz_kmh": np.clip(rng.normal(24.0, 4.0, n_t), 14.0, 40.0).round(0),
            "bekleme_saat": rng.uniform(8.0, 14.0, n_t).round(1),
        })
        for leg in range(N_PATROL_LEGS):
            teams[f"hedef_{leg + 1}"] = names_arr[seq[:, leg]]
            teams[f"enlem_{leg + 1}"] = pts_s[seq[:, leg], 0]
            teams[f"boylam_{leg + 1}"] = pts_s[seq[:, leg], 1]
        return RescueScenario(
            centre=self.centre, start=dt.datetime(2026, 8, 14, 6, 0, tzinfo=TR_TZ),
            wind_bearing_deg=self.wind, horizon_h=self.horizon_h, hotspots=hot,
            settlements=settlements, assembly=assembly, teams=teams, route_paths=paths,
            route_assembly={n: str(assembly["ad"].iloc[pick[i]]) for i, n in enumerate(_SETTLEMENT_NAMES)},
        )

    def snapshot(self, sc: RescueScenario, t_hours: float) -> RescueSnapshot:
        """Belirli bir saat için kriz durumunu hesaplar.

        Risk sınıfı kuralı (en yakın aktif anomaliye uzaklık ``d`` ve rüzgâr altı faktörü ``w``):
        Kritik: d < 2 km veya (d < 5 km ve w > 0.75); Yüksek: d < 4 km veya (d < 8 km ve w > 0.75);
        Orta: d < 8 km; aksi hâlde Düşük.

        Args:
            sc: ``build_scenario`` çıktısı.
            t_hours: Başlangıçtan geçen saat (0 – ufuk; kesirli olabilir).

        Returns:
            ``RescueSnapshot``.
        """
        t = float(np.clip(t_hours, 0, sc.horizon_h))
        hot = sc.hotspots
        ever = hot["ateslenme_saat"].to_numpy() <= t
        act_mask = ever & (hot["sonme_saat"].to_numpy() > t)
        active = hot.loc[act_mask].reset_index(drop=True)
        a_lat, a_lon = active["enlem"].to_numpy(), active["boylam"].to_numpy()

        weights = np.clip(active["frp_mw"].to_numpy() / 40.0, 0.15, 1.0)
        heat = [[float(a), float(b), float(w)] for a, b, w in zip(a_lat, a_lon, weights)]

        st = sc.settlements.copy()
        pts = st[["enlem", "boylam"]].to_numpy()
        dist = _min_distance_km(pts, a_lat, a_lon)
        if a_lat.size:
            c_lat, c_lon = float(a_lat.mean()), float(a_lon.mean())
            brg = np.arctan2((pts[:, 1] - c_lon) * math.cos(math.radians(c_lat)), pts[:, 0] - c_lat)
            downwind = 0.5 * (1.0 + np.cos(brg - math.radians(sc.wind_bearing_deg)))
        else:
            downwind = np.zeros(len(st))
        risk = np.select(
            [(dist < 2.0) | ((dist < 5.0) & (downwind > 0.75)),
             (dist < 4.0) | ((dist < 8.0) & (downwind > 0.75)),
             dist < 8.0],
            ["Kritik", "Yüksek", "Orta"], default="Düşük")
        share = np.array([_AFFECTED_SHARE[r] for r in risk])
        st["en_yakin_km"] = np.where(np.isfinite(dist), dist, 99.0).round(2)
        st["ruzgar_alti"] = downwind.round(2)
        st["risk"] = risk
        st["etkilenen_nufus"] = np.round(st["nufus"].to_numpy() * share).astype(int)

        rows: List[dict] = []
        for _, r in st.iterrows():
            if r["risk"] not in ("Kritik", "Yüksek"):
                continue
            path = sc.route_paths[r["yerlesim"]]
            dense = np.concatenate([path[:-1] + (path[1:] - path[:-1]) * f for f in (0.0, 0.5)]
                                   + [path[-1:]])
            d_min = float(_min_distance_km(dense, a_lat, a_lon).min()) if a_lat.size else 99.0
            length = _path_length_km(path)
            status = "Kapalı" if d_min < 0.8 else ("Riskli" if d_min < 2.5 else "Açık")
            speed = 28.0 if status == "Açık" else 18.0
            rows.append({"yerlesim": r["yerlesim"], "toplanma": sc.route_assembly[r["yerlesim"]],
                         "uzunluk_km": round(length, 1), "sure_dk": round(length / speed * 60.0),
                         "durum": status, "min_yangin_km": round(d_min, 2),
                         "nufus": int(r["nufus"])})
        routes = pd.DataFrame(rows, columns=["yerlesim", "toplanma", "uzunluk_km", "sure_dk",
                                             "durum", "min_yangin_km", "nufus"])

        tm = sc.teams.copy()
        cur_lat, cur_lon = tm["uss_enlem"].to_numpy(float), tm["uss_boylam"].to_numpy(float)
        speed, dwell = tm["hiz_kmh"].to_numpy(float), tm["bekleme_saat"].to_numpy(float)
        start_leg = tm["cikis_saat"].to_numpy(float)
        pos_lat, pos_lon = cur_lat.copy(), cur_lon.copy()
        state = np.where(t < start_leg, "Konuşlanıyor", "Sahada").astype(object)
        target = np.full(len(tm), "—", dtype=object)
        progress = np.zeros(len(tm))
        for leg in range(1, N_PATROL_LEGS + 1):
            t_lat, t_lon = tm[f"enlem_{leg}"].to_numpy(float), tm[f"boylam_{leg}"].to_numpy(float)
            travel = np.clip(haversine_km(cur_lat, cur_lon, t_lat, t_lon), 0.1, None) / speed
            frac = np.clip((t - start_leg) / travel, 0.0, 1.0)
            moving = (t >= start_leg) & (frac < 1.0)
            begun = t >= start_leg
            pos_lat = np.where(begun, cur_lat + (t_lat - cur_lat) * frac, pos_lat)
            pos_lon = np.where(begun, cur_lon + (t_lon - cur_lon) * frac, pos_lon)
            state = np.where(moving, "Yolda", np.where(begun, "Sahada", state))
            target = np.where(begun, tm[f"hedef_{leg}"].to_numpy(object), target)
            progress = np.where(begun, frac * 100.0, progress)
            cur_lat, cur_lon = t_lat, t_lon
            start_leg = start_leg + travel + dwell
        tm["enlem"], tm["boylam"] = np.round(pos_lat, 5), np.round(pos_lon, 5)
        tm["durum"], tm["hedef"] = state, target
        tm["ilerleme_pct"] = progress.round(0)

        kpis = {
            "affected_pop": float(st["etkilenen_nufus"].sum()),
            "total_pop": float(st["nufus"].sum()),
            "critical_zones": float((st["risk"] == "Kritik").sum()),
            "high_zones": float((st["risk"] == "Yüksek").sum()),
            "active_hotspots": float(len(active)),
            "frp_total_mw": float(active["frp_mw"].sum()) if len(active) else 0.0,
            "burned_ha": float(ever.sum()) * PIXEL_AREA_HA,
            "routes_open": float((routes["durum"] == "Açık").sum()),
            "routes_risky": float((routes["durum"] == "Riskli").sum()),
            "routes_closed": float((routes["durum"] == "Kapalı").sum()),
            "teams_on_site": float((tm["durum"] == "Sahada").sum()),
            "teams_total": float(len(tm)),
        }
        return RescueSnapshot(t, sc.start + dt.timedelta(hours=t), active, heat, st, routes, tm, kpis)


__all__ = [
    "ClaimRules", "DECISION_LABELS", "InsuranceSimulator", "InsuranceDataset",
    "AgricultureSimulator", "AgricultureDataset", "CropProfile", "CROPS", "StressPeriod",
    "MunicipalitySimulator", "MunicipalityDataset", "NEIGHBOURHOODS", "LC_LABELS",
    "RescueSimulator", "RescueScenario", "RescueSnapshot", "RISK_LEVELS",
    "SEVERITY_LABELS", "DNBR_EDGES", "ParcelGeometry", "ParcelGrid",
    "format_tr", "haversine_km", "parcels_to_geojson", "classify_dnbr",
]
