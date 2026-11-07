"""
yield_model.py — Çok değişkenli rekolte (verim) tahmini: geçmiş verim + meteoroloji + ürün türü + fenolojik evre + NDVI
======================================================================================================================

Rekolte yalnızca NDVI'dan çıkarılmaz. Model, ürün türüne özgü bir **ridge regresyonudur** ve şu bilgileri birlikte kullanır:

* **Geçmiş verim** (parselin önceki sezon/sezonlar verimi) — toprak ve yönetim etkisini taşır.
* **Meteoroloji**: sezon yağışı, büyüme derece-günü (GDD), sıcak gün sayısı, su açığı (yağış − ET0).
* **Fenolojik evre**: tahmin tarihi ürünün takvimine göre *erken* (vejetatif), *tepe* ya da *olgun* evredir; her evrenin kendi
  özellik kümesi ve modeli vardır (erken tahminin belirsizliği doğal olarak daha geniştir).
* **NDVI özellikleri**: tepe, sezon ortalaması, son değer.

Belirsizlik ve doğruluk
-----------------------
* **Aralık**: leave-one-season-out (LOSO) mutlak hatalarından *conformal* yarı-genişlik; aynı sezondaki parseller hava koşullarını
  paylaştığından sezon bazlı ayrılır. Takas edilebilirlik varsayımı altında hedef kapsama ≈ ``1 − alpha``.
* **Doğruluk**: LOSO ve son sezonları dışarıda bırakan zaman-bazlı sınama (RMSE, MAE, MAPE, R², sapma) + üç model karşılaştırması
  (yalnız geçmiş ortalama / yalnız NDVI / bu model).

Dürüstlük notu
--------------
Bu depoda **gerçek verim kaydı yoktur**. ``synthetic_history`` ile üretilen geçmiş, burada *tanımlanmış bir üretim sürecinden*
gelir; raporlanan hata ve kapsama değerleri o sürecin ürünüdür ve **sahadaki doğruluğu göstermez**. Yöntemi ve arayüzü
gösterir. Gerçek doğruluk için kullanıcı kendi geçmiş verim tablosunu yüklemelidir (``load_history_csv``); model o veriyle
yeniden eğitilir ve aynı ölçütler gerçek veri üzerinde hesaplanır. Az veriyle (< 3 sezon ya da < 30 gözlem) eğitim reddedilir.

Geliştirici: Elif Turut
"""

from __future__ import annotations

import datetime as dt
import functools
import io
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import data_engine as de

__all__ = ["STAGES", "STAGE_LABELS", "FEATURES", "FEATURE_LABELS", "YieldError", "Metrics", "YieldModel", "YieldEstimate", "crop_profile",
           "stage_for", "stage_cutoff_doy", "stage_date", "latest_usable_stage", "season_features", "synthetic_history", "fit_yield_model", "predict", "evaluate_holdout",
           "compare_models", "demo_model_pool", "history_template_csv", "load_history_csv", "estimate_from_series", "SYNTHETIC_NOTE", "MIN_SEASONS", "MIN_ROWS"]

STAGES: Tuple[str, ...] = ("erken", "tepe", "olgun")
STAGE_LABELS: Dict[str, str] = {"erken": "Vejetatif (tepe öncesi)", "tepe": "Tepe / çiçeklenme", "olgun": "Olgunlaşma (tepe sonrası)"}
_BASE = ("onceki_verim",)
FEATURES: Dict[str, Tuple[str, ...]] = {
    "erken": _BASE + ("ndvi_son", "yagis_mm", "gdd"),
    "tepe": _BASE + ("ndvi_tepe", "ndvi_ort", "yagis_mm", "gdd", "sicak_gun", "su_acigi_mm"),
    "olgun": _BASE + ("ndvi_tepe", "ndvi_ort", "ndvi_son", "yagis_mm", "gdd", "sicak_gun", "su_acigi_mm"),
}
FEATURE_LABELS: Dict[str, str] = {
    "onceki_verim": "Geçmiş verim (t/ha)", "ndvi_tepe": "NDVI tepe", "ndvi_ort": "NDVI sezon ortalaması", "ndvi_son": "NDVI son değer",
    "yagis_mm": "Sezon yağışı (mm)", "gdd": "Büyüme derece-günü", "sicak_gun": "Sıcak gün (Tmax ≥ 32 °C)", "su_acigi_mm": "Su dengesi (yağış − ET0, mm)",
}
SYNTHETIC_NOTE = ("Geçmiş verim tablosu SENTETİKTİR (tanımlı bir üretim sürecinden). Aşağıdaki hata ve kapsama değerleri o sürecin "
                  "ürünüdür; sahadaki doğruluğu göstermez. Gerçek doğruluk için kendi geçmiş verim tablonuzu yükleyin.")
MIN_SEASONS = 3
MIN_ROWS = 30
_HEAT_C = 32.0


class YieldError(ValueError):
    """Rekolte modeli girdisi ya da eğitimi geçersiz."""


def crop_profile(name: str) -> de.CropProfile:
    """Ürün profili (``data_engine.CROPS``).

    Raises:
        YieldError: Bilinmeyen ürün.
    """
    for c in de.CROPS:
        if c.name == name:
            return c
    raise YieldError(f"Bilinmeyen ürün: {name!r} (geçerli: {[c.name for c in de.CROPS]})")


def stage_cutoff_doy(crop: de.CropProfile, stage: str) -> float:
    """Evrenin eğitimde kullanılan nominal kesme günü (yılın günü)."""
    span = crop.eos - crop.sos
    return {"erken": crop.sos + 0.30 * span, "tepe": crop.sos + 0.60 * span, "olgun": crop.eos}[stage]


def stage_for(crop: de.CropProfile, doy: float) -> str:
    """Son gözlem gününe (yılın günü) göre fenolojik evre: sezonun %45'ine kadar erken, %80'ine kadar tepe, sonrası olgun."""
    span = crop.eos - crop.sos
    if doy < crop.sos + 0.45 * span:
        return "erken"
    return "tepe" if doy < crop.sos + 0.80 * span else "olgun"


def stage_date(crop: de.CropProfile, stage: str, year: int) -> dt.date:
    """Evrenin nominal kesme günü (tarih)."""
    return dt.date(year, 1, 1) + dt.timedelta(days=int(round(stage_cutoff_doy(crop, stage))) - 1)


def latest_usable_stage(crop: de.CropProfile, last_obs: dt.date) -> Optional[str]:
    """Son gözlemin ulaştığı en ileri evre (nominal kesme günü ≤ son gözlem); henüz hiçbirine ulaşılmadıysa ``None``.

    Özellikler her zaman evrenin nominal kesme gününde hesaplanır: eğitimdeki dağılımla birebir aynı tanım, ileri tarihli
    verinin (ör. sezon sonrası) özellikleri kaydırması önlenir.
    """
    best = None
    for st in STAGES:
        if stage_date(crop, st, last_obs.year) <= last_obs:
            best = st
    return best


# --------------------------------------------------------------------------- #
# Özellikler
# --------------------------------------------------------------------------- #
def season_features(ndvi: pd.DataFrame, weather: pd.DataFrame, crop: de.CropProfile, cutoff: dt.date,
                    prior_yield: float, max_gap_days: int = 20) -> Dict[str, float]:
    """Sezon başından ``cutoff`` gününe kadar olan serilerden özellik vektörü üretir.

    Pencere: ``[SOS − 30 gün, cutoff]`` (SOS = ürün takvimindeki sezon başı).

    Args:
        ndvi: ``tarih`` ve ``ortalama`` sütunlu NDVI serisi.
        weather: ``tarih, yagis_mm, tmax_c, tmin_c, et0_mm`` sütunlu günlük tablo.
        crop: Ürün profili.
        cutoff: Son gözlem günü.
        prior_yield: Geçmiş verim (t/ha).
        max_gap_days: Son NDVI gözlemi ile ``cutoff`` arasında izin verilen en büyük boşluk (gün).

    Raises:
        YieldError: Pencerede < 3 NDVI gözlemi, son gözlem çok eski ya da hava günlerinin < %80'i varsa.
    """
    sos = dt.date(cutoff.year, 1, 1) + dt.timedelta(days=int(round(crop.sos)) - 1)
    start = pd.Timestamp(sos - dt.timedelta(days=30))
    end = pd.Timestamp(cutoff)
    nd = ndvi[(ndvi["tarih"] >= start) & (ndvi["tarih"] <= end)].dropna(subset=["ortalama"]).sort_values("tarih")
    if len(nd) < 3:
        raise YieldError(f"Sezon penceresinde (≥ 3) NDVI gözlemi yok: {len(nd)} gözlem.")
    gap = (end - nd["tarih"].iloc[-1]).days
    if gap > max_gap_days:
        raise YieldError(f"Son NDVI gözlemi kesme gününden {gap} gün önce (en çok {max_gap_days} gün); tahmin güvenilmez.")
    days = (nd["tarih"] - nd["tarih"].iloc[0]).dt.days.to_numpy(dtype=float)
    vals = nd["ortalama"].to_numpy(dtype=float)
    span = max(days[-1], 1.0)
    ndvi_ort = float(np.sum((vals[1:] + vals[:-1]) / 2.0 * np.diff(days)) / span)  # yamuk kuralı
    w = weather[(weather["tarih"] >= start) & (weather["tarih"] <= end)]
    need = (end - start).days + 1
    if len(w) < 0.8 * need:
        raise YieldError(f"Sezon penceresinde hava verisi eksik: {len(w)}/{need} gün.")
    base = 5.0 if crop.sos < 120 else 10.0
    hi = np.clip(w["tmax_c"].to_numpy(dtype=float), base, 30.0)
    lo = np.clip(w["tmin_c"].to_numpy(dtype=float), base, 30.0)
    gdd = float(np.nansum((hi + lo) / 2.0 - base))
    precip = float(np.nansum(w["yagis_mm"]))
    et0 = float(np.nansum(w["et0_mm"])) if "et0_mm" in w.columns else math.nan
    return {"onceki_verim": float(prior_yield), "ndvi_tepe": float(vals.max()), "ndvi_ort": ndvi_ort, "ndvi_son": float(vals[-1]),
            "yagis_mm": precip, "gdd": gdd, "sicak_gun": float((w["tmax_c"] >= _HEAT_C).sum()), "su_acigi_mm": precip - et0}


# --------------------------------------------------------------------------- #
# Sentetik geçmiş (yöntem gösterimi)
# --------------------------------------------------------------------------- #
def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def synthetic_history(crop_name: str, n_fields: int = 40, years: Sequence[int] = tuple(range(2014, 2026)), seed: int = 5) -> pd.DataFrame:
    """Tanımlı bir üretim sürecinden geçmiş verim + özellik tablosu (uzun biçim: parsel × sezon × evre).

    Üretim süreci (hepsi varsayımdır): her parselin kalıcı bir toprak etkisi; her yılın hava anomalisi (yağış, sıcaklık); NDVI tepesi
    toprak ve su durumunun kısmi bir göstergesidir; verim, toprak etkisi + su dengesi − çiçeklenme dönemi sıcak günleri + gürültüdür.
    Özellikler, **gerçek tahminde kullanılan aynı işlevle** (``season_features``) günlük serilerden hesaplanır.

    Args:
        crop_name: Ürün adı.
        n_fields: Parsel sayısı.
        years: Sezon yılları.
        seed: Rastgelelik tohumu.

    Returns:
        Sütunlar: ``parsel_id, sezon, urun, asama, verim_t_ha`` + ``FEATURES`` birleşimi.
    """
    crop = crop_profile(crop_name)
    rng = np.random.default_rng([seed, int(crop.sos), 321])
    soil = rng.normal(0.0, 0.10, n_fields)
    rows = []
    prev: Dict[int, float] = {}
    for year in years:
        days = pd.date_range(dt.date(year, 1, 1), dt.date(year, 12, 31))
        doy = days.dayofyear.to_numpy().astype(float)
        wet = rng.normal(0.0, 1.0)
        warm = rng.normal(0.0, 1.0)
        tmax = 18.0 + 12.0 * np.sin(2 * np.pi * (doy - 105) / 365.0) + 1.3 * warm + rng.normal(0, 2.4, len(days))
        tmin = tmax - 10.0 + rng.normal(0, 1.2, len(days))
        p_wet = np.clip(0.26 - 0.12 * np.sin(2 * np.pi * (doy - 105) / 365.0) + 0.05 * wet, 0.04, 0.6)
        rain = np.where(rng.random(len(days)) < p_wet, rng.gamma(1.5, 4.2, len(days)), 0.0)
        et0 = np.clip(1.0 + 0.2 * tmax, 0.3, None)
        weather = pd.DataFrame({"tarih": days, "yagis_mm": rain, "tmax_c": tmax, "tmin_c": tmin, "et0_mm": et0})
        # sezon su dengesi ve çiçeklenme sıcak günleri (sezonun %50–%70'i)
        sos_d, eos_d = int(crop.sos), int(crop.eos)
        in_season = (doy >= sos_d - 30) & (doy <= eos_d)
        water_bal = float(rain[in_season].sum() - et0[in_season].sum())
        flower = (doy >= crop.sos + 0.5 * (crop.eos - crop.sos)) & (doy <= crop.sos + 0.7 * (crop.eos - crop.sos))
        for f in range(n_fields):
            heat_f = float((tmax[flower] + rng.normal(0, 1.0) >= _HEAT_C).sum())
            mult = 1.0 + soil[f] + 0.16 * math.tanh(water_bal / 250.0 + 0.5) * crop.water_need - 0.012 * heat_f + rng.normal(0, 0.055)
            verim = max(0.3, crop.base_yield * mult)
            peak = 0.88 * crop.peak_ndvi + 0.35 * soil[f] + 0.05 * math.tanh(water_bal / 250.0) + rng.normal(0, 0.025)
            curve = crop.base_ndvi + (peak - crop.base_ndvi) * (_sig((doy - crop.sos) / crop.rise_k) - _sig((doy - crop.eos) / crop.fall_k))
            obs_idx = np.arange(int(crop.sos) - 40, int(crop.eos) + 20, 5)
            obs = pd.DataFrame({"tarih": days[obs_idx], "ortalama": np.clip(curve[obs_idx] + rng.normal(0, 0.02, len(obs_idx)), 0.0, 0.98)})
            prior = prev.get(f, crop.base_yield * (1 + soil[f]) + rng.normal(0, 0.05 * crop.base_yield))
            for stage in STAGES:
                cut = dt.date(year, 1, 1) + dt.timedelta(days=int(round(stage_cutoff_doy(crop, stage))) - 1)
                feats = season_features(obs, weather, crop, cut, prior)
                rows.append({"parsel_id": f"P{f + 1:03d}", "sezon": year, "urun": crop.name, "asama": stage, "verim_t_ha": verim, **feats})
            prev[f] = verim
    df = pd.DataFrame(rows)
    # İlk sezonun "geçmiş verimi" gerçek bir kayıt değil, üretilmiş bir tahmindir; o sezonu eğitimden çıkar.
    first = df.groupby("parsel_id")["sezon"].transform("min")
    return df[df["sezon"] > first].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Metrics:
    """Doğruluk ölçütleri (yalnızca örnek-dışı tahminlerden).

    Attributes:
        rmse: Kök ortalama kare hata (t/ha).
        mae: Ortalama mutlak hata (t/ha).
        mape: Ortalama mutlak yüzde hata (%).
        r2: Belirlilik katsayısı (örnek-dışı).
        bias: Ortalama (tahmin − gerçek) (t/ha).
        n: Gözlem sayısı.
        coverage: Aralık kapsama oranı (%); yoksa ``nan``.
    """

    rmse: float
    mae: float
    mape: float
    r2: float
    bias: float
    n: int
    coverage: float = math.nan


def _metrics(y: np.ndarray, p: np.ndarray, coverage: float = math.nan) -> Metrics:
    err = p - y
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return Metrics(float(np.sqrt((err ** 2).mean())), float(np.abs(err).mean()), float(100.0 * np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-9))),
                   1.0 - float((err ** 2).sum()) / ss_tot if ss_tot > 0 else math.nan, float(err.mean()), int(len(y)), coverage)


def _ridge(X: np.ndarray, y: np.ndarray, lam: float) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Standartlaştırılmış ridge (kesişim cezasız). Dönüş: ``(katsayılar, kesişim, ortalama, std)``."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (X - mu) / sd
    ym = float(y.mean())
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    coef = np.linalg.solve(A, Z.T @ (y - ym))
    return coef, ym, mu, sd


@dataclass(frozen=True)
class YieldModel:
    """Eğitilmiş ürün+evre modeli.

    Attributes:
        crop: Ürün adı.
        stage: Evre.
        features: Özellik adları (sıralı).
        mu: Eğitim ortalamaları.
        sd: Eğitim std'leri.
        coef: Standartlaştırılmış katsayılar.
        intercept: Kesişim (eğitim verimi ortalaması).
        lam: Ridge cezası.
        alpha: Aralık hata düzeyi (0,10 → %90 aralık).
        half_width: Conformal yarı-genişlik (t/ha).
        n_train: Eğitim satırı.
        n_seasons: Sezon sayısı.
        loso: Leave-one-season-out örnek-dışı ölçütler.
        synthetic: Eğitim verisi sentetik mi.
        lo: Özellik başına eğitim alt sınırı.
        hi: Özellik başına eğitim üst sınırı.
    """

    crop: str
    stage: str
    features: Tuple[str, ...]
    mu: np.ndarray
    sd: np.ndarray
    coef: np.ndarray
    intercept: float
    lam: float
    alpha: float
    half_width: float
    n_train: int
    n_seasons: int
    loso: Metrics
    synthetic: bool
    lo: np.ndarray
    hi: np.ndarray


def _prep(df: pd.DataFrame, crop: str, stage: str) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    if stage not in STAGES:
        raise YieldError(f"Bilinmeyen evre: {stage!r}")
    feats = FEATURES[stage]
    need = {"parsel_id", "sezon", "urun", "verim_t_ha"} | set(feats)
    miss = need - set(df.columns)
    if miss:
        raise YieldError(f"Geçmiş tabloda eksik sütun: {sorted(miss)}")
    d = df[(df["urun"] == crop) & ((df["asama"] == stage) if "asama" in df.columns else True)].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=list(feats) + ["verim_t_ha"])
    if (d["verim_t_ha"] <= 0).any():
        raise YieldError("Verim değerleri pozitif olmalı.")
    if d["sezon"].nunique() < MIN_SEASONS or len(d) < MIN_ROWS:
        raise YieldError(f"Eğitim için en az {MIN_SEASONS} sezon ve {MIN_ROWS} gözlem gerekir (var: {d['sezon'].nunique()} sezon, {len(d)} gözlem).")
    return d.reset_index(drop=True), feats


def _loso(d: pd.DataFrame, feats: Tuple[str, ...], lam: float) -> np.ndarray:
    pred = np.zeros(len(d))
    X = d[list(feats)].to_numpy(dtype=float)
    y = d["verim_t_ha"].to_numpy(dtype=float)
    for s in d["sezon"].unique():
        te = (d["sezon"] == s).to_numpy()
        coef, ym, mu, sd = _ridge(X[~te], y[~te], lam)
        pred[te] = ym + ((X[te] - mu) / sd) @ coef
    return pred


def _conformal_q(abs_err: np.ndarray, alpha: float) -> float:
    """Bölünmüş-conformal yarı-genişlik: ``⌈(n+1)(1−α)⌉``. sıralı mutlak hata."""
    n = len(abs_err)
    k = min(n, int(math.ceil((n + 1) * (1.0 - alpha))))
    return float(np.sort(abs_err)[k - 1])


def fit_yield_model(df: pd.DataFrame, crop: str, stage: str, lam: float = 3.0, alpha: float = 0.10, synthetic: bool = False,
                    features: Optional[Sequence[str]] = None) -> YieldModel:
    """Ürün + evre modelini eğitir ve LOSO ölçütlerini hesaplar.

    Args:
        df: Geçmiş tablo (``synthetic_history`` ya da ``load_history_csv`` biçimi).
        crop: Ürün adı.
        stage: ``erken``/``tepe``/``olgun``.
        lam: Ridge cezası (> 0).
        alpha: Aralık hata düzeyi (0–0,5).
        synthetic: Veri sentetik mi (arayüz etiketi için).
        features: Özellik alt kümesi (karşılaştırma modelleri için); ``None`` = evrenin tam kümesi.

    Raises:
        YieldError: Veri yetersiz ya da parametre geçersizse.
    """
    if not lam > 0 or not 0 < alpha < 0.5:
        raise YieldError("lam > 0 ve 0 < alpha < 0,5 olmalı.")
    d, feats = _prep(df, crop, stage)
    if features is not None:
        feats = tuple(features)
        missing = [f for f in feats if f not in d.columns]
        if missing:
            raise YieldError(f"Eksik özellik: {missing}")
    X = d[list(feats)].to_numpy(dtype=float)
    y = d["verim_t_ha"].to_numpy(dtype=float)
    pred = _loso(d, feats, lam)
    q = _conformal_q(np.abs(pred - y), alpha)
    coef, ym, mu, sd = _ridge(X, y, lam)
    cover = float(100.0 * np.mean(np.abs(pred - y) <= q))
    return YieldModel(crop, stage, feats, mu, sd, coef, ym, lam, alpha, q, len(d), int(d["sezon"].nunique()), _metrics(y, pred, cover), synthetic,
                      X.min(axis=0), X.max(axis=0))


@dataclass(frozen=True)
class YieldEstimate:
    """Tek bir parsel için tahmin.

    Attributes:
        value: Tahmin (t/ha).
        lo: Alt sınır.
        hi: Üst sınır.
        level: Aralık düzeyi (örn. 0,90).
        stage: Evre.
        contributions: Özellik katkıları (t/ha; ortalama verime göre).
        warnings: Uyarılar.
    """

    value: float
    lo: float
    hi: float
    level: float
    stage: str
    contributions: pd.DataFrame
    warnings: Tuple[str, ...]


def predict(model: YieldModel, feats: Mapping[str, float]) -> YieldEstimate:
    """Özellik vektöründen verim tahmini, aralık ve özellik katkıları.

    Eğitim aralığının dışında kalan özellikler için uyarı verilir (ekstrapolasyon).

    Raises:
        YieldError: Gerekli özellik eksik ya da sonlu değilse.
    """
    miss = [f for f in model.features if f not in feats or not math.isfinite(float(feats[f]))]
    if miss:
        raise YieldError(f"Eksik/geçersiz özellik: {miss}")
    x = np.array([float(feats[f]) for f in model.features])
    z = (x - model.mu) / model.sd
    contrib = model.coef * z
    val = float(model.intercept + contrib.sum())
    warns: List[str] = []
    for f, xi, lo, hi in zip(model.features, x, model.lo, model.hi):
        tol = 0.05 * (hi - lo)  # aralığın %5'i kadar taşma gürültü sayılır
        if xi < lo - tol - 1e-9 or xi > hi + tol + 1e-9:
            warns.append(f"{FEATURE_LABELS.get(f, f)} = {xi:.2f}, eğitim aralığının ({lo:.2f}–{hi:.2f}) dışında: tahmin ekstrapolasyondur.")
    if val <= 0:
        warns.append("Tahmin ≤ 0: girdiler olağandışı; sonuca güvenmeyin.")
    if model.synthetic:
        warns.append("Model SENTETİK geçmiş veriyle eğitildi (yöntem gösterimi).")
    table = pd.DataFrame({"ozellik": [FEATURE_LABELS.get(f, f) for f in model.features], "deger": x, "katki_t_ha": contrib}).sort_values(
        "katki_t_ha", key=np.abs, ascending=False).reset_index(drop=True)
    return YieldEstimate(max(val, 0.0), max(val - model.half_width, 0.0), val + model.half_width, 1.0 - model.alpha, model.stage, table, tuple(warns))


def evaluate_holdout(df: pd.DataFrame, crop: str, stage: str, test_seasons: int = 2, lam: float = 3.0, alpha: float = 0.10) -> Metrics:
    """Zaman-bazlı sınama: son ``test_seasons`` sezon dışarıda; kalan sezonlarla eğit + conformal kalibre et, dışarıdakini tahmin et.

    Returns:
        Test sezonlarında RMSE/MAE/MAPE/R²/sapma ve aralık kapsaması (%).
    """
    d, feats = _prep(df, crop, stage)
    seasons = sorted(d["sezon"].unique())
    if len(seasons) < MIN_SEASONS + test_seasons:
        raise YieldError(f"Zaman-bazlı sınama için en az {MIN_SEASONS + test_seasons} sezon gerekir.")
    test = d["sezon"].isin(seasons[-test_seasons:]).to_numpy()
    tr = d[~test]
    m = fit_yield_model(tr.assign(urun=crop, asama=stage), crop, stage, lam, alpha)
    X = d.loc[test, list(feats)].to_numpy(dtype=float)
    y = d.loc[test, "verim_t_ha"].to_numpy(dtype=float)
    p = m.intercept + ((X - m.mu) / m.sd) @ m.coef
    return _metrics(y, p, float(100.0 * np.mean(np.abs(p - y) <= m.half_width)))


def compare_models(df: pd.DataFrame, crop: str, stage: str, lam: float = 3.0) -> pd.DataFrame:
    """Üç modeli LOSO örnek-dışı hatayla karşılaştırır: yalnız geçmiş verim, yalnız NDVI, çok değişkenli.

    Returns:
        Sütunlar: ``model, rmse, mae, mape, r2``.
    """
    ndvi_feat = "ndvi_son" if stage == "erken" else "ndvi_tepe"
    specs = [("Yalnız geçmiş verim", ("onceki_verim",)), (f"Yalnız NDVI ({FEATURE_LABELS[ndvi_feat]})", (ndvi_feat,)),
             ("Çok değişkenli (geçmiş verim + hava + NDVI)", FEATURES[stage])]
    out = []
    for name, feats in specs:
        m = fit_yield_model(df, crop, stage, lam, features=feats).loso
        out.append((name, m.rmse, m.mae, m.mape, m.r2))
    return pd.DataFrame(out, columns=["model", "rmse", "mae", "mape", "r2"])


@functools.lru_cache(maxsize=6)
def demo_model_pool(crop_name: str) -> Tuple[pd.DataFrame, Dict[str, YieldModel]]:
    """Sentetik geçmişle eğitilmiş üç evre modeli (önbellekli; yalnızca yöntem gösterimi).

    Returns:
        ``(geçmiş tablo, {evre: model})`` — modeller ``synthetic=True`` işaretlidir.
    """
    hist = synthetic_history(crop_name)
    return hist, {st: fit_yield_model(hist, crop_name, st, synthetic=True) for st in STAGES}


# --------------------------------------------------------------------------- #
# Kullanıcının kendi geçmiş verisi
# --------------------------------------------------------------------------- #
_CSV_REQUIRED = ("parsel_id", "sezon", "urun", "verim_t_ha")


def history_template_csv() -> str:
    """Örnek geçmiş-verim CSV şablonu (başlık + bir örnek satır; ``asama`` ve özellik sütunlarıyla)."""
    cols = list(_CSV_REQUIRED) + ["asama"] + [f for f in FEATURES["olgun"]]
    row = ["P001", "2023", "Buğday", "3.9", "olgun", "3.7", "0.71", "0.52", "0.58", "310", "1850", "6", "-40"]
    return ",".join(cols) + "\n" + ",".join(row) + "\n"


def load_history_csv(text: str) -> pd.DataFrame:
    """Kullanıcının geçmiş verim CSV'sini okur ve doğrular.

    ``asama`` sütunu yoksa tüm satırlar ``olgun`` sayılır. Özellik sütunları ``FEATURES`` adlarını taşımalıdır
    (``history_template_csv``).

    Raises:
        YieldError: Sütun eksik, sayısal olmayan değer, ya da ürün/evre geçersizse.
    """
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:  # noqa: BLE001
        raise YieldError(f"CSV okunamadı ({type(exc).__name__}).") from None
    df.columns = [str(c).strip() for c in df.columns]
    miss = [c for c in _CSV_REQUIRED if c not in df.columns]
    if miss:
        raise YieldError(f"CSV'de eksik sütun: {miss}")
    if "asama" not in df.columns:
        df["asama"] = "olgun"
    bad_stage = set(df["asama"].astype(str)) - set(STAGES)
    if bad_stage:
        raise YieldError(f"Geçersiz evre değerleri: {sorted(bad_stage)} (geçerli: {list(STAGES)})")
    known = {c.name for c in de.CROPS}
    bad_crop = set(df["urun"].astype(str)) - known
    if bad_crop:
        raise YieldError(f"Bilinmeyen ürün: {sorted(bad_crop)} (geçerli: {sorted(known)})")
    for col in ["sezon", "verim_t_ha"] + [f for f in FEATURES["olgun"] if f in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["sezon", "verim_t_ha"]].isna().any().any():
        raise YieldError("'sezon' ve 'verim_t_ha' sayısal ve dolu olmalı.")
    return df


# --------------------------------------------------------------------------- #
# Çalıştırmadan (iş akışı çıktısından) tahmin
# --------------------------------------------------------------------------- #
def estimate_from_series(model_pool: Mapping[str, YieldModel], ndvi: pd.DataFrame, weather: pd.DataFrame, crop_name: str,
                         prior_yield: float, min_valid_pct: float = 50.0) -> Tuple[YieldEstimate, Dict[str, float]]:
    """Analiz çalıştırmasının NDVI serisi ve hava tablosundan rekolte tahmini üretir.

    Evre, son *kullanılabilir* sahnenin ulaştığı en ileri evredir (``latest_usable_stage``); özellikler o evrenin nominal kesme
    gününde hesaplanır. Henüz ilk evreye ulaşılmadıysa ya da havuzda o evrenin modeli yoksa hata verilir.

    Args:
        model_pool: Evre → eğitilmiş model.
        ndvi: ``AnalysisRun.series`` (``tarih, ortalama, gecerli_oran``).
        weather: Günlük hava tablosu.
        crop_name: Ürün.
        prior_yield: Parselin geçmiş verimi (t/ha).
        min_valid_pct: Bundan az kullanılabilir pikselli sahneler seri dışı bırakılır.

    Returns:
        ``(tahmin, kullanılan özellikler)``.

    Raises:
        YieldError: Veri yetersizse.
    """
    crop = crop_profile(crop_name)
    nd = ndvi[ndvi["gecerli_oran"] >= min_valid_pct].dropna(subset=["ortalama"])
    if nd.empty:
        raise YieldError("Kullanılabilir NDVI gözlemi yok.")
    last = nd["tarih"].max().date()
    stage = latest_usable_stage(crop, last)
    if stage is None:
        raise YieldError(f"Son gözlem ({last:%d.%m.%Y}) henüz ilk evreye ({STAGE_LABELS['erken']}, "
                         f"{stage_date(crop, 'erken', last.year):%d.%m.%Y}) ulaşmadı; tahmin için daha geç bir tarih gerekir.")
    if stage not in model_pool:
        raise YieldError(f"'{stage}' evresi için eğitilmiş model yok.")
    feats = season_features(nd, weather, crop, stage_date(crop, stage, last.year), prior_yield)
    return predict(model_pool[stage], feats), feats
