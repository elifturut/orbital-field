"""yield_model.py çevrimdışı testleri (bilinen-cevap, sızıntı ve kapsama testleri)."""
import datetime as dt
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data_engine as de  # noqa: E402
import demo_imagery as di  # noqa: E402
import workflow as wf  # noqa: E402
import yield_model as ym  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  [{extra}]" if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)


def raises(fn, needle=""):
    try:
        fn()
    except ym.YieldError as e:
        return needle in str(e), str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"yanlış istisna: {type(e).__name__}: {e}"
    return False, "istisna yok"


D = dt.date
W = ym.crop_profile("Buğday")

print("== Ürün ve evre")
check("bilinmeyen ürün → hata", raises(lambda: ym.crop_profile("Muz"), "Bilinmeyen")[0])
check("kesme günleri sıralı: erken < tepe < olgun", ym.stage_cutoff_doy(W, "erken") < ym.stage_cutoff_doy(W, "tepe") < ym.stage_cutoff_doy(W, "olgun"))
span = W.eos - W.sos
check("evre sınırları: %45 ve %80", [ym.stage_for(W, W.sos + f * span) for f in (0.1, 0.44, 0.46, 0.79, 0.81, 1.2)] == ["erken", "erken", "tepe", "tepe", "olgun", "olgun"])
check("her evrenin kesme günü kendi evresine düşer", all(ym.stage_for(W, ym.stage_cutoff_doy(W, s)) == s for s in ym.STAGES))
check("özellik kümeleri iç içe (erken ⊂ ... olgun ⊂ tüm)", set(ym.FEATURES["tepe"]) <= set(ym.FEATURES["olgun"]) and all(f in ym.FEATURE_LABELS for s in ym.STAGES for f in ym.FEATURES[s]))

print("== Özellik çıkarımı (bilinen cevap)")
days = pd.date_range("2024-01-20", "2024-06-01")
wx = pd.DataFrame({"tarih": days, "yagis_mm": 2.0, "tmax_c": 20.0, "tmin_c": 10.0, "et0_mm": 3.0})
wx.loc[wx.tarih.isin(pd.to_datetime(["2024-04-10", "2024-04-11", "2024-04-12"])), "tmax_c"] = 35.0
nd = pd.DataFrame({"tarih": pd.date_range("2024-02-10", "2024-05-20", freq="10D")})
nd["ortalama"] = np.linspace(0.2, 0.8, len(nd))
cut = D(2024, 5, 20)
f = ym.season_features(nd, wx, W, cut, 4.5)
sos = D(2024, 1, 1) + dt.timedelta(days=int(round(W.sos)) - 1)
start = pd.Timestamp(sos - dt.timedelta(days=30))
n_days = (pd.Timestamp(cut) - start).days + 1
inwin = nd[nd.tarih >= start]
check("geçmiş verim aynen", f["onceki_verim"] == 4.5)
check("NDVI tepe/son pencere içinden", abs(f["ndvi_tepe"] - inwin.ortalama.max()) < 1e-12 and abs(f["ndvi_son"] - inwin.ortalama.iloc[-1]) < 1e-12)
check("yağış = 2 mm × gün sayısı", abs(f["yagis_mm"] - 2.0 * n_days) < 1e-9, f"{f['yagis_mm']} vs {2 * n_days}")
check("sıcak gün = 3", f["sicak_gun"] == 3)
check("su dengesi = yağış − ET0 = (2−3)×gün", abs(f["su_acigi_mm"] + n_days) < 1e-9)
exp_gdd = (n_days - 3) * ((20 + 10) / 2 - 5) + 3 * ((30 + 10) / 2 - 5)
check("GDD (taban 5, üst 30) bilinen", abs(f["gdd"] - exp_gdd) < 1e-6, f"{f['gdd']} vs {exp_gdd}")
lin = pd.DataFrame({"tarih": pd.date_range("2024-03-01", periods=11, freq="D"), "ortalama": np.linspace(0, 1, 11)})
check("ndvi_ort yamuk kuralı: doğrusal 0→1 ⇒ 0,5", abs(ym.season_features(lin, wx, W, D(2024, 3, 11), 4.0)["ndvi_ort"] - 0.5) < 1e-9)
check("< 3 NDVI gözlemi → hata", raises(lambda: ym.season_features(nd.iloc[:2], wx, W, cut, 4.5), "NDVI gözlemi")[0])
check("hava verisi eksikse → hata", raises(lambda: ym.season_features(nd, wx.iloc[:20], W, cut, 4.5), "hava verisi eksik")[0])

print("== Sızıntı yok: kesme gününden sonraki veri özellikleri değiştirmez")
cut2 = D(2024, 4, 5)
base = ym.season_features(nd, wx, W, cut2, 4.5)
nd2, wx2 = nd.copy(), wx.copy()
nd2.loc[nd2.tarih > "2024-04-05", "ortalama"] = 0.01
wx2.loc[wx2.tarih > "2024-04-05", ["yagis_mm", "tmax_c", "tmin_c"]] = [300.0, 50.0, 40.0]
check("kesmeden sonrasını bozmak özellikleri değiştirmez", ym.season_features(nd2, wx2, W, cut2, 4.5) == base)

print("== Sentetik geçmiş")
h = ym.synthetic_history("Buğday", n_fields=30, years=range(2016, 2026), seed=9)
h2 = ym.synthetic_history("Buğday", n_fields=30, years=range(2016, 2026), seed=9)
check("aynı tohum → aynı tablo", h.equals(h2))
check("farklı tohum → farklı", not h.equals(ym.synthetic_history("Buğday", n_fields=30, years=range(2016, 2026), seed=10)))
check("ilk sezon dışarıda (geçmiş verimi gerçek değil), 9 sezon × 30 parsel × 3 evre", h.sezon.min() == 2017 and len(h) == 9 * 30 * 3, f"{h.sezon.min()} {len(h)}")
check("geçmiş verim = aynı parselin önceki sezon verimi", (lambda t: np.allclose(t["onceki_verim"].to_numpy()[1:], t["verim_t_ha"].to_numpy()[:-1]))(h[(h.asama == "olgun") & (h.parsel_id == "P001")].sort_values("sezon")))
ol = h[h.asama == "olgun"].set_index(["parsel_id", "sezon"])
er = h[h.asama == "erken"].set_index(["parsel_id", "sezon"])
check("kümülatif özellikler evrelerle artar (yağış, GDD, NDVI tepe)", (er.yagis_mm <= ol.yagis_mm + 1e-9).all() and (er.gdd <= ol.gdd + 1e-9).all() and (er.ndvi_tepe <= ol.ndvi_tepe + 1e-9).all())
check("verimler pozitif ve makul (0,3–2×taban)", ((h.verim_t_ha > 0.3) & (h.verim_t_ha < 2 * W.base_yield)).all())
check("üretim sürecinde NDVI–verim ilişkisi pozitif", np.corrcoef(ol.ndvi_tepe, ol.verim_t_ha)[0, 1] > 0.2)

print("== Ridge ve conformal bilinen cevap")
rng = np.random.default_rng(1)
X = rng.normal(size=(200, 2))
y = 2 + 3 * X[:, 0] - 1 * X[:, 1]
coef, ym0, mu, sd = ym._ridge(X, y, 1e-9)
check("gürültüsüz doğrusal veri: tam geri kazanım", np.allclose(ym0 + ((X - mu) / sd) @ coef, y, atol=1e-6))
coef_big, *_ = ym._ridge(X, y, 1e9)
check("çok büyük ceza → katsayılar ≈ 0", np.abs(coef_big).max() < 1e-4)
check("conformal: |hata| 1..10, α=0,1 → 10", ym._conformal_q(np.arange(1, 11, dtype=float), 0.1) == 10.0)
check("conformal: |hata| 1..100, α=0,1 → ⌈101×0,9⌉=91. değer", ym._conformal_q(np.arange(1, 101, dtype=float), 0.1) == 91.0)
m_ = ym._metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
check("ölçütler: RMSE √(4/3), MAE 2/3, sapma 2/3", abs(m_.rmse - math.sqrt(4 / 3)) < 1e-9 and abs(m_.mae - 2 / 3) < 1e-9 and abs(m_.bias - 2 / 3) < 1e-9)

print("== Eğitim, doğruluk, kapsama")
models = {s: ym.fit_yield_model(h, "Buğday", s, synthetic=True) for s in ym.STAGES}
for s, m in models.items():
    check(f"{s}: LOSO kapsama %85–97 (hedef %90)", 85 <= m.loso.coverage <= 97, f"{m.loso.coverage:.1f}")
    ho = ym.evaluate_holdout(h, "Buğday", s)
    check(f"{s}: zaman-bazlı sınama kapsama %75–100 ve RMSE LOSO'nun 2 katından az", 75 <= ho.coverage <= 100 and ho.rmse < 2 * m.loso.rmse, f"{ho.coverage:.0f} {ho.rmse:.2f}")
cmp_ = ym.compare_models(h, "Buğday", "tepe")
check("çok değişkenli model yalnız-NDVI ve yalnız-geçmiş modelinden daha düşük hata verir", cmp_.rmse.iloc[2] < cmp_.rmse.iloc[0] and cmp_.rmse.iloc[2] < cmp_.rmse.iloc[1], str(cmp_.rmse.round(3).tolist()))
check("model tablosu 3 satır", len(cmp_) == 3 and cmp_.model.iloc[2].startswith("Çok değişkenli"))
check("LOSO ölçütleri örnek-dışı: eğitim-içi hatadan büyük", models["tepe"].loso.rmse > (lambda m: float(np.sqrt(np.mean((m.intercept + ((h[(h.asama == 'tepe')][list(m.features)].to_numpy() - m.mu) / m.sd) @ m.coef - h[h.asama == 'tepe'].verim_t_ha.to_numpy()) ** 2))))(models["tepe"]))
for label, kw, needle in [("lam", dict(lam=0.0), "lam"), ("alpha", dict(alpha=0.6), "alpha")]:
    check(f"geçersiz {label}", raises(lambda kw=kw: ym.fit_yield_model(h, "Buğday", "tepe", **kw), needle)[0])
check("az sezon → reddedilir", raises(lambda: ym.fit_yield_model(h[h.sezon <= 2018], "Buğday", "tepe"), "en az")[0])
check("az gözlem → reddedilir", raises(lambda: ym.fit_yield_model(h[(h.parsel_id <= "P002")], "Buğday", "tepe"), "en az")[0])
check("başka ürün için veri yok → reddedilir", raises(lambda: ym.fit_yield_model(h, "Mısır", "tepe"), "en az")[0])
neg = h.copy()
neg.loc[neg.index[neg.asama == "tepe"][0], "verim_t_ha"] = -1.0
check("negatif verim → hata", raises(lambda: ym.fit_yield_model(neg, "Buğday", "tepe"), "pozitif")[0])
check("eksik sütun → hata", raises(lambda: ym.fit_yield_model(h.drop(columns=["gdd"]), "Buğday", "tepe"), "eksik sütun")[0])
check("bilinmeyen evre → hata", raises(lambda: ym.fit_yield_model(h, "Buğday", "x"), "evre")[0])
check("holdout için sezon yetmezse hata", raises(lambda: ym.evaluate_holdout(h[h.sezon <= 2020], "Buğday", "tepe", test_seasons=2), "sezon gerekir")[0])

print("== Tahmin")
m = models["tepe"]
row = h[(h.asama == "tepe") & (h.parsel_id == "P005") & (h.sezon == 2022)].iloc[0]
feats = {k: float(row[k]) for k in m.features}
est = ym.predict(m, feats)
check("katkılar + kesişim = tahmin", abs(est.contributions.katki_t_ha.sum() + m.intercept - est.value) < 1e-9)
check("aralık simetrik ve genişlik = conformal yarı-genişlik", abs((est.hi - est.value) - m.half_width) < 1e-9 and abs((est.value - est.lo) - m.half_width) < 1e-9 and abs(est.level - 0.9) < 1e-12)
check("katkı tablosu mutlak değere göre sıralı", est.contributions.katki_t_ha.abs().is_monotonic_decreasing)
check("sentetik model uyarısı taşır", any("SENTETİK" in w for w in est.warnings))
bad = dict(feats)
bad["yagis_mm"] = 5000.0
mid_edge = dict(feats)
mid_edge["yagis_mm"] = float(m.hi[m.features.index("yagis_mm")]) + 0.02 * float(m.hi[m.features.index("yagis_mm")] - m.lo[m.features.index("yagis_mm")])
check("eğitim aralığı dışı → ekstrapolasyon uyarısı", any("ekstrapolasyon" in w for w in ym.predict(m, bad).warnings))
check("aralığın %2'si kadar taşma uyarı üretmez (gürültü), %5'ten fazlası üretir", not any("ekstrapolasyon" in w for w in ym.predict(m, mid_edge).warnings))
check("eksik özellik → hata", raises(lambda: ym.predict(m, {k: v for k, v in feats.items() if k != "gdd"}), "Eksik")[0])
check("NaN özellik → hata", raises(lambda: ym.predict(m, {**feats, "gdd": float("nan")}), "Eksik")[0])
hi_prior = ym.predict(m, {**feats, "onceki_verim": feats["onceki_verim"] + 1.0}).value
check("geçmiş verim artınca tahmin artar (pozitif katsayı)", hi_prior > est.value)
check("tahmin ≤ 0 ise 0'a kırpılır + uyarı", (lambda e: e.value == 0 and any("≤ 0" in w for w in e.warnings))(ym.predict(m, {**feats, "onceki_verim": -100.0})))

print("== Kullanıcı CSV'si")
tmpl = ym.history_template_csv()
t_df = ym.load_history_csv(tmpl)
check("şablon okunur: olgun evre, ürün Buğday", len(t_df) == 1 and t_df.asama.iloc[0] == "olgun" and t_df.urun.iloc[0] == "Buğday")
csv = h[h.asama == "olgun"].to_csv(index=False)
back = ym.load_history_csv(csv)
mb = ym.fit_yield_model(back, "Buğday", "olgun")
check("CSV gidiş-dönüş: aynı model (sentetik bayrağı hariç)", np.allclose(mb.coef, models["olgun"].coef) and not mb.synthetic)
no_stage = ym.load_history_csv(h[h.asama == "olgun"].drop(columns=["asama"]).to_csv(index=False))
check("asama sütunu yoksa 'olgun' sayılır", (no_stage.asama == "olgun").all())
check("eksik zorunlu sütun", raises(lambda: ym.load_history_csv("a,b\n1,2\n"), "eksik sütun")[0])
check("geçersiz evre", raises(lambda: ym.load_history_csv(tmpl.replace("olgun,", "yok,", 1)), "evre")[0])
check("bilinmeyen ürün", raises(lambda: ym.load_history_csv(tmpl.replace("Buğday", "Muz")), "Bilinmeyen ürün")[0])
check("sayısal olmayan verim", raises(lambda: ym.load_history_csv(tmpl.replace("3.9", "x")), "sayısal")[0])
check("bozuk CSV → YieldError", raises(lambda: ym.load_history_csv("\x00\x01"), "")[0])

print("== İş akışı çıktısından tahmin (sentetik tarım senaryosu)")
scn = di.preset("tarim")
ws, we = di.default_window(scn)
run = wf.run_analysis(di.DemoSource(scn), wf.AnalysisRequest("tarim", scn.region, ws, we, scn.event_date, max_cloud=40.0))
hist, pool = ym.demo_model_pool("Buğday")
check("demo havuzu: 3 evre, hepsi sentetik", set(pool) == set(ym.STAGES) and all(p.synthetic for p in pool.values()))
check("demo havuzu önbellekli (aynı nesne)", ym.demo_model_pool("Buğday")[1] is pool)
est2, feats2 = ym.estimate_from_series(pool, run.series, run.weather, "Buğday", 4.9)
check("seriden tahmin: sonlu, aralık içinde, evre geçerli", math.isfinite(est2.value) and est2.lo <= est2.value <= est2.hi and est2.stage in ym.STAGES and 0 < est2.value < 15, f"{est2.value} {est2.stage}")
check("kullanılan özellikler evre kümesiyle aynı", set(feats2) >= set(pool[est2.stage].features))
early = run.series[run.series.tarih <= pd.Timestamp(2026, 4, 22)]
est3, f3 = ym.estimate_from_series(pool, early, run.weather, "Buğday", 4.9)
check("erken seri → erken evre modeli, aralık daha geniş ya da eşit", est3.stage == "erken" and (est3.hi - est3.lo) >= (est2.hi - est2.lo) - 0.2, f"{est3.stage} {est3.hi - est3.lo:.2f} vs {est2.hi - est2.lo:.2f}")
too_early = run.series[run.series.tarih <= pd.Timestamp(2026, 3, 25)]
check("ilk evreye ulaşılmadıysa açık hata", raises(lambda: ym.estimate_from_series(pool, too_early, run.weather, "Buğday", 4.9), "ilk evreye")[0])
check("özellikler nominal kesme gününde: sezon sonrası veri tahmini değiştirmez", ym.estimate_from_series(pool, run.series[run.series.tarih <= pd.Timestamp(2026, 6, 10)], run.weather, "Buğday", 4.9)[0].value == est2.value)
check("evre seçimi: nominal kesme günleri", [ym.latest_usable_stage(W, ym.stage_date(W, s, 2026)) for s in ym.STAGES] == list(ym.STAGES) and ym.latest_usable_stage(W, D(2026, 3, 1)) is None)
sparse = nd.iloc[[3, 4, 5]].copy()
check("son NDVI gözlemi kesmeden çok önceyse hata", raises(lambda: ym.season_features(sparse, wx, W, D(2024, 5, 20), 4.5), "gün önce")[0])
check("kullanılabilir gözlem yoksa hata", raises(lambda: ym.estimate_from_series(pool, run.series.assign(gecerli_oran=0.0), run.weather, "Buğday", 4.9), "Kullanılabilir")[0])
check("havuzda evre yoksa hata", raises(lambda: ym.estimate_from_series({k: v for k, v in pool.items() if k != est2.stage}, run.series, run.weather, "Buğday", 4.9), "model yok")[0])
check("ürün profili data_engine.CROPS ile aynı", {c.name for c in de.CROPS} >= {"Buğday", "Mısır"})

print()
print("BAŞARISIZ:", FAILS if FAILS else "yok — tüm kontroller geçti")
sys.exit(1 if FAILS else 0)
