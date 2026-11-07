"""analysis.py ve demo_imagery.py çevrimdışı testleri (ağ yok; bilinen-cevap testleri)."""
import datetime as dt
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analysis as an  # noqa: E402
import data_engine as de  # noqa: E402
import demo_imagery as di  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  [{extra}]" if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)


def raises(fn, exc=an.AnalysisError):
    try:
        fn()
    except exc as e:
        return True, str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"yanlış istisna: {type(e).__name__}: {e}"
    return False, "istisna yok"


D = dt.date
print("== Bölge ve ızgara")
reg = an.region_from_center("kare", 39.0, 35.0, 1.0)
check("2×2 km kare ≈ 400 ha", abs(reg.area_ha - 400) < 4, f"{reg.area_ha:.1f}")
check("merkez doğru", abs(reg.centre[0] - 39.0) < 1e-9 and abs(reg.centre[1] - 35.0) < 1e-9)
gj = reg.geojson()
check("GeoJSON halkası kapalı", gj["coordinates"][0][0] == gj["coordinates"][0][-1] and len(gj["coordinates"][0]) == 5)
reg2 = an.region_from_geojson({"type": "Feature", "geometry": gj})
check("GeoJSON gidiş-dönüş", len(reg2.ring) == 4 and abs(reg2.area_ha - reg.area_ha) < 1e-6)
check("kapalı halka açılır", len(an.Region("x", reg.ring + (reg.ring[0],)).ring) == 4)
for label, fn in [
    ("2 köşe", lambda: an.Region("x", ((30, 39), (31, 39)))),
    ("NaN", lambda: an.Region("x", ((30, 39), (31, float("nan")), (31, 40)))),
    ("aralık dışı", lambda: an.Region("x", ((30, 39), (31, 99), (31, 40)))),
    ("çok büyük", lambda: an.region_from_bbox("x", 30, 39, 31, 40)),
    ("çok küçük", lambda: an.region_from_bbox("x", 35.0, 39.0, 35.0002, 39.0002)),
    ("Point geometrisi", lambda: an.region_from_geojson({"type": "Point", "coordinates": [1, 2]})),
    ("bozuk koordinat", lambda: an.region_from_geojson({"type": "Polygon", "coordinates": [["a"]]})),
    ("ters bbox", lambda: an.region_from_bbox("x", 31, 39, 30, 40)),
    ("aynı köşeler", lambda: an.Region("x", ((30, 39), (30, 39), (30, 39)))),
]:
    ok, msg = raises(fn, an.RegionError)
    check(f"geçersiz bölge reddedilir: {label}", ok, msg)
grid = an.make_grid(reg.bbox, 100)
check("ızgara uzun kenar ≤ 100 px", max(grid.rows, grid.cols) <= 100 and min(grid.rows, grid.cols) >= 90)
check("piksel alanı × piksel sayısı = kutu alanı", abs(grid.pixel_area_ha * grid.rows * grid.cols - 400) < 8)
check("min_res_m 10 m'den ince piksel üretmez", an.make_grid(reg.bbox, 1000).rows <= 202)
check("max_px sınırı", raises(lambda: an.make_grid(reg.bbox, 4))[0])
tri = an.Region("üçgen", ((reg.bbox[0], reg.bbox[1]), (reg.bbox[2], reg.bbox[1]), (reg.bbox[0], reg.bbox[3])))
m = an.polygon_mask(grid, tri.ring)
frac = m.sum() * grid.pixel_area_ha / tri.area_ha
check("üçgen maskesi alanı ≈ çokgen alanı (±3%)", abs(frac - 1) < 0.03, f"{frac:.3f}")
check("kare maskesi tüm pikseller", an.polygon_mask(grid, reg.ring).all())
lon = np.array([reg.bbox[2] + 1000 / an._m_per_deg_lon(39.0)])
dist = an.distance_to_ring_m(lon, np.array([39.0]), reg.ring)
check("kenara uzaklık ≈ 1000 m", abs(dist[0] - 1000) < 10, f"{dist[0]:.1f}")
check("kenar üstündeki nokta ≈ 0 m", an.distance_to_ring_m(np.array([reg.bbox[0]]), np.array([39.0]), reg.ring)[0] < 1.0)

print("== İndeksler")
b = {"red": np.array([[0.1]]), "nir": np.array([[0.5]]), "blue": np.array([[0.03]]), "green": np.array([[0.2]]),
     "swir16": np.array([[0.3]]), "swir22": np.array([[0.1]])}
check("NDVI = 0,6667", abs(an.compute_index("ndvi", b)[0, 0] - 2 / 3) < 1e-6)
check("NBR = 0,6667", abs(an.compute_index("nbr", b)[0, 0] - 2 / 3) < 1e-6)
check("NDMI = 0,25", abs(an.compute_index("ndmi", b)[0, 0] - 0.25) < 1e-6)
check("MNDWI = −0,2", abs(an.compute_index("mndwi", b)[0, 0] + 0.2) < 1e-6)
e = {"nir": np.array([[0.45]]), "red": np.array([[0.04]]), "blue": np.array([[0.03]])}
check("EVI bilinen değer", abs(an.compute_index("evi", e)[0, 0] - 2.5 * 0.41 / (0.45 + 0.24 - 0.225 + 1)) < 1e-5)
check("payda ~0 → NaN", np.isnan(an.compute_index("ndvi", {"red": np.zeros((1, 1)), "nir": np.zeros((1, 1))})[0, 0]))
check("eksik bant → hata", raises(lambda: an.compute_index("nbr", {"nir": b["nir"]}))[0])
check("bilinmeyen indeks → hata", raises(lambda: an.compute_index("xyz", b))[0])
scl = np.array([[4, 5, 6, 3, 8, 9, 10, 11, 0, 1]], dtype=np.uint8)
vm = an.valid_mask(scl, {"red": np.ones((1, 10))})
check("SCL maskesi: bitki/çıplak/su geçerli, bulut/gölge/kar/doygun/yok geçersiz", vm.tolist() == [[True, True, True, False, False, False, False, False, False, False]])
check("bantta NaN geçersiz", not an.valid_mask(None, {"red": np.array([[np.nan, 1.0]])})[0, 0])
check("bands_for yangın", an.bands_for("yangin") == ("red", "nir", "swir22"))
check("bands_for sel+rgb", set(an.bands_for("sel", True)) == {"blue", "green", "red", "nir", "swir16"})
check("bands_for bilinmeyen → hata", raises(lambda: an.bands_for("x"))[0])

print("== Bulut düzeyi ve süzgeç")
check("bulut düzeyleri", [an.cloud_level(x)[1] for x in (0, 10, 10.1, 20, 20.1, 90)] == ["ok", "ok", "mid", "mid", "low", "low"])
check("bilinmeyen bulut", an.cloud_level(float("nan"))[1] == "unknown")
S = lambda i, d, c: an.SceneMeta(i, d, "S2", c, "t")  # noqa: E731
sc = [S("a", D(2026, 6, 1), 5), S("b", D(2026, 6, 6), 50), S("c", D(2026, 6, 11), float("nan")), S("d", D(2026, 5, 1), 1), S("e", D(2026, 6, 3), 20)]
f = an.filter_scenes(sc, 20, D(2026, 5, 20), D(2026, 6, 30))
check("süzgeç: bulut, tarih, NaN dışlanır; sıralı", [x.scene_id for x in f] == ["a", "e"])

print("== Sahne istatistiği (bilinen cevap)")
g = an.make_grid(reg.bbox, 40)
mask = an.polygon_mask(g, reg.ring)


def scene(date, nir, swir22, red=0.05, scl_arr=None, cloud=5.0, sid=None):
    shp = (g.rows, g.cols)
    bands = {"nir": np.full(shp, nir, np.float32), "swir22": np.full(shp, swir22, np.float32), "red": np.full(shp, red, np.float32),
             "green": np.full(shp, 0.1, np.float32), "blue": np.full(shp, 0.03, np.float32), "swir16": np.full(shp, 0.2, np.float32)}
    return an.SceneData(an.SceneMeta(sid or f"s{date}", date, "S2A", cloud, "t"), bands, scl_arr)


s1 = scene(D(2026, 7, 1), 0.4, 0.1)
st, idx = an.scene_stats(s1, "nbr", mask)
check("sabit NBR: ort = medyan = 0,6", abs(st.mean - 0.6) < 1e-6 and abs(st.median - 0.6) < 1e-6 and st.std < 1e-6)
check("scl yoksa yerel bulut NaN, geçerli %100", math.isnan(st.cloud_local) and st.valid_pct == 100.0)
scl_arr = np.full((g.rows, g.cols), 4, np.uint8)
scl_arr[: g.rows // 4] = 9
s2 = scene(D(2026, 7, 1), 0.4, 0.1, scl_arr=scl_arr)
st2, idx2 = an.scene_stats(s2, "nbr", mask)
check("bulutlu çeyrek: geçerli ≈ %75, yerel bulut ≈ %25", abs(st2.valid_pct - 75) < 2 and abs(st2.cloud_local - 25) < 2, f"{st2.valid_pct:.1f}/{st2.cloud_local:.1f}")
check("bulut pikselleri raster'da NaN", np.isnan(idx2[0, 0]) and np.isfinite(idx2[-1, -1]))
ser = an.build_series([st2, st])
check("seri tarihe göre sıralı, sütunlar tam", list(ser.columns)[:3] == ["tarih", "sahne", "uydu"] and ser["tarih"].is_monotonic_increasing)

print("== Önce/sonra seçimi")
series = pd.DataFrame({"tarih": pd.to_datetime(["2026-06-01", "2026-06-20", "2026-07-10", "2026-07-20", "2026-08-30"]),
                       "sahne": list("abcde"), "gecerli_oran": [95, 70, 50, 99, 98]})
pc = an.pick_pair(series, D(2026, 7, 5))
check("önce: yakın ve berrak (b: 70, 15 gün) > a (95, 34 gün)?", pc.pre_id in ("a", "b"))
check("sonra: d (99, 15 gün) seçilir, c (50) elenir", pc.post_id == "d")
nopair = an.pick_pair(series, D(2026, 7, 5), min_valid_pct=99.5)
check("aday yoksa None + açıklama", nopair.pre_id is None and nopair.post_id is None and len(nopair.notes) == 2)
check("max_gap_days uygulanır", an.pick_pair(series, D(2026, 7, 5), max_gap_days=10).pre_id is None)
yo = pd.DataFrame({"tarih": pd.to_datetime(["2025-07-01", "2025-07-14", "2025-09-01", "2026-07-05", "2026-07-25"]),
                   "sahne": list("vwxyz"), "gecerli_oran": [90, 95, 99, 80, 97]})
py = an.pick_pair(yo, D(2026, 6, 15), mode="yoy")
check("yoy: sonra = olaydan ~5 hafta, önce = geçen yıl aynı dönem", py.post_id in ("y", "z") and py.pre_id in ("v", "w"), str(py))
check("yoy: geçen yıl yoksa açıklama", an.pick_pair(yo.iloc[3:], D(2026, 6, 15), mode="yoy").pre_id is None)
check("bilinmeyen kip → hata", raises(lambda: an.pick_pair(series, D(2026, 7, 5), mode="x"))[0])

print("== Değişim sınıflandırma — eşik sınırları")
f1 = lambda d: int(an.classify_change("yangin", np.array([d], np.float32) + 0.0, np.array([0.0], np.float32))[1][0])  # noqa: E731
check("dNBR sınıfları (USGS sınırları)", [f1(x) for x in (-0.11, -0.10, 0.09, 0.10, 0.269, 0.27, 0.439, 0.44, 0.659, 0.66, 1.0)] == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
check("yangın sınıfı dNBR eşikleriyle uyumlu", an.KINDS["yangin"].classes.__len__() == len(de.DNBR_EDGES) + 2)
sw = lambda a, b_: int(an.classify_change("sel", np.array([a], np.float32), np.array([b_], np.float32))[1][0])  # noqa: E731
check("sel: yeni su / kalıcı / çekilen / yok", [sw(-0.3, 0.4), sw(0.4, 0.4), sw(0.4, -0.3), sw(-0.3, -0.3)] == [2, 1, 3, 0])
kr = lambda p, q: int(an.classify_change("kuraklik", np.array([p], np.float32), np.array([q], np.float32))[1][0])  # noqa: E731
check("kuraklık: bitkisiz, artış, hafif, orta, şiddetli", [kr(0.1, 0.1), kr(0.6, 0.6), kr(0.6, 0.5), kr(0.6, 0.4), kr(0.6, 0.2)] == [4, 0, 1, 2, 3])
tr = lambda d: int(an.classify_change("tarim", np.array([0.5], np.float32), np.array([0.5 + d], np.float32))[1][0])  # noqa: E731
check("tarım: ΔNDVI sınıfları", [tr(-0.2), tr(-0.1), tr(0.0), tr(0.1), tr(0.2)] == [0, 1, 2, 3, 4])
dd, cc = an.classify_change("yangin", np.array([np.nan, 0.5], np.float32), np.array([0.1, 0.1], np.float32))
check("NaN piksel → sınıf −1", cc[0] == -1 and cc[1] == 3)
check("bilinmeyen tür → hata", raises(lambda: an.classify_change("x", np.zeros(1), np.zeros(1)))[0])
for k, kd in an.KINDS.items():
    check(f"{k}: sınıf/ağırlık sayısı eşit, etkilenen indeksler geçerli", len(kd.classes) == len(kd.weights) and all(0 <= i < len(kd.classes) for i in kd.affected) and all(0 <= w <= 1 for w in kd.weights))

print("== Analiz + sektör görünümleri (bilinen cevap)")
reg_s = an.region_from_center("test", 39.0, 35.0, 1.0)
gs = an.make_grid(reg_s.bbox, 40)
g = gs  # scene() yardımcısı bu ızgarayı kullanır
mask = an.polygon_mask(gs, reg_s.ring)


def fire_result(dnbr, scl_arr=None, cloud=3.0, ev=None, pre_d=None, post_d=None):
    ev, pre_d, post_d = ev or D(2026, 7, 22), pre_d or D(2026, 7, 16), post_d or D(2026, 7, 31)
    # NBR_önce = 0,6 (nir .4, swir .1); NBR_sonra = 0,6 − dnbr → swir22 çözülür: (n−s)/(n+s)=x → s = n(1−x)/(1+x)
    x = 0.6 - dnbr
    post_swir = 0.4 * (1 - x) / (1 + x)
    pre = scene(pre_d, 0.4, 0.1, cloud=cloud, sid="pre")
    post = scene(post_d, 0.4, post_swir, scl_arr=scl_arr, cloud=cloud, sid="post")
    return an.analyse_change("yangin", pre, post, reg_s, gs, ev)


R = de.ClaimRules()
for dn, want in ((0.60, "ONAY"), (0.35, "KISMİ ONAY"), (0.05, "RET"), (0.18, "EKSPERTİZ")):
    rs = fire_result(dn)
    iv = an.insurance_view(rs, R, 1_000_000)
    check(f"yangın tek-değer dNBR {dn} → {want}", iv.decision == want, f"{iv.decision} dnbr={iv.mean_dnbr:.3f} rel={rs.reliability.score}")
rs = fire_result(0.60)
iv = an.insurance_view(rs, R, 1_000_000)
want_claim = float(de.net_claim_tl(np.array([0.60]), np.array([1_000_000.0]), R)[0])
check("tazminat, uygulamanın tek formülüyle (net_claim_tl) aynı", abs(iv.indicative_claim_tl - want_claim) < 1.0, f"{iv.indicative_claim_tl} vs {want_claim}")
check("hasar oranı = damage_ratio(0,60)", abs(iv.damage_ratio - float(de.damage_ratio(np.array([0.60]), R)[0])) < 1e-4)
check("etkilenen alan ≈ bölge alanı (tek-değer)", abs(rs.affected_ha - rs.region_ha) / rs.region_ha < 1e-9 and abs(rs.region_ha - reg_s.area_ha) / reg_s.area_ha < 0.03)
scl_cloudy = np.full((gs.rows, gs.cols), 4, np.uint8)
scl_cloudy[: int(gs.rows * 0.5)] = 9
rc = fire_result(0.60, scl_arr=scl_cloudy, cloud=60.0)
ivc = an.insurance_view(rc, R, 1_000_000)
check("yarı bulutlu + karo bulutu %60 → otomatik ONAY verilmez", ivc.decision == "EKSPERTİZ" and rc.reliability.level != "Yüksek", f"{ivc.decision} {rc.reliability.score}")
check("bulutlu sonuç: kullanılabilir alan ≈ %50", abs(rc.valid_pct - 50) < 3, f"{rc.valid_pct:.1f}")
check("karo bulutu > %20 → güvenilirlik uyarısı metni", any("> %20" in n for n in rc.reliability.notes))
full_cloud = np.full((gs.rows, gs.cols), 9, np.uint8)
rf = fire_result(0.6, scl_arr=full_cloud, cloud=95.0)
ivf = an.insurance_view(rf, R, 1_000_000)
check("tam bulutlu sonra sahnesi → ölçülemedi, EKSPERTİZ, tazminat 0", ivf.decision == "EKSPERTİZ" and ivf.indicative_claim_tl == 0 and math.isnan(ivf.damage_ratio))
check("insurance_view geçersiz bedel → hata", raises(lambda: an.insurance_view(rs, R, float("nan")))[0])
rg = an.analyse_change("sel", scene(D(2026, 7, 1), 0.4, 0.1), scene(D(2026, 7, 15), 0.4, 0.1), reg_s, gs, D(2026, 7, 8))
ivs = an.insurance_view(rg, R, 1_000_000)
check("sel: değişim yok → RET; otomatik ONAY yok", ivs.decision == "RET" and rg.affected_ha == 0)
check("agri_view: değişmeyen parsel Sağlıklı, stres 0", an.agri_view(rg).health == "Sağlıklı" and an.agri_view(rg).stressed_ha == 0)
rd = an.analyse_change("tarim", scene(D(2026, 5, 1), 0.5, 0.1, red=0.05), scene(D(2026, 5, 20), 0.3, 0.1, red=0.1), reg_s, gs, D(2026, 5, 10))
av = an.agri_view(rd)
check("agri_view: NDVI 0,82→0,50 → Kritik/Stresli, tüm bitkili alan stresli", av.health in ("Stresli", "Kritik") and av.stressed_pct > 99, f"{av.health} {av.stressed_pct}")
check("agri_view bilinen NDVI ortalamaları", abs(av.mean_pre - 0.8182) < 1e-3 and abs(av.mean_post - 0.5) < 1e-3, f"{av.mean_pre} {av.mean_post}")
check("aynı sonuçtan iki görünüm", an.insurance_view(rd, R, 1e6).decision in ("EKSPERTİZ", "RET") and av.veg_ha > 0)
check("ters sıralı sahneler → hata", raises(lambda: an.analyse_change("yangin", scene(D(2026, 7, 31), 0.4, 0.1), scene(D(2026, 7, 1), 0.4, 0.1), reg_s, gs))[0])

print("== Güvenilirlik puanı")
def stat(valid=100.0, cloud=0.0, d=None):
    d = d or D(2026, 7, 1)
    return an.SceneStats("x", d, "S2", cloud, 0.0, valid, 0.5, 0.5, 0.4, 0.6, 0.05, 1000)


check("ağırlıklar toplamı 1", abs(sum(an.RELIABILITY_WEIGHTS.values()) - 1) < 1e-12)
good = an.assess_reliability("yangin", stat(100, 2, D(2026, 7, 15)), stat(100, 2, D(2026, 8, 5)), 2000, D(2026, 7, 22))
check("berrak, yakın tarihli, büyük örnek → 100 / Yüksek", good.score >= 95 and good.level == "Yüksek", str(good.score))
bad = an.assess_reliability("yangin", stat(30, 70, D(2026, 5, 1)), stat(30, 70, D(2026, 10, 1)), 50, D(2026, 7, 22))
check("bulutlu, uzak tarihli, küçük örnek → Düşük", bad.level == "Düşük" and bad.score < 50, str(bad.score))
check("bileşenler 0–1", all(0 <= v <= 1 for v in bad.parts.values()) and all(0 <= v <= 1 for v in good.parts.values()))
neutral = an.assess_reliability("tarim", stat(), stat(d=D(2026, 7, 20)), 2000, None)
check("olay tarihi yoksa zamanlama nötr + not", abs(neutral.parts["zamanlama"] - 0.7) < 1e-9 and any("nötr" in n for n in neutral.notes))
wrong = an.assess_reliability("yangin", stat(d=D(2026, 8, 1)), stat(d=D(2026, 8, 10)), 2000, D(2026, 7, 22))
check("önce görüntüsü olaydan sonraysa uyarı", any("olaydan önce değil" in n for n in wrong.notes))
check("her türün sınırı nota eklenir", all(an.KINDS[k].caveat in an.assess_reliability(k, stat(), stat(d=D(2026, 7, 20)), 500, D(2026, 7, 10)).notes for k in an.KINDS))
mono = [an.assess_reliability("yangin", stat(v, 2, D(2026, 7, 15)), stat(v, 2, D(2026, 8, 5)), 2000, D(2026, 7, 22)).score for v in (20, 50, 80, 100)]
check("berraklık arttıkça puan azalmaz", mono == sorted(mono), str(mono))

print("== Hava kanıtı ve FIRMS eşleştirme")
days = pd.date_range("2026-06-01", "2026-09-01")
wx = pd.DataFrame({"tarih": days, "yagis_mm": 1.0, "tmax_c": 28.0, "tmin_c": 15.0, "ruzgar_kmh": 10.0, "et0_mm": 5.0})
ev = D(2026, 7, 14)
wx.loc[wx.tarih == "2026-07-12", "yagis_mm"] = 30.0
wx.loc[wx.tarih == "2026-07-13", "yagis_mm"] = 40.0
wx.loc[wx.tarih == "2026-07-14", "yagis_mm"] = 20.0
rows = an.weather_evidence("sel", wx, ev)
check("sel: 3 gün 90 mm → destekler", rows[0].flag is True and "90" in rows[0].value, rows[0].value)
wx2 = wx.copy()
wx2["yagis_mm"] = 0.5
check("sel: yağış yok → desteklemez", an.weather_evidence("sel", wx2, ev)[0].flag is False)
fire_wx = wx.copy()
fire_wx.loc[fire_wx.tarih == "2026-07-14", ["tmax_c", "ruzgar_kmh"]] = [39.0, 50.0]
fire_wx["yagis_mm"] = 0.0
fr = an.weather_evidence("yangin", fire_wx, ev)
check("yangın: sıcak + rüzgâr + kuru → hepsi destekler", [r.flag for r in fr] == [True, True, True], str([(r.label, r.value) for r in fr]))
check("kuraklık: kısa pencere → veri yok bayrağı", an.weather_evidence("kuraklik", wx.iloc[:30], D(2026, 6, 20))[0].flag is None)
dry = wx.copy()
dry.loc[(dry.tarih >= "2026-06-14") & (dry.tarih <= "2026-07-14"), "yagis_mm"] = 0.0
check("kuraklık: 30 gün kuru → destekler", an.weather_evidence("kuraklik", dry, ev)[0].flag is True)
check("hava yok → tek satır veri yok", an.weather_evidence("sel", None, ev)[0].flag is None)
check("GDD bilinen: Tmax 20 Tmin 10 taban 5 → 10", abs(an.growing_degree_days([20], [10]) - 10) < 1e-9)
check("GDD üst sınır 30", abs(an.growing_degree_days([40], [30], 5, 30) - 25) < 1e-9)
hs = pd.DataFrame({"enlem": [39.0, 39.0, 39.0, 39.30, 39.0], "boylam": [35.0, 35.0 + 1.2 / 86.5, 35.0 + 3 / 86.5, 35.0, 35.0],
                   "frp_mw": 5.0, "parlaklik_k": 340.0, "kaynak": "t", "guven": 0.9,
                   "tarih_utc": pd.to_datetime(["2026-07-14 10:00", "2026-07-14 11:00", "2026-07-14 12:00", "2026-07-14 10:00", "2026-07-30 10:00"], utc=True),
                   "gece_gunduz": "D", "uydu": "t"})
mt = an.match_hotspots(hs, reg, D(2026, 7, 13), D(2026, 7, 16), buffer_km=1.0)
check("FIRMS: içeride + 0,2 km tampon içinde eşleşir; 2 km uzak, çok uzak ve tarih dışı elenir", len(mt) == 2 and int(mt.icinde.sum()) == 1, str(mt[["boylam", "icinde", "uzaklik_m"]]))
check("FIRMS: boş girdi → boş tablo", an.match_hotspots(None, reg, D(2026, 7, 13), D(2026, 7, 16)).empty)

print("== Renklendirme")
rgba = an.colorize(np.array([[0.0, 0.5], [1.0, np.nan]], np.float32), 0.0, 1.0)
check("rampa uçları kırmızı/yeşil, NaN saydam", rgba[0, 0, 0] > 200 and rgba[1, 0, 1] > 120 and rgba[1, 1, 3] == 0 and rgba[0, 0, 3] == 235)
cr = an.colorize(None, 0, 1, classes=["#ff0000", "#00ff00"], class_idx=np.array([[0, 1, -1]]))
check("sınıf boyama", cr[0, 0, 0] == 255 and cr[0, 1, 1] == 255 and cr[0, 2, 3] == 0)
tc = an.true_color(scene(D(2026, 7, 1), 0.4, 0.1))
check("gerçek renk: uint8 (r,c,3)", tc is not None and tc.dtype == np.uint8 and tc.shape == (g.rows, g.cols, 3))
check("gerçek renk: bant yoksa None", an.true_color(an.SceneData(an.SceneMeta("x", D(2026, 1, 1), "S2", 1, "t"), {"nir": np.ones((2, 2))})) is None)

print("== Sentetik demo")
for kind in an.KINDS:
    sc_ = di.preset(kind)
    gr = an.make_grid(sc_.region.bbox, 64)
    cat = di.catalog(sc_)
    check(f"{kind}: katalog 5 günde bir, kimlikler benzersiz", len({c.scene_id for c in cat}) == len(cat) and all((b_.date - a_.date).days == 5 for a_, b_ in zip(cat, cat[1:])))
    pre_ok = [c for c in cat if c.date <= sc_.event_date - dt.timedelta(days=3) and c.cloud_pct < 15]
    post_ok = [c for c in cat if c.date >= sc_.event_date + dt.timedelta(days=7) and c.cloud_pct < 15]
    check(f"{kind}: olay çevresinde berrak önce/sonra sahnesi var", bool(pre_ok) and bool(post_ok))
    d1 = di.load_scene(sc_, gr, cat[3])
    d2 = di.load_scene(sc_, gr, cat[3])
    check(f"{kind}: aynı tohum → birebir aynı sahne", all(np.array_equal(d1.bands[k], d2.bands[k]) for k in d1.bands) and np.array_equal(d1.scl, d2.scl))
    check(f"{kind}: bantlar 0–1, SCL geçerli kümede", all(0 < v.min() and v.max() <= 1 for v in d1.bands.values()) and set(np.unique(d1.scl)) <= {3, 4, 5, 6, 8, 9, 10})
    other = di.load_scene(di.DemoScenario(kind, sc_.region, sc_.event_date, sc_.start, sc_.end, sc_.seed + 1, ""), gr, cat[3])
    check(f"{kind}: farklı tohum → farklı sahne", not np.array_equal(d1.bands["nir"], other.bands["nir"]))
ysc = di.preset("yangin")
yg = an.make_grid(ysc.region.bbox, 96)
ycat = {c.scene_id: c for c in di.catalog(ysc)}
tm_before = di.truth_masks(ysc, yg, ysc.event_date - dt.timedelta(days=1))
tm_after = di.truth_masks(ysc, yg, ysc.event_date + dt.timedelta(days=1))
check("yangın izi olaydan önce yok, sonra var", tm_before["burn"].max() == 0 and tm_after["burn"].max() > 0.5)
ssc = di.preset("sel")
sg = an.make_grid(ssc.region.bbox, 96)
w0 = di.truth_masks(ssc, sg, ssc.event_date - dt.timedelta(days=1))["flooded"].sum()
w1 = di.truth_masks(ssc, sg, ssc.event_date + dt.timedelta(days=1))["flooded"].sum()
w2 = di.truth_masks(ssc, sg, ssc.event_date + dt.timedelta(days=15))["flooded"].sum()
check("sel: sel öncesi yalnız nehir; sonra genişler; sonra çekilir", w0 < w1 and w2 < w1 and w2 >= w0, f"{w0} {w1} {w2}")
ksc = di.preset("kuraklik")
kg = an.make_grid(ksc.region.bbox, 64)
check("kuraklık: başlangıçtan önce stres 0, sonra artar", di.truth_masks(ksc, kg, ksc.event_date)["stress"].max() == 0 and di.truth_masks(ksc, kg, ksc.event_date + dt.timedelta(days=40))["stress"].max() > 0.3)
check("kuraklık varsayılan pencere geçen yılı içermez", di.default_window(ksc)[0] > ksc.start and di.default_window(di.preset("yangin")) == (ysc.start, ysc.end))
big = an.make_grid(ysc.region.bbox, 160)
cloudy = next(c for c in di.catalog(ysc) if 25 < di._plan(ysc)[1][c.scene_id] < 60)
dc = di.load_scene(ysc, big, cloudy)
measured = 100.0 * np.isin(dc.scl, [8, 9]).mean()
check("bulut hedefi ≈ ölçülen bulut oranı (±8 puan)", abs(measured - di._plan(ysc)[1][cloudy.scene_id]) < 8, f"{measured:.1f} vs {di._plan(ysc)[1][cloudy.scene_id]:.1f}")
hs_d = di.demo_hotspots(ysc, yg)
check("demo yangın noktaları: şema + 'demo' etiketi + izin içinde", list(hs_d.columns) == ["enlem", "boylam", "frp_mw", "parlaklik_k", "kaynak", "guven", "tarih_utc", "gece_gunduz", "uydu"] and (hs_d.kaynak == "demo").all() and len(hs_d) > 10)
check("demo yangın dışı türlerde nokta yok", di.demo_hotspots(di.preset("sel"), sg).empty)
dw = di.demo_weather(ssc)
check("demo hava: sel olayında 3 günlük yağış ≥ 40 mm", an.weather_evidence("sel", dw, ssc.event_date)[0].flag is True)
check("demo hava: yangın olayında sıcak+rüzgâr", [r.flag for r in an.weather_evidence("yangin", di.demo_weather(ysc), ysc.event_date)][:2] == [True, True])
check("etiket açıkça SENTETİK", "SENTETİK" in di.DEMO_LABEL and "gerçek" in di.DEMO_LABEL)
check("bilinmeyen senaryo → hata", raises(lambda: di.preset("x"))[0])

print()
print("BAŞARISIZ:", FAILS if FAILS else "yok — tüm kontroller geçti")
sys.exit(1 if FAILS else 0)
