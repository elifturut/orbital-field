"""live_sources.py çevrimdışı testleri: sahte HTTP oturumu ve sahte COG okuyucu (ağ yok, rasterio yok).

Bu testler istek/yanıt biçimlerini ve mantığı sınar; gerçek sunucu ya da gerçek rasterio davranışını KANITLAMAZ.
"""
import datetime as dt
import math
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analysis as an  # noqa: E402
import data_sources as dsrc  # noqa: E402
import live_sources as ls  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  [{extra}]" if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)


def raises(fn, needle="", exc=dsrc.DataSourceError):
    try:
        fn()
    except exc as e:
        return needle in str(e), str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"yanlış istisna: {type(e).__name__}: {e}"
    return False, "istisna yok"


D = dt.date
KEY = "FIRMSKEY123456"


class Resp:
    def __init__(self, status=200, text="", data=None):
        self.status_code, self.text, self._data = status, text, data

    def json(self):
        if self._data is None:
            raise ValueError("json değil")
        return self._data


class Sess:
    """URL parçasına göre yanıt üreten sahte oturum; tüm çağrıları kaydeder."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def _go(self, verb, url, kw):
        self.calls.append((verb, url, kw))
        for needle, fn in self.routes:
            if needle in url:
                out = fn(verb, url, kw) if callable(fn) else fn
                if isinstance(out, Exception):
                    raise out
                return out
        return Resp(404, text="yok")

    def get(self, url, **kw):
        return self._go("GET", url, kw)

    def post(self, url, **kw):
        return self._go("POST", url, kw)


def feat(i, day, cloud=5.0, bbox=(34.0, 38.0, 36.0, 40.0), platform="sentinel-2a", with_rn=True, offset=-0.1):
    rb = [{"scale": 0.0001, "offset": offset}]
    a = {k: {"href": f"https://example.test/{i}/{k}.tif", **({"raster:bands": rb} if with_rn else {})}
         for k in ("red", "nir", "green", "blue", "swir16", "swir22", "scl")}
    p = {"datetime": f"{day}T08:30:00Z", "platform": platform}
    if cloud is not None:
        p["eo:cloud_cover"] = cloud
    return {"id": i, "bbox": list(bbox), "properties": p, "assets": a}


print("== STAC ayrıştırma")
payload = {"features": [feat("A", "2026-07-01", 3.2), feat("B", "2026-07-06", None, platform="sentinel-2b"),
                        {"id": "bozuk"}, {"id": "x", "properties": {"datetime": "2026-07-01T00:00:00Z"}, "assets": {}},
                        feat("C", "2026-07-11", 12.0, with_rn=False)]}
ms = ls.parse_stac_items(payload)
check("bozuk ve bant eksik kayıtlar atlanır (3 geçerli)", [m.scene_id for m in ms] == ["A", "B", "C"])
check("tarih, platform, bulut", ms[0].date == D(2026, 7, 1) and ms[0].platform == "Sentinel-2A" and ms[0].cloud_pct == 3.2 and ms[1].platform == "Sentinel-2B")
check("bulut yoksa NaN", math.isnan(ms[1].cloud_pct))
check("ölçek/kayma raster:bands'ten", ms[0].band_scale["red"] == (0.0001, -0.1) and "scl" not in ms[0].band_scale)
check("raster:bands yoksa varsayılan (1e-4, 0)", ms[2].band_scale["red"] == (0.0001, 0.0))
check("adresler ve bbox", ms[0].assets["nir"].endswith("/A/nir.tif") and ms[0].bbox == (34.0, 38.0, 36.0, 40.0))
check("UTC dışı saat dilimi gün sınırında doğru güne yazılır", ls.parse_stac_items({"features": [feat("T", "2026-07-01")]})[0].date == D(2026, 7, 1))
f_tz = feat("Z", "2026-07-01")
f_tz["properties"]["datetime"] = "2026-07-01T23:30:00-03:00"
check("saat dilimli zaman UTC gününe çevrilir (02:30 UTC ertesi gün)", ls.parse_stac_items({"features": [f_tz]})[0].date == D(2026, 7, 2))
target = (34.8, 38.8, 35.2, 39.2)
dup = [an.SceneMeta("cov", D(2026, 7, 1), "S2", 20, "t", bbox=(34, 38, 36, 40)), an.SceneMeta("part", D(2026, 7, 1), "S2", 1, "t", bbox=(35.0, 38, 37, 40)),
       an.SceneMeta("other", D(2026, 7, 6), "S2", 9, "t", bbox=None)]
bp = ls.best_per_date(dup, target)
check("aynı gün: bölgeyi tam kapsayan, bulutlu da olsa seçilir", [m.scene_id for m in bp] == ["cov", "other"])
check("kapsayan yoksa en az bulutlu", ls.best_per_date(dup[1:2] + [an.SceneMeta("p2", D(2026, 7, 1), "S2", 30, "t", bbox=(35.0, 38, 37, 40))], target)[0].scene_id == "part")

print("== STAC arama")
page1 = {"features": [feat("A", "2026-07-01"), feat("B", "2026-07-06")],
         "links": [{"rel": "next", "method": "POST", "href": "https://x.test/search", "body": {"next": "tok2"}, "merge": True}]}
page2 = {"features": [feat("C", "2026-07-11")], "links": []}
seen = []


def search_h(verb, url, kw):
    seen.append(kw["json"])
    return Resp(200, data=page2 if kw["json"].get("next") == "tok2" else page1)


sess = Sess([("/search", search_h)])
cl = ls.StacClient(sess)
out = cl.search((34.9, 38.9, 35.1, 39.1), D(2026, 7, 1), D(2026, 7, 31))
b0 = seen[0]
check("istek gövdesi: koleksiyon, bbox, tarih aralığı, sıralama", b0["collections"] == ["sentinel-2-c1-l2a"] and b0["bbox"] == [34.9, 38.9, 35.1, 39.1]
      and b0["datetime"] == "2026-07-01T00:00:00Z/2026-07-31T23:59:59Z" and b0["sortby"][0]["field"] == "properties.datetime", str(b0))
check("sayfalama: next gövdesi birleştirilir, 3 sahne", len(seen) == 2 and seen[1].get("next") == "tok2" and seen[1]["collections"] == b0["collections"] and [m.scene_id for m in out] == ["A", "B", "C"])
check("koleksiyon kaydı", cl.collection_used == "sentinel-2-c1-l2a")
calls = {"n": 0}


def first_fails(verb, url, kw):
    calls["n"] += 1
    return Resp(400, text="collection yok") if kw["json"]["collections"][0] == "sentinel-2-c1-l2a" else Resp(200, data=page2)


cl2 = ls.StacClient(Sess([("/search", first_fails)]))
o2 = cl2.search((34.9, 38.9, 35.1, 39.1), D(2026, 7, 1), D(2026, 7, 31))
check("ilk koleksiyon 400 → eski koleksiyon denenir", cl2.collection_used == "sentinel-2-l2a" and len(o2) == 1 and calls["n"] == 2)
ok, msg = raises(lambda: ls.StacClient(Sess([("/search", Resp(500, text="boom"))])).search((1, 1, 2, 2), D(2026, 7, 1), D(2026, 7, 2)), "HTTP 500")
check("tüm koleksiyonlar başarısız → LiveSourceError + ayrıntı", ok, msg)
ok, msg = raises(lambda: ls.StacClient(Sess([("/search", ConnectionError("dns"))])).search((1, 1, 2, 2), D(2026, 7, 1), D(2026, 7, 2)), "bağlantı kurulamadı")
check("bağlantı hatası → anlaşılır ileti", ok, msg)
check("bitiş < başlangıç → hata", raises(lambda: cl.search((1, 1, 2, 2), D(2026, 7, 5), D(2026, 7, 1)), "Bitiş", ls.LiveSourceError)[0])
pages = {"n": 0}


def endless(verb, url, kw):
    pages["n"] += 1
    return Resp(200, data={"features": [feat(f"P{pages['n']}", "2026-07-01")], "links": [{"rel": "next", "method": "POST", "href": "https://x.test/search", "body": {"t": pages["n"]}}]})


ls.StacClient(Sess([("/search", endless)])).search((1, 1, 2, 2), D(2026, 7, 1), D(2026, 7, 2), max_pages=3)
check("sonsuz sayfalama max_pages ile durur", pages["n"] == 3)
get_next = {"features": [feat("G", "2026-07-01")], "links": [{"rel": "next", "method": "GET", "href": "https://x.test/search?token=1"}]}
n_get = {"n": 0}


def gn(verb, url, kw):
    n_get["n"] += 1
    return Resp(200, data=get_next)


ls.StacClient(Sess([("/search", gn)])).search((1, 1, 2, 2), D(2026, 7, 1), D(2026, 7, 2))
check("GET türü next bağlantısı izlenmez (yanlış gövdeyle döngüye girmez)", n_get["n"] == 1)

print("== Open-Meteo")
TODAY = D(2026, 11, 7)


def meteo_h(verb, url, kw):
    p = kw["params"]
    s, e = D.fromisoformat(p["start_date"]), D.fromisoformat(p["end_date"])
    n = (e - s).days + 1
    t = [(s + dt.timedelta(days=i)).isoformat() for i in range(n)]
    return Resp(200, data={"daily": {"time": t, "precipitation_sum": [1.5] * n, "temperature_2m_max": [30.0] * n, "temperature_2m_min": [15.0] * n,
                                      "wind_speed_10m_max": [20.0] * n, "et0_fao_evapotranspiration": [None] * n}})


ms_ = Sess([("archive-api", meteo_h), ("api.open-meteo.com", meteo_h)])
om = ls.OpenMeteoClient(ms_, TODAY)
df = om.daily(39.0, 35.0, D(2026, 7, 1), D(2026, 7, 10))
check("geçmiş aralık yalnızca arşiv uç noktası", len(ms_.calls) == 1 and "archive-api.open-meteo.com" in ms_.calls[0][1] and len(df) == 10)
q = ms_.calls[0][2]["params"]
check("parametreler: konum 4 ondalık, UTC, günlük değişkenler", q["latitude"] == "39.0000" and q["timezone"] == "UTC" and "et0_fao_evapotranspiration" in q["daily"] and "wind_speed_10m_max" in q["daily"])
check("sütunlar ve NaN (null ET0)", list(df.columns) == ["tarih", "yagis_mm", "tmax_c", "tmin_c", "ruzgar_kmh", "et0_mm"] and df.et0_mm.isna().all() and df.yagis_mm.iloc[0] == 1.5)
ms2 = Sess([("archive-api", meteo_h), ("api.open-meteo.com", meteo_h)])
df2 = ls.OpenMeteoClient(ms2, TODAY).daily(39.0, 35.0, D(2026, 10, 25), D(2026, 11, 12))
urls = [c[1] for c in ms2.calls]
check("yakın aralık arşiv + tahmin olarak bölünür, tekrar yok", len(urls) == 2 and "archive-api" in urls[0] and "archive-api" not in urls[1] and df2.tarih.is_unique and len(df2) == 19, str(urls))
check("tahmin parçası arşiv bitişinin ertesinden başlar", ms2.calls[1][2]["params"]["start_date"] == "2026-11-02" and ms2.calls[0][2]["params"]["end_date"] == "2026-11-01")
df3 = ls.OpenMeteoClient(Sess([("open-meteo", meteo_h)]), TODAY).daily(39.0, 35.0, D(2026, 11, 1), D(2027, 3, 1))
check("bitiş bugün+15 günle sınırlanır", df3.tarih.max() == pd.Timestamp(2026, 11, 22))
ok, msg = raises(lambda: ls.OpenMeteoClient(Sess([("open-meteo", Resp(200, data={"error": True, "reason": "Date out of range"}))]), TODAY).daily(39, 35, D(2026, 7, 1), D(2026, 7, 5)), "Date out of range", ls.LiveSourceError)
check("daily yok → neden ileti içinde", ok, msg)
check("429 → istek sınırı iletisi", raises(lambda: ls.OpenMeteoClient(Sess([("open-meteo", Resp(429))]), TODAY).daily(39, 35, D(2026, 7, 1), D(2026, 7, 5)), "429")[0])
check("bitiş < başlangıç → hata", raises(lambda: om.daily(39, 35, D(2026, 7, 5), D(2026, 7, 1)), "Bitiş", ls.LiveSourceError)[0])

print("== FIRMS parçalama ve kaynak")
check("12 gün → 5+5+2", ls.firms_chunks(D(2026, 7, 1), D(2026, 7, 12)) == [(D(2026, 7, 1), 5), (D(2026, 7, 6), 5), (D(2026, 7, 11), 2)])
check("1 gün → tek parça", ls.firms_chunks(D(2026, 7, 1), D(2026, 7, 1)) == [(D(2026, 7, 1), 1)])
check("parçalar boşluksuz ve çakışmasız", (lambda ch: sum(n for _, n in ch) == 37 and all(ch[i][0] + dt.timedelta(days=ch[i][1]) == ch[i + 1][0] for i in range(len(ch) - 1)))(ls.firms_chunks(D(2026, 7, 1), D(2026, 8, 6))))
check("60 günden uzun → hata", raises(lambda: ls.firms_chunks(D(2026, 1, 1), D(2026, 4, 1)), "en çok", ls.LiveSourceError)[0])
check("kaynak: yeni olay NRT, eski olay SP", ls.pick_firms_source(D(2026, 10, 20), TODAY) == "VIIRS_SNPP_NRT" and ls.pick_firms_source(D(2026, 7, 22), TODAY) == "VIIRS_SNPP_SP")
check("seçilen kaynaklar FIRMS_SOURCES'ta", ls.pick_firms_source(D(2026, 10, 20), TODAY) in dsrc.FIRMS_SOURCES and ls.pick_firms_source(D(2020, 1, 1), TODAY) in dsrc.FIRMS_SOURCES)

print("== Canlı kaynak: pencere okuma ve yansıtım")
reg = an.region_from_center("t", 39.0, 35.0, 0.6)
grid = an.make_grid(reg.bbox, 30)
meta = ls.parse_stac_items({"features": [feat("A", "2026-07-01", 4.0)]})[0]
reads = []


def fake_reader(href, bbox, g, categorical):
    reads.append((href, bbox, categorical))
    shp = (g.rows, g.cols)
    if href.endswith("scl.tif"):
        a = np.full(shp, 4, np.uint8)
        a[0, 0] = 9
        return a
    base = {"red": 1500, "nir": 5100}.get(href.rsplit("/", 1)[1][:-4], 1200)
    a = np.full(shp, base, np.uint16)
    a[1, 1] = 0  # nodata
    return a


src = ls.LiveSource(Sess([]), dsrc.Credentials(firms_map_key=KEY), reader=fake_reader, today=TODAY)
data = src.load(meta, reg, grid, ("red", "nir"))
check("DN 1500 × 1e-4 − 0,1 = 0,05 (kırmızı); DN 5100 → 0,41 (YKI)", abs(data.bands["red"][3, 3] - 0.05) < 1e-6 and abs(data.bands["nir"][3, 3] - 0.41) < 1e-6)
check("DN 0 → NaN (nodata), SCL 0'a çekilir", np.isnan(data.bands["red"][1, 1]) and data.scl[1, 1] == 0 and data.scl[0, 0] == 9 and data.scl[3, 3] == 4)
check("her bant bir kez okundu (red, nir, scl); SCL kategorik", sorted(r[0].rsplit("/", 1)[1] for r in reads) == ["nir.tif", "red.tif", "scl.tif"] and [r[2] for r in reads if r[0].endswith("scl.tif")] == [True])
check("okuma bölge sınır kutusuyla", all(r[1] == reg.bbox for r in reads))
check("yansıtım 0–1,5 arasına kırpılır", np.nanmax(data.bands["nir"]) <= 1.5)
st, idx = an.scene_stats(data, "ndvi", an.polygon_mask(grid, reg.ring))
check("okunan sahneden NDVI = (0,41−0,05)/(0,41+0,05)", abs(st.median - (0.41 - 0.05) / (0.41 + 0.05)) < 1e-4)
check("yanlış boyut → hata", raises(lambda: ls.LiveSource(Sess([]), reader=lambda h, b, g, c: np.zeros((3, 3))).load(meta, reg, grid, ("red", "nir")), "beklenen", ls.LiveSourceError)[0])
no_green = ls.parse_stac_items({"features": [feat("N", "2026-07-02")]})[0]
no_green.assets.pop("green")
check("eksik bant → hata", raises(lambda: src.load(no_green, reg, grid, ("green",)), "eksik bant", ls.LiveSourceError)[0])
no_scl = ls.parse_stac_items({"features": [feat("S", "2026-07-03")]})[0]
no_scl.assets.pop("scl")
d_ns = src.load(no_scl, reg, grid, ("red", "nir"))
check("SCL yoksa maske yok + not", d_ns.scl is None and any("SCL yok" in n for n in src.notes))
check("adres yoksa hata", raises(lambda: src.load(an.SceneMeta("e", D(2026, 7, 1), "S2", 1, "t"), reg, grid, ("red",)), "adresi yok", ls.LiveSourceError)[0])
bad = ls.LiveSource(Sess([]), reader=lambda h, b, g, c: (_ for _ in ()).throw(ls.LiveSourceError("okunamadı (OSError)")))
check("okuyucu hatası yukarı çıkar", raises(lambda: bad.load(meta, reg, grid, ("red", "nir")), "okunamadı", ls.LiveSourceError)[0])

print("== rasterio_reader (yüklü değilse açık hata; sahte modülle çağrı biçimi)")
saved = {k: sys.modules.get(k) for k in ("rasterio", "rasterio.enums", "rasterio.warp", "rasterio.windows")}
for k in saved:
    sys.modules.pop(k, None)
try:
    import rasterio  # noqa: F401
    have_real = True
except ImportError:
    have_real = False
if not have_real:
    ok, msg = raises(lambda: ls.rasterio_reader("https://x/a.tif", reg.bbox, grid, False), "rasterio", ls.LiveSourceError)
    check("rasterio yok → 'pip install rasterio' içeren hata", ok and "pip install" in msg, msg)
rec = {}


class FakeSrc:
    crs, transform = "EPSG:32636", "T"

    def read(self, band, **kw):
        rec["read"] = (band, kw)
        return np.full(kw["out_shape"], 7, np.uint16)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Env:
    def __init__(self, **kw):
        rec["env"] = kw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


fake = types.ModuleType("rasterio")
fake.Env = Env
fake.open = lambda href: (rec.__setitem__("href", href), FakeSrc())[1]
fe = types.ModuleType("rasterio.enums")
fe.Resampling = types.SimpleNamespace(nearest="NEAREST", bilinear="BILINEAR")
fw = types.ModuleType("rasterio.warp")
fw.transform_bounds = lambda crs_from, crs_to, w, s, e, n, densify_pts=21: (w * 1000, s * 1000, e * 1000, n * 1000)
fwin = types.ModuleType("rasterio.windows")
fwin.from_bounds = lambda lft, b, r, t, transform=None: ("WIN", lft, b, r, t, transform)
sys.modules.update({"rasterio": fake, "rasterio.enums": fe, "rasterio.warp": fw, "rasterio.windows": fwin})
try:
    a1 = ls.rasterio_reader("https://x/red.tif", reg.bbox, grid, False)
    kw = rec["read"][1]
    check("sahte rasterio: çıktı boyutu ızgara", a1.shape == (grid.rows, grid.cols) and kw["out_shape"] == (grid.rows, grid.cols))
    check("bant verisi: çift doğrusal, sınır dışı 0 dolgu, pencere dönüştürülmüş sınırlarla",
          kw["resampling"] == "BILINEAR" and kw["boundless"] is True and kw["fill_value"] == 0 and kw["window"][0] == "WIN" and kw["window"][1] == reg.bbox[0] * 1000)
    ls.rasterio_reader("https://x/scl.tif", reg.bbox, grid, True)
    check("kategorik bant: en yakın komşu", rec["read"][1]["resampling"] == "NEAREST")
    check("anahtarsız S3 ortamı ayarı", rec["env"].get("AWS_NO_SIGN_REQUEST") == "YES")
    fake.open = lambda href: (_ for _ in ()).throw(OSError("secret-token-123 failed"))
    ok, msg = raises(lambda: ls.rasterio_reader("https://x/red.tif", reg.bbox, grid, False), "okunamadı", ls.LiveSourceError)
    check("okuma hatası: yalnızca tür bildirilir (ham ileti sızmaz)", ok and "secret-token" not in msg and "OSError" in msg, msg)
finally:
    for k in saved:
        sys.modules.pop(k, None)
        if saved[k] is not None:
            sys.modules[k] = saved[k]

print("== FIRMS (canlı kaynak üzerinden)")
csv_text = ("latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,confidence,version,bright_ti5,frp,daynight\n"
            "39.001,35.001,340.1,0.4,0.4,2026-07-22,1030,N,h,2.0NRT,300.0,5.5,D\n")


def firms_h(verb, url, kw):
    return Resp(200, text=csv_text)


fs = Sess([("firms.modaps", firms_h)])
lsrc = ls.LiveSource(fs, dsrc.Credentials(firms_map_key=KEY), reader=fake_reader, today=TODAY)
hh = lsrc.hotspots(reg, D(2026, 7, 19), D(2026, 7, 30), D(2026, 7, 22))
urls = [c[1] for c in fs.calls]
check("12 gün → 3 istek, her biri 5/5/2 gün ve doğru kaynak", len(urls) == 3 and urls[0].endswith("/5/2026-07-19") and urls[2].endswith("/2/2026-07-29") and "VIIRS_SNPP_SP" in urls[0], str(urls))
check("aynı nokta 3 parçada dönse bile tekilleşir", len(hh) == 1 and list(hh.columns) == list(dsrc.HOTSPOT_COLUMNS))
ok, msg = raises(lambda: ls.LiveSource(Sess([]), dsrc.Credentials(), reader=fake_reader).hotspots(reg, D(2026, 7, 19), D(2026, 7, 20)), "MAP_KEY", ls.LiveSourceError)
check("anahtar yoksa açık hata", ok, msg)
ok, msg = raises(lambda: ls.LiveSource(Sess([("firms", Resp(403, text=f"bad key {KEY}"))]), dsrc.Credentials(firms_map_key=KEY), reader=fake_reader).hotspots(reg, D(2026, 7, 19), D(2026, 7, 20)), "HTTP 403")
check("403 iletisinde anahtar maskeli", ok and KEY not in msg, msg)
empty = ls.LiveSource(Sess([("firms", Resp(200, text=""))]), dsrc.Credentials(firms_map_key=KEY), reader=fake_reader).hotspots(reg, D(2026, 7, 19), D(2026, 7, 20))
check("boş yanıt → boş tablo (şema korunur)", empty.empty and list(empty.columns) == list(dsrc.HOTSPOT_COLUMNS))

print("== Tanılama")
good_routes = [("/search", lambda v, u, k: Resp(200, data={"features": [feat("OK1", (TODAY - dt.timedelta(days=5)).isoformat(), 2.0)], "links": []})),
               ("open-meteo", meteo_h), ("firms.modaps", firms_h)]
rows = ls.run_diagnostics(ls.LiveSource(Sess(good_routes), dsrc.Credentials(firms_map_key=KEY), reader=fake_reader, today=TODAY))
check("4 adım, hepsi başarılı", len(rows) == 4 and all(r.ok for r in rows), str([(r.step, r.ok, r.detail) for r in rows]))
check("ayrıntılar anlamlı (sahne sayısı, piksel, yağış)", "1 sahne" in rows[0].detail and "OK1" in rows[1].detail and "mm" in rows[2].detail)
rows2 = ls.run_diagnostics(ls.LiveSource(Sess([("/search", Resp(500, text="down")), ("open-meteo", meteo_h)]), dsrc.Credentials(), reader=fake_reader, today=TODAY))
check("katalog çökerse pencere adımı nedenini söyler, hava yine denenir", not rows2[0].ok and not rows2[1].ok and "sahne bulamadı" in rows2[1].detail and rows2[2].ok)
check("FIRMS anahtarı yoksa açıkça belirtilir", not rows2[3].ok and "FIRMS_MAP_KEY" in rows2[3].detail)
check("tanılama iletilerinde anahtar yok", all(KEY not in r.detail for r in rows + rows2))
boom = ls.run_diagnostics(ls.LiveSource(Sess(good_routes), dsrc.Credentials(firms_map_key=KEY), reader=lambda *a: (_ for _ in ()).throw(RuntimeError(f"gizli {KEY}")), today=TODAY))
check("beklenmeyen hata türü bildirilir, ham ileti sızmaz", (not boom[1].ok) and KEY not in boom[1].detail and "RuntimeError" in boom[1].detail, boom[1].detail)

print()
print("BAŞARISIZ:", FAILS if FAILS else "yok — tüm kontroller geçti")
sys.exit(1 if FAILS else 0)
