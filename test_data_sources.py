"""data_sources.py çevrimdışı testleri (sahte HTTP oturumu; ağ yok)."""
import datetime as dt
import json
import math
import sys

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd  # noqa: E402

import data_sources as ds  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  [{extra}]" if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)


def raises(fn, needle=""):
    try:
        fn()
    except ds.DataSourceError as exc:
        return needle in str(exc), str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, f"yanlış istisna: {type(exc).__name__}: {exc}"
    return False, "istisna yok"


class Resp:
    def __init__(self, status=200, text="", data=None):
        self.status_code, self.text, self._data = status, text, data

    def json(self):
        if self._data is None:
            raise ValueError("json değil")
        return self._data


class Sess:
    """Sıradaki yanıtı döndüren sahte oturum; çağrıları kaydeder."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls = []

    def _next(self, verb, url, kw):
        self.calls.append((verb, url, kw))
        item = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    def post(self, url, **kw):
        return self._next("POST", url, kw)

    def get(self, url, **kw):
        return self._next("GET", url, kw)


POLY = {"type": "Polygon", "coordinates": [[[28.30, 37.00], [28.31, 37.00], [28.31, 37.01], [28.30, 37.01], [28.30, 37.00]]]}
SECRET, KEY = "s3cr3t-VALUE", "ABCDEF1234KEY"

print("== Sentinel-2 istek gövdesi")
p = ds.build_statistics_payload(POLY, dt.date(2026, 5, 1), dt.date(2026, 7, 31))
json.dumps(p)
agg = p["aggregation"]
check("JSON'a çevrilebilir", True)
check("aralık P5D", agg["aggregationInterval"] == {"of": "P5D"})
check("to tam dilim sınırı (92 gün → 18×5=90)", agg["timeRange"] == {"from": "2026-05-01T00:00:00Z", "to": "2026-07-30T00:00:00Z"}, str(agg["timeRange"]))
check("crs EPSG:4326", p["input"]["bounds"]["properties"]["crs"].endswith("/EPSG/0/4326"))
check("veri türü sentinel-2-l2a", p["input"]["data"][0]["type"] == "sentinel-2-l2a")
check("bulut filtresi", p["input"]["data"][0]["dataFilter"]["maxCloudCoverage"] == 60.0 and p["input"]["data"][0]["dataFilter"]["mosaickingOrder"] == "leastCC")
check("derece çözünürlüğü (10 m)", 1.0e-4 < agg["resx"] < 1.3e-4 and 8.5e-5 < agg["resy"] < 9.5e-5, f"{agg['resx']} {agg['resy']}")
check("resx > resy (enlem 37°)", agg["resx"] > agg["resy"])
check("evalscript NDVI", "ndvi" in agg["evalscript"] and "SCL" in agg["evalscript"] and agg["evalscript"].startswith("//VERSION=3"))
p2 = ds.build_statistics_payload(POLY, dt.date(2026, 5, 1), dt.date(2026, 5, 1), index="nbr", max_cloud_pct=None)
check("tek gün: P1D, to=ertesi gün", p2["aggregation"]["aggregationInterval"]["of"] == "P1D" and p2["aggregation"]["timeRange"]["to"] == "2026-05-02T00:00:00Z")
check("bulut filtresi yok", "maxCloudCoverage" not in p2["input"]["data"][0]["dataFilter"])
check("NBR evalscript B12", "B12" in p2["aggregation"]["evalscript"] and 'id: "nbr"' in p2["aggregation"]["evalscript"])
p3 = ds.build_statistics_payload(POLY, dt.date(2026, 5, 1), dt.date(2026, 5, 3), interval_days=5)
check("pencere < dilim → P3D", p3["aggregation"]["aggregationInterval"]["of"] == "P3D" and p3["aggregation"]["timeRange"]["to"] == "2026-05-04T00:00:00Z")
bad = {"type": "Point", "coordinates": [1, 2]}
for label, fn, needle in (
    ("bitiş<başlangıç", lambda: ds.build_statistics_payload(POLY, dt.date(2026, 5, 2), dt.date(2026, 5, 1)), "Bitiş"),
    ("bilinmeyen indeks", lambda: ds.build_statistics_payload(POLY, dt.date(2026, 5, 1), dt.date(2026, 5, 2), index="evi"), "indeks"),
    ("Point geometri", lambda: ds.build_statistics_payload(bad, dt.date(2026, 5, 1), dt.date(2026, 5, 2)), "Polygon"),
    ("interval 0", lambda: ds.build_statistics_payload(POLY, dt.date(2026, 5, 1), dt.date(2026, 5, 2), interval_days=0), "interval"),
    ("koordinat aralık dışı", lambda: ds.build_statistics_payload({"type": "Polygon", "coordinates": [[[500, 1], [501, 1], [501, 2], [500, 1]]]}, dt.date(2026, 5, 1), dt.date(2026, 5, 2)), "derece"),
    ("az köşe", lambda: ds.build_statistics_payload({"type": "Polygon", "coordinates": [[[1, 1], [2, 2]]]}, dt.date(2026, 5, 1), dt.date(2026, 5, 2)), "köşe"),
):
    ok, msg = raises(fn, needle)
    check(f"hata: {label}", ok, msg)
mp = {"type": "MultiPolygon", "coordinates": [POLY["coordinates"], POLY["coordinates"]]}
check("MultiPolygon kabul", ds.build_statistics_payload(mp, dt.date(2026, 5, 1), dt.date(2026, 5, 6))["input"]["bounds"]["geometry"]["type"] == "MultiPolygon")

print("== Sentinel-2 yanıt ayrıştırma")
RESP = {"data": [
    {"interval": {"from": "2026-05-01T00:00:00Z", "to": "2026-05-06T00:00:00Z"},
     "outputs": {"ndvi": {"bands": {"B0": {"stats": {"min": 0.2, "max": 0.8, "mean": 0.55, "stDev": 0.1, "sampleCount": 100, "noDataCount": 20}}}}}},
    {"interval": {"from": "2026-05-06T00:00:00Z", "to": "2026-05-11T00:00:00Z"},
     "outputs": {"ndvi": {"bands": {"B0": {"stats": {"min": "NaN", "max": "NaN", "mean": "NaN", "stDev": "NaN", "sampleCount": 100, "noDataCount": 100}}}}}},
    {"interval": {"from": "2026-05-11T00:00:00Z", "to": "2026-05-16T00:00:00Z"}, "error": {"type": "x"}},
], "status": "OK"}
df = ds.parse_statistics_response(RESP, "ndvi")
check("3 satır, sütunlar", len(df) == 3 and list(df.columns) == list(ds.STATS_COLUMNS))
check("ortalama 0,55", abs(df.loc[0, "ortalama"] - 0.55) < 1e-9)
check("geçerli oran 0,8", abs(df.loc[0, "gecerli_oran"] - 0.8) < 1e-9)
check("NaN metni → nan, oran 0", math.isnan(df.loc[1, "ortalama"]) and df.loc[1, "gecerli_oran"] == 0.0)
check("hatalı dilim nan satırı", math.isnan(df.loc[2, "ortalama"]) and math.isnan(df.loc[2, "gecerli_oran"]))
check("zaman UTC", str(df["baslangic"].dt.tz) == "UTC" and df.loc[0, "baslangic"] == pd.Timestamp("2026-05-01", tz="UTC"))
empty = ds.parse_statistics_response({"data": []}, "ndvi")
check("boş yanıt → boş tablo, sütunlar", empty.empty and list(empty.columns) == list(ds.STATS_COLUMNS))
check("yanlış çıktı kimliği → nan satırı", math.isnan(ds.parse_statistics_response(RESP, "nbr").loc[0, "ortalama"]))

print("== CDSE kimlik doğrulama")
clock = [0.0]
tok = Resp(200, data={"access_token": "TKN1", "expires_in": 100})
s = Sess(tok)
auth = ds.CdseAuth("cid", SECRET, session=s, clock=lambda: clock[0])
check("belirteç alındı", auth.token() == "TKN1" and len(s.calls) == 1)
check("form alanları", s.calls[0][2]["data"] == {"grant_type": "client_credentials", "client_id": "cid", "client_secret": SECRET} and s.calls[0][1] == ds.CDSE_TOKEN_URL)
clock[0] = 60.0
auth.token()
check("önbellek (60 sn < 100−30)", len(s.calls) == 1)
clock[0] = 80.0
auth.token()
check("süre dolmadan 30 sn önce yenilenir", len(s.calls) == 2)
check("repr sırrı göstermez", SECRET not in repr(auth))
ok, msg = raises(lambda: ds.CdseAuth("", SECRET, session=s), "eksik")
check("boş kimlik → hata", ok, msg)
ok, msg = raises(lambda: ds.CdseAuth("cid", SECRET, session=Sess(Resp(401, text=f"bad client_secret={SECRET}"))).token(), "HTTP 401")
check("401 → hata", ok, msg)
ok, msg = raises(lambda: ds.CdseAuth("cid", SECRET, session=Sess(Resp(401, text=f"bad client_secret={SECRET}"))).token(), "***")
check("401 gövdesinde sır maskelenir", ok and SECRET not in msg, msg)
ok, msg = raises(lambda: ds.CdseAuth("cid", SECRET, session=Sess(Resp(429, text="slow down"))).token(), "429")
check("429 → anlaşılır ileti", ok, msg)
ok, msg = raises(lambda: ds.CdseAuth("cid", SECRET, session=Sess(Resp(200, data={"x": 1}))).token(), "access_token")
check("access_token yok → hata", ok, msg)
ok, msg = raises(lambda: ds.CdseAuth("cid", SECRET, session=Sess(Resp(200, text="<html>"))).token(), "JSON")
check("JSON olmayan yanıt → hata", ok, msg)
ok, msg = raises(lambda: ds.CdseAuth("cid", SECRET, session=Sess(ConnectionError(f"https://x?client_secret={SECRET}"))).token(), "ConnectionError")
check("bağlantı hatasında sır sızmaz", ok and SECRET not in msg, msg)

print("== CDSE istatistik istemcisi")
s2 = Sess(Resp(200, data={"access_token": "TKN", "expires_in": 600}))
s2.queue = [Resp(200, data={"access_token": "TKN", "expires_in": 600}), Resp(200, data=RESP)]
# token POST'u sıra 1, istatistik POST'u sıra 2 — aynı sahte oturum
auth2 = ds.CdseAuth("cid", SECRET, session=s2)
cli = ds.CdseStatisticsClient(auth2)
out = cli.series(POLY, dt.date(2026, 5, 1), dt.date(2026, 5, 15))
check("istatistik POST adresi", s2.calls[1][1] == ds.CDSE_STATISTICS_URL)
check("Bearer başlığı", s2.calls[1][2]["headers"]["Authorization"] == "Bearer TKN")
check("gövde JSON ve indeks sütunu", s2.calls[1][2]["json"]["aggregation"]["aggregationInterval"] == {"of": "P5D"} and set(out["indeks"]) == {"ndvi"})


def nbr_resp(mean):
    return Resp(200, data={"data": [{"interval": {"from": "2026-05-01T00:00:00Z", "to": "2026-05-11T00:00:00Z"},
                                   "outputs": {"nbr": {"bands": {"B0": {"stats": {"mean": mean, "min": 0, "max": 1, "stDev": 0.1, "sampleCount": 50, "noDataCount": 5}}}}}}]})


s3 = Sess(Resp(200, data={"access_token": "T", "expires_in": 600}))
s3.queue = [Resp(200, data={"access_token": "T", "expires_in": 600}), nbr_resp(0.62), nbr_resp(0.18)]
d = ds.CdseStatisticsClient(ds.CdseAuth("cid", SECRET, session=s3)).dnbr(POLY, (dt.date(2026, 5, 1), dt.date(2026, 5, 10)), (dt.date(2026, 5, 20), dt.date(2026, 5, 29)))
check("dNBR = önce − sonra = 0,44", abs(d - 0.44) < 1e-9, str(d))
check("dNBR istekleri NBR + P10D", all(c[2]["json"]["aggregation"]["aggregationInterval"] == {"of": "P10D"} for c in s3.calls[1:]))
s4 = Sess(Resp(200, data={"access_token": "T", "expires_in": 600}))
s4.queue = [Resp(200, data={"access_token": "T", "expires_in": 600}), nbr_resp("NaN")]
ok, msg = raises(lambda: ds.CdseStatisticsClient(ds.CdseAuth("cid", SECRET, session=s4)).dnbr(POLY, (dt.date(2026, 5, 1), dt.date(2026, 5, 10)), (dt.date(2026, 5, 20), dt.date(2026, 5, 29))), "bulutsuz")
check("bulutsuz gözlem yoksa dNBR hatası", ok, msg)

print("== NASA FIRMS")
BBOX = (28.0, 36.8, 28.6, 37.2)
url = ds.build_firms_url(KEY, "VIIRS_SNPP_NRT", BBOX, 2)
check("URL biçimi", url == f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{KEY}/VIIRS_SNPP_NRT/28,36.8,28.6,37.2/2", url)
check("URL + tarih", ds.build_firms_url(KEY, "MODIS_NRT", BBOX, 1, dt.date(2026, 8, 14)).endswith("/1/2026-08-14"))
for label, fn, needle in (
    ("anahtar yok", lambda: ds.build_firms_url("", "MODIS_NRT", BBOX), "MAP_KEY"),
    ("kaynak yok", lambda: ds.build_firms_url(KEY, "LANDSAT_NRT", BBOX), "kaynağı"),
    ("gün 6", lambda: ds.build_firms_url(KEY, "MODIS_NRT", BBOX, 6), "day_range"),
    ("gün 0", lambda: ds.build_firms_url(KEY, "MODIS_NRT", BBOX, 0), "day_range"),
    ("bbox ters", lambda: ds.build_firms_url(KEY, "MODIS_NRT", (28.6, 36.8, 28.0, 37.2)), "bbox"),
    ("bbox enlem>90", lambda: ds.build_firms_url(KEY, "MODIS_NRT", (28.0, 36.8, 28.6, 91)), "bbox"),
):
    ok, msg = raises(fn, needle)
    check(f"hata: {label}", ok, msg)

VIIRS = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
    "37.01234,28.31234,330.5,0.39,0.36,2026-08-14,52,N,VIIRS,n,2.0NRT,290.1,12.4,N\n"
    "37.02,28.32,360.1,0.4,0.37,2026-08-14,1304,N,VIIRS,h,2.0NRT,300.0,45.0,D\n"
    "37.03,28.33,310.2,0.4,0.37,2026-08-13,2359,N,VIIRS,l,2.0NRT,295.0,3.1,D\n")
v = ds.parse_firms_csv(VIIRS, "VIIRS_SNPP_NRT")
check("VIIRS 3 satır + şema", len(v) == 3 and list(v.columns) == list(ds.HOTSPOT_COLUMNS))
check("zamana göre sıralı", list(v["tarih_utc"]) == sorted(v["tarih_utc"]))
check("acq_time 52 → 00:52 UTC", pd.Timestamp("2026-08-14 00:52", tz="UTC") in set(v["tarih_utc"]))
check("güven l/n/h → 0,30/0,70/0,95", sorted(v["guven"]) == [0.30, 0.70, 0.95], str(list(v["guven"])))
check("kaynak etiketi", set(v["kaynak"]) == {"VIIRS-375m"})
check("FRP ve parlaklık sayısal", v["frp_mw"].dtype.kind == "f" and v["parlaklik_k"].max() == 360.1)
check("gece/gündüz", set(v["gece_gunduz"]) == {"N", "D"})
MODIS = (
    "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight\n"
    "37.0,28.3,320.0,1.1,1.0,2026-08-14,1010,Terra,MODIS,85,6.1NRT,295.0,20.5,D\n")
m = ds.parse_firms_csv(MODIS, "MODIS_NRT")
check("MODIS güven 85 → 0,85, kaynak", m.loc[0, "guven"] == 0.85 and m.loc[0, "kaynak"] == "MODIS-1km" and m.loc[0, "parlaklik_k"] == 320.0)
check("başlık-only → boş tablo", ds.parse_firms_csv(VIIRS.splitlines()[0] + "\n", "VIIRS_SNPP_NRT").empty)
check("boş metin → boş tablo + sütunlar", list(ds.parse_firms_csv("", "MODIS_NRT").columns) == list(ds.HOTSPOT_COLUMNS))
ok, msg = raises(lambda: ds.parse_firms_csv("Invalid MAP_KEY.", "MODIS_NRT"), "CSV değil")
check("geçersiz anahtar iletisi → hata", ok, msg)
ok, msg = raises(lambda: ds.parse_firms_csv("latitude,longitude\n1,2\n", "MODIS_NRT"), "eksik")
check("zorunlu sütun eksik → hata", ok, msg)
required = {"enlem", "boylam", "frp_mw", "parlaklik_k", "kaynak", "guven"}
check("data_engine termal şemasını kapsar", required <= set(ds.HOTSPOT_COLUMNS))

fc = ds.FirmsClient(KEY, session=Sess(Resp(200, text=VIIRS)))
hs = fc.hotspots(BBOX, "VIIRS_SNPP_NRT", 1)
check("istemci GET + ayrıştırma", len(hs) == 3 and fc.session.calls[0][0] == "GET" and KEY in fc.session.calls[0][1])
ok, msg = raises(lambda: ds.FirmsClient(KEY, session=Sess(Resp(403, text=f"forbidden key {KEY}"))).hotspots(BBOX), "HTTP 403")
check("403 → hata, anahtar maskeli", ok and KEY not in msg, msg)
ok, msg = raises(lambda: ds.FirmsClient(KEY, session=Sess(TimeoutError(f"GET /{KEY}/ zaman aşımı"))).hotspots(BBOX), "TimeoutError")
check("zaman aşımında anahtar sızmaz", ok and KEY not in msg, msg)
ok, msg = raises(lambda: ds.FirmsClient(""), "MAP_KEY")
check("boş anahtar → hata", ok, msg)
check("FirmsClient repr anahtarı göstermez", KEY not in repr(vars(ds.FirmsClient(KEY, session=Sess(Resp(200, text=""))))).replace(f"'_map_key': '{KEY}'", ""))

print("== Kimlik bilgileri ve hat özeti")
c = ds.load_credentials({"CDSE_CLIENT_ID": " id ", "CDSE_CLIENT_SECRET": SECRET, "FIRMS_MAP_KEY": KEY})
check("ortam okuma + kırpma", c.cdse_client_id == "id" and c.has_cdse and c.has_firms)
check("Credentials repr sır göstermez", SECRET not in repr(c) and KEY not in repr(c))
e = ds.load_credentials({})
check("boş ortam → yok", not e.has_cdse and not e.has_firms)
check("PIPELINE 6 kaynak, hepsi dolu", len(ds.PIPELINE) == 6 and all(all(isinstance(x, str) and x for x in vars(r).values()) for r in ds.PIPELINE))

print()
print("BAŞARISIZ:", FAILS if FAILS else "yok — tüm kontroller geçti")
sys.exit(1 if FAILS else 0)
