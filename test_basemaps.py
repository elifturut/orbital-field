"""basemaps.py çevrimdışı testleri (ağ, Streamlit ve Folium gerekmez)."""
import datetime as dt
import os
import re
import sys
from dataclasses import FrozenInstanceError, replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import basemaps as bm  # noqa: E402

FAILS = []
TODAY = dt.date(2026, 11, 4)


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  [{extra}]" if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)


# --- tanımlar
check("dört altlık", len(bm.BASEMAPS) == 4 and bm.LABELS == tuple(b.label for b in bm.BASEMAPS))
check("anahtarlar benzersiz", len({b.key for b in bm.BASEMAPS}) == 4)
check("etiketler benzersiz", len(set(bm.LABELS)) == 4)
check("ilk iki etiket eski sürümle aynı", bm.LABELS[:2] == ("Koyu harita (CARTO)", "Uydu altlığı (Esri)"))
check("varsayılan koyu harita", bm.by_label(bm.LABELS[0]).key == bm.DEFAULT_KEY)
check("yalnızca VIIRS canlı", [b.key for b in bm.BASEMAPS if b.live] == ["viirs"])
check("yalnızca VIIRS tarihli", [b.key for b in bm.BASEMAPS if b.dated] == ["viirs"])
check("tarihli katmanın alt sınırı var", all(b.first_day is not None for b in bm.BASEMAPS if b.dated))
check("tüm adresler https", all(b.url.startswith("https://") for b in bm.BASEMAPS))
check("her katmanda açıklama ve atıf", all(b.note and b.attribution for b in bm.BASEMAPS))
check("Esri ve EOX notu canlı olmadığını söyler", all("canlı değil" in bm.by_key(k).note for k in ("esri", "eox")))
check("VIIRS notu çözünürlük sınırını söyler", "250 m" in bm.by_key("viirs").note and "parsel" in bm.by_key("viirs").note)
check("EOX notu lisansı söyler", "lisans" in bm.by_key("eox").note)
try:
    bm.BASEMAPS[0].key = "x"  # type: ignore[misc]
    check("tanım değiştirilemez", False)
except FrozenInstanceError:
    check("tanım değiştirilemez", True)

# --- adres kurma
viirs = bm.by_key("viirs")
day = dt.date(2026, 11, 3)
url = bm.tile_url(viirs, day)
check("VIIRS adresi tarihi içerir", "/default/2026-11-03/" in url, url)
check("VIIRS adresi doğru katman ve matris", f"/{bm.VIIRS_LAYER}/" in url and f"/{bm.VIIRS_MATRIX}/" in url, url)
check("VIIRS adresinde z/y/x sırası", url.endswith("/{z}/{y}/{x}.jpg"), url)
check("VIIRS adresi EPSG:3857 ve best", "/wmts/epsg3857/best/" in url, url)
check("adreste doldurulmamış {date} kalmaz", "{date}" not in url)
check("VIIRS yerel yakınlaştırma 9", viirs.max_native_zoom == 9 and viirs.max_zoom > 9)
check("tarihsiz katman adresi değişmez", bm.tile_url(bm.by_key("esri"), day) == bm.by_key("esri").url)
check("Esri z/y/x sırası", bm.by_key("esri").url.endswith("/tile/{z}/{y}/{x}"))
check("CARTO alt alanlı", "{s}" in bm.by_key("dark").url and bm.by_key("dark").subdomains == "abcd")
for spec in bm.BASEMAPS:
    try:
        bm.tile_url(spec, day)
        ok = True
    except ValueError:
        ok = False
    check(f"adres geçerli: {spec.key}", ok)
for bad_url, label in (("https://x/{date}/{z}/{x}", "eksik y"), ("https://x/{z}/{y}/{x}/{date}", "doldurulamaz tarih")):
    broken = replace(viirs, url=bad_url) if label == "eksik y" else replace(viirs, url=bad_url, dated=False)
    try:
        bm.tile_url(broken, day)
        check(f"hatalı adres reddedilir: {label}", False)
    except ValueError:
        check(f"hatalı adres reddedilir: {label}", True)

# --- tarih mantığı
check("varsayılan gün dün", bm.default_date(TODAY) == dt.date(2026, 11, 3))
check("utc_today: UTC+3 gece yarısından sonra UTC günü", bm.utc_today(dt.datetime(2026, 11, 4, 1, 30, tzinfo=dt.timezone(dt.timedelta(hours=3)))) == dt.date(2026, 11, 3))
check("utc_today: saat dilimsiz değer olduğu gibi", bm.utc_today(dt.datetime(2026, 11, 4, 1, 30)) == dt.date(2026, 11, 4))
check("utc_today parametresiz", isinstance(bm.utc_today(), dt.date))
check("coerce: date", bm.coerce_date(dt.date(2026, 5, 1), TODAY) == dt.date(2026, 5, 1))
check("coerce: datetime", bm.coerce_date(dt.datetime(2026, 5, 1, 23, 0), TODAY) == dt.date(2026, 5, 1))
check("coerce: ISO metni", bm.coerce_date("2026-05-01", TODAY) == dt.date(2026, 5, 1))
check("coerce: zaman damgalı ISO metni", bm.coerce_date("2026-05-01T10:00:00Z", TODAY) == dt.date(2026, 5, 1))
check("coerce: liste ve demet", bm.coerce_date([dt.date(2026, 5, 1)], TODAY) == dt.date(2026, 5, 1) and bm.coerce_date((dt.date(2026, 5, 2),), TODAY) == dt.date(2026, 5, 2))
check("coerce: gelecek bugüne kırpılır", bm.coerce_date(dt.date(2030, 1, 1), TODAY) == TODAY)
check("coerce: çok eski ilk güne kırpılır", bm.coerce_date(dt.date(1999, 1, 1), TODAY) == bm.GIBS_FIRST_DAY)
check("coerce: None dünü verir", bm.coerce_date(None, TODAY) == dt.date(2026, 11, 3))
check("coerce: boş liste dünü verir", bm.coerce_date([], TODAY) == dt.date(2026, 11, 3))
check("coerce: bozuk metin dünü verir", bm.coerce_date("yarın", TODAY) == dt.date(2026, 11, 3))
check("coerce: sayı dünü verir", bm.coerce_date(20260501, TODAY) == dt.date(2026, 11, 3))
check("coerce: bugünün kendisine izin verir", bm.coerce_date(TODAY, TODAY) == TODAY)
check("coerce: özel alt sınır", bm.coerce_date(dt.date(2020, 1, 1), TODAY, first_day=dt.date(2021, 1, 1)) == dt.date(2021, 1, 1))
check("coerce: ayın ilk günü yılbaşında dün", bm.default_date(dt.date(2026, 1, 1)) == dt.date(2025, 12, 31))

# --- seçim
c0 = bm.make_choice(bm.LABELS[0], None, TODAY)
check("seçim: koyu harita uydu değil", c0.key == "dark" and not c0.is_satellite)
c1 = bm.make_choice(bm.LABELS[1], None, TODAY)
check("seçim: Esri uydu", c1.key == "esri" and c1.is_satellite)
c2 = bm.make_choice("Canlı uydu · NASA VIIRS (günlük)", dt.date(2026, 10, 1), TODAY)
check("seçim: VIIRS ve günü", c2.key == "viirs" and c2.day == dt.date(2026, 10, 1) and c2.is_satellite and c2.spec.live)
check("seçim: bilinmeyen etiket koyu harita", bm.make_choice("eski etiket", None, TODAY).key == "dark")
check("seçim: None etiket koyu harita", bm.make_choice(None, None, TODAY).key == "dark")
check("seçim: bozuk anahtar koyu harita", bm.by_key("yok").key == "dark" and bm.by_key(None).key == "dark")
check("seçim: gün her zaman aralıkta", all(bm.GIBS_FIRST_DAY <= bm.make_choice(lbl, v, TODAY).day <= TODAY for lbl in bm.LABELS for v in (None, "x", dt.date(1990, 1, 1), dt.date(2099, 1, 1), TODAY)))
try:
    c2.key = "dark"  # type: ignore[misc]
    check("seçim değiştirilemez", False)
except FrozenInstanceError:
    check("seçim değiştirilemez", True)

# --- etiketler
check("katman adı tarihli", bm.layer_name(viirs, day) == "Canlı uydu · NASA VIIRS (günlük) · 2026-11-03")
check("katman adı tarihsiz", bm.layer_name(bm.by_key("esri"), day) == "Uydu altlığı (Esri)")
check("atıf tarihli", bm.attribution(viirs, day).endswith("· 2026-11-03") and "NASA" in bm.attribution(viirs, day))
check("atıf tarihsiz", bm.attribution(bm.by_key("esri"), day) == "Tiles © Esri")
check("katman adları benzersiz", len({bm.layer_name(b, day) for b in bm.BASEMAPS}) == 4)
check("kenar çubuğu notu seçime göre", bm.sidebar_note(c2) == viirs.note and bm.sidebar_note(c0) == bm.by_key("dark").note)
check("etiketlerde ham yer tutucu yok", not any(re.search(r"[{}]", t) for b in bm.BASEMAPS for t in (b.label, b.note, b.attribution)))

# --- aktif yangın katmanı (NASA GIBS WMS)
fires = bm.fire_wms_options(dt.date(2026, 11, 3))
check("iki yangın katmanı (Suomi NPP, NOAA-20)", len(fires) == 2 and len(bm.FIRE_LAYERS) == 2)
check("yangın kimlikleri 375 m _All", all(o["layers"].startswith("VIIRS_") and o["layers"].endswith("_Thermal_Anomalies_375m_All") for o in fires))
check("yangın kimlikleri farklı uydular", {o["layers"] for o in fires} == {"VIIRS_SNPP_Thermal_Anomalies_375m_All", "VIIRS_NOAA20_Thermal_Anomalies_375m_All"})
check("yangın adresi https ve GIBS WMS EPSG:3857", all(o["url"] == "https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi" for o in fires))
check("yangın katmanı saydam PNG", all(o["fmt"] == "image/png" and o["transparent"] is True for o in fires))
check("yangın WMS sürümü 1.1.1", all(o["version"] == "1.1.1" for o in fires))
check("yangın zaman parametresi ISO günü", all(o["time"] == "2026-11-03" for o in fires))
check("yangın adı ve atıf gün içerir", all("2026-11-03" in o["name"] and "2026-11-03" in o["attr"] for o in fires))
check("yangın adları benzersiz ve uydu adını içerir", len({o["name"] for o in fires}) == 2 and any("Suomi NPP" in o["name"] for o in fires) and any("NOAA-20" in o["name"] for o in fires))
check("yangın katmanı adları altlık adlarıyla çakışmaz", not ({o["name"] for o in fires} & {bm.layer_name(b, dt.date(2026, 11, 3)) for b in bm.BASEMAPS}))
check("yangın notu sentetik senaryoyla ilgisizliği ve boşluk uyarısını söyler", "sentetik" in bm.FIRE_NOTE and "kanıtı değildir" in bm.FIRE_NOTE)
check("yangın etiketi gerçek olduğunu belirtir", "gerçek" in bm.FIRE_LABEL)
check("yangın günü değişince değerler değişir", bm.fire_wms_options(dt.date(2026, 1, 2))[0]["time"] == "2026-01-02")
check("yangın etiketinde ham yer tutucu yok", not re.search(r"[{}]", bm.FIRE_LABEL + bm.FIRE_NOTE))

print()
print("BAŞARISIZ: " + ", ".join(FAILS) if FAILS else "BAŞARISIZ: yok — tüm kontroller geçti")
sys.exit(1 if FAILS else 0)
