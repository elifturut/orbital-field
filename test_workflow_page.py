"""workflow_page.py ve app.py bağlantısı: saf yardımcılar + sahte Streamlit ile uçtan uca ekran sınaması.

UYARI: Gerçek Streamlit/Folium/Plotly çalıştırılmaz (sahte düzenek); bu test arayüz *mantığını* (adımlar, hata yolları, sızıntı yok,
bayat sonuç, PDF üretimi) sınar, tarayıcıda gerçek çizimi değil.
"""
import datetime as dt
import json
import os
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["ORBITAL_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "t.sqlite3")
import fakest  # noqa: E402

st = fakest.install()
import analysis as an  # noqa: E402
import app  # noqa: E402
import auth  # noqa: E402
import data_sources as dsrc  # noqa: E402
import workflow_page as wp  # noqa: E402
import yield_model as ym  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  [{extra}]" if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)


def raises(fn, needle="", exc=an.RegionError):
    try:
        fn()
    except exc as e:
        return needle in str(e), str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"yanlış istisna: {type(e).__name__}: {e}"
    return False, "istisna yok"


POLY = {"type": "Polygon", "coordinates": [[[35.00, 39.00], [35.01, 39.00], [35.01, 39.01], [35.00, 39.01], [35.00, 39.00]]]}

print("— saf yardımcılar")
r = wp.parse_geojson_text(json.dumps(POLY))
check("Polygon okunur", isinstance(r, an.Region) and 90 < r.area_ha < 110, r.area_ha)
check("Feature okunur", wp.parse_geojson_text(json.dumps({"type": "Feature", "properties": {}, "geometry": POLY})).area_ha > 50)
fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [35, 39]}}, {"type": "Feature", "geometry": POLY}]}
check("FeatureCollection: ilk çokgen özellik", wp.parse_geojson_text(json.dumps(fc)).area_ha > 50)
check("MultiPolygon: ilk çokgen", wp.parse_geojson_text(json.dumps({"type": "MultiPolygon", "coordinates": [POLY["coordinates"]]})).area_ha > 50)
ok, msg = raises(lambda: wp.parse_geojson_text(""), "boş")
check("boş GeoJSON reddedilir", ok, msg)
ok, msg = raises(lambda: wp.parse_geojson_text("{bozuk"), "geçerli JSON")
check("bozuk JSON reddedilir", ok, msg)
ok, msg = raises(lambda: wp.parse_geojson_text(json.dumps({"type": "Point", "coordinates": [1, 2]})), "çokgen")
check("nokta reddedilir", ok, msg)
ok, msg = raises(lambda: wp.parse_geojson_text('{"x":"' + "a" * (wp.MAX_GEOJSON_BYTES + 10) + '"}'), "büyük")
check("çok büyük dosya reddedilir", ok, msg)
big = {"type": "Polygon", "coordinates": [[[35, 39], [36, 39], [36, 40], [35, 40], [35, 39]]]}
ok, msg = raises(lambda: wp.parse_geojson_text(json.dumps(big)), "çok büyük")
check("25 km üstü bölge reddedilir", ok, msg)
ok, msg = raises(lambda: wp.parse_geojson_text(json.dumps({"type": "Polygon", "coordinates": "x"})), "")
check("bozuk koordinatlar RegionError verir (başka istisna değil)", ok, msg)

today = dt.date(2026, 11, 7)
d = wp.default_dates("yangin", False)
check("demo tarihleri hazır pencere + olay", d[0] < d[2] < d[1], d)
c = wp.default_dates("kuraklik", True, today)
check("özel bölge tarihleri bugünden geriye, olay aralıkta", c[1] < today and c[0] < c[2] < c[1], c)
check("özel pencere 400 günü aşmaz", all((x[1] - x[0]).days <= 400 for x in (wp.default_dates(k, True, today) for k in wp.KIND_ORDER)))
scn = wp.scenario_for("sel", r, dt.date(2026, 6, 1), dt.date(2026, 8, 1), dt.date(2026, 7, 1))
check("senaryo kullanıcı bölge ve tarihlerini taşır", scn.region == r and scn.start == dt.date(2026, 6, 1) and scn.event_date == dt.date(2026, 7, 1) and scn.kind == "sel")
scn_k = wp.scenario_for("kuraklik", r, dt.date(2026, 4, 20), dt.date(2026, 8, 31), dt.date(2026, 6, 15))
check("kuraklık senaryosu kataloğu geçen yıla uzatır", (dt.date(2026, 4, 20) - scn_k.start).days >= 365, scn_k.start)
import workflow as wf  # noqa: E402

rq = wf.AnalysisRequest(kind="sel", region=r, start=dt.date(2026, 6, 1), end=dt.date(2026, 8, 1), event_date=dt.date(2026, 7, 1))
f1 = wp.fingerprint(rq, False, "Buğday", 4.8)
check("parmak izi kararlı", f1 == wp.fingerprint(rq, False, "Buğday", 4.8))
import dataclasses  # noqa: E402

check("parmak izi: tarih/kaynak/ürün/verim/bulut değişince değişir",
      len({f1, wp.fingerprint(dataclasses.replace(rq, event_date=dt.date(2026, 7, 2)), False, "Buğday", 4.8), wp.fingerprint(rq, True, "Buğday", 4.8),
           wp.fingerprint(rq, False, "Arpa", 4.8), wp.fingerprint(rq, False, "Buğday", 5.0), wp.fingerprint(dataclasses.replace(rq, max_cloud=10.0), False, "Buğday", 4.8)}) == 6)
check("parsel kimliği kararlı", wp.parcel_id_for(r) == wp.parcel_id_for(r) and wp.parcel_id_for(r).startswith("PRS-"))

print("— uygulama bağlantısı")
check("sekme kaydı ile auth.INSURER_TABS birebir", set(app.TAB_RENDERERS) == set(auth.INSURER_TABS) and "ANALİZ İŞ AKIŞI" in app.TAB_RENDERERS)

acc = [a for a in auth.demo_accounts() if a.role is auth.Role.INSURER][0]
principal = auth.Principal(acc.username, acc.role, acc.display_name, None, None, acc.company, acc.title)


def login():
    st.session_state[app.SESSION_USER] = principal.to_session()
    st.session_state[app.SESSION_SEEN] = time.time()


def fresh(**state):
    st.reset()
    login()
    st.session_state.update(state)


def page(**kw):
    """Sekmeyi çizer; ``pressed`` düğmeleri tıklanmış sayılır."""
    st.out.clear()
    st.frames.clear()
    st.downloads.clear()
    for k in kw.get("press", ()):
        st.pressed.add(k)
    app.render_workflow_page()
    return st.html(), st.frame_html()


fresh()
html0, fr0 = page()
check("ilk çizim: adım başlıkları ve izleyici", all(x in html0 for x in ("ADIM 1", "ADIM 2", "ADIM 3", "ADIM 4", "wf-step")), html0[:200])
check("ilk çizim: henüz analiz yok notu", "Henüz analiz çalıştırılmadı" in html0)
check("ilk çizim: doğrulama paneli (eşik dayanağı) görünür", "EŞİKLER NEDEN BU DEĞERLER" in fr0)
check("ilk çizim: hata kartı yok", 'class="note err"' not in html0)

print("— her tür demo senaryoda uçtan uca")
for kind in wp.KIND_ORDER:
    fresh(wf_kind=kind)
    html, fr = page(press=["wf_run"])
    run = st.session_state.get("wf_run")
    check(f"{kind}: analiz çalıştı ve oturuma yazıldı", run is not None and run[1].request.kind == kind, html[-300:])
    if run is None:
        continue
    html, fr = page()
    check(f"{kind}: önce/sonra kaydırıcı (görüntüler gömülü)", "ÖNCE / SONRA" in fr and "data:image/png;base64," in fr and 'type="range"' in fr)
    check(f"{kind}: sahne listesi bulut rozetli + zaman serisi", "wf-scene" in fr and "wf-cloud" in fr and "<svg" in fr and "İNDEKS ZAMAN SERİSİ" in fr)
    check(f"{kind}: sınıf tablosu ve güvenilirlik göstergesi", "DEĞİŞİM SINIFLARI" in fr and "GÜVENİLİRLİK GÖSTERGELERİ" in fr)
    check(f"{kind}: hava doğrulaması", "HAVA DURUMU DOĞRULAMASI" in fr)
    check(f"{kind}: aynı parsel iki sektör görünümü", "SİGORTA GÖRÜNÜMÜ" in fr and "TARIM GÖRÜNÜMÜ" in fr)
    check(f"{kind}: uyarı önizlemesi simülasyon etiketli", "BİLDİRİM ÖNİZLEMESİ (SİMÜLASYON)" in fr and "gerçek SMS" in fr)
    check(f"{kind}: demo çıktı SENTETİK etiketli", "SENTETİK" in fr or "SENTETİK" in html)
    check(f"{kind}: yangın noktası paneli yalnızca yangında", ("YANGIN NOKTALARI" in fr) == (kind == "yangin"))
    check(f"{kind}: destek simülasyonu yalnızca kuraklıkta", ("KURAKLIK DESTEK ÖDEMESİ" in fr) == (kind == "kuraklik") and ("VARSAYIMSAL" in fr or "varsayımsal" in fr or kind != "kuraklik"))
    check(f"{kind}: rekolte paneli yalnızca tarımda", ("REKOLTE TAHMİNİ" in fr) == (kind == "tarim"), kind)
    check(f"{kind}: hata kartı yok", 'class="note err"' not in html)
    check(f"{kind}: izleyici 4-5. adımları tamamlandı sayar", html.count("wf-step done") >= 4 or "wf-step done" in html)
    pdf_before = len(st.downloads)
    html, fr = page(press=["wf_make_pdf"])
    saved = st.session_state.get("wf_pdf")
    check(f"{kind}: PDF üretildi ve indirme düğmesi var", saved is not None and saved[1][:4] == b"%PDF" and any(x["mime"] == "application/pdf" for x in st.downloads), html[-250:])
    del pdf_before

print("— kuraklık destek simülasyonu")
fresh(wf_kind="kuraklik")
page(press=["wf_run"])
kr = st.session_state["wf_run"][1]
sv = an.drought_support(kr.result, kr.insurance, 1000.0, 0.30)
check("destek: oran uygunluk eşiğine göre karar verir", sv.eligible == (kr.insurance.damage_ratio >= 0.30), (sv.eligible, kr.insurance.damage_ratio))
if sv.eligible and not sv.needs_review:
    check("destek: ödeme = ölçülebilir alan × oran × birim", abs(sv.payment_tl - kr.result.valid_ha * kr.insurance.damage_ratio * 1000.0) < 1e-6, sv.payment_tl)
sv_hi = an.drought_support(kr.result, kr.insurance, 1000.0, 1.0)
check("destek: eşik 1,0 ise ödeme yok (uygun değil)", (not sv_hi.eligible) and sv_hi.payment_tl == 0.0)
check("destek: birim 0 ise ödeme 0", an.drought_support(kr.result, kr.insurance, 0.0, 0.05).payment_tl == 0.0)
check("destek: notlar varsayımsal olduğunu söyler", any("VARSAYIMSAL" in n for n in sv.notes))
low = dataclasses.replace(kr.result, reliability=dataclasses.replace(kr.result.reliability, level="Düşük", score=30))
sv_low = an.drought_support(low, kr.insurance, 1000.0, 0.05)
check("destek: güvenilirlik düşükse otomatik ödeme yok, doğrulama istenir", sv_low.needs_review and sv_low.payment_tl == 0.0)
for bad in ((float("nan"), 0.3), (-1.0, 0.3), (1000.0, 0.0), (1000.0, 1.5)):
    ok, msg = raises(lambda bad=bad: an.drought_support(kr.result, kr.insurance, *bad), "", an.AnalysisError)
    check(f"destek: geçersiz parametre reddedilir {bad}", ok, msg)
fresh(wf_kind="yangin")
page(press=["wf_run"])
yr = st.session_state["wf_run"][1]
ok, msg = raises(lambda: an.drought_support(yr.result, yr.insurance, 1000.0), "yalnızca kuraklık", an.AnalysisError)
check("destek: kuraklık dışı tür reddedilir", ok, msg)

print("— hızlı demo girişi")
for role_key, role in (("login_try_customer", auth.Role.CUSTOMER), ("login_try_insurer", auth.Role.INSURER)):
    st.reset()
    st.pressed.add(role_key)
    try:
        app.render_login(time.time())
        hit = False
    except fakest.Rerun:
        hit = True
    sess = st.session_state.get(app.SESSION_USER)
    check(f"{role_key}: oturum açılır ve yeniden çizim istenir", hit and sess is not None, sess)
    pr2 = app.session_principal()
    check(f"{role_key}: doğru rol", pr2 is not None and pr2.role is role, pr2)
st.reset()
app.render_login(time.time())
check("giriş ekranı: iki hızlı deneme düğmesi etiketi", "HIZLI DENEME" in st.html())
os.environ["ORBITAL_ACCOUNTS_JSON"] = json.dumps([{"username": "x", "role": "sigortaci", "display_name": "X", "salt": "00" * 16, "hash": "00" * 32, "iterations": 200000}])
try:
    st.reset()
    app.render_login(time.time())
    check("gerçek hesap yapılandırılınca hızlı deneme düğmeleri yok", "HIZLI DENEME" not in st.html())
finally:
    os.environ.pop("ORBITAL_ACCOUNTS_JSON", None)

print("— bayat sonuç, kaynak hataları, sızıntı")
fresh(wf_kind="yangin")
page(press=["wf_run"])
st.session_state["wf_kind"] = "sel"
html, _ = page()
check("girdi değişince 'önceki çalıştırmaya aittir' uyarısı", "önceki çalıştırmaya" in html)

# canlı kaynak hatası: sessizce demoya düşülmez
SECRET = "SECRETKEY123"


class BrokenLive:
    name = "kırık canlı"
    is_demo = False
    _today = None

    def __init__(self, session=None, creds=None, **kw):
        self.creds = creds

    def catalog(self, region, start, end):
        raise dsrc.DataSourceError("STAC bağlantısı kurulamadı (sınama)")


orig_live = wp.ls.LiveSource
wp.ls.LiveSource = BrokenLive
try:
    fresh(wf_region_mode=wp.MODE_COORD, wf_source=wp.SRC_LIVE, wf_kind="yangin")
    st.secrets = {"FIRMS_MAP_KEY": SECRET}
    html, fr = page(press=["wf_run"])
    check("canlı hata: kullanıcıya iletilir, demoya düşülmez", "Analiz yapılamadı" in html and "sentetik veriye geçilmedi" in html and st.session_state.get("wf_run") is None, html[-300:])
    check("gizli anahtar ekrana sızmaz", SECRET not in html and SECRET not in fr)
    html, fr = page(press=["wf_diag"])
    check("tanılama: adımlar tablo olarak gösterilir, anahtar sızmaz", "CANLI BAĞLANTI SINAMASI" in fr and SECRET not in html + fr, fr[-300:])
finally:
    wp.ls.LiveSource = orig_live
    st.secrets = {}

print("— bölge kipleri")
fresh(wf_region_mode=wp.MODE_COORD, wf_kind="sel", wf_lat=41.0, wf_lon=40.5, wf_half=1.0)
html, _ = page(press=["wf_run"])
check("koordinat kipi: bölge kartı + demo kaynakla çalışır", "SEÇİLİ BÖLGE" in html and st.session_state.get("wf_run") is not None and st.session_state["wf_run"][1].is_demo, html[-300:])
fresh(wf_region_mode=wp.MODE_COORD, wf_lat=39.0, wf_lon=35.0, wf_half=0.03)
html, _ = page()
check("geçersiz yarı kenar hata verir, analiz başlamaz", 'class="note err"' in html and "Yarı kenar" in html and st.session_state.get("wf_run") is None, html[-200:])
fresh(wf_region_mode=wp.MODE_COORD, wf_lat=39.0, wf_lon=35.0, wf_half=0.05)
html, _ = page(press=["wf_run"])
check("en küçük geçerli bölge çalışır (0,05 km yarı kenar ≈ 1 ha)", st.session_state.get("wf_run") is not None or "Analiz yapılamadı" in html, html[-200:])

fresh(wf_region_mode=wp.MODE_GEOJSON, wf_geojson_text=json.dumps(POLY), wf_kind="tarim")
html, _ = page()
check("GeoJSON kipi: bölge kuruldu", "GeoJSON bölgesi" in html)
fresh(wf_region_mode=wp.MODE_GEOJSON, wf_geojson_text="{bozuk")
html, _ = page()
check("GeoJSON kipi: geçersiz metin hata kartı", 'class="note err"' in html and "geçerli JSON" in html)
fresh(wf_region_mode=wp.MODE_GEOJSON)
html, _ = page()
check("GeoJSON kipi: boşken yönlendirme (hata değil)", "yapıştırın" in html and 'class="note err"' not in html)

fresh(wf_region_mode=wp.MODE_MAP)
html, _ = page()
check("harita kipi: çizim yokken yönlendirme", "çokgen ya da dikdörtgen çizin" in html and any(k.get("key") == "wf_map_0" for _, k in st.folium_calls))
st.map_return["wf_map_0"] = {"all_drawings": [{"type": "Feature", "geometry": POLY}]}
html, _ = page()
check("harita kipi: çizilen çokgen bölge olur ve oturumda kalır", "Haritadan çizilen bölge" in html and st.session_state.get("wf_drawn") is not None)
st.map_return.clear()
html, _ = page()
check("harita kipi: sonraki çalıştırmada (çizim dönmese de) bölge korunur", "Haritadan çizilen bölge" in html)
html, _ = page(press=["wf_clear_draw"])
check("harita kipi: temizle bölgeyi siler ve harita anahtarını değiştirir", st.session_state.get("wf_drawn") is None and st.session_state.get("wf_map_n") == 1)
st.map_return["wf_map_1"] = {"all_drawings": [{"type": "Feature", "geometry": big}]}
html, _ = page()
check("harita kipi: aşırı büyük çizim hata kartı", "çok büyük" in html and 'class="note err"' in html)

print("— rekolte, doğrulama, rapor")
fresh(wf_kind="tarim")
page(press=["wf_run"])
check("rekolte paketi oturumda", st.session_state.get("wf_yield") is not None)
run = st.session_state["wf_run"][1]
b = wp.yield_for_run(run, "Buğday", 4.8)
check("rekolte: sentetik model etiketli, aralık sıralı", b.synthetic and b.estimate.lo <= b.estimate.value <= b.estimate.hi and b.compare is not None)
bad_run = dataclasses.replace(run, weather=None)
ok, msg = raises(lambda: wp.yield_for_run(bad_run, "Buğday", 4.8), "Hava verisi yok", ym.YieldError)
check("rekolte: hava yoksa açık hata", ok, msg)


class Up:
    def __init__(self, data):
        self._d = data

    def getvalue(self):
        return self._d


st.session_state["wf_history"] = Up(b"a,b\n1,2\n")
html, fr = page()
check("geçmiş verim CSV'si bozuksa hata kartı (analiz çökmez)", "okunamadı" in html and 'class="note err"' in html, html[-300:])
st.session_state["wf_history"] = None

html, fr = page(press=["wf_synth"])
check("sentetik doğrulama çalışır ve SENTETİK etiketli", "SENTETİK DOĞRULAMA" in html and "ROC" in fr and "hasarlı tanımı" in html.lower() + " ".join(str(t.columns.tolist()) for t in st.tables).lower())
rows = ["parsel_id,dnbr,hasarli"] + [f"P{i},{0.05 + 0.012 * i:.3f},{1 if i > 25 else 0}" for i in range(60)]
st.session_state["wf_truth"] = Up("\n".join(rows).encode())
html, fr = page()
check("kendi saha CSV'si değerlendirilir", "SAHA DOĞRULAMASI" in html and "EŞİK BAŞINA HATA ORANLARI" in fr, html[-300:])
st.session_state["wf_truth"] = Up(b"x,y\n1,2\n")
html, _ = page()
check("bozuk doğrulama CSV'si hata kartı", "Doğrulama yapılamadı" in html)
st.session_state["wf_truth"] = None

# PDF fotoğraf yolu
import io  # noqa: E402

from PIL import Image  # noqa: E402

buf = io.BytesIO()
Image.new("RGB", (200, 120), (30, 120, 60)).save(buf, "JPEG")
st.session_state["wf_photo"] = Up(buf.getvalue())
st.session_state["wf_expert"] = "Mehmet Kaya"
html, _ = page(press=["wf_make_pdf"])
pdf = st.session_state["wf_pdf"][1]
check("fotoğraflı PDF üretilir", pdf[:4] == b"%PDF" and len(pdf) > 5000)
st.session_state["wf_photo"] = Up(b"bu bir fotograf degil")
html, _ = page(press=["wf_make_pdf"])
check("geçersiz fotoğraf: kullanıcıya anlaşılır hata, çökme yok", "Rapor üretilemedi" in html and "okunamadı" in html, html[-300:])

print("— sekme hata yalıtımı")
fresh()
app.safe_render(lambda: (_ for _ in ()).throw(RuntimeError("deneme")), "Analiz iş akışı")
check("safe_render: sigortacıya ayrıntı gösterir", "çizilemedi" in st.html())

print()
if FAILS:
    print(f"BAŞARISIZ ({len(FAILS)}):")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("BAŞARISIZ: yok — tüm kontroller geçti")
del types
