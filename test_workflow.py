"""workflow.py çevrimdışı testleri (sentetik kaynak + hatalı kaynak sarmalayıcıları)."""
import dataclasses
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import analysis as an  # noqa: E402
import data_engine as de  # noqa: E402
import data_sources as dsrc  # noqa: E402
import demo_imagery as di  # noqa: E402
import workflow as wf  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  [{extra}]" if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)


def raises(fn, needle="", exc=wf.WorkflowError):
    try:
        fn()
    except exc as e:
        return needle in str(e), str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"yanlış istisna: {type(e).__name__}: {e}"
    return False, "istisna yok"


D = dt.date


def request(kind, **kw):
    scn = di.preset(kind)
    ws, we = di.default_window(scn)
    base = dict(kind=kind, region=scn.region, start=ws, end=we, event_date=scn.event_date, max_cloud=40.0)
    base.update(kw)
    return scn, wf.AnalysisRequest(**base)


class Wrap:
    """DemoSource'u sarar; belirli adımları bozar."""

    is_demo = True
    name = "wrap"

    def __init__(self, inner, fail_ids=(), fail_all=False, cat_error=None, wx_error=None, hs_error=None, empty_cat=False, cloudy=False, unknown_cloud=False):
        self.inner, self.fail_ids, self.fail_all = inner, set(fail_ids), fail_all
        self.cat_error, self.wx_error, self.hs_error, self.empty_cat = cat_error, wx_error, hs_error, empty_cat
        self.cloudy, self.unknown_cloud = cloudy, unknown_cloud
        self.notes = ["not-1"]

    def catalog(self, region, start, end):
        if self.cat_error:
            raise dsrc.DataSourceError(self.cat_error)
        if self.empty_cat:
            return []
        out = self.inner.catalog(region, start, end)
        if self.cloudy:
            out = [dataclasses.replace(m, cloud_pct=90.0) for m in out]
        if self.unknown_cloud:
            out = [dataclasses.replace(m, cloud_pct=float("nan")) if i % 4 == 0 else m for i, m in enumerate(out)]
        return out

    def load(self, meta, region, grid, bands):
        if self.fail_all or meta.scene_id in self.fail_ids:
            raise dsrc.DataSourceError("okuma yok")
        return self.inner.load(meta, region, grid, bands)

    def weather(self, region, start, end):
        if self.wx_error:
            raise dsrc.DataSourceError(self.wx_error)
        return self.inner.weather(region, start, end)

    def hotspots(self, region, start, end, event_day=None, grid=None):
        if self.hs_error:
            raise dsrc.DataSourceError(self.hs_error)
        return self.inner.hotspots(region, start, end, event_day, grid)


print("== Her tür için uçtan uca (sentetik)")
runs = {}
for kind in an.KINDS:
    scn, req = request(kind)
    log = []
    run = wf.run_analysis(di.DemoSource(scn), req, progress=lambda f, m, log=log: log.append((f, m)))
    runs[kind] = run
    check(f"{kind}: ilerleme 0→1 azalmaz, son ileti Tamam", [f for f, _ in log] == sorted(f for f, _ in log) and log[-1] == (1.0, "Tamam"))
    pre, post = run.result.pre_meta, run.result.post_meta
    if kind == "kuraklik":
        check(f"{kind}: önce = geçen yıl aynı dönem (≈365 gün ±15)", abs((post.date - pre.date).days - 365) <= 15, f"{pre.date} {post.date}")
    else:
        check(f"{kind}: önce < olay < sonra", pre.date < req.event_date < post.date, f"{pre.date} {req.event_date} {post.date}")
    check(f"{kind}: katalog durumları tutarlı (tek ÖNCE, tek SONRA)", (run.catalog.durum == "ÖNCE (seçildi)").sum() == 1 and (run.catalog.durum == "SONRA (seçildi)").sum() == 1)
    check(f"{kind}: elenen sahnelerin hepsi gerçekten eşik üstü bulutlu", (run.catalog[run.catalog.durum == "elendi: bulut"].bulut_karo > 40).all())
    check(f"{kind}: seçilen sahneler bulut eşiği altında", (run.catalog[run.catalog.durum.str.contains("seçildi")].bulut_karo <= 40).all())
    check(f"{kind}: sentetik uyarısı var", any("SENTETİK" in w for w in run.warnings) and run.is_demo)
    check(f"{kind}: hava kanıtı satırları var, hata yok", len(run.weather_rows) >= 1 and run.weather_error is None)
    check(f"{kind}: gerçek renkli önizleme", run.rgb_pre is not None and run.rgb_pre.shape[2] == 3 and run.rgb_post.shape == run.rgb_pre.shape)
    check(f"{kind}: iki sektör görünümü de dolu", run.insurance.decision in de.DECISION_LABELS and run.agri.health in ("Sağlıklı", "İzlemede", "Stresli", "Kritik"))
    check(f"{kind}: seri bölge istatistikleri sonlu ve −1..1", run.series["ortalama"].dropna().between(-1, 1.5).all() and len(run.series) >= 5)
r = runs["yangin"]
check("yangın: demo noktaları bölgeyle eşleşti, sonuç çerçevesinde", r.hotspots is not None and len(r.hotspots) >= 10 and r.hotspots_error is None)
check("yangın: ortalama dNBR > 0,27 ve karar KISMİ ONAY/ONAY", r.insurance.mean_dnbr > 0.27 and r.insurance.decision in ("KISMİ ONAY", "ONAY"), f"{r.insurance.mean_dnbr:.2f} {r.insurance.decision}")
check("yangın dışında nokta tablosu yok", all(runs[k].hotspots is None for k in ("sel", "kuraklik", "tarim")))
check("sel: yeni su alanı > 0, ana sınıf 'Yeni su'", runs["sel"].result.class_ha[2] > 5)
check("aynı sonuçtan sigorta + tarım: aynı sahne çifti", runs["yangin"].insurance and runs["yangin"].agri and runs["yangin"].agri.veg_ha > 0)

print("== Gerçeğe (bilinen durum) karşı sınıflandırma — sentetik")
scn = di.preset("yangin")
run = runs["yangin"]
truth = di.truth_masks(scn, run.result.grid, run.result.post_meta.date)["burn"]
cls = run.result.cls
inside = run.result.region_mask & (cls >= 0)
burned_true = (truth >= 0.5) & inside
pred = np.isin(cls, [3, 4, 5]) & inside
recall = (pred & burned_true).sum() / max(burned_true.sum(), 1)
unburned_true = (truth == 0) & inside
fpr = (pred & unburned_true).sum() / max(unburned_true.sum(), 1)
check("yangın: gerçek yanmış (şiddet ≥ 0,5) piksellerin ≥ %80'i orta-düşük ve üstü", recall >= 0.8, f"{recall:.2f}")
check("yangın: yanmamış piksellerde yanlış alarm ≤ %5", fpr <= 0.05, f"{fpr:.3f}")

print("== Hata politikası")
scn, req = request("yangin")
src = di.DemoSource(scn)
check("katalog hatası → WorkflowError, sentetiğe düşmez", raises(lambda: wf.run_analysis(Wrap(src, cat_error="ulaşılamadı"), req), "Uydu kataloğuna ulaşılamadı")[0])
check("boş katalog → açık ileti", raises(lambda: wf.run_analysis(Wrap(src, empty_cat=True), req), "sahne bulunamadı")[0])
check("hiçbir sahne okunamazsa ilk hata iletilir", raises(lambda: wf.run_analysis(Wrap(src, fail_all=True), req), "okuma yok")[0])
check("tüm sahneler eşikten bulutluysa → açık ileti", raises(lambda: wf.run_analysis(Wrap(src, cloudy=True), req), "Bulut eşiği")[0])
ru = wf.run_analysis(Wrap(src, unknown_cloud=True), req)
check("bulut bilgisi olmayan sahne elenir + uyarı (belirsizlik sessizce kabul edilmez)", any("bulut bilgisi yok" in w for w in ru.warnings) and (ru.catalog.durum == "elendi: bulut bilgisi yok").sum() >= 5)
base = wf.run_analysis(src, req)
bad_ids = [i for i in base.series["sahne"] if i not in (base.pair.pre_id, base.pair.post_id)][:2]
r2 = wf.run_analysis(Wrap(src, fail_ids=bad_ids), req)
check("bazı sahneler okunamazsa analiz sürer + uyarı + durum", any("okunamadı" in w for w in r2.warnings) and (r2.catalog.durum.str.startswith("okunamadı")).sum() == 2)
r3 = wf.run_analysis(Wrap(src, wx_error="hava yok", hs_error="FIRMS yok"), req)
check("hava/FIRMS hatası analizi durdurmaz, bildirilir", r3.weather_error == "hava yok" and r3.hotspots_error == "FIRMS yok" and r3.weather_rows[0].flag is None and r3.insurance.decision)
check("önce/sonra elle seçimi", wf.run_analysis(src, dataclasses.replace(req, pre_id=base.pair.pre_id, post_id=base.pair.post_id)).pair.notes == ("Sahneler elle seçildi.",))
check("elle seçim: seride olmayan kimlik → hata", raises(lambda: wf.run_analysis(src, dataclasses.replace(req, pre_id="YOK", post_id=base.pair.post_id)), "seride yok")[0])
mid = base.catalog[base.catalog.durum == "seride"].iloc[0]
late = base.catalog[base.catalog.durum == "seride"].iloc[-1]
check("elle seçim: ters tarih → hata", raises(lambda: wf.run_analysis(src, dataclasses.replace(req, pre_id=base.pair.post_id, post_id=base.pair.pre_id)), "önce olmalı")[0])
tight = dataclasses.replace(req, start=req.event_date - dt.timedelta(days=2), end=req.event_date + dt.timedelta(days=2))
check("olayın çevresinde sahne yoksa çift bulunamadı iletisi", raises(lambda: wf.run_analysis(src, tight), "")[0])

print("== Girdi doğrulama")
scn, ok_req = request("sel")
for label, kw, needle in [("kind", dict(kind="x"), "Bilinmeyen"), ("sıra", dict(start=D(2026, 8, 1), end=D(2026, 7, 1)), "Bitiş"),
                          ("uzun aralık", dict(start=D(2025, 1, 1), end=D(2026, 9, 1)), "400"), ("olay dışarıda", dict(event_date=D(2026, 12, 1)), "içinde"),
                          ("bulut", dict(max_cloud=101.0), "Bulut"), ("piksel", dict(max_px=4), "Izgara"), ("bedel", dict(insured_tl=float("nan")), "Sigorta")]:
    check(f"geçersiz girdi: {label}", raises(lambda kw=kw: dataclasses.replace(ok_req, **kw).validate(), needle)[0])
check("geçerli girdi hata vermez", ok_req.validate() is None)
r_ins = wf.run_analysis(di.DemoSource(scn), dataclasses.replace(ok_req, insured_tl=2_000_000.0))
check("sigorta bedeli tazminatı ölçekler (sel: 2×)", abs(r_ins.insurance.indicative_claim_tl / max(wf.run_analysis(di.DemoSource(scn), ok_req).insurance.indicative_claim_tl, 1) - 2) < 0.01)

print()
print("BAŞARISIZ:", FAILS if FAILS else "yok — tüm kontroller geçti")
sys.exit(1 if FAILS else 0)
