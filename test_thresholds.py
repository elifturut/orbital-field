"""thresholds.py çevrimdışı testleri (bilinen-cevap; ROC/AUC/Wilson/bootstrap; sentetik doğrulama özellikleri)."""
import dataclasses
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data_engine as de  # noqa: E402
import thresholds as th  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  [{extra}]" if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)


def raises(fn, needle=""):
    try:
        fn()
    except th.ThresholdError as e:
        return needle in str(e), str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"yanlış istisna: {type(e).__name__}: {e}"
    return False, "istisna yok"


print("== Dayanak tablosu")
R = de.ClaimRules()
ev = th.evidence_table(R)
check("varsayılan eşikler data_engine sınıf sınırlarıyla aynı", th.DEFAULT_THRESHOLDS == de.DNBR_EDGES)
check("tablo dört dNBR sınırını içerir (0,10 / 0,27 / 0,44 / 0,66)", sorted(ev[ev.esik.isin(["Yanmamış / düşük sınırı", "Orta-düşük şiddet başlangıcı", "Orta-yüksek şiddet başlangıcı", "Yüksek şiddet / tam hasar"])].deger.tolist()) == [0.10, 0.27, 0.44, 0.66])
dn = ev[(ev.alan == "Yangın (dNBR)")].set_index("esik")
check("0,27 ve 0,44 satırları literatür kaynaklı (Key & Benson)", "Key & Benson" in dn.loc["Orta-düşük şiddet başlangıcı", "dayanak"] and "Key & Benson" in dn.loc["Orta-yüksek şiddet başlangıcı", "dayanak"])
check("değerler kurallardan okunur", dn.loc["Orta-düşük şiddet başlangıcı", "deger"] == R.partial_dnbr and dn.loc["Orta-yüksek şiddet başlangıcı", "deger"] == R.approve_dnbr)
ev2 = th.evidence_table(dataclasses.replace(R, approve_dnbr=0.5, partial_dnbr=0.3))
check("ayarlanan kural tabloya yansır", ev2[ev2.esik == "Orta-yüksek şiddet başlangıcı"].deger.iloc[0] == 0.5 and ev2[ev2.esik == "Orta-düşük şiddet başlangıcı"].deger.iloc[0] == 0.3)
check("hiçbir satırın dayanağı/doğrulama durumu boş değil", (ev.dayanak.str.len() > 10).all() and (ev.dogrulama.str.len() > 3).all())
check("sezgisel eşikler açıkça 'Sezgisel' etiketli", ev[ev.alan.str.startswith("Kuraklık")].dogrulama.iloc[0] == "Sezgisel" and ev[ev.alan.str.startswith("Hava")].dogrulama.iloc[0] == "Sezgisel")
check("yerel doğrulama yapılmadığı yazılı", ev.dogrulama.str.contains("yerel doğrulama yok").any())
check("modül belgesi tarım arazisi uyarısını içerir", "tarım arazisine" in th.__doc__)

print("== Wilson aralığı (bilinen)")
lo, hi = th.wilson(0, 10)
check("0/10 → [0; 0,2775]", lo == 0.0 and abs(hi - 3.8416 / 13.8416) < 1e-4, f"{lo} {hi}")
lo, hi = th.wilson(5, 10)
check("5/10 → [0,2366; 0,7634]", abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3, f"{lo} {hi}")
check("n=0 → NaN", all(math.isnan(x) for x in th.wilson(0, 0)))
check("aralık [0,1] içinde, alt ≤ üst", all(0 <= th.wilson(k, 40)[0] <= th.wilson(k, 40)[1] <= 1 for k in range(0, 41, 5)))

print("== Karışıklık matrisi, ROC, AUC (bilinen)")
d = np.array([0.1, 0.3, 0.5, 0.7])
y = np.array([False, False, True, True])
c = th.confusion_at(d, y, 0.4)
check("eşik 0,4: TP2 FP0 TN2 FN0, tüm oranlar mükemmel", (c.tp, c.fp, c.tn, c.fn) == (2, 0, 2, 0) and c.sensitivity == 1 and c.specificity == 1 and c.youden == 1 and c.accuracy == 1)
c = th.confusion_at(d, y, 0.2)
check("eşik 0,2: bir yanlış pozitif → FPR 0,5", (c.fp, c.tn) == (1, 1) and c.fpr == 0.5 and c.precision == 2 / 3)
c = th.confusion_at(d, y, 0.6)
check("eşik 0,6: bir yanlış negatif → FNR 0,5", (c.fn, c.tp) == (1, 1) and c.fnr == 0.5 and c.sensitivity == 0.5)
check("eşit değer 'hasarlı' sayılır (≥), karar kuralıyla tutarlı", th.confusion_at(np.array([0.44]), np.array([True]), 0.44).tp == 1)
check("FPR + özgüllük = 1, FNR + duyarlılık = 1", abs(c.fpr + c.specificity - 1) < 1e-12 and abs(c.fnr + c.sensitivity - 1) < 1e-12)
check("AUC klasik örnek [.1,.4,.35,.8] → 0,75", abs(th.auc_score(np.array([0.1, 0.4, 0.35, 0.8]), np.array([False, False, True, True])) - 0.75) < 1e-12)
check("AUC ayrışık = 1, ters = 0, tümü eşit = 0,5", th.auc_score(d, y) == 1.0 and th.auc_score(-d, y) == 0.0 and th.auc_score(np.ones(4), y) == 0.5)
roc = th.roc_curve(d, y)
check("ROC (0,0)'dan (1,1)'e, tpr/fpr azalmaz", roc.iloc[0][["tpr", "fpr"]].tolist() == [0, 0] and roc.iloc[-1][["tpr", "fpr"]].tolist() == [1, 1] and roc.tpr.is_monotonic_increasing and roc.fpr.is_monotonic_increasing)
check("ROC alanı (yamuk) = AUC", abs(float(np.sum(np.diff(roc.fpr.to_numpy()) * (roc.tpr.to_numpy()[1:] + roc.tpr.to_numpy()[:-1]) / 2)) - th.auc_score(d, y)) < 1e-9)
check("en iyi eşik: ayrışık veride en küçük pozitif değer (0,5)", th.best_threshold(d, y) == 0.5)
check("tek sınıf → hata", raises(lambda: th.auc_score(d, np.ones(4, bool)), "hem hasarlı")[0] and raises(lambda: th.roc_curve(d, np.zeros(4, bool)), "hem hasarlı")[0])

print("== evaluate_thresholds — bilinen dağılım")
rng = np.random.default_rng(3)
neg = rng.normal(0.10, 0.10, 400)
pos = rng.normal(0.50, 0.15, 400)
df = pd.DataFrame({"dnbr": np.r_[neg, pos], "hasarli": np.r_[np.zeros(400, bool), np.ones(400, bool)]})
rep = th.evaluate_thresholds(df, n_boot=200)
analytic_auc = 0.5 * (1 + math.erf((0.4 / math.sqrt(0.10 ** 2 + 0.15 ** 2)) / math.sqrt(2)))
check("AUC ≈ analitik (0,987) ±0,015", abs(rep.auc - analytic_auc) < 0.015, f"{rep.auc:.4f} vs {analytic_auc:.4f}")
check("AUC güven aralığı AUC'yi içerir ve sıralı", rep.auc_ci[0] <= rep.auc <= rep.auc_ci[1])
check("en iyi eşik analitik Youden bölgesinde (0,25–0,36) ve CI içinde", 0.25 <= rep.best <= 0.36 and rep.best_ci[0] <= rep.best <= rep.best_ci[1], f"{rep.best:.3f} {rep.best_ci}")
p27 = rep.points[rep.points.threshold == 0.27].iloc[0]
exp_fpr = 1 - 0.5 * (1 + math.erf(((0.27 - 0.10) / 0.10) / math.sqrt(2)))
exp_fnr = 0.5 * (1 + math.erf(((0.27 - 0.50) / 0.15) / math.sqrt(2)))
check("0,27'de FPR ≈ P(N(.1,.1) ≥ .27) ve FNR ≈ P(N(.5,.15) < .27)", abs(p27.fpr - exp_fpr) < 0.025 and abs(p27.fnr - exp_fnr) < 0.04, f"{p27.fpr:.3f}/{exp_fpr:.3f} {p27.fnr:.3f}/{exp_fnr:.3f}")
check("Wilson aralığı gözlenen oranı içerir", p27.fpr_alt <= p27.fpr <= p27.fpr_ust and p27.fnr_alt <= p27.fnr <= p27.fnr_ust)
check("eşik arttıkça FPR azalır, FNR artar (monoton)", rep.points.fpr.is_monotonic_decreasing and rep.points.fnr.is_monotonic_increasing)
check("tablo, confusion_at ile birebir", all(rep.points.iloc[i].tp == th.confusion_at(df.dnbr.to_numpy(), df.hasarli.to_numpy(), t).tp for i, t in enumerate(th.DEFAULT_THRESHOLDS)))
check("aynı tohum → aynı sonuç", th.evaluate_thresholds(df, n_boot=100, seed=4).best_ci == th.evaluate_thresholds(df, n_boot=100, seed=4).best_ci)
small = df.iloc[list(range(0, 400, 40)) + list(range(400, 800, 40))]
rs = th.evaluate_thresholds(small, n_boot=100)
check("20 örnek: çalışır ama 'az örnek' uyarısı verir", rs.n == 20 and any("30'dan az" in n for n in rs.notes))
check("NaN dNBR atılır ve bildirilir", th.evaluate_thresholds(pd.concat([df, pd.DataFrame({"dnbr": [np.nan], "hasarli": [True]})]), n_boot=60).dropped == 1)
for label, frame, needle in [("sütun eksik", df.drop(columns=["hasarli"]), "sütunları"), ("çok az örnek", df.iloc[:10], "En az"),
                             ("tek sınıf", df[df.hasarli], "En az")]:
    check(f"geçersiz girdi: {label}", raises(lambda f=frame: th.evaluate_thresholds(f), needle)[0])
check("n_boot < 50 → hata", raises(lambda: th.evaluate_thresholds(df, n_boot=10), "n_boot")[0])

print("== Saha doğrulama CSV")
t = th.load_ground_truth_csv(th.ground_truth_template_csv())
check("şablon okunur", len(t) == 2 and t.hasarli.tolist() == [True, False])
t2 = th.load_ground_truth_csv("DNBR,Hasarli\n0.3,Evet\n0.1,hayır\n0.5,TRUE\n0.0,0\n")
check("büyük/küçük harf, Türkçe ve İngilizce etiketler", t2.hasarli.tolist() == [True, False, True, False])
check("tanınmayan etiket → hata", raises(lambda: th.load_ground_truth_csv("dnbr,hasarli\n0.3,belki\n"), "tanınmadı")[0])
check("eksik sütun → hata", raises(lambda: th.load_ground_truth_csv("a,b\n1,2\n"), "eksik sütun")[0])
check("sayısal olmayan dNBR → hata", raises(lambda: th.load_ground_truth_csv("dnbr,hasarli\nx,1\ny,0\n"), "sayısal değil")[0])

print("== Sentetik doğrulama (ölçüm zinciri)")
sv = th.synthetic_validation(n_scenes=8)
sv2 = th.synthetic_validation(n_scenes=8)
check("deterministik", sv.equals(sv2))
check("sütunlar ve yeterli blok", list(sv.columns) == ["sahne", "blok", "dnbr", "hasarli", "gercek_siddet", "gecerli_oran"] and len(sv) > 300)
check("etiket = gerçek şiddet ≥ kesme", ((sv.gercek_siddet >= 0.5) == sv.hasarli).all())
check("her blokta kullanılabilir piksel ≥ %50", (sv.gecerli_oran >= 0.5).all())
check("her iki sınıf da yeterli", sv.hasarli.sum() >= 50 and (~sv.hasarli).sum() >= 50)
check("blok dNBR'si gerçek şiddetle pozitif ilişkili (r > 0,5)", np.corrcoef(sv.dnbr, sv.gercek_siddet)[0, 1] > 0.5)
sw = th.synthetic_sweep(n_scenes=8)
check("hasarlı tanımı sıkılaştıkça en iyi eşik yükselir (eşik evrensel değil)", sw.en_iyi_esik.is_monotonic_increasing and sw.en_iyi_esik.iloc[-1] - sw.en_iyi_esik.iloc[0] > 0.1, str(sw.en_iyi_esik.round(3).tolist()))
check("tanım sıkılaştıkça 0,27'de yanlış pozitif oranı artar", sw.fpr_027.is_monotonic_increasing)
check("0,44'te yanlış negatif oranı 0,27'dekinden yüksek (daha temkinli)", (sw.fnr_044 >= sw.fnr_027).all())
check("not metni 'DEĞİLDİR' içerir", "DEĞİLDİR" in th.SYNTHETIC_VALIDATION_NOTE and "SENTETİK" in th.SYNTHETIC_VALIDATION_NOTE)
check("geçersiz parametreler", raises(lambda: th.synthetic_validation(n_scenes=0), "aralık")[0] and raises(lambda: th.synthetic_validation(damage_cut=1.5), "aralık")[0])

print()
print("BAŞARISIZ:", FAILS if FAILS else "yok — tüm kontroller geçti")
sys.exit(1 if FAILS else 0)
