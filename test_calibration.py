"""calibration.py çevrimdışı testleri (gerçek numpy/pandas; ağ ve Streamlit gerekmez)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import calibration as cal  # noqa: E402
import data_engine as de  # noqa: E402
from data_engine import ClaimRules  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  [{extra}]" if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)


def raises(fn, needle=""):
    try:
        fn()
    except cal.CalibrationError as exc:
        return needle in str(exc), str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, f"yanlış istisna: {type(exc).__name__}: {exc}"
    return False, "istisna yok"


def labelled(n, hidden, seed=1, noise=0.0):
    """Gizli kurallarla etiketlenmiş rastgele geçmiş."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "dnbr": rng.uniform(-0.1, 1.0, n), "guven": rng.uniform(0.4, 1.0, n),
        "police_aktif": rng.random(n) > 0.05,
        "sigorta_bedeli_tl": rng.uniform(1e6, 5e6, n), "gercek_odeme_tl": rng.uniform(0, 1e6, n),
    })
    codes = de.InsuranceSimulator.decide_codes(frame.dnbr.to_numpy(), frame.guven.to_numpy(),
                                               frame.police_aktif.to_numpy(), hidden).astype(int)
    if noise:
        flip = rng.random(n) < noise
        codes = np.where(flip, rng.integers(0, 4, n), codes)
    frame["gercek_kod"] = codes.astype(np.int8)
    frame["gercek_karar"] = [de.DECISION_LABELS[c] for c in codes]
    return frame


HIDDEN = ClaimRules(approve_dnbr=0.53, partial_dnbr=0.34, review_dnbr=0.15, min_confidence=0.73)

print("== motor: karar mantığının tek kaynağı")
ds = de.InsuranceSimulator(seed=5).simulate()
rng = np.random.default_rng(3)
same = True
for _ in range(60):
    a = float(rng.uniform(0.3, 0.7))
    p = float(rng.uniform(0.15, 0.43))
    r = float(rng.uniform(0.02, 0.2))
    rules = ClaimRules(approve_dnbr=a, partial_dnbr=p, review_dnbr=r, min_confidence=float(rng.uniform(0.5, 0.99)))
    claims = de.InsuranceSimulator.evaluate_claims(ds.parcels, rules)
    codes = de.InsuranceSimulator.decide_codes(ds.parcels.dnbr.to_numpy(), ds.parcels.guven.to_numpy(),
                                               ds.parcels.police_aktif.to_numpy(), rules)
    same &= bool((claims.karar.to_numpy() == np.array(de.DECISION_LABELS, dtype=object)[codes]).all())
check("decide_codes ile evaluate_claims 60 rastgele kuralda özdeş", same)
check("karar etiket sırası", de.DECISION_LABELS == ("RET", "EKSPERTİZ", "KISMİ ONAY", "ONAY"))

print("== yardımcılar")
lo, hi = cal.wilson_interval(50, 100)
check("Wilson 50/100 ≈ (0.404, 0.596)", abs(lo - 0.404) < 0.002 and abs(hi - 0.596) < 0.002, f"{lo:.3f},{hi:.3f}")
check("Wilson boş örneklem (0,1)", cal.wilson_interval(0, 0) == (0.0, 1.0))
lo0, hi0 = cal.wilson_interval(0, 40)
lo1, hi1 = cal.wilson_interval(40, 40)
check("Wilson uç değerler sınır içinde", lo0 == 0.0 and 0 < hi0 < 0.15 and hi1 == 1.0 and 0.85 < lo1 < 1)
check("Wilson n büyüdükçe daralır", (lambda a, b: a[1] - a[0] > b[1] - b[0])(cal.wilson_interval(80, 100), cal.wilson_interval(800, 1000)))
check("tutarlı eşik düzeltmesi", cal.consistent_thresholds(0.40, 0.45, 0.50) == (0.40, 0.39, 0.38, True))
check("tutarlı eşik değişmez", cal.consistent_thresholds(0.44, 0.27, 0.10) == (0.44, 0.27, 0.10, False))
check("sayı çözümleme TR/EN", [cal._parse_number(x) for x in ("0,52", "0.52", "1.234,5", "1,234.5", " ₺12 ")][:4] == [0.52, 0.52, 1234.5, 1234.5]
      and cal._parse_number(" ₺12 ") == 12.0 and np.isnan(cal._parse_number("abc")) and np.isnan(cal._parse_number("")))
check("karar çözümleme", [cal._normalise_decision(x) for x in ("onay", "Kısmi Onay", "KISMI_ONAY", "red", "Eksper", "ekspertiz", "?")]
      == ["ONAY", "KISMİ ONAY", "KISMİ ONAY", "RET", "EKSPERTİZ", "EKSPERTİZ", None])

print("== CSV okuma ve doğrulama")
hist = cal.sample_history(2026)
rt = cal.parse_history(cal.history_csv(hist))
check("örnek geçmiş CSV gidiş-dönüş: satır sayısı", len(rt.frame) == len(hist) and rt.dropped == 0 and not rt.warnings, str(rt.warnings))
check("gidiş-dönüş: dNBR ve karar korunur", np.allclose(rt.frame.dnbr, hist.dnbr) and (rt.frame.gercek_karar.to_numpy() == hist.gercek_karar.to_numpy()).all())
check("finansal sütunlar tanındı", rt.has_financials)
tpl = cal.parse_history(cal.template_csv(2026, 6))
check("şablon okunabilir (6 satır, küçük örneklem uyarısı)", len(tpl.frame) == 6 and any("Küçük örneklem" in w for w in tpl.warnings))
check("şablon BOM'lu UTF-8 ve ; ayraçlı", cal.template_csv(2026).startswith(b"\xef\xbb\xbfdnbr;guven;"))

alias = "DNBR,Güven,Karar,Aktif\n0.60,0.9,Tam Onay,evet\n0.30,0.85,kısmi,Hayır\n0.05,0.9,red,1\n0.5,0.9,eksper,0\n"
a = cal.parse_history(alias)
check("takma adlı sütunlar, virgül ayraçlı", list(a.frame.gercek_karar) == ["ONAY", "KISMİ ONAY", "RET", "EKSPERTİZ"], str(a.frame.gercek_karar.tolist()))
check("police_aktif evet/hayır/1/0", list(a.frame.police_aktif) == [True, False, True, False])
pct = cal.parse_history("dnbr;guven;gercek_karar\n0,6;90;ONAY\n0,2;80;RET\n")
lone = cal.parse_history("dnbr;guven;gercek_karar\n0,6;0,9;ONAY\n0,5;0,9;RET\n0,4;1,5;RET\n0,3;0,8;RET\n")
check("tek bozuk güven değeri sütunu yüzdeye çevirmez", np.allclose(lone.frame.guven, [0.9, 0.9, 0.8]) and lone.dropped == 1, str(lone.frame.guven.tolist()))
check("güven yüzde ise ÷100 + uyarı", np.allclose(pct.frame.guven, [0.9, 0.8]) and any("yüzde" in w for w in pct.warnings))
check("police_aktif yoksa varsayım uyarısı", any("tüm poliçeler aktif" in w for w in pct.warnings) and pct.frame.police_aktif.all())
bad = cal.parse_history("dnbr;guven;gercek_karar\n0,6;0,9;ONAY\n9;0,9;ONAY\nabc;0,9;RET\n0,5;0,9;BELKI\n0,5;1,5;RET\n0,4;0,9;RET\n")
check("geçersiz satırlar atılır (2 geçerli kalır: ONAY, RET)", len(bad.frame) == 2 and bad.dropped == 4, f"{len(bad.frame)},{bad.dropped}")
check("tanınmayan karar uyarısı adı geçer", any("BELKI" in w for w in bad.warnings))
flag = cal.parse_history("dnbr;guven;gercek_karar;police_aktif\n0,6;0,9;ONAY;evet\n0,4;0,9;RET;belki\n")
check("tanınmayan police_aktif satırı atılır", len(flag.frame) == 1 and any("police_aktif" in w for w in flag.warnings))
cp = "dnbr;guven;gercek_karar\n0,6;0,9;KISMİ ONAY\n0,2;0,9;EKSPERTİZ\n".encode("cp1254")
check("Windows-1254 kodlu dosya okunur", list(cal.parse_history(cp).frame.gercek_karar) == ["KISMİ ONAY", "EKSPERTİZ"])
ok, msg = raises(lambda: cal.parse_history("dnbr;guven\n0,6;0,9\n"), "gercek_karar")
check("eksik zorunlu sütun açıkça bildirilir", ok, msg)
ok, msg = raises(lambda: cal.parse_history("   \n"), "boş")
check("boş dosya", ok, msg)
ok, msg = raises(lambda: cal.parse_history("dnbr;guven;gercek_karar\nx;y;z\n"), "Geçerli satır")
check("hiç geçerli satır yok", ok, msg)
ok, msg = raises(lambda: cal.parse_history("dnbr;karar;gercek_karar;guven\n0,5;ONAY;ONAY;0,9\n"), "birden fazla")
check("yinelenen (takma adlı) sütun", ok, msg)
ok, msg = raises(lambda: cal.parse_history(b"x" * (cal.MAX_BYTES + 1)), "büyük")
check("dosya boyutu sınırı", ok, msg)
check("dengesiz sınıf uyarısı", any("Dengesiz" in w for w in cal.parse_history(
    "dnbr;guven;gercek_karar\n" + "0,6;0,9;ONAY\n" * 30).warnings))

print("== ölçümler")
act = np.array([3, 3, 3, 2, 2, 0, 0, 0, 1, 1])
prd = np.array([3, 2, 0, 2, 3, 0, 2, 3, 1, 3])
m = cal.score(act, prd)
check("tam uyum 4/10", m.n == 10 and abs(m.agreement - 0.4) < 1e-12)
check("haksız ödeme riski 2 (RET→KISMİ,ONAY)", m.false_approval == 2)
check("haksız ret 1 (ONAY→RET)", m.wrongful_denial == 1)
check("kritik oran 0,3", abs(m.critical_rate - 0.3) < 1e-12)
check("ekspertiz payı 0,1", abs(m.expert_share - 0.1) < 1e-12 and abs(m.automation - 0.9) < 1e-12)
check("karışıklık matrisi toplamı n ve köşegen 4", m.confusion.sum() == 10 and np.trace(m.confusion) == 4)
check("kesinlik/duyarlılık", abs(m.precision[3] - 1 / 4) < 1e-12 and abs(m.recall[3] - 1 / 3) < 1e-12 and abs(m.recall[0] - 1 / 3) < 1e-12)
check("güven aralığı oranı kapsar", m.agreement_low < m.agreement < m.agreement_high)
cf = m.confusion_frame()
check("karışıklık tablosu etiketli", list(cf.index) == list(de.DECISION_LABELS) and cf.loc["ONAY", "ONAY"] == 1)
empty = cal.score(np.array([3, 3]), np.array([3, 3]))
check("boş sınıf için kesinlik NaN", np.isnan(empty.precision[0]) and empty.agreement == 1.0)
perfect = labelled(500, HIDDEN)
mp = cal.backtest(perfect, HIDDEN)
check("kendi kuralıyla geriye dönük test: %100 uyum, 0 kritik", mp.agreement == 1.0 and mp.critical_rate == 0.0)

print("== sapma listesi")
noisy = labelled(800, HIDDEN, seed=2, noise=0.15)
dis = cal.disagreements(noisy, ClaimRules())
mm = cal.backtest(noisy, ClaimRules())
check("sapma sayısı = n − uyum", len(dis) == noisy.shape[0] - round(mm.agreement * mm.n))
kinds = set(dis.tur)
check("sapma türleri tanımlı", kinds <= {"Haksız ödeme riski", "Haksız ret", "Ekspertize düştü", "Diğer sapma"})
risk = dis[dis.tur == "Haksız ödeme riski"]
den = dis[dis.tur == "Haksız ret"]
check("haksız ödeme: uzman RET, platform ödüyor", (risk.gercek_karar == "RET").all() and risk.platform_karar.isin(["ONAY", "KISMİ ONAY"]).all())
check("haksız ret: platform RET, uzman ödemiş", (den.platform_karar == "RET").all() and den.gercek_karar.isin(["ONAY", "KISMİ ONAY"]).all())
check("kritik hata sayısı metriklerle aynı", len(risk) == mm.false_approval and len(den) == mm.wrongful_denial)
order = ["Haksız ödeme riski", "Haksız ret", "Diğer sapma", "Ekspertize düştü"]
pos = [order.index(t) for t in dis.tur]
check("kritik hatalar üstte sıralı", pos == sorted(pos))

print("== kalibrasyon: gizli kuralı geri bulma (gürültüsüz)")
t0 = time.time()
res = cal.calibrate(perfect, ClaimRules(), 0.02, 0.3, 11)
dur = time.time() - t0
t = res.tuned_rules
check("eğitimde tam uyuma yakın (≥ %97)", res.train_after.agreement >= 0.97, f"{res.train_after.agreement:.3f}")
check("testte tam uyuma yakın (≥ %96)", res.test_after.agreement >= 0.96, f"{res.test_after.agreement:.3f}")
check("eşikler gizli değerlerin ±0,03 yakınında",
      abs(t.approve_dnbr - 0.53) <= 0.03 and abs(t.partial_dnbr - 0.34) <= 0.03 and abs(t.review_dnbr - 0.15) <= 0.03
      and abs(t.min_confidence - 0.73) <= 0.03, str(t))
check("eğitim uyumu mevcut kuraldan kötü değil", res.train_after.agreement >= res.train_before.agreement)
check("test uyumu belirgin iyileşti ve öneri verildi", res.test_after.agreement > res.test_before.agreement and res.recommended)
check("kritik hata sınırı sağlandı", res.feasible and res.train_after.critical_rate <= 0.02)
check("poliçe şartları (muafiyet/teminat/avans) korunur",
      (t.deductible_pct, t.coverage_pct, t.advance_pct) == (2.0, 90.0, 70.0))
check("süre makul (< 10 sn)", dur < 10, f"{dur:.1f}s")
check("eğitim+test = n", res.n_train + res.n_test == len(perfect) and res.n_test == 150)

print("== kalibrasyon: kendi kuralı ve iyileşme olmayan durum")
same_res = cal.calibrate(perfect, HIDDEN, 0.02, 0.3, 11)
check("zaten doğru kural değişmez", same_res.tuned_rules == HIDDEN and (same_res.changes["fark"] == 0).all())
check("iyileşme yoksa öneri verilmez + not", not same_res.recommended and any("uyum artmadı" in n for n in same_res.notes))

print("== kalibrasyon: sınırlar, sıralama, belirlilik")
rand = labelled(300, HIDDEN, seed=4, noise=1.0)  # etiketler tamamen rastgele
rr = cal.calibrate(rand, ClaimRules(), 0.02, 0.3, 5)
tt = rr.tuned_rules
within = all(cal.BOUNDS[k][0] - 1e-9 <= getattr(tt, k) <= cal.BOUNDS[k][1] + 1e-9 for k in cal.BOUNDS)
check("rastgele etiketlerde bile eşikler sınırlar içinde", within, str(tt))
check("rastgele etiketlerde bile ret < kısmi < tam sırası", tt.review_dnbr < tt.partial_dnbr < tt.approve_dnbr)
check("rastgele etiketlerde mevcut kuraldan kötü değil (eğitim)", rr.train_after.agreement >= rr.train_before.agreement - 1e-12
      or rr.train_after.critical_rate < rr.train_before.critical_rate)
tight = cal.calibrate(noisy, ClaimRules(), 0.0, 0.3, 6)
check("sıfır kritik hata isteği: sağlanırsa gerçekten 0",
      (not tight.feasible) or tight.train_after.false_approval + tight.train_after.wrongful_denial == 0)
loose = cal.calibrate(noisy, ClaimRules(), 0.5, 0.3, 6)
check("gevşek sınırda uyum ≥ sıkı sınır uyumu", loose.train_after.agreement >= tight.train_after.agreement - 1e-12)
again = cal.calibrate(noisy, ClaimRules(), 0.02, 0.3, 6)
check("aynı girdi → aynı öneri", again.tuned_rules == cal.calibrate(noisy, ClaimRules(), 0.02, 0.3, 6).tuned_rules)
ok, msg = raises(lambda: cal.calibrate(noisy.head(20), ClaimRules()), "en az")
check("30 kayıttan az → hata", ok, msg)

print("== bölme")
ident = noisy.assign(_id=np.arange(len(noisy)))
tr, te = cal.split_history(ident, 0.3, 9)
check("bölmeler ayrık ve toplam n", set(tr._id).isdisjoint(set(te._id)) and len(tr) + len(te) == len(ident))
check("test payı ≈ %30", len(te) == 240)
check("bölme tekrarlanabilir", (cal.split_history(ident, 0.3, 9)[1]._id.to_numpy() == te._id.to_numpy()).all())
check("farklı tohum farklı bölme", not (cal.split_history(ident, 0.3, 10)[1]._id.to_numpy() == te._id.to_numpy()).all())
check("test payı 0,1–0,5'e kırpılır", len(cal.split_history(ident, 0.9, 1)[1]) == 400 and len(cal.split_history(ident, 0.0, 1)[1]) == 80)

print("== duyarlılık eğrileri")
sens = res.sensitivity
check("dört parametre için eğri", set(sens) == set(cal.BOUNDS))
check("eğri uzunluğu = ızgara", len(sens["review_dnbr"]) == 19 and len(sens["min_confidence"]) == 50 and len(sens["approve_dnbr"]) == 41)
tr_res, _ = cal.split_history(perfect, 0.3, 11)
at = sens["min_confidence"]
val = float(at.loc[(at.deger - t.min_confidence).abs().idxmin(), "uyum_pct"])
check("eğri önerilen noktada eğitim uyumuna eşit", abs(val - 100 * res.train_after.agreement) < 1e-6, f"{val}")
check("eğri yüzdeleri 0–100", all(((c.uyum_pct >= 0) & (c.uyum_pct <= 100) & (c.kritik_pct >= 0) & (c.kritik_pct <= 100)).all() for c in sens.values()))

print("== finansal etki")
fin = res.financial
check("finansal etki hesaplandı", fin is not None and fin.rows == res.n_test)
check("finansal alanlar sonlu ve ≥ 0", fin is not None and all(np.isfinite(v) and v >= 0 for v in (
    fin.human_tl, fin.base_paid_tl, fin.base_pending_tl, fin.tuned_paid_tl, fin.tuned_pending_tl)))
check("sapma = ödenecek + bekleyen − uzman", fin is not None and abs(fin.tuned_gap_tl - (fin.tuned_paid_tl + fin.tuned_pending_tl - fin.human_tl)) < 1e-6)
nofin = perfect.assign(gercek_odeme_tl=np.nan)
check("ödeme sütunu yoksa finansal etki None", cal.calibrate(nofin, ClaimRules(), 0.02, 0.3, 11).financial is None)

print("== örnek (SENTETİK) geçmiş")
h2 = cal.sample_history(2026)
check("örnek geçmiş tekrarlanabilir", h2.equals(cal.sample_history(2026)) and not h2.equals(cal.sample_history(2027)))
check("1296 satır (3 olay), 6 sütun", h2.shape == (1296, 6) and list(h2.columns) == ["dnbr", "guven", "police_aktif", "sigorta_bedeli_tl", "gercek_karar", "gercek_odeme_tl"])
check("dört karar sınıfı da var", set(h2.gercek_karar) == set(de.DECISION_LABELS))
check("RET kararlarında ödeme 0, diğerlerinde ≥ 0", (h2.loc[h2.gercek_karar == "RET", "gercek_odeme_tl"] == 0).all() and (h2.gercek_odeme_tl >= 0).all())
sr = cal.calibrate(cal.parse_history(cal.history_csv(h2)).frame, ClaimRules(), 0.02, 0.3, 2026)
check("örnek geçmişte test uyumu artar ve öneri verilir", sr.test_after.agreement >= sr.test_before.agreement and sr.recommended,
      f"{sr.test_before.agreement:.3f}→{sr.test_after.agreement:.3f}")
check("örnek geçmişte tam onay eşiği 0,50'ye yaklaşır (±0,06)", abs(sr.tuned_rules.approve_dnbr - cal.ADJUSTER_RULES.approve_dnbr) <= 0.06,
      str(sr.tuned_rules))
check("platform varsayılanı gizli eksper kuralından farklı", ClaimRules() != cal.ADJUSTER_RULES)

print("== eşleştirilmiş işaret testi")
check("işaret testi 9-1 → 0,0215", abs(cal.sign_test_p(9, 1) - 0.021484375) < 1e-12)
check("işaret testi 10-0 → 2/1024", abs(cal.sign_test_p(10, 0) - 2 / 1024) < 1e-12)
check("işaret testi eşitlik ve boş → 1", cal.sign_test_p(5, 5) == 1.0 and cal.sign_test_p(0, 0) == 1.0)
check("işaret testi simetrik", cal.sign_test_p(3, 12) == cal.sign_test_p(12, 3))
exact, approx = cal.sign_test_p(120, 80), cal._normal_sign_p(120, 80)
check("normal yaklaşım n=200'de kesine ±%15 yakın", abs(exact - approx) / exact < 0.15, f"{exact:.5f} vs {approx:.5f}")
check("büyük n yaklaşık yoldan gider ve (0,1] aralığında", 0 < cal.sign_test_p(250, 150) < 1e-5)
check("kazanç − kayıp = uyum farkı (sayım)", res.test_wins - res.test_losses == round(res.test_after.agreement * res.n_test) - round(res.test_before.agreement * res.n_test))
check("gizli kuralı bulan kalibrasyon belirgin (p < 0,10)", res.significant and res.p_value < cal.SIGNIFICANCE, f"p={res.p_value:.4f}")
check("belirgin olmayan iyileşmede uyarı notu", (not sr.significant) and any("belirgin değil" in n for n in sr.notes) if sr.recommended and not sr.significant else True)
check("iyileşme yoksa kazanç ≤ kayıp", same_res.test_wins <= same_res.test_losses and not same_res.significant)

print("== büyük girdi")
big = labelled(40_000, HIDDEN, seed=8)
t0 = time.time()
bres = cal.calibrate(big, ClaimRules(), 0.02, 0.3, 3)
dur = time.time() - t0
check("40k satırda örnekleme notu ve doğru eşikler", any("hız için" in n for n in bres.notes) and bres.test_after.agreement >= 0.995,
      f"{bres.test_after.agreement:.4f}")
check("40k satır < 15 sn", dur < 15, f"{dur:.1f}s")

print()
print("BAŞARISIZ: " + ", ".join(FAILS) if FAILS else "BAŞARISIZ: yok — tüm kontroller geçti")
sys.exit(1 if FAILS else 0)
