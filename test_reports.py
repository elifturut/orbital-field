"""reports.py çevrimdışı testleri (gerçek numpy/pandas; ağ ve Streamlit gerekmez)."""
import datetime as dt
import io
import json
import os
import re
import sys
import zipfile
from dataclasses import replace
from html.parser import HTMLParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data_engine as de  # noqa: E402
import reports as rep  # noqa: E402
from data_engine import ClaimRules, InsuranceSimulator  # noqa: E402

FAILS = []
SEV = ("#2f8f83", "#dcc060", "#ec9a4d", "#e0452f", "#9a2f7a")
NOW = dt.datetime(2026, 11, 3, 10, 30, tzinfo=de.TR_TZ)


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  [{extra}]" if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)


class Balance(HTMLParser):
    """Açılış/kapanış etiket dengesini denetler."""
    VOID = {"meta", "br", "hr", "img", "input", "link"}

    def __init__(self):
        super().__init__()
        self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"beklenmeyen </{tag}> (yığın: {self.stack[-3:]})")
        else:
            self.stack.pop()


def balanced(doc):
    b = Balance()
    b.feed(doc)
    b.close()
    return not b.errors and not b.stack, (b.errors[:2], b.stack[:3])


ds = InsuranceSimulator(seed=2026).simulate()
RULESETS = {
    "varsayılan": ClaimRules(),
    "sıkı": ClaimRules(approve_dnbr=0.60, partial_dnbr=0.40, review_dnbr=0.18, min_confidence=0.95, deductible_pct=5.0, coverage_pct=70.0),
    "gevşek": ClaimRules(approve_dnbr=0.32, partial_dnbr=0.2, review_dnbr=0.04, min_confidence=0.55, deductible_pct=0.0, coverage_pct=100.0, advance_pct=40.0),
    "düşük güven": ClaimRules(min_confidence=0.99),
}


def ctx_for(rules, **kw):
    base = dict(rules=rules, event_label=ds.event_label, event_date=ds.event_date, generated_at=NOW, severity_colors=SEV)
    base.update(kw)
    return rep.EvidenceContext(**base)


print("== bağımsız doğrulama: iz ve tutar = motor (tüm parseller × 4 kural seti)")
all_ok, seen = True, set()
for name, rules in RULESETS.items():
    claims = InsuranceSimulator.evaluate_claims(ds.parcels, rules)
    bad = 0
    for _, row in claims.iterrows():
        try:
            trace, pay = rep.verify(row, rules)
            seen.add(trace.decision)
        except ValueError:
            bad += 1
    check(f"{name}: {len(claims)} kararın hepsi doğrulandı", bad == 0, f"{bad} uyuşmazlık")
check("dört karar türü de sınandı", seen == set(de.DECISION_LABELS), str(seen))

print("== kural izi")
claims = InsuranceSimulator.evaluate_claims(ds.parcels, ClaimRules())
by = {d: claims[claims.karar == d] for d in de.DECISION_LABELS}
t_ok = rep.build_trace(by["ONAY"].iloc[0], ClaimRules())
check("ONAY izi: ilk adım poliçe aktif ✓, güven ✓, kısmi onay değerlendirilmedi",
      t_ok.steps[0].passed is True and t_ok.steps[3].passed is True and t_ok.steps[-1].passed is None and t_ok.reason_key == "approve")
inactive = claims[~claims.police_aktif]
t_in = rep.build_trace(inactive.iloc[0], ClaimRules())
check("pasif poliçe: ilk adım ✕, kalanlar değerlendirilmedi", t_in.steps[0].passed is False and all(s.passed is None for s in t_in.steps[1:]) and t_in.decision == "RET")
t_low = rep.build_trace(claims[(claims.karar == "RET") & claims.police_aktif].iloc[0], ClaimRules())
check("düşük dNBR: ret eşiği ✕, sonrası değerlendirilmedi", t_low.steps[1].passed is False and all(s.passed is None for s in t_low.steps[2:]) and t_low.reason_key == "below_review")
ex = by["EKSPERTİZ"]
low_conf = ex[ex.guven < 0.80]
if len(low_conf):
    t_lc = rep.build_trace(low_conf.iloc[0], ClaimRules())
    check("düşük güven → EKSPERTİZ ve neden low_confidence (dNBR yüksekse güven adımı ✕)",
          t_lc.decision == "EKSPERTİZ" and t_lc.reason_key == "low_confidence")
try:
    rep.build_trace({"dnbr": 0.5}, ClaimRules())
    check("eksik alanlı satır hata verir", False)
except ValueError as exc:
    check("eksik alanlı satır hata verir", "eksik" in str(exc))

print("== tutar hesabı")
row_ok = by["ONAY"].iloc[0]
p_ok = rep.build_payment(row_ok, ClaimRules(), "ONAY")
check("ONAY: ödenecek = net, bekleyen 0", abs(p_ok.paid - p_ok.net) < 1e-6 and p_ok.pending == 0.0)
row_pt = by["KISMİ ONAY"].iloc[0]
p_pt = rep.build_payment(row_pt, ClaimRules(), "KISMİ ONAY")
check("KISMİ: ödenecek = %70 net, kalan = %30", abs(p_pt.paid - 0.7 * p_pt.net) < 1e-6 and abs(p_pt.pending - 0.3 * p_pt.net) < 1e-6)
p_ret = rep.build_payment(by["RET"].iloc[0], ClaimRules(), "RET")
check("RET: ödeme 0", p_ret.paid == 0.0 and p_ret.pending == 0.0)
check("net = max(brüt − muafiyet, 0)", all(abs(p.net - max(p.gross - p.deductible, 0)) < 1e-6 for p in (p_ok, p_pt, p_ret)))

print("== karar sınırı analizi")
def decision_for(row, **over):
    r = row.copy()
    for k, v in over.items():
        r[k] = v
    return InsuranceSimulator.evaluate_claims(pd.DataFrame([r]), ClaimRules()).iloc[0]["karar"]

checked, good = 0, True
for _, row in claims[claims.police_aktif].head(40).iterrows():
    flips = rep.flip_points(row, ClaimRules())
    for key, sign in (("dnbr_down", -1), ("dnbr_up", 1)):
        f = flips[key]
        if f is None:
            continue
        checked += 1
        at = decision_for(row, dnbr=round(float(row.dnbr) + sign * f.delta, 6))
        before = decision_for(row, dnbr=round(float(row.dnbr) + sign * (f.delta - 0.005), 6))
        good &= (at == f.to_decision) and (before == row.karar)
check(f"dNBR sınır noktaları motorla doğrulandı ({checked} nokta)", good and checked > 20)
checked, good = 0, True
for _, row in claims[claims.police_aktif].iterrows():
    flips = rep.flip_points(row, ClaimRules())
    for key, sign in (("guven_down", -1), ("guven_up", 1)):
        f = flips[key]
        if f is None:
            continue
        checked += 1
        at = decision_for(row, guven=round(float(np.clip(row.guven + sign * f.delta, 0, 1)), 6))
        before = decision_for(row, guven=round(float(np.clip(row.guven + sign * (f.delta - 0.01), 0, 1)), 6))
        good &= (at == f.to_decision) and (before == row.karar)
check(f"güven sınır noktaları motorla doğrulandı ({checked} nokta)", good and checked > 5)
fi = rep.flip_points(inactive.iloc[0], ClaimRules())
check("pasif poliçe: hiçbir değişim kararı etkilemez → 'Etkisiz'", all(v is None for v in fi.values()) and rep.fragility(fi) == ("Etkisiz", None))
edge = claims[claims.police_aktif].iloc[0].copy()
edge["dnbr"], edge["guven"] = ClaimRules().approve_dnbr + 0.004, 0.95
edge = InsuranceSimulator.evaluate_claims(pd.DataFrame([edge]), ClaimRules()).iloc[0]
check("eşiğin hemen üstünde → 'Sınırda'", rep.fragility(rep.flip_points(edge, ClaimRules()))[0] == "Sınırda" and edge.karar == "ONAY")
mid = edge.copy()
mid["dnbr"] = 0.70
mid = InsuranceSimulator.evaluate_claims(pd.DataFrame([mid]), ClaimRules()).iloc[0]
check("eşikten uzakta → 'Sağlam'", rep.fragility(rep.flip_points(mid, ClaimRules()))[0] == "Sağlam")

print("== özet kodu")
r0 = claims.iloc[5]
c0 = ctx_for(ClaimRules())
d0 = rep.evidence_digest(r0, c0)
check("64 haneli onaltılık", re.fullmatch(r"[0-9a-f]{64}", d0) is not None)
check("deterministik, üretim zamanından bağımsız", d0 == rep.evidence_digest(r0, ctx_for(ClaimRules(), generated_at=NOW + dt.timedelta(days=9))))
r1 = r0.copy()
r1["dnbr"] = float(r0.dnbr) + 0.001
check("dNBR değişince değişir", rep.evidence_digest(r1, c0) != d0)
check("kural değişince değişir", rep.evidence_digest(r0, ctx_for(replace(ClaimRules(), coverage_pct=80.0))) != d0)
check("sentetik bayrağı/kaynak değişince değişir", rep.evidence_digest(r0, ctx_for(ClaimRules(), synthetic=False)) != d0
      and rep.evidence_digest(r0, ctx_for(ClaimRules(), provenance="başka")) != d0)

print("== HTML belge")
geo = ds.geometry
row = by["KISMİ ONAY"].iloc[0]
pos = int(np.flatnonzero((claims.police_no == row.police_no).to_numpy())[0])
doc = rep.render_claim_html(row, c0, geo, claims, pos)
check("doctype + Türkçe dil", doc.startswith("<!doctype html>") and 'lang="tr"' in doc)
check("poliçe no, karar ve özet kodu belgede", row.police_no in doc and "KISMİ ONAY" in doc and rep.evidence_digest(row, c0) in doc)
check("etiket dengesi", balanced(doc)[0], str(balanced(doc)[1]))
check("sentetik veri bandı görünür", "DEMO · SENTETİK VERİ" in doc)
check("harici kaynak/script yok", "http://" not in doc and "https://" not in doc and "<script" not in doc and "<link" not in doc and "<img" not in doc)
real = rep.render_claim_html(row, ctx_for(ClaimRules(), synthetic=False, provenance="Sentinel-2 L2A, tile 35SNA"), geo, claims, pos)
check("gerçek veri: sentetik bandı yok, kaynak yazılı", "DEMO · SENTETİK VERİ" not in real and "Sentinel-2 L2A, tile 35SNA" in real and balanced(real)[0])
evil = claims.copy()
evil.loc[evil.index[pos], "sigortali"] = '<b>X</b>&"\''
evil_doc = rep.render_claim_html(evil.iloc[pos], c0, geo, evil, pos)
check("HTML kaçışlama (sigortalı alanı)", "<b>X</b>" not in evil_doc and "&lt;b&gt;X&lt;/b&gt;&amp;" in evil_doc)
cal_doc = rep.render_claim_html(row, ctx_for(ClaimRules(), calibration_lines=("Test uyumu %91 <script>alert(1)</script>",)), geo, claims, pos)
check("kalibrasyon satırı belgede ve kaçışlı", "Test uyumu %91" in cal_doc and "<script>alert" not in cal_doc and balanced(cal_doc)[0])
check("kalibrasyonsuz belge bunu açıkça söyler", "kalibre edilmemiştir" in doc and "kalibre edilmemiştir" not in cal_doc)
check("tüm bölümler var", all(h in doc for h in ("1 · Poliçe ve parsel", "2 · Uydu ölçümleri", "3 · Kural izi", "4 · Tutar hesabı",
                                                  "5 · Karar sınırına uzaklık", "6 · Kural seti", "7 · Onay")))
check("yazdırma düğmesi ve @page var", "window.print()" in doc and "@page" in doc)
for d in de.DECISION_LABELS:
    r = by[d].iloc[0]
    p = int(np.flatnonzero((claims.police_no == r.police_no).to_numpy())[0])
    check(f"{d} belgesi üretilir ve dengeli", balanced(rep.render_claim_html(r, c0, geo, claims, p))[0])
check("pasif poliçe belgesi üretilir", balanced(rep.render_claim_html(inactive.iloc[0], c0, geo, claims,
      int(np.flatnonzero((claims.police_no == inactive.iloc[0].police_no).to_numpy())[0])))[0])

print("== uyuşmazlık engeli")
tam = claims.copy()
tam.loc[tam.index[pos], "karar"] = "ONAY"
try:
    rep.render_claim_html(tam.iloc[pos], c0, geo, tam, pos)
    check("motor kararı kurcalanırsa belge üretilmez", False)
except ValueError as exc:
    check("motor kararı kurcalanırsa belge üretilmez", "kural izi" in str(exc))
tam = claims.copy()
tam.loc[tam.index[pos], "odenecek_tl"] = float(tam.loc[tam.index[pos], "odenecek_tl"]) + 500.0
try:
    rep.render_claim_html(tam.iloc[pos], c0, geo, tam, pos)
    check("ödeme kurcalanırsa belge üretilmez", False)
except ValueError as exc:
    check("ödeme kurcalanırsa belge üretilmez", "tutar" in str(exc))

print("== SVG çizimleri")
svg = rep.parcel_context_svg(geo, claims, pos, SEV)
n_poly = svg.count("<polygon")
r0_, c0_ = int(geo.row[pos]), int(geo.col[pos])
expect = int(((np.abs(geo.row - r0_) <= 3) & (np.abs(geo.col - c0_) <= 3)).sum())
check("çevre parsel sayısı doğru", n_poly == expect, f"{n_poly} vs {expect}")
nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", " ".join(re.findall(r'points="([^"]+)"', svg)))]
check("koordinatlar sonlu ve görünüm kutusu içinde", nums and all(np.isfinite(nums)) and min(nums) >= -1 and max(nums) <= 421 + 400)
check("seçili parsel kalın çerçeveli, en son çizilir", svg.rfind("stroke-width=\"2.6\"") > svg.rfind("stroke-width=\"0.8\""))
vb = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
check("viewBox pozitif", vb is not None and int(vb.group(1)) > 0 and int(vb.group(2)) > 0)
corner = rep.parcel_context_svg(geo, claims, 0, SEV)
check("köşe parselde de çalışır", corner.count("<polygon") == int(((geo.row <= 3) & (geo.col <= 3)).sum()))
ruler = rep.threshold_ruler_svg(5.0, ClaimRules())
check("cetvel: aralık dışı değer kırpılır, NaN yok", "nan" not in ruler.lower() and "dNBR 5,000" in ruler)
check("cetvel: üç eşik etiketi", all(t in ruler for t in ("0,10", "0,27", "0,44")))

print("== ZIP paket")
sample = list(claims.police_no.iloc[:12]) + [claims.police_no.iloc[0]]
blob = rep.build_pack_zip(claims, sample, c0, geo)
zf = zipfile.ZipFile(io.BytesIO(blob))
names = zf.namelist()
htmls = [n for n in names if n.endswith(".html")]
check("12 belge + manifest + ozet + OKUBENI (yineleme atıldı)", len(htmls) == 12 and {"manifest.json", "ozet.csv", "OKUBENI.txt"} <= set(names), str(names[-4:]))
manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
check("manifest: adet, sentetik bayrağı, kural seti", manifest["adet"] == 12 and manifest["sentetik"] is True and manifest["kural_seti"]["approve_dnbr"] == 0.44)
import hashlib  # noqa: E402
check("manifest belge özetleri dosyalarla eşleşir", all(hashlib.sha256(zf.read(e["dosya"])).hexdigest() == e["belge_sha256"] for e in manifest["dosyalar"]))
check("ZIP içindeki belgeler dengeli HTML", all(balanced(zf.read(n).decode("utf-8"))[0] for n in htmls))
ozet = zf.read("ozet.csv").decode("utf-8-sig").strip().splitlines()
check("özet.csv: başlık + 12 satır, ; ayraçlı", len(ozet) == 13 and ozet[0].startswith("police_no;sigortali;karar"))
check("ZIP baytları deterministik", blob == rep.build_pack_zip(claims, sample, c0, geo))
check("OKUBENI sentetik uyarısı taşır", "SENTETİK" in zf.read("OKUBENI.txt").decode("utf-8"))
check("dosya adları ASCII-güvenli", all(re.fullmatch(r"[A-Za-z0-9_.-]+", n) for n in names), str([n for n in names if not re.fullmatch(r"[A-Za-z0-9_.-]+", n)]))
for bad_call, label in ((lambda: rep.build_pack_zip(claims, [], c0, geo), "boş liste"),
                        (lambda: rep.build_pack_zip(claims, ["YOK-1"], c0, geo), "bilinmeyen poliçe"),
                        (lambda: rep.build_pack_zip(claims, [f"X{i}" for i in range(rep.MAX_PACK_FILES + 1)], c0, geo), "dosya sayısı sınırı")):
    try:
        bad_call()
        check(f"hata: {label}", False)
    except (ValueError, KeyError):
        check(f"hata: {label}", True)
t0_names = rep.claim_filename(claims.iloc[0].copy().rename({"police_no": "police_no"}))
check("dosya adı biçimi", re.fullmatch(r"kanit_POL-2026-\d{5}_(ONAY|KISMI-ONAY|EKSPERTIZ|RET)\.html", t0_names) is not None, t0_names)
full = rep.build_pack_zip(claims, list(claims.police_no), c0, geo)
check("tüm 192 poliçe tek pakette", len(zipfile.ZipFile(io.BytesIO(full)).namelist()) == len(claims) + 3, f"{len(full) / 1e6:.2f} MB")

print()
print("BAŞARISIZ: " + ", ".join(FAILS) if FAILS else "BAŞARISIZ: yok — tüm kontroller geçti")
sys.exit(1 if FAILS else 0)
