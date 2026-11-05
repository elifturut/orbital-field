"""portal.py çevrimdışı testleri: müşteri verisi süzme, sızıntı yokluğu, karar bildirimi (gerçek numpy/pandas)."""
import datetime as dt
import json
import os
import re
import sys
from dataclasses import replace
from html.parser import HTMLParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import auth  # noqa: E402
import data_engine as de  # noqa: E402
import portal as pt  # noqa: E402
from data_engine import ClaimRules, InsuranceSimulator  # noqa: E402

FAILS = []
SEV = ("#2f8f83", "#dcc060", "#ec9a4d", "#e0452f", "#9a2f7a")
NOW = dt.datetime(2026, 11, 4, 12, 0)


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  [{extra}]" if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)


class Balance(HTMLParser):
    """Açılış/kapanış etiket dengesini denetler."""
    VOID = {"meta", "br", "hr", "img", "input", "link"}

    def __init__(self):
        super().__init__()
        self.stack, self.errors, self.tags = [], [], []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"beklenmeyen </{tag}>")
        else:
            self.stack.pop()


def balanced(markup):
    p = Balance()
    p.feed(markup)
    p.close()
    return not p.errors and not p.stack, p


def world(seed=2026, rules=None):
    rules = rules or ClaimRules()
    ds = InsuranceSimulator(seed=seed).simulate()
    return ds, rules, InsuranceSimulator.evaluate_claims(ds.parcels, rules)


def notice(ds, rules, claims, pos, owner=None):
    row = claims.iloc[pos]
    return pt.build_notice_html(row, owner or row["sigortali"], rules, ds.event_label, ds.event_date, NOW, ds.geometry, pos, SEV)


# ============================== eşleme ve süzme ==============================
ds, rules, claims = world()
names = pt.customer_ranking(claims)
check("sıralama: her müşteri bir kez", len(names) == claims["sigortali"].nunique() == len(set(names)))
check("sıralama kararlı (aynı girdi aynı sonuç)", names == pt.customer_ranking(claims.copy()))
g = claims.groupby("sigortali")["karar"].agg(["nunique", "size"])
check("sıralama: önce karar çeşitliliği", all(g.at[a, "nunique"] >= g.at[b, "nunique"] for a, b in zip(names, names[1:])))
check("1. müşteri birden çok karar türü görür", g.at[names[0], "nunique"] >= 3, g.at[names[0], "nunique"])
tab = pt.customers_table(claims)
check("müşteri tablosu sıralamayla aynı sırada, sayılar doğru", tab["musteri"].tolist() == names and int(tab["police_sayisi"].sum()) == len(claims))
check("müşteri no biçimi ve kararlılık", re.fullmatch(r"MÜŞ-\d{5}", pt.customer_no("H. Kaya")) is not None and pt.customer_no("H. Kaya") == pt.customer_no("H. Kaya") and pt.customer_no("H. Kaya") != pt.customer_no("M. Aslan"))

ins = auth.Principal("sigortaci", auth.Role.INSURER, "S")
c1 = auth.Principal("musteri1", auth.Role.CUSTOMER, "M1", customer_rank=1)
c9 = auth.Principal("musteri9", auth.Role.CUSTOMER, "M9", customer_rank=9999)
cn = auth.Principal("gercek", auth.Role.CUSTOMER, "G", customer_name=names[3])
cx = auth.Principal("hayalet", auth.Role.CUSTOMER, "H", customer_name="Yok Kişi")
check("eşleme: sıra → ad", pt.resolve_customer(claims, c1) == names[0])
check("eşleme: ad → ad", pt.resolve_customer(claims, cn) == names[3])
check("eşleme: sıra aşımı → None", pt.resolve_customer(claims, c9) is None)
check("eşleme: olmayan ad → None", pt.resolve_customer(claims, cx) is None)
check("eşleme: sigortacı için müşteri yok", pt.resolve_customer(claims, ins) is None)
check("eşleme: sıra 1..3 farklı müşteriler", len({pt.resolve_customer(claims, auth.Principal("m", auth.Role.CUSTOMER, "m", customer_rank=k)) for k in (1, 2, 3)}) == 3)

v = pt.build_view(claims, names[0], rules)
own = claims[claims["sigortali"] == names[0]]
check("görünüm: yalnızca kendi satırları", len(v.policies) == len(own) == len(v.positions) and set(v.policies["police_no"]) == set(own["police_no"]))
check("görünüm: konumlar geometri sırasıyla aynı", all(claims.iloc[p]["police_no"] == pn for p, pn in zip(v.positions, v.policies["police_no"])))
check("görünüm: sütunlar beyaz liste + durum/aciklama", list(v.policies.columns) == list(pt.CUSTOMER_COLUMNS) + ["durum", "aciklama"])
check("görünüm: iç sütun yok", not (set(pt.INTERNAL_COLUMNS) & set(v.policies.columns)))
check("görünüm: toplamlar satırların toplamı", abs(v.total_insured - own["sigorta_bedeli_tl"].sum()) < 1e-6 and abs(v.total_paid - own["odenecek_tl"].sum()) < 1e-6 and abs(v.total_pending - own["bekleyen_tl"].sum()) < 1e-6)
check("görünüm: karar sayıları toplamı poliçe sayısı", sum(v.counts.values()) == len(v.policies))
check("görünüm: durum etiketleri tanımlı", v.policies["durum"].isin(pt.STATUS_LABELS.values()).all() and not v.policies["durum"].isna().any())
check("görünüm: açıklama tanımlı", v.policies["aciklama"].isin(pt.REASON_TEXT.values()).all())
check("görünüm: NaN yok", not v.policies.isna().any().any())
check("görünüm: müşteri no ve ad", v.name == names[0] and v.number == pt.customer_no(names[0]))
try:
    pt.build_view(claims, "Yok Kişi", rules)
    check("görünüm: olmayan müşteri KeyError", False)
except KeyError:
    check("görünüm: olmayan müşteri KeyError", True)
for partial in (names[0][:3], names[0].lower(), names[0] + " ", "", "."):
    try:
        pt.build_view(claims, partial, rules)
        check(f"görünüm: kısmi ad eşleşmez ({partial!r})", False)
    except KeyError:
        check(f"görünüm: kısmi ad eşleşmez ({partial!r})", True)

# alt dize tuzağı: benzer adlar karışmaz
trap = claims.copy()
trap.loc[trap.index[0], "sigortali"] = "H. Kaya"
trap.loc[trap.index[1], "sigortali"] = "H. Kaya-Öz"
trap.loc[trap.index[2], "sigortali"] = "XH. Kaya"
tv = pt.build_view(trap, "H. Kaya", rules)
check("benzer adlar karışmaz (alt dize/önek/sonek)", set(tv.policies["police_no"]) == set(trap[trap["sigortali"] == "H. Kaya"]["police_no"]) and "H. Kaya-Öz" not in set(trap.iloc[list(tv.positions)]["sigortali"]))

# bölünme özelliği: tüm tohumlar, tüm müşteriler
for seed in (2026, 1, 7, 99999):
    ds_s, rules_s, claims_s = world(seed)
    seen, total, ok_rows = set(), 0, True
    for name in claims_s["sigortali"].unique():
        view = pt.build_view(claims_s, name, rules_s)
        pns = set(view.policies["police_no"])
        ok_rows &= not (seen & pns) and all(claims_s.iloc[p]["sigortali"] == name for p in view.positions)
        seen |= pns
        total += len(view.policies)
    check(f"seed {seed}: müşteri görünümleri poliçeleri ayrık ve eksiksiz bölüyor", ok_rows and total == len(claims_s) == len(seen))

# ============================== metin güvenliği ==============================
digit = re.compile(r"\d")
forbidden_words = ("dnbr", "nbr", "ndvi", "eşik", "esik", "risk", "skor", "güven", "guven", "kural", "sha", "karar sınır")
check("müşteri gerekçelerinde sayı yok", not any(digit.search(t) for t in pt.REASON_TEXT.values()))
check("müşteri gerekçelerinde iç terim yok", not any(w in t.lower() for t in pt.REASON_TEXT.values() for w in forbidden_words))
check("başlıklarda sayı ve iç terim yok", not any(digit.search(t) or any(w in t.lower() for w in forbidden_words) for t in list(pt.HEADLINES.values()) + list(pt.STATUS_LABELS.values())))
check("gerekçe sözlüğü kural izinin tüm nedenlerini kapsar", set(pt.REASON_TEXT) == {"inactive", "below_review", "approve", "partial", "low_confidence", "borderline"})
check("her karar için başlık ve durum var", set(pt.HEADLINES) == set(pt.STATUS_LABELS) == set(de.DECISION_LABELS))
check("görünürlük tablosu: müşterinin gördüğü her şeyi sigortacı da görür", all(ins_ok for _, cust, ins_ok in pt.VISIBILITY if cust))
check("görünürlük tablosu: risk ve ölçüm müşteriden gizli", all(not cust for label, cust, _ in pt.VISIBILITY if any(w in label.lower() for w in ("risk", "dnbr", "eşik", "kalibrasyon", "kanıt", "diğer müşteri"))))

# ============================== adımlar ==============================
d = dt.date(2026, 8, 14)
expect = {"ONAY": ["done", "done", "done", "current"], "KISMİ ONAY": ["done", "done", "done", "current", "waiting"],
          "EKSPERTİZ": ["done", "done", "done", "current", "waiting"], "RET": ["done", "done", "done", "closed"]}
for dec, states in expect.items():
    st_ = pt.steps_for(dec, d, 1_234_567.0, 234_000.0, 70.0)
    check(f"adımlar: {dec}", [s.state for s in st_] == states)
check("adım: olay tarihi görünür", "14.08.2026" in pt.steps_for("ONAY", d, 1.0, 0.0, 70.0)[0].detail)
check("adım: onayda ödenecek tutar", "₺1.234.567" in pt.steps_for("ONAY", d, 1_234_567.0, 0.0, 70.0)[3].detail)
check("adım: kısmi onayda avans oranı ve kalan", "%70" in pt.steps_for("KISMİ ONAY", d, 700.0, 300.0, 70.0)[3].detail and "₺300" in pt.steps_for("KISMİ ONAY", d, 700.0, 300.0, 70.0)[4].detail)
check("adım: retle dosya kapanır, ödeme geçmez", "ödeme yapılmaz" in pt.steps_for("RET", d, 0.0, 0.0, 70.0)[3].detail.lower())
check("adım: bekleyen tutar yoksa 'en çok ₺0' denmez", "en çok" not in pt.steps_for("EKSPERTİZ", d, 0.0, 0.0, 70.0)[4].detail and "en çok" not in pt.steps_for("KISMİ ONAY", d, 5.0, 0.2, 70.0)[4].detail)
check("adım: bekleyen tutar varsa üst sınır yazılır", "en çok ₺250" in pt.steps_for("EKSPERTİZ", d, 0.0, 250.0, 70.0)[4].detail)
try:
    pt.steps_for("BİLİNMEYEN", d, 0, 0, 70)
    check("adım: bilinmeyen karar KeyError", False)
except KeyError:
    check("adım: bilinmeyen karar KeyError", True)

# ============================== harita verisi ==============================
gj = pt.own_geojson(ds.geometry, v, SEV)
text = json.dumps(gj, ensure_ascii=False)
check("geojson: yalnızca kendi parselleri", len(gj["features"]) == len(v.policies) and {f["id"] for f in gj["features"]} == set(v.positions))
check("geojson: özellikler beyaz listeden", all(set(f["properties"]) == {"police_no", "varlik_turu", "alan_txt", "bedel_txt", "siddet", "durum", "odeme_txt", "bekleyen_txt", "renk"} for f in gj["features"]))
check("geojson: iç alan adı yok", not any(w in text for w in ("risk", "dnbr", "guven", "ndvi", "nbr", "sigortali", "gerekce")))
other = next(n for n in names if n != names[0])
check("geojson: başka müşterinin poliçe no'su yok", not any(pn in text for pn in claims[claims["sigortali"] == other]["police_no"]))
check("geojson: halka kapalı, koordinat sonlu", all(f["geometry"]["coordinates"][0][0] == f["geometry"]["coordinates"][0][-1] and np.isfinite(np.array(f["geometry"]["coordinates"][0])).all() for f in gj["features"]))
check("geojson: renk şiddet sınıfından", all(f["properties"]["renk"] in SEV for f in gj["features"]))
check("geojson: JSON'a çevrilebilir (NaN yok)", json.loads(json.dumps(gj, allow_nan=False)) == gj)

svg = pt.own_parcel_svg(ds.geometry, v.positions[0], SEV[3], "P-001")
check("svg: tek çokgen (komşu yok)", svg.count("<polygon") == 1 and svg.startswith("<svg") and svg.endswith("</svg>"))
check("svg: dış kaynak ve betik yok", "http" not in svg and "<script" not in svg and "href" not in svg)
check("svg: etiket kaçışlı", "&lt;b&gt;" in pt.own_parcel_svg(ds.geometry, 0, SEV[0], "<b>"))
check("svg: çizim sonlu", "nan" not in svg.lower() and "inf" not in svg.lower())

# ============================== karar bildirimi ==============================
found = {}
for seed in (2026, 1, 7, 99999):
    ds_s, rules_s, claims_s = world(seed)
    for pos in range(len(claims_s)):
        dec = claims_s.iloc[pos]["karar"]
        found.setdefault(dec, (ds_s, rules_s, claims_s, pos))
check("dört karar türü de örneklendi", set(found) == set(de.DECISION_LABELS))
leak_words = ("risk", "dnbr", "nbr", "ndvi", "eşik", "güven", "kırılgan", "sha-256", "kural izi", "karar sınırı", "kanıt")
for dec, (ds_s, rules_s, claims_s, pos) in found.items():
    row = claims_s.iloc[pos]
    doc = notice(ds_s, rules_s, claims_s, pos)
    low = doc.lower()
    ok_bal, parsed = balanced(doc)
    check(f"bildirim [{dec}]: HTML dengeli", ok_bal, parsed.errors[:2] + parsed.stack[:2])
    check(f"bildirim [{dec}]: tam belge ve Türkçe", doc.startswith("<!doctype html>") and 'lang="tr"' in doc)
    check(f"bildirim [{dec}]: betik ve dış kaynak yok", "<script" not in low and "src=" not in low and not re.search(r"(href|src)=\"https?:", doc) and "@import" not in low)
    check(f"bildirim [{dec}]: poliçe, ad, tutar var", row["police_no"] in doc and row["sigortali"] in doc and f"₺{de.format_tr(float(row['sigorta_bedeli_tl']), 0)}" in doc)
    check(f"bildirim [{dec}]: durum ve başlık", pt.STATUS_LABELS[dec] in doc and pt.HEADLINES[dec] in doc)
    check(f"bildirim [{dec}]: iç terim yok", not any(w in low for w in leak_words), [w for w in leak_words if w in low])
    check(f"bildirim [{dec}]: risk skoru ve ölçüm değerleri yok", f"{de.format_tr(float(row['risk_skoru']), 1)} /" not in doc and str(row["risk_sinifi"]) + ")" not in doc)
    check(f"bildirim [{dec}]: eşik sayıları yok", not re.search(r"(?<![\d,.])0,(44|27|10|80)(?![\d])", re.sub(r"<svg.*?</svg>", "", doc, flags=re.S)))
    check(f"bildirim [{dec}]: yalnızca kendi parseli (tek çokgen)", doc.count("<polygon") == 1)
    others = [pn for pn in claims_s["police_no"] if pn != row["police_no"]]
    check(f"bildirim [{dec}]: başka poliçe no yok", not any(pn in doc for pn in others))
    other_names = set(claims_s["sigortali"]) - {row["sigortali"]}
    check(f"bildirim [{dec}]: başka müşteri adı yok", not any(re.search(rf"(?<![\wÇĞİÖŞÜçğıöşü]){re.escape(n)}(?![\wÇĞİÖŞÜçğıöşü])", doc) for n in other_names))
    check(f"bildirim [{dec}]: DEMO bandı", "DEMO · SENTETİK VERİ" in doc)
    check(f"bildirim [{dec}]: bildirim kodu", re.search(r"BLD-[0-9A-F]{10}", doc) is not None)
    check(f"bildirim [{dec}]: yazdır düğmesi satır içi", doc.count("window.print()") == 1)
    check(f"bildirim [{dec}]: dosya adı ASCII", re.fullmatch(r"hasar_bildirimi_POL-2026-\d{5}_[A-Za-z-]+\.html", pt.notice_filename(row["police_no"], dec)) is not None, pt.notice_filename(row["police_no"], dec))
    if dec == "RET":
        check("bildirim [RET]: ödeme hesaplanmaz notu", "ödeme hesaplanmaz" in doc)
    else:
        check(f"bildirim [{dec}]: tutar hesabı satırları", "hesaplanan tazminat" in doc and "muafiyet" in doc and "teminat oranı" in doc)
    if dec == "KISMİ ONAY":
        check("bildirim [KISMİ]: avans ve kalan", "Avans (%70)" in doc and "Kalan" in doc)
    check(f"bildirim [{dec}]: ödenecek tutar belgede", f"₺{de.format_tr(float(row['odenecek_tl']), 0)}" in doc)

# ortak güvenlik
ds_s, rules_s, claims_s, pos = found["ONAY"]
a = notice(ds_s, rules_s, claims_s, pos)
check("bildirim deterministik", a == notice(ds_s, rules_s, claims_s, pos))
check("bildirim kodu tutara bağlı", re.search(r"BLD-([0-9A-F]{10})", a).group(1) != re.search(r"BLD-([0-9A-F]{10})", notice(*found["RET"][:3], found["RET"][3])).group(1))
row = claims_s.iloc[pos]
try:
    pt.build_notice_html(row, "Başka Kişi", rules_s, ds_s.event_label, ds_s.event_date, NOW, ds_s.geometry, pos, SEV)
    check("sahiplik: başkasının bildirimi reddedilir", False)
except PermissionError:
    check("sahiplik: başkasının bildirimi reddedilir", True)
for other_name in list(set(claims_s["sigortali"]) - {row["sigortali"]})[:6]:
    try:
        pt.build_notice_html(row, other_name, rules_s, ds_s.event_label, ds_s.event_date, NOW, ds_s.geometry, pos, SEV)
        check(f"sahiplik: {other_name} reddedilir", False)
    except PermissionError:
        pass
check("sahiplik: tüm diğer müşteriler reddedildi", True)
tampered = row.copy()
tampered["karar"] = "RET"
try:
    pt.build_notice_html(tampered, tampered["sigortali"], rules_s, ds_s.event_label, ds_s.event_date, NOW, ds_s.geometry, pos, SEV)
    check("doğrulama: değiştirilmiş karar reddedilir", False)
except ValueError:
    check("doğrulama: değiştirilmiş karar reddedilir", True)
tampered = row.copy()
tampered["odenecek_tl"] = float(row["odenecek_tl"]) + 5000
try:
    pt.build_notice_html(tampered, tampered["sigortali"], rules_s, ds_s.event_label, ds_s.event_date, NOW, ds_s.geometry, pos, SEV)
    check("doğrulama: değiştirilmiş tutar reddedilir", False)
except ValueError:
    check("doğrulama: değiştirilmiş tutar reddedilir", True)
try:
    pt.build_notice_html(row.drop("siddet"), row["sigortali"], rules_s, ds_s.event_label, ds_s.event_date, NOW, ds_s.geometry, pos, SEV)
    check("eksik alan reddedilir", False)
except ValueError:
    check("eksik alan reddedilir", True)
try:
    pt.build_view(claims_s.assign(karar="ONAY"), row["sigortali"], rules_s)
    check("doğrulama: görünüm de motorla uyuşmayan kararı reddeder", False)
except ValueError:
    check("doğrulama: görünüm de motorla uyuşmayan kararı reddeder", True)

# XSS: kötü niyetli ad ve metinler kaçışlanır
evil = '<script>alert(1)</script>"><img src=x onerror=alert(1)>'
ev = claims_s.copy()
ev.loc[ev.index[pos], "sigortali"] = evil
doc = pt.build_notice_html(ev.iloc[pos], evil, rules_s, '<b>olay</b>', ds_s.event_date, NOW, ds_s.geometry, pos, SEV)
check("XSS: kötü niyetli ad kaçışlanır", "<script>alert" not in doc and "<img src=x" not in doc and "&lt;script&gt;" in doc)
check("XSS: olay etiketi kaçışlanır", "<b>olay</b>" not in doc and "&lt;b&gt;olay&lt;/b&gt;" in doc)
check("XSS: bozuk ad belgeyi bozmaz", balanced(doc)[0])
check("XSS: geojson özellikleri düz metin kalır (kaçış tarayıcıda)", isinstance(json.dumps(pt.own_geojson(ds_s.geometry, pt.build_view(ev, evil, rules_s), SEV)), str))

# kurallar değişince bildirim de değişir ama eşik sızdırmaz
strict = replace(ClaimRules(), approve_dnbr=0.437, partial_dnbr=0.263, review_dnbr=0.117, min_confidence=0.813)
ds2, _, claims2 = world(2026, strict)
changed = sum(1 for i in range(len(claims2)) if claims2.iloc[i]["karar"] != world(2026)[2].iloc[i]["karar"])
check("kurallar değişince kararlar değişir (denetim)", changed > 0, changed)
leaks = 0
for i in range(0, len(claims2), 7):
    html_i = re.sub(r"<svg.*?</svg>", "", notice(ds2, strict, claims2, i), flags=re.S)
    leaks += bool(re.search(r"(?<![\d,.])0,(44|26|12|81)(?![\d])", html_i))
check("özel eşiklerle de bildirimde eşik sayısı yok", leaks == 0, leaks)

# geometri/sıra uyumsuzluğu
try:
    pt.own_parcel_svg(ds.geometry, 10_000, SEV[0], "x")
    check("svg: geçersiz konum hata verir", False)
except IndexError:
    check("svg: geçersiz konum hata verir", True)

print()
print("BAŞARISIZ: " + ", ".join(FAILS) if FAILS else "BAŞARISIZ: yok — tüm kontroller geçti")
sys.exit(1 if FAILS else 0)
