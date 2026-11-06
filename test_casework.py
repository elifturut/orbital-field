"""tenancy.py, casework.py ve store.py çevrimdışı testleri (Streamlit gerekmez; gerçek numpy/pandas/sqlite3)."""
import datetime as dt
import json
import os
import sqlite3
import sys
import tempfile
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import casework as cw  # noqa: E402
import data_engine as de  # noqa: E402
import portal as pt  # noqa: E402
import store as stmod  # noqa: E402
import tenancy  # noqa: E402
from data_engine import ClaimRules, InsuranceSimulator  # noqa: E402

FAILS = []
SEV = ("#2f8f83", "#dcc060", "#ec9a4d", "#e0452f", "#9a2f7a")
NOW = "2026-11-06T10:30:00+03:00"
SECRET = "ICNOT-GIZLI-9137"


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  [{extra}]" if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:       # noqa: BLE001 — beklenenden farklı hata da başarısızlıktır
        return False
    return False


# ============================== şirket modeli ==============================
print("== şirketler")
ids = tenancy.company_ids()
check("üç kurgusal şirket, adlarında (demo)", len(ids) == 3 and all("(demo)" in tenancy.company_name(i) for i in ids))
check("kimlikler benzersiz, kısa kodlar benzersiz", len(set(ids)) == 3 and len({tenancy.company_short(i) for i in ids}) == 3)
check("geçerli şirket denetimi", tenancy.valid_company("boztepe") and not any(tenancy.valid_company(v) for v in ("", "yok", None, 3, ["boztepe"], "BOZTEPE", " boztepe")))
check("bilinmeyen şirket adı ve kodu '—'", tenancy.company_name("yok") == "—" == tenancy.company_short(None))
check("varsayılan şirket geçerli", tenancy.valid_company(tenancy.DEFAULT_COMPANY))
check("eşleme kararlı", tenancy.assign_company("POL-2026-10007") == tenancy.assign_company("POL-2026-10007"))
check("eşleme geçerli şirket verir", all(tenancy.valid_company(tenancy.assign_company(f"POL-{i}")) for i in range(200)))
spread = {c: sum(tenancy.assign_company(f"POL-{i}") == c for i in range(3000)) for c in ids}
check("eşleme dengeli (her şirket %25–%42)", all(750 <= v <= 1260 for v in spread.values()), spread)
check("assign_companies liste eşlemesi", tenancy.assign_companies(["a", "b"]) == (tenancy.assign_company("a"), tenancy.assign_company("b")))

# ============================== veri ==============================
print("== veri")
ds = InsuranceSimulator(seed=2026).simulate()
auto = InsuranceSimulator.evaluate_claims(ds.parcels, ClaimRules())
check("parsel tablosunda sirket sütunu ve geçerli değerler", "sirket" in ds.parcels and all(tenancy.valid_company(c) for c in ds.parcels["sirket"]))
check("şirket poliçe numarasından türer", list(ds.parcels["sirket"]) == list(tenancy.assign_companies(list(ds.parcels["police_no"]))))
ds2 = InsuranceSimulator(seed=7).simulate()
same = ds.parcels.set_index("police_no")["sirket"].eq(ds2.parcels.set_index("police_no")["sirket"]).all()
check("şirket atanması tohumdan bağımsız (rastgele sayı üretecini etkilemez)", bool(same))
check("her şirkette yeterli poliçe", all(int((ds.parcels["sirket"] == c).sum()) >= 50 for c in ids), ds.parcels["sirket"].value_counts().to_dict())
ranking = pt.customer_ranking(auto)
for k in range(3):
    own = auto[auto["sigortali"] == ranking[k]]
    check(f"demo müşteri {k + 1} birden çok şirkette poliçe sahibi", own["sirket"].nunique() >= 2, own["sirket"].tolist())
check("müşteri başına birden çok şirket (portföy genelinde)", (auto.groupby("sigortali")["sirket"].nunique() >= 2).sum() >= 10)

# net_claim_tl, otomatik motorla aynı formül
net = de.net_claim_tl(auto["dnbr"].to_numpy(), auto["sigorta_bedeli_tl"].to_numpy(), ClaimRules())
onay = (auto["karar"] == "ONAY").to_numpy()
kis = (auto["karar"] == "KISMİ ONAY").to_numpy()
check("net_claim_tl: ONAY ödemesi motorla aynı", np.allclose(np.round(net[onay]), auto.loc[onay, "odenecek_tl"]))
check("net_claim_tl: KISMİ avansı motorla aynı", np.allclose(np.round(net[kis] * 0.7), auto.loc[kis, "odenecek_tl"], atol=1))
check("damage_ratio sınırları", de.damage_ratio(np.array([0.0, 0.10, 0.66, 2.0]), ClaimRules()).tolist() == [0.0, 0.0, 1.0, 1.0])

# altküme GeoJSON
mine_pos = np.flatnonzero((auto["sirket"] == "boztepe").to_numpy())
sub = auto.iloc[mine_pos]
gj = ds.geojson(sub, positions=list(mine_pos))
check("altküme GeoJSON: yalnızca o şirketin parselleri", len(gj["features"]) == len(mine_pos)
      and [f["id"] for f in gj["features"]] == [int(p) for p in mine_pos])
check("altküme GeoJSON: öznitelik ve geometri aynı parselin", all(
    f["properties"]["parsel_id"] == auto["parsel_id"].iloc[f["id"]]
    and abs(f["geometry"]["coordinates"][0][0][1] - float(ds.geometry.corners_lat[f["id"]][0])) < 1e-9 for f in gj["features"]))
check("tam GeoJSON hâlâ çalışır", len(ds.geojson(auto)["features"]) == ds.geometry.count)
check("altküme GeoJSON: konumsuz verilirse reddedilir", raises(lambda: ds.geojson(sub), ValueError))
check("altküme GeoJSON: sayı uyuşmazlığı", raises(lambda: ds.geojson(sub, positions=list(mine_pos[:-1])), ValueError))
check("altküme GeoJSON: yinelenen konum", raises(lambda: ds.geojson(sub.iloc[:2], positions=[3, 3]), ValueError))
check("altküme GeoJSON: aralık dışı/eksi konum", raises(lambda: ds.geojson(sub.iloc[:1], positions=[10_000]), ValueError)
      and raises(lambda: ds.geojson(sub.iloc[:1], positions=[-1]), ValueError))

# ============================== girdi doğrulama ==============================
print("== girdi doğrulama")
check("not: boşluklar birleşir, denetim karakterleri atılır", cw.clean_text("  a \n\t b\x00c d  ", "x") == "a b c d")
check("not: en çok 500 karakter", cw.clean_text("a" * 500, "x") == "a" * 500 and raises(lambda: cw.clean_text("a" * 501, "x"), cw.CaseworkError))
check("not: en az uzunluk temizlenmiş metne uygulanır", raises(lambda: cw.clean_text("  ab \n ", "x", min_len=5), cw.CaseworkError)
      and cw.clean_text("abcde", "x", min_len=5) == "abcde")
check("not: metin olmayan reddedilir", all(raises(lambda v=v: cw.clean_text(v, "x"), cw.CaseworkError) for v in (None, 5, b"a", ["a"])))
check("not: HTML ve SQL metni olduğu gibi (kaçışlama gösterimde) saklanır",
      cw.clean_text("<b>x</b>'; DROP TABLE decisions;--", "x") == "<b>x</b>'; DROP TABLE decisions;--")

# karar tutarları
row = auto[(auto["karar"] == "EKSPERTİZ")].iloc[0]
rules = ClaimRules()
n_tl = float(de.net_claim_tl(np.array([row["dnbr"]]), np.array([row["sigorta_bedeli_tl"]]), rules)[0])
for dec, paid, pend in (("ONAY", n_tl, 0.0), ("KISMİ ONAY", n_tl * 0.7, n_tl * 0.3), ("EKSPERTİZ", 0.0, n_tl), ("RET", 0.0, 0.0)):
    d = cw.plan_decision(row, dec, rules, None, "", "iç gerekçe")
    check(f"tutar: {dec}", d.decision == dec and abs(d.paid_tl - round(paid)) <= 1 and abs(d.pending_tl - round(pend)) <= 1, d)
dm = cw.plan_decision(row, "ONAY", rules, 12_345.0, "  Tutar elle  ", "iç gerekçe")
check("elle tutar ONAY: ödenecek = girilen, bekleyen 0, not temizlenir", dm.paid_tl == 12345.0 and dm.pending_tl == 0.0 and dm.public_note == "Tutar elle")
dk = cw.plan_decision(row, "KISMİ ONAY", rules, 1000.0, "", "iç gerekçe")
check("elle tutar KISMİ: bekleyen = hesaplanan − girilen (en az 0)", dk.paid_tl == 1000.0 and abs(dk.pending_tl - round(max(n_tl - 1000, 0))) <= 1)
dk2 = cw.plan_decision(row, "KISMİ ONAY", rules, float(row["sigorta_bedeli_tl"]), "", "iç gerekçe")
check("elle tutar hesaplananı aşarsa bekleyen 0", dk2.pending_tl == 0.0)
bad_plans = {
    "bilinmeyen karar": ("YOK", None, "", "iç gerekçe"), "boş karar": ("", None, "", "iç gerekçe"),
    "RET ile tutar": ("RET", 5.0, "", "iç gerekçe"), "EKSPERTİZ ile tutar": ("EKSPERTİZ", 5.0, "", "iç gerekçe"),
    "eksi tutar": ("ONAY", -1.0, "", "iç gerekçe"), "bedeli aşan tutar": ("ONAY", float(row["sigorta_bedeli_tl"]) + 1, "", "iç gerekçe"),
    "NaN tutar": ("ONAY", float("nan"), "", "iç gerekçe"), "sonsuz tutar": ("ONAY", float("inf"), "", "iç gerekçe"),
    "bool tutar": ("ONAY", True, "", "iç gerekçe"), "metin tutar": ("ONAY", "5", "", "iç gerekçe"),
    "iç gerekçe yok": ("ONAY", None, "", ""), "iç gerekçe kısa": ("ONAY", None, "", "abc"),
    "iç gerekçe boşluk": ("ONAY", None, "", "      "), "müşteri notu çok uzun": ("ONAY", None, "x" * 501, "iç gerekçe"),
    "iç gerekçe çok uzun": ("ONAY", None, "", "x" * 501), "not metin değil": ("ONAY", None, 5, "iç gerekçe"),
}
for label, (dec, man, pub, internal) in bad_plans.items():
    check(f"karar reddedilir: {label}", raises(lambda a=(dec, man, pub, internal): cw.plan_decision(row, a[0], rules, a[1], a[2], a[3]), cw.CaseworkError))
check("tutar sınırı dahil: 0 ve sigorta bedeli kabul", cw.plan_decision(row, "ONAY", rules, 0.0, "", "iç gerekçe").paid_tl == 0.0
      and cw.plan_decision(row, "ONAY", rules, float(row["sigorta_bedeli_tl"]), "", "iç gerekçe").paid_tl == round(float(row["sigorta_bedeli_tl"])))

# ============================== kayıt doğrulama ==============================
def rec(**kw):
    base = dict(company="boztepe", police_no="POL-2026-10007", seed=2026, action="karar", decision="ONAY", paid_tl=100.0,
                pending_tl=0.0, public_note="Not", internal_note=SECRET, decided_by="sigortaci",
                decided_by_name="Hasar yöneticisi", decided_at=NOW, auto_decision="EKSPERTİZ", auto_paid_tl=0.0)
    base.update(kw)
    return cw.Decision(**base)


cw.validate_record(rec())
check("geçerli kayıt kabul", True)
bad_recs = {
    "bilinmeyen şirket": rec(company="yok"), "şirket None": rec(company=None), "bilinmeyen işlem": rec(action="sil"),
    "geçersiz karar": rec(decision="YOK"), "boş karar (karar işleminde)": rec(decision=""),
    "SQL'li poliçe": rec(police_no="x'; DROP TABLE decisions;--"), "boş poliçe": rec(police_no=""),
    "çok uzun poliçe": rec(police_no="A" * 41), "tohum metin": rec(seed="2026"), "tohum bool": rec(seed=True),
    "eksi tutar": rec(paid_tl=-1.0), "NaN tutar": rec(pending_tl=float("nan")), "bool tutar": rec(paid_tl=True),
    "iç gerekçe kısa": rec(internal_note="abc"), "not çok uzun": rec(public_note="x" * 501),
    "veren boş": rec(decided_by=""), "veren çok uzun": rec(decided_by="u" * 65), "zaman bozuk": rec(decided_at="dün"),
    "zaman None": rec(decided_at=None),
}
for label, r_ in bad_recs.items():
    check(f"kayıt reddedilir: {label}", raises(lambda r_=r_: cw.validate_record(r_), cw.CaseworkError))
cw.validate_record(rec(action="geri_al", decision="", paid_tl=0.0, pending_tl=0.0))
check("geri alma kaydı kabul (karar boş olabilir; gerekçe şart)", True)
check("geri alma gerekçesiz reddedilir", raises(lambda: cw.validate_record(rec(action="geri_al", decision="", internal_note="")), cw.CaseworkError))

# ============================== geçerli karar ==============================
print("== geçerli karar")
h = [rec(id=1, decision="ONAY"), rec(id=2, decision="RET", paid_tl=0.0), rec(id=3, police_no="POL-2", decision="ONAY")]
eff = cw.effective(h)
check("son kayıt kazanır", eff["POL-2026-10007"].decision == "RET" and eff["POL-2"].decision == "ONAY" and len(eff) == 2)
eff2 = cw.effective(h + [rec(id=4, action="geri_al", decision="")])
check("geri alma kararı kaldırır, diğer poliçe kalır", "POL-2026-10007" not in eff2 and "POL-2" in eff2)
eff3 = cw.effective(h + [rec(id=4, action="geri_al", decision=""), rec(id=5, decision="ONAY")])
check("geri aldıktan sonra yeni karar geçerli olur", eff3["POL-2026-10007"].decision == "ONAY")
check("boş kayıt listesi", cw.effective([]) == {})

# ============================== karar tablosuna uygulama ==============================
print("== uygulama")
multi_company = auto["sigortali"].map(auto.groupby("sigortali")["sirket"].nunique()) >= 2
target = auto[(auto["sirket"] == "boztepe") & (auto["karar"] == "EKSPERTİZ") & multi_company].iloc[0]
tno = str(target["police_no"])
other = auto[(auto["sirket"] == "yaylacik")].iloc[0]
plan = cw.plan_decision(target, "KISMİ ONAY", rules, None, "Dosya tamamlandı", "Eksper raporu geldi")
dec = rec(police_no=tno, decision=plan.decision, paid_tl=plan.paid_tl, pending_tl=plan.pending_tl,
          public_note=plan.public_note, internal_note=SECRET, auto_decision="EKSPERTİZ")
before = auto.copy(deep=True)
final = cw.apply_decisions(auto, {tno: dec})
check("giriş tablosu değişmez", auto.equals(before))
i = int(np.flatnonzero((final["police_no"] == tno).to_numpy())[0])
check("karar, tutar, gerekçe, kaynak ve müşteri notu uygulanır",
      final.at[i, "karar"] == "KISMİ ONAY" and final.at[i, "odenecek_tl"] == plan.paid_tl and final.at[i, "bekleyen_tl"] == plan.pending_tl
      and final.at[i, "gerekce"] == cw.REASON_INSURER and final.at[i, "karar_kaynagi"] == "Sigortacı" and final.at[i, "musteri_notu"] == "Dosya tamamlandı")
check("otomatik karar ve tutar ayrıca saklanır", final.at[i, "otomatik_karar"] == "EKSPERTİZ" and final.at[i, "otomatik_odenecek_tl"] == auto.at[i, "odenecek_tl"])
others = final.drop(index=i)
check("diğer satırlar otomatik kalır", (others["karar_kaynagi"] == "Otomatik").all() and (others["musteri_notu"] == "").all()
      and others["karar"].equals(auto.drop(index=i)["karar"]))
check("iç not ve kararı veren kullanıcı tabloya girmez", SECRET not in final.to_json() and "sigortaci" not in " ".join(final.columns)
      and not {"ic_not", "karar_veren", "decided_by", "internal_note"} & set(final.columns))
check("satır sayısı ve sırası korunur", len(final) == len(auto) and list(final["police_no"]) == list(auto["police_no"]))
# yanlış şirket / sahte kayıtlar yok sayılır
forged = {tno: replace(dec, company="yaylacik"), str(other["police_no"]): replace(dec, police_no=str(other["police_no"]), company="boztepe"),
          "POL-YOK": replace(dec, police_no="POL-YOK"), tno + "x": dec}
f2 = cw.apply_decisions(auto, forged)
check("şirketi uyuşmayan, bilinmeyen poliçeli ve anahtarı uyuşmayan kayıtlar yok sayılır", (f2["karar_kaynagi"] == "Otomatik").all() and f2["karar"].equals(auto["karar"]))
check("geri alma kaydı uygulanmaz", (cw.apply_decisions(auto, {tno: replace(dec, action="geri_al")})["karar_kaynagi"] == "Otomatik").all())
check("geçersiz karar etiketli kayıt uygulanmaz", (cw.apply_decisions(auto, {tno: replace(dec, decision="YOK")})["karar_kaynagi"] == "Otomatik").all())
check("karar yokken tablo aynı, yalnızca ek sütunlar", cw.apply_decisions(auto, {})["karar"].equals(auto["karar"])
      and {"otomatik_karar", "karar_kaynagi", "musteri_notu"} <= set(cw.apply_decisions(auto, {}).columns))

mine = final[final["sirket"] == "boztepe"]
s_auto = cw.summarize_final(cw.apply_decisions(auto, {}))
s_fin = cw.summarize_final(final)
check("özet: sigortacı kararı sayılır, otomasyon oranı yalnızca motor kararlarını sayar",
      s_fin["manual"] == 1.0 and s_auto["manual"] == 0.0 and abs(s_fin["automation_pct"] - s_auto["automation_pct"]) < 1e-9
      and InsuranceSimulator.summarize_claims(final)["automation_pct"] > s_fin["automation_pct"])
check("özet: kısmi onay ekspertizden çıkmış sayılır", s_fin["review"] == s_auto["review"] - 1 and s_fin["partial"] == s_auto["partial"] + 1)
cl = cw.customer_list(mine)
check("müşteri listesi: yalnızca şirketin müşterileri, poliçe sayıları toplamı eşit", set(cl["musteri"]) == set(mine["sigortali"]) and int(cl["police_sayisi"].sum()) == len(mine))
check("müşteri listesi: ekspertiz bekleyenler önde", list(cl["ekspertiz"]) == sorted(cl["ekspertiz"], reverse=True))
check("müşteri listesi: sigortacı kararı sayılır", int(cl["sigortaci_karari"].sum()) == 1)
check("müşteri listesi: yalnızca şirketin bedeli (başka şirket poliçesi toplama girmez)",
      abs(cl["bedel_tl"].sum() - mine["sigorta_bedeli_tl"].sum()) < 1e-6)
check("müşteri listesi: boş tablo çökertmez", cw.customer_list(mine.iloc[0:0]).empty)

# ============================== portal ile bütünleşme ==============================
print("== portal")
name = str(target["sigortali"])
view_all = pt.build_view(final, name, rules)
view_boz = pt.build_view(final, name, rules, company="boztepe")
check("müşteri tüm şirketlerdeki poliçelerini görür", len(view_all.policies) == int((final["sigortali"] == name).sum())
      and view_all.policies["sirket_adi"].nunique() >= 2)
check("sigortacı önizlemesi yalnızca kendi şirketini görür", set(view_boz.policies["sirket_adi"]) == {tenancy.company_name("boztepe")}
      and len(view_boz.policies) < len(view_all.policies))
check("önizleme konumları tam tablo konumudur", all(final.iloc[p]["police_no"] == pn for p, pn in zip(view_boz.positions, view_boz.policies["police_no"])))
check("başka şirkette poliçesi olmayan müşteri için KeyError", raises(lambda: pt.build_view(final, name, rules, company="yok-sirket"), KeyError))
vrow = view_all.policies[view_all.policies["police_no"] == tno].iloc[0]
check("sigortacı kararı müşteri görünümünde: durum, kaynak, not, açıklama",
      vrow["karar"] == "KISMİ ONAY" and vrow["karar_kaynagi"] == "Sigortacı" and vrow["musteri_notu"] == "Dosya tamamlandı"
      and vrow["aciklama"] == pt.REASON_TEXT["insurer"])
check("müşteri görünümünde iç not, veren kullanıcı ve iç sütunlar yok",
      SECRET not in view_all.policies.to_json() and not set(view_all.policies.columns) & set(pt.INTERNAL_COLUMNS)
      and set(view_all.policies.columns) == set(pt.CUSTOMER_COLUMNS) | {"durum", "aciklama"})
gj_c = pt.own_geojson(ds.geometry, view_all, SEV)
check("müşteri haritası: şirket adı var, iç not yok", all(f["properties"]["sirket"] for f in gj_c["features"]) and SECRET not in json.dumps(gj_c))
steps = pt.steps_for("KISMİ ONAY", dt.date(2026, 8, 14), 5000.0, 2000.0, 70.0, insurer=True)
check("sigortacı kısmi onayında avans oranı anlatılmaz, tutar anlatılır",
      "%70" not in steps[3].detail and "₺5.000" in steps[3].detail and "sigorta şirketi kararı" in steps[2].detail, steps[3].detail)
check("otomatik kısmi onayda oran anlatılır (eski davranış)", "%70" in pt.steps_for("KISMİ ONAY", dt.date(2026, 8, 14), 5000.0, 2000.0, 70.0)[3].detail)
# bildirim
notice = pt.build_notice_html(final.iloc[view_all.positions[list(view_all.policies["police_no"]).index(tno)]], name, rules,
                              ds.event_label, ds.event_date, dt.datetime(2026, 11, 6, 12, 0), ds.geometry,
                              view_all.positions[list(view_all.policies["police_no"]).index(tno)], SEV)
check("bildirim: şirket adı, şirket notu ve sigortacı kararı açıklaması", tenancy.company_name("boztepe") in notice and "Dosya tamamlandı" in notice
      and "yetkilisi tarafından belirlenmiştir" in notice and pt.REASON_TEXT["insurer"] in notice)
check("bildirim: iç not ve kullanıcı adı yok", SECRET not in notice and "@sigortaci" not in notice)
xss = replace(dec, public_note="<script>alert(1)</script>")
fx = cw.apply_decisions(auto, {tno: xss})
posx = int(np.flatnonzero((fx["police_no"] == tno).to_numpy())[0])
nx = pt.build_notice_html(fx.iloc[posx], name, rules, ds.event_label, ds.event_date, dt.datetime(2026, 11, 6, 12, 0), ds.geometry, posx, SEV)
check("bildirim: not HTML'e kaçışlanır", "<script>alert(1)</script>" not in nx and "&lt;script&gt;" in nx)
# otomatik satır bildirimi hâlâ motorla doğrulanır
autorow = final[(final["karar_kaynagi"] == "Otomatik") & (final["karar"] == "ONAY")].iloc[0]
apos = int(np.flatnonzero((final["police_no"] == autorow["police_no"]).to_numpy())[0])
na = pt.build_notice_html(autorow, str(autorow["sigortali"]), rules, ds.event_label, ds.event_date, dt.datetime(2026, 11, 6, 12, 0), ds.geometry, apos, SEV)
check("otomatik karar bildirimi: tutar tablosu ve şirket satırı", "= hesaplanan tazminat" in na and tenancy.company_name(str(autorow["sirket"])) in na)
tampered = autorow.copy()
tampered["odenecek_tl"] = float(tampered["odenecek_tl"]) + 500
check("otomatik karar tutarı değiştirilirse bildirim üretilmez (motorla doğrulama sürüyor)",
      raises(lambda: pt.build_notice_html(tampered, str(autorow["sigortali"]), rules, ds.event_label, ds.event_date,
                                          dt.datetime(2026, 11, 6, 12, 0), ds.geometry, apos, SEV), ValueError))
check("karar_kaynagi sahteciliği: 'Sigortacı' yazılmış otomatik satır doğrulamayı atlar ama kayıt defterinden gelmez",
      "karar_kaynagi" in final and set(final["karar_kaynagi"]) <= {"Otomatik", "Sigortacı"})

# ============================== kayıt defteri (SQLite) ==============================
print("== kayıt defteri")
tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "alt", "k.sqlite3")
db = stmod.Store(path)
check("klasör ve dosya oluşturulur", os.path.exists(path))
i1 = db.add(rec(police_no="POL-A", decision="ONAY"))
i2 = db.add(rec(police_no="POL-A", decision="RET", paid_tl=0.0, internal_note="ikinci gerekçe"))
i3 = db.add(rec(police_no="POL-B", company="yaylacik", decision="ONAY"))
i4 = db.add(rec(police_no="POL-C", seed=7))
check("kayıt numaraları artar", i1 < i2 < i3 < i4)
check("geçerli karar: son kayıt", db.effective("boztepe", 2026)["POL-A"].decision == "RET")
check("şirket sınırı: başka şirketin kaydı görünmez", "POL-B" not in db.effective("boztepe", 2026) and "POL-B" in db.effective("yaylacik", 2026))
check("tohum sınırı: başka tohumun kaydı görünmez", "POL-C" not in db.effective("boztepe", 2026) and "POL-C" in db.effective("boztepe", 7))
hist = db.history("boztepe", 2026)
check("geçmiş yeniden eskiye, yalnızca o şirket ve tohum", [d.id for d in hist] == [i2, i1])
check("geçmişte iç not var (sigortacı ekranı)", hist[0].internal_note == "ikinci gerekçe")
check("tek poliçe geçmişi", [d.police_no for d in db.history("boztepe", 2026, police_no="POL-A")] == ["POL-A", "POL-A"] and db.history("boztepe", 2026, "POL-Z") == [])
check("geçmiş sınırı", len(db.history("boztepe", 2026, limit=1)) == 1)
check("bilinmeyen şirket sorgusu reddedilir", raises(lambda: db.history("yok", 2026), cw.CaseworkError) and raises(lambda: db.effective("x' OR '1'='1", 2026), cw.CaseworkError))
pub = db.public_effective(2026)
check("genel görünüm: tüm şirketler, iç not ve veren bilgisi boş", set(pub) == {"POL-A", "POL-B"}
      and all(d.internal_note == "" and d.decided_by == "-" and d.decided_by_name == "-" for d in pub.values()))
check("genel görünümde iç not hiçbir alanda geçmez", "ikinci gerekçe" not in repr(pub) and SECRET not in repr(pub))
db.add(rec(police_no="POL-A", action="geri_al", decision="", paid_tl=0.0, pending_tl=0.0, internal_note="vazgeçildi"))
check("geri alma sonrası geçerli karar yok, geçmiş korunur", "POL-A" not in db.effective("boztepe", 2026) and len(db.history("boztepe", 2026, "POL-A")) == 3)
# SQL enjeksiyonu
for evil in ("x'; DROP TABLE decisions;--", "' OR 1=1 --", "a\"b"):
    check(f"kötü poliçe numarası reddedilir: {evil[:18]}", raises(lambda e=evil: db.add(rec(police_no=e)), cw.CaseworkError))
db.add(rec(police_no="POL-Q", public_note="'; DROP TABLE decisions;--", internal_note="\"; DELETE FROM decisions;--"))
check("SQL içeren not parametreyle güvenle saklanır, tablo ayakta",
      db.effective("boztepe", 2026)["POL-Q"].public_note == "'; DROP TABLE decisions;--" and len(db.history("boztepe", 2026)) >= 4)
check("geçersiz kayıt yazılmaz (doğrulama)", raises(lambda: db.add(rec(decision="YOK")), cw.CaseworkError)
      and raises(lambda: db.add(rec(company="yok")), cw.CaseworkError))
n_before = len(db.history("boztepe", 2026))
raises(lambda: db.add(rec(internal_note="x")), cw.CaseworkError)
check("reddedilen kayıt satır eklemez", len(db.history("boztepe", 2026)) == n_before)
# veritabanı kısıtları (doğrulama atlansa bile)
conn = sqlite3.connect(path)
bad_sql = [
    "INSERT INTO decisions (company,police_no,seed,action,decision,paid_tl,pending_tl,public_note,internal_note,decided_by,decided_by_name,decided_at,auto_decision,auto_paid_tl) VALUES ('boztepe','P',1,'sil','ONAY',0,0,'','x','u','U','t','RET',0)",
    "INSERT INTO decisions (company,police_no,seed,action,decision,paid_tl,pending_tl,public_note,internal_note,decided_by,decided_by_name,decided_at,auto_decision,auto_paid_tl) VALUES ('boztepe','P',1,'karar','YOK',0,0,'','x','u','U','t','RET',0)",
    "INSERT INTO decisions (company,police_no,seed,action,decision,paid_tl,pending_tl,public_note,internal_note,decided_by,decided_by_name,decided_at,auto_decision,auto_paid_tl) VALUES ('boztepe','P',1,'karar','ONAY',-5,0,'','x','u','U','t','RET',0)",
]
for k, q in enumerate(bad_sql):
    try:
        conn.execute(q)
        check(f"veritabanı kısıtı {k + 1}", False)
    except sqlite3.IntegrityError:
        check(f"veritabanı kısıtı {k + 1}", True)
conn.close()
db2 = stmod.Store(path)
check("kalıcılık: yeniden açınca kayıtlar duruyor", db2.effective("yaylacik", 2026)["POL-B"].decision == "ONAY")
check("kayıtlar eklemelidir: eski satır değişmedi", db2.history("boztepe", 2026, "POL-A")[-1].id == i1 and db2.history("boztepe", 2026, "POL-A")[-1].decision == "ONAY")
check("açık ama yazılamayan yol StoreError verir", raises(lambda: stmod.Store(os.path.join(path, "icinde", "x.db")), stmod.StoreError))
check("açık bozuk yol için geçici klasöre düşülmez", raises(lambda: stmod.open_store(os.path.join(path, "icinde", "x.db")), stmod.StoreError))
old_env = os.environ.get(stmod.DB_ENV)
os.environ[stmod.DB_ENV] = os.path.join(tmp, "env.sqlite3")
check("ORBITAL_DB_PATH öncelikli", stmod.default_path() == os.path.join(tmp, "env.sqlite3") and stmod.open_store().path == os.path.join(tmp, "env.sqlite3"))
os.environ[stmod.DB_ENV] = os.path.join(path, "icinde", "x.db")          # yazılamaz → geçici klasöre düşer
fb = stmod.open_store()
check("varsayılan yol yazılamazsa geçici klasöre düşer", fb.path.startswith(tempfile.gettempdir()))
if old_env is None:
    os.environ.pop(stmod.DB_ENV, None)
else:
    os.environ[stmod.DB_ENV] = old_env
check("varsayılan yol uygulama klasörü altında", stmod.default_path().endswith(os.path.join("orbital_data", "orbital.sqlite3")) or bool(os.environ.get(stmod.DB_ENV)))
# eşzamanlı yazma
import threading  # noqa: E402

errs = []


def worker(k):
    try:
        for j in range(15):
            stmod.Store(path).add(rec(police_no=f"POL-T{k}-{j}"))
    except Exception as exc:     # noqa: BLE001
        errs.append(exc)


threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
[t.start() for t in threads]
[t.join() for t in threads]
check("eşzamanlı yazma: hata yok, 60 kayıt eklendi", not errs and sum(d.police_no.startswith("POL-T") for d in db2.history("boztepe", 2026, limit=500)) == 60, errs)

# ============================== CSV güvenliği ve kayıt tablosu ==============================
print("== kayıt tablosu")
check("csv_safe: formül başlangıçları korunur", all(cw.csv_safe(v) == "'" + v for v in ("=1+1", "+5", "-2", "@SUM(A1)", "\tx", "\rx")))
check("csv_safe: normal metin ve sayı aynen", cw.csv_safe("Merhaba") == "Merhaba" and cw.csv_safe(5) == 5 and cw.csv_safe("") == "")
lf = cw.log_frame([rec(id=9, public_note="=HYPERLINK(\"http://x\")", internal_note="@kötü", decision="ONAY")], safe=True)
check("kayıt tablosu: sütunlar, korumalı hücreler", list(lf.columns) == list(cw.LOG_COLUMNS) and lf.at[0, "müşteri_notu"].startswith("'=")
      and lf.at[0, "iç_gerekçe"].startswith("'@") and lf.at[0, "zaman"] == "2026-11-06 10:30")
lf2 = cw.log_frame([rec(id=1, action="geri_al", decision="", paid_tl=0.0)])
check("kayıt tablosu: geri alma etiketi, korumasız mod", lf2.at[0, "işlem"] == "Geri alma" and lf2.at[0, "karar"] == "—")
check("kayıt tablosu: boş liste", cw.log_frame([]).empty and list(cw.log_frame([]).columns) == list(cw.LOG_COLUMNS))

print()
print("BAŞARISIZ: " + ", ".join(FAILS) if FAILS else "BAŞARISIZ: yok — tüm kontroller geçti")
sys.exit(1 if FAILS else 0)
