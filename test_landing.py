"""landing.py çevrimdışı testleri (Streamlit gerekmez): iletişim adresi doğrulama, mailto, HTML kaçışı, içerik dürüstlüğü."""
import os
import re
import sys
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import landing  # noqa: E402
from html.parser import HTMLParser  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  [{extra}]" if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)


# --- adres doğrulama
for good in ("info@example.com", "  Ad.Soyad+demo@alt.ornek.com.tr  ", "a@b.co", "x_y-z@kurum-adi.org"):
    check(f"geçerli adres: {good.strip()}", landing.valid_email(good) == good.strip())
BAD = [
    "", "   ", "yok", "a@b", "a@b.", "@b.com", "a@@b.com", "a b@c.com", "a@b .com", "a@b.com\nBcc: x@y.com",
    'a@b.co"><script>alert(1)</script>', "a@b.co?bcc=kotu@x.io", "a@b.co&subject=x", "a@b.co,kotu@x.io", "<a@b.co>",
    "javascript:alert(1)", "a@b.c", "a@" + "x" * 130 + ".com", "ü@örnek.com", None, 123, ["a@b.co"], {"a": 1}, b"a@b.co",
]
for bad in BAD:
    check(f"geçersiz adres reddedilir: {bad!r:.50}", landing.valid_email(bad) == "")

check("uçlardaki boşluk ve satır sonu kırpılır", landing.valid_email("a@b.com\n") == "a@b.com")

# --- mailto
link = landing.build_mailto("info@example.com")
parsed = urlparse(link)
qs = parse_qs(parsed.query)
check("mailto şeması ve adres", parsed.scheme == "mailto" and parsed.path == "info@example.com")
check("mailto konu ve gövde çözülür", qs["subject"] == [landing.MAILTO_SUBJECT] and "ORBITAL FIELD" in qs["body"][0])
check("mailto gövdesi doldurulacak alanları içerir", all(w in qs["body"][0] for w in ("Ad soyad:", "Kurum:", "sigorta")))
check("mailto'da ham satır sonu/boşluk yok", not re.search(r"[\s\"<>]", link), link)
check("geçersiz adreste mailto boş", landing.build_mailto("a@b.co?bcc=x@y.io") == "" and landing.build_mailto("") == "")

# --- iletişim bölümü
with_mail = landing.build_contact_html("info@example.com")
check("adres varsa düğme ve adres görünür", 'class="lp-cta"' in with_mail and 'href="mailto:info@example.com?' in with_mail and "info@example.com</span>" in with_mail)
check("adres varsa href kaçışlı (& → &amp;)", "&amp;body=" in with_mail and "&body=" not in with_mail)
for bad in ('x"><img src=x onerror=alert(1)>', "a@b.co?bcc=kotu@x.io", "", None):
    out = landing.build_contact_html(bad)
    check(f"adres geçersizse düğme ve adres yok: {bad!r:.40}", "lp-cta" not in out and "mailto" not in out and "@" not in out and "<img" not in out)
check("adressiz metin uydurma adres içermez", "yakında" in landing.build_contact_html("") and "@" not in landing.build_contact_html(""))
check("demo cümlesi yalnızca demo açıkken", "demo hesaplarla" in landing.build_contact_html("", True) and "demo hesaplarla" not in landing.build_contact_html("", False))
check("durum notu demo cümlesi yalnızca demo açıkken",
      "demo hesaplarla" in landing.build_status_html(True) and "demo hesaplarla" not in landing.build_status_html(False))

# --- kartlar
cards = landing.build_cards(landing.SECTORS, columns=4, core=2)
check("4 sektör kartı, ilk ikisi çekirdek", cards.count('class="step') == 4 and cards.count("step core") == 2 and 'flow c4' in cards)
check("3 sütun varsayılan sınıf", landing.build_cards(landing.HOW_STEPS).startswith('<div class="flow">'))
try:
    landing.build_cards(landing.HOW_STEPS, columns=5)
    check("geçersiz sütun sayısı reddedilir", False)
except ValueError:
    check("geçersiz sütun sayısı reddedilir", True)
esc = landing.build_cards((("<b>x</b>", '"><script>', "a & b", "<i>"),), columns=2)
check("kart metni kaçışlanır", "<script>" not in esc and "&lt;script&gt;" in esc and "a &amp; b" in esc and "<b>x</b>" not in esc)
check("kartlar tek satır HTML", "\n" not in cards and "\n" not in landing.build_about_html())
check("dört sektör: sigorta, tarım, belediye, afet", [c[0].split(" · ")[1] for c in landing.SECTORS] == ["SİGORTA", "TARIM", "BELEDİYE", "AFET VE KURTARMA"])
check("üç adım ve altı özellik", len(landing.HOW_STEPS) == 3 and len(landing.FEATURES) == 6 and len(landing.ABOUT) == 2)
hero = landing.build_home_hero() + landing.build_hero_copy()
check("kahraman: prototip ve sentetik veri etiketi", "PROTOTİP" in hero and "SENTETİK" in hero)

# --- dürüstlük: yapılmamış iş, uydurma rakam ya da müşteri iddiası yok
alltext = " ".join(" ".join(c) for group in (landing.SECTORS, landing.HOW_STEPS, landing.FEATURES, landing.ABOUT,
                                                landing.PRINCIPLES, landing.ROADMAP, landing.HOW_DETAIL) for c in group)
alltext += " " + landing.STATUS_TEXT + " " + hero + " " + landing.HOME_ABOUT + " " + landing.STATS_NOTE + " " + landing.SERVICES_COPY
alltext += " " + " ".join(n + " " + lab + " " + sub for n, lab, sub in landing.STATS)
alltext += " " + landing.build_bento_html() + landing.build_steps_html() + landing.build_offer_html(True) + landing.build_stats_html()
alltext += " " + " ".join(t + " " + sub for t, sub in landing.PILLARS) + " " + " ".join(landing.PIPELINE)
alltext += " " + " ".join(q + " " + a for q, a in landing.FAQ)
for _sid, kicker, title, who, problem, items, status in landing.SERVICES:
    alltext += " " + " ".join((kicker, title, who, problem, status) + tuple(items))
for kicker, title, items in landing.ROLES:
    alltext += " " + " ".join((kicker, title) + tuple(items))
for pid in landing.PAGES:
    alltext += " " + landing.build_page_header(pid[0])
check("rakamsal iddia yok (yüzde, müşteri/kurum sayısı)", not re.search(r"\d+\s*(%|müşteri|kurum|sigortacı|belediye|poliçe)|%\s*\d", alltext, re.I), re.findall(r".{12}\d+\s*%.{6}", alltext))
check("referans/müşteri/ödül iddiası yok", not re.search(r"referans|müşterilerimiz|ödül|sertifika|kullanan\s+\w+\s+kurum", alltext, re.I))
check("durum notu: prototip, sentetik, canlı denenmedi", all(w in landing.STATUS_TEXT for w in ("prototip", "sentetiktir", "denenmedi")))
check("ekspertizin yerine geçmediği açık", "ekspertizin yerine geçmez" in alltext)
check("yüklenen parsel dosyası vaadi yok (henüz yok)", not re.search(r"dosyan[ıi]z[ıi]\s+yükle|verinizi\s+yükle|kendi parsel", alltext, re.I))
check("iletişim ayar adı", landing.CONTACT_ENV == "ORBITAL_CONTACT_EMAIL")


check("kişi biyografisi, ekip/ofis ya da üstünlük iddiası yok",
      not re.search(r"mezun|üniversite|yıllık deneyim|ekibimiz|kadromuz|ofisimiz|lider|en iyi|ilk ve tek|garanti|patent|%\s*100", alltext, re.I),
      re.findall(r".{10}(?:mezun|üniversite|ekibimiz|kadromuz|lider|en iyi|garanti).{10}", alltext, re.I))
check("hiçbir yerde e-posta adresi ya da bağlantı sabit yazılı değil", "@" not in alltext and "http" not in alltext.lower())

# --- sayfalar ve menü
ids = [p for p, _ in landing.PAGES]
check("beş sayfa, sıra: ana, hakkımızda, hizmetler, nasıl, iletişim", ids == ["ana", "hakkimizda", "hizmetler", "nasil", "iletisim"])
check("sayfa kimlikleri küçük harf ASCII ve tekil", len(set(ids)) == 5 and all(re.fullmatch(r"[a-z]+", i) for i in ids))
check("etiketler dolu ve tekil", all(lab.strip() for _, lab in landing.PAGES) and len({lab for _, lab in landing.PAGES}) == 5)
check("varsayılan sayfa ana", landing.DEFAULT_PAGE == "ana" and landing.QUERY_KEY == "sayfa")
for i in ids:
    check(f"geçerli sayfa kimliği korunur: {i}", landing.valid_page(i) == i and landing.page_label(i) == dict(landing.PAGES)[i])
for bad in ("", "Ana", "ana ", "admin", "../x", "ana\nx", "<script>", None, 3, ["ana"], {"ana": 1}, b"ana"):
    check(f"geçersiz sayfa ana sayfaya düşer: {bad!r:.30}", landing.valid_page(bad) == "ana" and landing.page_label(bad) == "ANA SAYFA")

# --- HTML kuralları: iyi biçimli, betik yok, satır sonu yok
VOID = {"br", "hr", "img", "input", "meta", "link"}


class _Bal(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.errors, self.tags = [], [], []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(tag)
        else:
            self.stack.pop()


def balanced(markup):
    p = _Bal()
    p.feed(markup)
    p.close()
    return not p.errors and not p.stack, p


builders = {
    "hero": landing.build_page_header("ana"),
    "nav_brand": landing.build_nav_brand(),
    "hero_copy": landing.build_hero_copy(),
    "about_head": landing.build_about_head(),
    "stats": landing.build_stats_html(),
    "services_head": landing.build_services_head(),
    "bento": landing.build_bento_html(),
    "steps_head": landing.build_steps_head(),
    "steps": landing.build_steps_html(),
    "offer_demo": landing.build_offer_html(True),
    "offer": landing.build_offer_html(False),
    "glass_cube": landing.build_glass("cube", "g-hero"),
    "glass_pill": landing.build_glass("pill"),
    "pillars": landing.build_pillars(),
    "about": landing.build_about_html(),
    "services": landing.build_services_html(),
    "roles": landing.build_roles_html(),
    "pipeline": landing.build_pipeline_html(),
    "cta": landing.build_cta_band(True),
    "contact": landing.build_contact_html("info@example.com", True),
    "hints": landing.build_contact_hints(),
    "status": landing.build_status_html(True),
    "faq": landing.build_faq_answer("a & b"),
    "roadmap": landing.build_cards(landing.ROADMAP, 3),
    "principles": landing.build_cards(landing.PRINCIPLES, 4),
}
for pid in ids[1:]:
    builders["header_" + pid] = landing.build_page_header(pid)
for name, markup in builders.items():
    good, parsed = balanced(markup)
    check(f"{name}: etiketler dengeli", good, (parsed.errors[:2], parsed.stack[-2:]))
    check(f"{name}: tek satır, betik ve olay öznitelikleri yok",
          "\n" not in markup and not re.search(r"<script|<iframe|<object|<embed|javascript:|\son\w+\s*=", markup, re.I))

# --- ana sayfa: kahraman, cam şekiller, ızgara, istatistik
check("Türkçe büyük harf: i→İ, ı→I", landing.tr_upper("hizmetlerimiz") == "HİZMETLERİMİZ" and landing.tr_upper("ılık iş") == "ILIK İŞ"
      and landing.tr_upper("Nasıl çalışır") == "NASIL ÇALIŞIR")
check("menü etiketleri CSS'e bırakılmadan büyük harf yazılı", all(lab == landing.tr_upper(lab) for _, lab in landing.PAGES))
check("menü etiketlerinde Türkçe harfler doğru", dict(landing.PAGES)["hizmetler"] == "HİZMETLERİMİZ" and dict(landing.PAGES)["hakkimizda"] == "HAKKIMIZDA"
      and dict(landing.PAGES)["iletisim"] == "İLETİŞİM")
hh = builders["hero"]
check("kahraman: dört satırlı başlık (sıra korunur), cam küp ve etiket", landing.HERO_LINES == ("UYDUDAN", "PARSELE,", "PARSELDEN", "KARARA.")
      and all(f'<span class="l{i}">{t}</span>' in hh for i, t in enumerate(landing.HERO_LINES, 1)) and hh.count('class="lp-glass cube') == 1
      and 'role="heading" aria-level="1"' in hh)
check("kahraman SVG/resim içermez (yalnızca CSS)", not re.search(r"<svg|<img|<image|<canvas|<video", hh + builders["bento"] + builders["offer_demo"], re.I))
check("cam şekiller ekran okuyuculardan gizli", all('aria-hidden="true"' in builders[k] for k in ("glass_cube", "glass_pill")))
check("cam şekil türleri: cube/pill", 'lp-glass cube' in builders["glass_cube"] and 'lp-glass pill' in builders["glass_pill"])
for bad_kind in ("", "sphere", None, 3):
    try:
        landing.build_glass(bad_kind)
        check(f"geçersiz cam şekil türü reddedilir: {bad_kind!r}", False)
    except ValueError:
        check(f"geçersiz cam şekil türü reddedilir: {bad_kind!r}", True)
for bad_cls in ('x"><script>alert(1)</script>', "A", "a;b", "a\nb", "x" * 41, "a<b"):
    try:
        landing.build_glass("cube", bad_cls)
        check(f"geçersiz sınıf adı reddedilir: {bad_cls!r:.30}", False)
    except ValueError:
        check(f"geçersiz sınıf adı reddedilir: {bad_cls!r:.30}", True)
check("cam şekil sabittir (iki çağrı aynı)", landing.build_glass("cube", "g-hero") == landing.build_glass("cube", "g-hero"))
check("ürün kapsamı sayıları gerçekle tutarlı (4 alan, 3 indeks, 4 karar türü)",
      landing.STATS[0][0] == str(len(landing.SECTORS)) and landing.STATS[1][0] == "3" and landing.STATS[1][2] == "NDVI · NBR · dNBR"
      and landing.STATS[2][0] == "4" and landing.STATS[2][2].count("·") == 3)
check("sayılar kullanım/müşteri iddiası olmadığını söyler", "müşteri ya da kullanım sayısı değildir" in landing.STATS_NOTE
      and "müşteri ya da kullanım sayısı değildir" in builders["stats"])
check("istatistik: üç kutu ve iki cam şekil", builders["stats"].count('class="lp-stat"') == 3 and builders["stats"].count("lp-glass cube") == 2)
bt = builders["bento"]
check("ızgara: dört kart + cam kart; ilk kart beyaz", len(re.findall(r'class="lp-bcard b[1-4]', bt)) == 4 and bt.count('class="lp-bcard bg"') == 1 and bt.count(" white") == 1
      and 'class="lp-bcard b1 white"' in bt)
check("ızgara: 01-04 numaraları ve ÇEKİRDEK/GENİŞLEME etiketleri", all(f'<span class="num">0{i}</span>' in bt for i in range(1, 5))
      and bt.count('<span class="st">ÇEKİRDEK</span>') == 2 and bt.count('<span class="st">GENİŞLEME</span>') == 2)
check("ızgara başlıkları Türkçe büyük harf", all(landing.tr_upper(c[1]) in bt for c in landing.SECTORS) and "KAÇAK YAPILAŞMA VE YEŞİL ALAN" in bt)
check("ızgara cam kartı gizli ve metinsiz", re.search(r'<div class="lp-bcard bg" aria-hidden="true"><span class="lp-glass pill g-bento" aria-hidden="true"></span></div>', bt) is not None)
check("adımlar: üç çerçeveli kart, numaralı", builders["steps"].count('class="lp-bcard"') == 3 and all(f'<span class="num">0{i}</span>' in builders["steps"] for i in (1, 2, 3)))
check("çağrı bölümü: başlık, cam hap; demo cümlesi yalnızca demoda",
      "UYGULAMAYI DENEYİN" in builders["offer_demo"] and "lp-glass pill g-offer" in builders["offer_demo"]
      and "demo hesaplarla" in builders["offer_demo"] and "demo hesaplarla" not in builders["offer"])
check("bölüm başlıkları anlamsal (aria-level 2)", all('role="heading" aria-level="2"' in builders[k] for k in ("about_head", "services_head", "steps_head", "offer")))
check("yazı logosu: ad ve kırmızı nokta", builders["nav_brand"] == '<div class="lp-logo">orbital field<i>.</i></div>')
check("__all__ içindeki her ad tanımlı", all(hasattr(landing, n) for n in landing.__all__), [n for n in landing.__all__ if not hasattr(landing, n)])
check("ana sayfa dışı başlıklarda cam şekil yok", all("lp-glass" not in builders["header_" + p] for p in ids[1:]))
for pid in ids[1:]:
    h = builders["header_" + pid]
    check(f"başlık {pid}: etiket, başlık ve açıklama", all(k in h for k in ('hero-eyebrow', 'hero-title', 'hero-sub')))
check("bilinmeyen sayfa ana sayfa başlığına düşer", landing.build_page_header("yok") == builders["hero"] and landing.build_page_header(None) == builders["hero"])

# --- hizmetler
sv = landing.SERVICES
check("dört hizmet: sigorta, tarım, belediye, afet (özet kartlarla aynı sıra)",
      [s[0] for s in sv] == ["sigorta", "tarim", "belediye", "afet"] and [s[1] for s in sv] == [c[0] for c in landing.SECTORS]
      and [s[2] for s in sv] == [c[1] for c in landing.SECTORS])
check("ilk ikisi çekirdek, ikisi genişleme", [s[1].split(" · ")[0] for s in sv] == ["ÇEKİRDEK", "ÇEKİRDEK", "GENİŞLEME", "GENİŞLEME"])
check("her hizmette en az üç madde, kim için ve neden dolu", all(len(s[5]) >= 3 and s[3].strip() and s[4].strip() for s in sv))
check("her hizmet durumu prototip ya da prototip modülü (hiçbiri 'üretimde' denmez)",
      all("PROTOTİP" in s[6] for s in sv) and not re.search(r"üretim|canlı kullanım", " ".join(s[6] for s in sv), re.I))
sh = builders["services"]
check("hizmet bölümleri: 4 bölüm, 2 çekirdek", sh.count('<section class="lp-svc') == 4 and sh.count('<section class="lp-svc core"') == 2 and all(f'id="{s[0]}"' in sh for s in sv))
check("hizmet maddeleri sayısı kadar <li>", sh.count("<li>") == sum(len(s[5]) for s in sv))
saved = landing.SERVICES
landing.SERVICES = (("x\"><b>", "<i>k</i>", "<script>alert(1)</script>", "a & b", "<img src=x onerror=1>", ("<u>m</u>",), "\"s\""),)
mal = landing.build_services_html()
landing.SERVICES = saved
check("hizmet metinleri kaçışlanır (kimlik ve öznitelik dahil)", "<script>" not in mal and "<img" not in mal and "<b>" not in mal
      and "<u>" not in mal and "&lt;script&gt;" in mal and 'id="x&quot;&gt;&lt;b&gt;"' in mal)
roles = builders["roles"]
check("iki rol: sigortacı ve müşteri; iç gerekçe ve şirket ayrımı belirtilir",
      roles.count('class="step"') == 2 and "SİGORTACI" in roles and "MÜŞTERİ" in roles and "İç gerekçeniz yalnızca sizde kalır" in roles)
check("müşteri rolü: birden çok şirket ve başkasının verisini görmeme", "birden çok sigorta şirketinde" in roles and "hiçbir verisini göremezsiniz" in roles)

# --- nasıl çalışır, SSS, yol haritası
check("beş ayrıntılı adım, sıralı etiket", [c[0] for c in landing.HOW_DETAIL] == [f"ADIM 0{i}" for i in range(1, 6)])
check("akış çizgisi: altı düğüm, beş ok", builders["pipeline"].count('class="lp-node"') == 6 and builders["pipeline"].count("lp-arrow") == 5)
check("yedi SSS, soru işaretiyle biter ve cevaplar dolu", len(landing.FAQ) == 7 and all(q.endswith("?") and len(a) > 40 for q, a in landing.FAQ))
faq = " ".join(a for _, a in landing.FAQ)
check("SSS: gerçek veri denenmedi, sentetik, ekspertiz, güvenlik sınırı, kalıcılık açıkça yazılı",
      all(w in faq for w in ("sentetiktir", "henüz denenmedi", "ekspertizin yerine geçmez", "henüz yoktur", "kalıcı olmayabilir", "kurgusaldır")))
check("SSS: güvenlik cevabı 'güvenli' garantisi vermez", not re.search(r"tamamen güvenli|%\s*100|kesinlikle güvenli|hacklenemez", faq, re.I))
check("SSS: kendi veri yükleme sorusuna 'henüz değil' denir", any(q.startswith("Kendi verimle") and a.startswith("Henüz değil") for q, a in landing.FAQ))
check("SSS cevabı kaçışlanır", landing.build_faq_answer("<b>&</b>") == '<p class="lp-faq">&lt;b&gt;&amp;&lt;/b&gt;</p>')
rm = landing.ROADMAP
check("yol haritası: yalnızca ilk aşama yapıldı, diğerleri 'henüz yapılmadı'",
      "YAPILDI" in rm[0][0] and all("PLANLANDI" in r[0] and "HENÜZ YAPILMADI" in r[3] and "Henüz yapılmadı" in r[2] for r in rm[1:]))
check("geliştirici kartı yalnızca ad içerir", len(landing.TEAM) == 1 and landing.TEAM[0][1] == "Elif Turut" and landing.TEAM[0][3] == "")
check("iletişim ipuçları e-posta toplamadığını söyler", "toplamaz" in builders["hints"] and "<li>" in builders["hints"])
check("çağrı bandı: demo açıkken demo cümlesi, kapalıyken yok",
      "demo hesaplarla" in landing.build_cta_band(True) and "demo hesaplarla" not in landing.build_cta_band(False))
check("değer önermeleri dört tane ve rakam içermez", len(landing.PILLARS) == 4 and builders["pillars"].count('class="lp-pillar"') == 4 and not re.search(r"\d", builders["pillars"]))

# --- CSS
for key in (".flow.c2", ".flow.c4", ".lp-cta", ".lp-contact", ".lp-hero", ".st-key-lp_nav", "st-key-navon_", ".st-key-lp_nav .st-key-lp_login",
            "st-key-lp_pri_", "st-key-lp_sec_", ".lp-hero2", ".lp-display", ".lp-glass", ".lp-glass::before", ".lp-glass::after", ".lp-bento", ".lp-bcard",
            ".lp-bcard.white", ".lp-steps", ".lp-offer", ".lp-stat", ".lp-h2", ".lp-logo", ".lp-svc", ".lp-svc-head .tag", ".lp-list",
            ".lp-pipeline", ".lp-band", ".lp-pillars", ".st-key-lp_center_btns", ".st-key-lp_right_btn", "prefers-reduced-motion", "@keyframes lpfloat"):
    check(f"CSS tanımı: {key}", key in landing.LANDING_CSS)
check("CSS: dar ekranda menü yapışkan değil ve satıra sarılır", "max-width:640px" in landing.LANDING_CSS and "position:static" in landing.LANDING_CSS and "flex-wrap:wrap" in landing.LANDING_CSS)
check("CSS: ızgara dar ekranda tek sütun", "@media (max-width:820px)" in landing.LANDING_CSS and ".lp-bento{ grid-template-columns:1fr; }" in landing.LANDING_CSS)
check("CSS dışarıdan kaynak yüklemez", not re.search(r"@import|url\(", landing.LANDING_CSS + landing.LANDING_BG_CSS))
check("CSS: Türkçe i/ı'yı bozan text-transform kullanılmaz", "text-transform" not in landing.LANDING_CSS)
check("CSS: cam şekil tarayıcı öneki ve maske, animasyon azaltılmış harekette kapanır",
      "-webkit-backdrop-filter" in landing.LANDING_CSS and "-webkit-mask-composite" in landing.LANDING_CSS
      and re.search(r"prefers-reduced-motion:reduce\)\{ \.lp-glass\{ animation:none; \}", landing.LANDING_CSS) is not None)
check("CSS: cam şekiller tıklamayı engellemez ve yazının altında kalır", "pointer-events:none" in landing.LANDING_CSS and ".lp-display{ position:relative; z-index:2;" in landing.LANDING_CSS)
check("arka plan CSS'i yalnızca siyah zemin ve genişlik", ".stApp{ background:#000" in landing.LANDING_BG_CSS and ".block-container" in landing.LANDING_BG_CSS
      and "<" not in landing.LANDING_BG_CSS)
check("CSS parantezleri dengeli", landing.LANDING_CSS.count("{") == landing.LANDING_CSS.count("}") and landing.LANDING_BG_CSS.count("{") == landing.LANDING_BG_CSS.count("}"))

print()
print("BAŞARISIZ: " + ", ".join(FAILS) if FAILS else "BAŞARISIZ: yok — tüm kontroller geçti")
sys.exit(1 if FAILS else 0)
