"""landing.py çevrimdışı testleri (Streamlit gerekmez): iletişim adresi doğrulama, mailto, HTML kaçışı, içerik dürüstlüğü."""
import os
import re
import sys
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import landing  # noqa: E402

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
hero = landing.build_landing_hero()
check("başlık: prototip ve sentetik veri etiketi", "PROTOTİP" in hero and "SENTETİK" in hero)

# --- dürüstlük: yapılmamış iş, uydurma rakam ya da müşteri iddiası yok
alltext = " ".join(" ".join(c) for group in (landing.SECTORS, landing.HOW_STEPS, landing.FEATURES, landing.ABOUT) for c in group)
alltext += " " + landing.STATUS_TEXT + " " + hero
check("rakamsal iddia yok (yüzde, müşteri/kurum sayısı)", not re.search(r"\d+\s*(%|müşteri|kurum|sigortacı|belediye|poliçe)|%\s*\d", alltext, re.I), re.findall(r".{12}\d+\s*%.{6}", alltext))
check("referans/müşteri/ödül iddiası yok", not re.search(r"referans|müşterilerimiz|ödül|sertifika|kullanan\s+\w+\s+kurum", alltext, re.I))
check("durum notu: prototip, sentetik, canlı denenmedi", all(w in landing.STATUS_TEXT for w in ("prototip", "sentetiktir", "denenmedi")))
check("ekspertizin yerine geçmediği açık", "ekspertizin yerine geçmez" in alltext)
check("yüklenen parsel dosyası vaadi yok (henüz yok)", not re.search(r"dosyan[ıi]z[ıi]\s+yükle|kendi parsel", alltext, re.I))
check("CSS yeni sınıfları tanımlar", all(k in landing.LANDING_CSS for k in (".flow.c2", ".flow.c4", ".lp-cta", ".lp-contact", ".lp-hero")))
check("iletişim ayar adı", landing.CONTACT_ENV == "ORBITAL_CONTACT_EMAIL")

print()
print("BAŞARISIZ: " + ", ".join(FAILS) if FAILS else "BAŞARISIZ: yok — tüm kontroller geçti")
sys.exit(1 if FAILS else 0)
