"""Giriş öncesi karşılama sayfası: içerik ve HTML üreticileri (Streamlit'e bağlı değildir).

Sayfa herkese açıktır; hesap, müşteri ya da veri bilgisi içermez. İletişim adresi koda yazılmaz: ``ORBITAL_CONTACT_EMAIL``
ayarıyla verilir, geçerli değilse iletişim düğmesi hiç çizilmez (uydurma adres gösterilmez).
"""
from __future__ import annotations

import html
import re
from typing import Sequence, Tuple
from urllib.parse import quote

__all__ = [
    "CONTACT_ENV", "MAILTO_SUBJECT", "LANDING_CSS", "SECTORS", "HOW_STEPS", "FEATURES", "valid_email", "build_mailto",
    "build_landing_hero", "build_about_html", "build_cards", "build_status_html", "build_contact_html",
]

#: İletişim adresini veren ayar (ortam değişkeni ya da ``st.secrets``).
CONTACT_ENV = "ORBITAL_CONTACT_EMAIL"
MAILTO_SUBJECT = "ORBITAL FIELD — demo / pilot görüşmesi"
_MAILTO_BODY = ("Merhaba,\n\nORBITAL FIELD hakkında görüşmek istiyorum.\n\nAd soyad:\nKurum:\n"
                "İlgilendiğim alan (sigorta / tarım / belediye / afet):\n")
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,24}$")
MAX_EMAIL = 120

# (üst etiket, başlık, açıklama, alt etiket)
Card = Tuple[str, str, str, str]

#: "Neler yapıyoruz": ilk ikisi çekirdek pazar, sonrakiler genişleme (yol haritasıyla aynı sıra).
SECTORS: Tuple[Card, ...] = (
    ("ÇEKİRDEK · SİGORTA", "Hasar doğrulama ve tazminat kararı",
     "Parsel bazlı hasar haritası ve poliçe risk skoru. Muafiyet, teminat ve eşikleri siz girersiniz; her poliçe için "
     "onay, kısmi onay, ekspertiz ya da ret kararı çıkar ve kararın kanıt belgesi hazırlanır.", "SİGORTACILAR"),
    ("ÇEKİRDEK · TARIM", "Bitki gelişimi ve rekolte izleme",
     "Mevsimlik gelişim eğrisi, su stresi dönemleri ve parsel bazlı rekolte tahmini.", "TARIM EKİPLERİ"),
    ("GENİŞLEME · BELEDİYE", "Kaçak yapılaşma ve yeşil alan",
     "İki tarihli görüntüden yapılaşma ve yeşil alan değişimi; mahalle bazlı önceliklendirme.", "BELEDİYELER"),
    ("GENİŞLEME · AFET VE KURTARMA", "Yangın, risk ve tahliye",
     "Termal anomali izleme, yerleşim riski, tahliye rotaları ve saha ekibi koordinasyonu.", "AFET EKİPLERİ"),
)

HOW_STEPS: Tuple[Card, ...] = (
    ("ADIM 01", "Parsel ve olay", "Parsel sınırları ve olay tarihi seçilir.", ""),
    ("ADIM 02", "İndeks ve sınıf",
     "Uydu görüntüsünden bitki ve hasar indeksleri (NDVI, NBR, dNBR) hesaplanır; parseller hasar sınıflarına ayrılır.", ""),
    ("ADIM 03", "Kural, karar ve kanıt",
     "Kurallarınız arayüzden girilir; her parsel için karar çıkar ve nasıl çıktığı belgelenir.", ""),
)

FEATURES: Tuple[Card, ...] = (
    ("KOD YOK", "Kuralları siz belirlersiniz",
     "Eşik, muafiyet, teminat ve avans oranları arayüzden değişir; harita ve tablo anında yeniden hesaplanır.", ""),
    ("KARAR", "Poliçe düzeyinde karar tablosu",
     "Onay, kısmi onay, ekspertiz ve ret; filtrelenir, sıralanır, CSV olarak indirilir.", ""),
    ("KANIT", "Her karar için belge",
     "Kural izi, tutar hesabı ve içerik özetiyle indirilebilir kanıt belgesi; toplu paket olarak da alınır.", ""),
    ("DOĞRULAMA", "Geçmiş kararlarla sınama",
     "Kurallarınızı geçmiş uzman kararlarına karşı sınayın; eşikler yalnızca eğitim kayıtlarında aranır, hiç "
     "görülmemiş kayıtlarda doğrulanır.", ""),
    ("MÜŞTERİ", "Müşteri portalı",
     "Müşteriniz yalnızca kendi poliçelerini, hasar kararını, ödeme adımlarını ve parsellerinin verimlilik takibini görür.", ""),
    ("HARİTA", "Canlı uydu altlığı",
     "NASA günlük uydu görüntüsü ve gün seçici; görsel bağlam sağlar, hesaplara girmez.", ""),
)

ABOUT: Tuple[Card, ...] = (
    ("BİZ NEYİZ", "Uydu verisini karara çeviren araç",
     "ORBITAL FIELD, uydu görüntülerinden hesaplanan bitki ve hasar indekslerini ekiplerin kendi kurallarına bağlayan, "
     "kod yazmayı gerektirmeyen bir analiz platformudur. Uydu verisi parsel ve tarih bazında hasarı ve gelişimi ölçmeye "
     "yeter; eksik olan, bu ölçümü uzman gerektirmeden bir karara çeviren araçtır.", ""),
    ("AMACIMIZ", "Daha hızlı ve belgelenebilir karar",
     "Sahada yapılan incelemelerin bir kısmını uydu verisiyle hızlandırmak ve her kararın nasıl çıktığını belgelenebilir "
     "kılmak. Platform ekspertizin yerine geçmez; kararları ekspertizle yan yana koyup karşılaştırmayı sağlar.", ""),
)

STATUS_TEXT = ("<b>Şu anki durum: prototip.</b> Gösterilen tüm parsel, hasar ve müşteri verileri sentetiktir; sonuçlar gerçek "
               "bir hasar dosyasıyla karşılaştırılmadı. Gerçek uydu verisi bağlantıları (Sentinel-2, NASA FIRMS) yazıldı "
               "ancak canlı hesaplarla denenmedi.")
DEMO_TEXT = " Giriş ekranındaki demo hesaplarla uygulamayı hemen deneyebilirsiniz."


def _h(value: object) -> str:
    """HTML metnini ve özniteliğini kaçışlar."""
    return html.escape(str(value), quote=True)


def valid_email(value: object) -> str:
    """İletişim adresini doğrular.

    Args:
        value: Ayardan gelen ham değer.

    Returns:
        Kırpılmış adres; metin değilse, çok uzunsa ya da biçimi geçersizse (boşluk, satır sonu, ``?``, ``&``, ``<``,
        ``"`` vb. dahil) boş metin.
    """
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text or len(text) > MAX_EMAIL or not _EMAIL_RE.fullmatch(text):
        return ""
    return text


def build_mailto(email: str) -> str:
    """Önceden doldurulmuş konu ve gövdeyle ``mailto:`` bağlantısı kurar (adres doğrulanmamışsa boş döner).

    Args:
        email: ``valid_email`` çıktısı.

    Returns:
        ``mailto:`` adresi ya da boş metin.
    """
    clean = valid_email(email)
    if not clean:
        return ""
    return f"mailto:{clean}?subject={quote(MAILTO_SUBJECT)}&body={quote(_MAILTO_BODY)}"


def build_landing_hero() -> str:
    """Karşılama başlığı: etiket, büyük başlık, açıklama ve hap etiketleri.

    Returns:
        HTML dizgesi.
    """
    chips = "".join(f'<span class="chip">{_h(c)}</span>' for c in ("PROTOTİP", "DEMO · SENTETİK VERİ"))
    return (
        '<div class="hero lp-hero"><div><div class="hero-eyebrow">UYDU VERİSİ · KARAR PLATFORMU</div>'
        '<div class="hero-title">UYDUDAN PARSELE, <span>PARSELDEN KARARA.</span></div>'
        '<div class="hero-sub">ORBITAL FIELD, parsel bazlı uydu analizini kod yazmadan sigorta, tarım, belediye ve afet '
        "ekiplerinin karar sürecine taşır.</div></div>"
        f'<div class="chips">{chips}</div></div>')


def build_cards(items: Sequence[Card], columns: int = 3, core: int = 0) -> str:
    """Numaralı/etiketli kart ızgarası (uygulamadaki ``.flow`` / ``.step`` düzeniyle aynı).

    Args:
        items: ``(üst etiket, başlık, açıklama, alt etiket)`` dörtlüleri.
        columns: Sütun sayısı (2, 3 ya da 4).
        core: İlk kaç kartın "çekirdek" vurgusu alacağı.

    Returns:
        HTML dizgesi.
    """
    if columns not in (2, 3, 4):
        raise ValueError("columns 2, 3 ya da 4 olmalı")
    cards = []
    for i, (kicker, title, text, tag) in enumerate(items):
        tag_html = f'<span class="tag">{_h(tag)}</span>' if tag else ""
        cards.append(
            f'<div class="step{" core" if i < core else ""}"><div class="n">{_h(kicker)}</div>'
            f'<div class="t">{_h(title)}</div><p>{_h(text)}</p>{tag_html}</div>')
    cls = "" if columns == 3 else f" c{columns}"
    return f'<div class="flow{cls}">{"".join(cards)}</div>'


def build_about_html() -> str:
    """"Biz neyiz" ve "Amacımız" kartları."""
    return build_cards(ABOUT, columns=2)


def build_status_html(demo: bool = False) -> str:
    """Prototip durumunu açıkça söyleyen not kutusu.

    Args:
        demo: Giriş ekranında demo hesapları gösteriliyorsa ``True`` (o zaman denemeye davet eklenir).
    """
    return f'<div class="note info">{STATUS_TEXT}{DEMO_TEXT if demo else ""}</div>'


def build_contact_html(email: str = "", demo: bool = False) -> str:
    """"Hizmet alın" bölümü: adres geçerliyse ``mailto:`` düğmesi, değilse yalnızca açıklama.

    Args:
        email: Ayardan gelen iletişim adresi (doğrulanır).
        demo: Demo hesapları giriş ekranında gösteriliyorsa ``True``.

    Returns:
        HTML dizgesi.
    """
    link = build_mailto(email)
    if link:
        action = (f'<a class="lp-cta" href="{_h(link)}">BİZE ULAŞIN · HİZMET ALIN</a>'
                  f'<span class="lp-mail">{_h(valid_email(email))}</span>')
        text = ("Pilot görüşmesi ve demo için bize yazın; hangi alanla ilgilendiğinizi kısaca belirtmeniz yeterli. "
                "Düğme e-posta uygulamanızı hazır bir iletiyle açar.")
    else:
        action = ""
        text = "Pilot görüşmesi ve demo talepleri için iletişim adresi yakında burada yayımlanacak." + (
            " O zamana kadar demo hesaplarla uygulamayı kendiniz deneyebilirsiniz." if demo else "")
    return f'<div class="lp-contact"><p>{_h(text)}</p><div class="lp-actions">{action}</div></div>'


LANDING_CSS = """
.flow.c2{ grid-template-columns:repeat(2,minmax(0,1fr)); }
.flow.c4{ grid-template-columns:repeat(4,minmax(0,1fr)); }
@media (max-width:1280px){ .flow.c4{ grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width:820px){ .flow.c2, .flow.c4{ grid-template-columns:1fr; } }
.lp-hero{ padding-top:26px; }
.lp-hero .hero-title{ font-size:clamp(2.2rem,4.2vw,3.5rem); }
.lp-contact{ padding:22px 24px; border-radius:var(--r-lg,16px); border:1px solid var(--edge,rgba(255,255,255,.12));
  background:linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.012)), var(--glass,rgba(16,19,26,.6));
  -webkit-backdrop-filter:blur(16px); backdrop-filter:blur(16px); }
.lp-contact p{ margin:0 0 14px; color:var(--muted); font:400 .95rem/1.6 'Inter',sans-serif; max-width:720px; }
.lp-actions{ display:flex; flex-wrap:wrap; align-items:center; gap:16px; }
.lp-cta{ display:inline-block; padding:10px 22px; border-radius:999px; background:var(--red); color:#fff !important;
  font:600 .74rem 'Inter',sans-serif; letter-spacing:.14em; text-decoration:none !important; }
.lp-cta:hover{ filter:brightness(1.1); }
.lp-mail{ color:var(--soft); font:500 .82rem ui-monospace,Menlo,Consolas,monospace; }
"""
