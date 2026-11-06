"""Giriş öncesi web sitesi: sayfa içerikleri ve HTML üreticileri (Streamlit'e bağlı değildir).

Site beş sayfadan oluşur (Ana Sayfa, Hakkımızda, Hizmetlerimiz, Nasıl çalışır, İletişim) ve üstteki menü çubuğuyla gezilir;
"Giriş yap" ayrı bir görünümdür. Sayfalar herkese açıktır; hesap, müşteri ya da veri bilgisi içermez.

İçerik kuralı: yapılmamış iş, uydurma rakam, müşteri/referans iddiası ya da kişi biyografisi YOKTUR. Her özellik için
"prototipte çalışıyor" ile "planlandı, henüz yapılmadı" ayrımı açıkça yazılır. İletişim adresi koda yazılmaz:
``ORBITAL_CONTACT_EMAIL`` ayarıyla verilir, geçerli değilse iletişim düğmesi hiç çizilmez.
"""
from __future__ import annotations

import html
import re
from typing import Dict, Sequence, Tuple
from urllib.parse import quote

__all__ = [
    "CONTACT_ENV", "MAILTO_SUBJECT", "LANDING_CSS", "LANDING_BG_CSS", "SECTORS", "HOW_STEPS", "FEATURES", "ABOUT", "PRINCIPLES",
    "ROADMAP", "SERVICES", "ROLES", "TEAM", "PILLARS", "HOW_DETAIL", "PIPELINE", "FAQ", "PAGES", "DEFAULT_PAGE", "QUERY_KEY",
    "STATUS_TEXT", "HERO_LINES", "HOME_ABOUT", "STATS", "STATS_NOTE", "valid_email", "build_mailto", "valid_page", "page_label",
    "tr_upper", "build_glass", "build_nav_brand", "build_home_hero", "build_hero_copy", "build_about_head", "build_stats_html",
    "build_services_head", "build_bento_html", "build_steps_head", "build_steps_html", "build_offer_html", "build_page_header",
    "build_pillars", "build_about_html", "build_cards", "build_services_html", "build_roles_html", "build_pipeline_html",
    "build_faq_answer", "build_status_html", "build_contact_html", "build_contact_hints", "build_cta_band",
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

# --------------------------------------------------------------------------------------------- sayfalar
#: Menü sırası: (kimlik, etiket). Kimlik adres çubuğundaki ``?sayfa=`` değeridir.
#: Etiketler CSS'e bırakılmadan büyük harf yazılır (CSS, Türkçe i/ı harflerini yanlış büyütür).
PAGES: Tuple[Tuple[str, str], ...] = (
    ("ana", "ANA SAYFA"), ("hakkimizda", "HAKKIMIZDA"), ("hizmetler", "HİZMETLERİMİZ"),
    ("nasil", "NASIL ÇALIŞIR"), ("iletisim", "İLETİŞİM"),
)
DEFAULT_PAGE = PAGES[0][0]
QUERY_KEY = "sayfa"
_PAGE_LABELS: Dict[str, str] = dict(PAGES)

#: Ana sayfa dışındaki sayfaların başlığı: (üst etiket, koyu kısım, soluk kısım, açıklama).
_HEADERS: Dict[str, Tuple[str, str, str, str]] = {
    "hakkimizda": ("HAKKIMIZDA", "UYDU VERİSİNİ", "ANLAŞILIR BİR KARARA ÇEVİRİYORUZ.",
                   "ORBITAL FIELD, kod yazmadan uydu analizi yapmak isteyen sigorta, tarım, belediye ve afet ekipleri "
                   "için geliştirilen bir prototiptir. Ne olduğumuzu, neye inandığımızı ve nerede olduğumuzu açıkça yazdık."),
    "hizmetler": ("HİZMETLERİMİZ", "NE YAPIYORUZ,", "AYRINTILI.",
                  "Dört alan, tek uydu çekirdeği. Her alanda ne sunduğumuzu ve hangi aşamada olduğumuzu ayrı ayrı belirttik."),
    "nasil": ("NASIL ÇALIŞIR", "PARSELDEN KARARA", "BEŞ ADIM.",
              "Uydu görüntüsünden müşterinin ekranındaki karara kadar yol, ve bu yolda sizin elinizde olanlar."),
    "iletisim": ("İLETİŞİM", "BİZİMLE", "GÖRÜŞÜN.",
                 "Demo ya da pilot görüşmesi için yazın. Önce uygulamayı kendiniz de deneyebilirsiniz."),
}


def valid_page(value: object) -> str:
    """Sayfa kimliğini doğrular; bilinmeyen ya da metin olmayan değer ana sayfaya düşer.

    Args:
        value: Oturumdan ya da adres çubuğundan gelen ham değer.

    Returns:
        Geçerli sayfa kimliği.
    """
    return value if isinstance(value, str) and value in _PAGE_LABELS else DEFAULT_PAGE


def page_label(page: object) -> str:
    """Sayfa kimliğinin menü etiketi (bilinmiyorsa ana sayfanınki)."""
    return _PAGE_LABELS[valid_page(page)]


# --------------------------------------------------------------------------------------------- içerik
#: Üç dört kelimelik değer önermeleri (ana sayfada şerit olarak görünür; rakam içermez).
PILLARS: Tuple[Tuple[str, str], ...] = (
    ("PARSEL BAZLI", "Hasar ve gelişim, tarla değil parsel düzeyinde ölçülür."),
    ("KURALLAR SİZDE", "Eşik, muafiyet ve teminatı kod yazmadan siz belirlersiniz."),
    ("KANITLI KARAR", "Her karar kural izi ve tutar hesabıyla belgelenir."),
    ("EKSPERTİZLE YAN YANA", "Otomatik öneri bir öneridir; son kararı insan verir."),
)

#: "Neler yapıyoruz" özeti: ilk ikisi çekirdek pazar, sonrakiler genişleme (yol haritasıyla aynı sıra).
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

PRINCIPLES: Tuple[Card, ...] = (
    ("İLKE 01", "Şeffaflık", "Her karar için kural izi, tutar hesabı ve içerik özeti belgelenir; kararın nasıl çıktığı "
                              "sonradan açıklanabilir.", ""),
    ("İLKE 02", "Kural sizin", "Eşikleri, muafiyeti ve teminatı ekip belirler. Platform yalnızca sizin kurallarınızı "
                                "uygular.", ""),
    ("İLKE 03", "İnsan karar verir", "Otomatik sonuç bir öneridir. Son kararı sigortacı verir; verilen her karar, "
                                      "kimin ne zaman verdiğiyle birlikte kaydedilir.", ""),
    ("İLKE 04", "Dürüst sınırlar", "Neyin çalıştığını, neyin denenmediğini ve neyin henüz yapılmadığını açıkça yazarız.", ""),
)

#: Geliştirici bilgisi: yalnızca ad; unvan, geçmiş ya da kurum iddiası yazılmaz.
TEAM: Tuple[Card, ...] = (("GELİŞTİRİCİ", "Elif Turut", "ORBITAL FIELD prototipinin geliştiricisi.", ""),)

#: Yol haritası: yalnızca niyet beyanıdır; "yapıldı" yazan tek aşama prototiptir.
ROADMAP: Tuple[Card, ...] = (
    ("ŞİMDİ · YAPILDI", "Çalışan prototip",
     "Sentetik veriyle çalışan sigorta, tarım, belediye ve afet modülleri; sigortacı ve müşteri girişi; şirket bazlı "
     "ayrım; müşteri dosyaları ve karar kaydı.", "PROTOTİP"),
    ("SIRADA · PLANLANDI", "Gerçek veriyle doğrulama",
     "Gerçek uydu veri bağlantılarını canlı hesaplarla denemek ve kuralları gerçek hasar dosyalarıyla karşılaştırmak. "
     "Henüz yapılmadı.", "HENÜZ YAPILMADI"),
    ("SONRA · PLANLANDI", "Sınırlı pilot",
     "Bir ya da birkaç kurumla sınırlı pilot; kalıcı veritabanı ve iki adımlı doğrulama gibi üretim gereksinimleri. "
     "Henüz yapılmadı.", "HENÜZ YAPILMADI"),
)

# (kimlik, üst etiket, başlık, kimin için, sorun, sunduklarımız, durum)
Service = Tuple[str, str, str, str, str, Tuple[str, ...], str]

SERVICES: Tuple[Service, ...] = (
    ("sigorta", "ÇEKİRDEK · SİGORTA", "Hasar doğrulama ve tazminat kararı",
     "Tarım, orman ve mülk sigortası yapan hasar ve ekspertiz ekipleri.",
     "Büyük bir olaydan sonra yüzlerce parselin hasarını sahada tek tek görmek haftalar alır ve kararlar ekipten ekibe "
     "değişebilir. Uydu verisi hangi parsele önce gidileceğini ve her kararın dayanağını gösterebilir.",
     ("Parsel bazlı hasar haritası: NDVI, NBR ve dNBR indekslerinden hasar sınıfı",
      "Poliçe risk skoru ve hasar şiddeti",
      "Muafiyet, teminat, eşik ve avans oranları arayüzden girilir; sonuç anında yeniden hesaplanır",
      "Her poliçe için otomatik öneri: onay, kısmi onay, ekspertiz ya da ret",
      "Müşteri dosyaları: istediğiniz müşteriyi açar, poliçelerini ve haritasını inceler, kararı kendiniz verirsiniz; "
      "otomatik öneri değişmez, kararınız ayrıca kaydedilir",
      "Her karar için indirilebilir kanıt belgesi ve toplu paket",
      "Kalibrasyon: kurallarınızı geçmiş uzman kararlarına karşı sınama",
      "Yönetim: profil, şirket özeti, ekip ve karar kaydı"),
     "PROTOTİPTE ÇALIŞIYOR · SENTETİK VERİ"),
    ("tarim", "ÇEKİRDEK · TARIM", "Bitki gelişimi ve rekolte izleme",
     "Tarım ekipleri ve parsel sahiplerine hizmet veren kurumlar.",
     "Bir sezonun gidişini görmek için tarlayı sık sık gezmek gerekir. Uydu serisi, gelişimin beklenen seyrin altında "
     "kaldığı dönemleri ve parselleri öne çıkarabilir.",
     ("Mevsimlik bitki gelişim eğrisi",
      "Su stresi dönemlerinin işaretlenmesi",
      "Parsel bazlı rekolte tahmini",
      "Müşteri portalında parsellerin verimlilik takibi"),
     "PROTOTİPTE ÇALIŞIYOR · SENTETİK VERİ"),
    ("belediye", "GENİŞLEME · BELEDİYE", "Kaçak yapılaşma ve yeşil alan",
     "Belediyelerin imar ve fen işleri ekipleri.",
     "Yapılaşma ve yeşil alan değişimini denetim turlarıyla yakalamak zordur. İki tarihli görüntü, değişen bölgeleri "
     "önceliklendirmeye yardım edebilir.",
     ("İki tarihli görüntüden yapılaşma değişimi",
      "Yeşil alan kaybı ve artışı",
      "Mahalle bazlı önceliklendirme"),
     "GENİŞLEME · PROTOTİP MODÜLÜ"),
    ("afet", "GENİŞLEME · AFET VE KURTARMA", "Yangın, risk ve tahliye",
     "Afet ve acil durum koordinasyon ekipleri.",
     "Bir olay sırasında nereye öncelik verileceği hızla belirlenmelidir. Termal anomali ve yerleşim bilgisi bu "
     "önceliklendirmeyi destekleyebilir.",
     ("Termal anomali (yangın) izleme",
      "Yerleşim riski",
      "Tahliye rotaları",
      "Saha ekibi koordinasyonu"),
     "GENİŞLEME · PROTOTİP MODÜLÜ"),
)

#: İki kullanıcı rolü: (üst etiket, başlık, maddeler)
Role = Tuple[str, str, Tuple[str, ...]]
ROLES: Tuple[Role, ...] = (
    ("SİGORTACI", "Şirketinizin poliçeleri, sizin kararınız", (
        "Yalnızca kendi şirketinizin poliçelerini görürsünüz; başka şirketin verisi size gelmez",
        "Müşteri dosyasını açar, poliçelerini ve haritasını incelersiniz",
        "Onay, kısmi onay, ekspertiz ya da ret kararı verir, müşteriye görünecek notu yazarsınız",
        "İç gerekçeniz yalnızca sizde kalır; müşteri görmez",
        "Verdiğiniz kararı geri alabilirsiniz; eski kayıt silinmez, yeni kayıt eklenir")),
    ("MÜŞTERİ", "Tüm poliçeleriniz tek ekranda", (
        "Poliçeleriniz birden çok sigorta şirketinde olabilir; hepsini şirket adıyla birlikte görürsünüz",
        "Karar, ödeme adımları ve parsellerinizin verimlilik takibi burada",
        "Kararı veren şirketin notunu okur, karar bildirimini indirirsiniz",
        "Başka bir müşterinin hiçbir verisini göremezsiniz")),
)

# Adım ayrıntıları: (etiket, başlık, açıklama, "")
HOW_DETAIL: Tuple[Card, ...] = (
    ("ADIM 01", "Parsel ve olay", "Parsel sınırları ve olay tarihi (yangın, dolu, kuraklık gibi) belirlenir.", ""),
    ("ADIM 02", "Önce ve sonra", "Olay öncesi ve sonrası uydu ölçümleri karşılaştırılır. Bu prototipte ölçümler "
                                 "sentetiktir.", ""),
    ("ADIM 03", "İndeks ve hasar sınıfı",
     "NDVI, NBR ve dNBR hesaplanır; parseller hasar sınıflarına ayrılır ve harita boyanır.", ""),
    ("ADIM 04", "Kural ve karar",
     "Muafiyet, teminat ve eşikler sizin kurallarınızla uygulanır; her poliçe için bir öneri çıkar. Sigortacı öneriyi "
     "olduğu gibi bırakır ya da kendi kararını verir.", ""),
    ("ADIM 05", "Kanıt ve müşteri",
     "Karar, nasıl çıktığıyla birlikte belgelenir. Müşteri kendi ekranında sonucu ve şirketin notunu görür.", ""),
)

PIPELINE: Tuple[str, ...] = ("Uydu görüntüsü", "Hasar indeksi", "Hasar sınıfı", "Sizin kurallarınız", "Karar", "Kanıt belgesi")

#: Sıkça sorulan sorular: (soru, cevap). Cevaplar prototip durumunu açıkça söyler.
FAQ: Tuple[Tuple[str, str], ...] = (
    ("Gerçek uydu verisiyle mi çalışıyor?",
     "Şu an hayır. Uygulamadaki tüm parsel, hasar ve müşteri verileri sentetiktir. Gerçek Sentinel-2 ve NASA FIRMS "
     "bağlantıları yazıldı ve çevrimdışı test edildi, ancak canlı hesaplarla henüz denenmedi. Haritadaki NASA günlük "
     "uydu görüntüsü yalnızca görsel bağlam sağlar, hesaplara girmez."),
    ("Ekspertizin yerine geçer mi?",
     "Hayır, ekspertizin yerine geçmez. Hangi parsele önce gidileceğini ve kararın dayanağını gösterir; sonuçları "
     "ekspertizle yan yana koymanızı sağlar. Sonuçlar gerçek bir hasar dosyasıyla henüz karşılaştırılmadı."),
    ("Kod yazmam gerekir mi?",
     "Hayır. Eşikler, muafiyet, teminat ve avans oranları arayüzden girilir; harita ve tablo anında yeniden hesaplanır."),
    ("Bir müşterinin poliçeleri farklı sigorta şirketlerinde olabilir mi?",
     "Evet. Şirket bilgisi müşteriye değil poliçeye bağlıdır: müşteri tüm poliçelerini şirket adıyla görür, sigortacı "
     "yalnızca kendi şirketinin poliçelerini. Demodaki şirketler kurgusaldır ve gerçek bir kurumla ilgisi yoktur."),
    ("Verilerim ne kadar güvende?",
     "Bu bir prototiptir. Parolalar özetlenerek saklanır, art arda hatalı girişte oturum geçici kilitlenir, işlem "
     "yapılmazsa oturum kapanır ve her sigortacı yalnızca kendi şirketinin poliçelerini görür. İki adımlı doğrulama, "
     "kurumsal tek oturum ve bağımsız güvenlik denetimi henüz yoktur; gerçek müşteri verisiyle kullanılmadan önce "
     "yapılmalıdır."),
    ("Karar kayıtları nerede tutuluyor?",
     "Prototipte sunucudaki bir veritabanı dosyasında. Kayıtlar eklemelidir: bir kararı geri almak eskisini silmez. "
     "Ücretsiz barındırmada bu dosya kalıcı olmayabilir; gerçek kullanımda kalıcı bir veritabanı gerekir."),
    ("Kendi verimle deneyebilir miyim?",
     "Henüz değil. Uygulama şimdilik yalnızca sentetik demo veriyle çalışır."),
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


# --------------------------------------------------------------------------------------------- ana sayfa
def tr_upper(text: str) -> str:
    """Türkçe kurallarıyla büyük harfe çevirir (i → İ, ı → I).

    CSS ``text-transform: uppercase`` sayfa dili İngilizce olduğunda ``i`` harfini noktasız ``I`` yapar; bu yüzden büyük
    harfli başlıklar CSS'e bırakılmaz, burada üretilir.
    """
    return text.replace("i", "İ").replace("ı", "I").upper()


#: Kahraman başlığı: dört satır, CSS ile kademeli girintilenir (satır sırası anlam taşır).
HERO_LINES: Tuple[str, ...] = ("UYDUDAN", "PARSELE,", "PARSELDEN", "KARARA.")
HERO_COPY = ("ORBITAL FIELD, parsel bazlı uydu analizini kod yazmadan sigorta, tarım, belediye ve afet ekiplerinin karar "
             "sürecine taşır. Kuralları siz belirlersiniz; her karar belgelenir.")
HOME_ABOUT = ("ORBITAL FIELD, uydu görüntülerinden hesaplanan bitki ve hasar indekslerini ekiplerin kendi kurallarına bağlayan, "
              "kod yazmayı gerektirmeyen bir analiz platformudur. Platform ekspertizin yerine geçmez; kararları ekspertizle "
              "yan yana koymayı sağlar.")
#: Ürün kapsamı sayıları (gerçek bir kullanım ya da müşteri sayısı DEĞİL): (sayı, etiket, açıklama).
STATS: Tuple[Tuple[str, str, str], ...] = (
    ("4", "ALAN", "sigorta · tarım · belediye · afet"),
    ("3", "UYDU İNDEKSİ", "NDVI · NBR · dNBR"),
    ("4", "KARAR TÜRÜ", "onay · kısmi onay · ekspertiz · ret"),
)
STATS_NOTE = "Prototipin kapsamını gösterir; müşteri ya da kullanım sayısı değildir."
SERVICES_COPY = ("Dört alan, tek uydu çekirdeği. Sigorta ve tarım çekirdek pazarımız; belediye ve afet genişleme alanlarımız. "
                 "Her biri prototip aşamasındadır.")
_CLASS_RE = re.compile(r"^[a-z0-9 \-]{0,40}$")


def _h2(text: str, cls: str = "") -> str:
    """Büyük, ince bölüm başlığı (anlamsal olarak ikinci düzey başlık)."""
    return f'<div class="lp-h2{" " + cls if cls else ""}" role="heading" aria-level="2">{_h(text)}</div>'


def build_glass(kind: str = "cube", cls: str = "") -> str:
    """Süs amaçlı cam (yarı saydam, gökkuşağı kenarlı) 3B şekil; yalnızca CSS ile çizilir.

    Args:
        kind: ``cube`` (yuvarlatılmış kare) ya da ``pill`` (hap).
        cls: Konumlandırma için ek sınıf adları (küçük harf, rakam, boşluk, tire).

    Returns:
        HTML dizgesi (ekran okuyuculardan gizli).
    """
    if kind not in ("cube", "pill"):
        raise ValueError("kind cube ya da pill olmalı")
    if not _CLASS_RE.fullmatch(cls):
        raise ValueError("cls yalnızca küçük harf, rakam, boşluk ve tire içerebilir")
    extra = f" {cls}" if cls else ""
    return f'<span class="lp-glass {kind}{extra}" aria-hidden="true"></span>'


def build_nav_brand() -> str:
    """Menü çubuğunun sol üstündeki yazı logosu."""
    return '<div class="lp-logo">orbital field<i>.</i></div>'


def build_home_hero() -> str:
    """Ana sayfa kahramanı: üst etiket, kademeli dev başlık ve başlığın üstünde yüzen cam şekil."""
    lines = "".join(f'<span class="l{i}">{_h(t)}</span>' for i, t in enumerate(HERO_LINES, 1))
    return ('<div class="lp-hero2"><div class="lp-eyebrow">UYDU VERİSİ · KARAR PLATFORMU</div>'
            f'<div class="lp-display" role="heading" aria-level="1">{lines}</div>{build_glass("cube", "g-hero")}</div>')


def build_hero_copy() -> str:
    """Kahraman altındaki kısa açıklama ve durum etiketleri (sağ sütunda durur)."""
    chips = "".join(f'<span class="chip">{_h(c)}</span>' for c in ("PROTOTİP", "DEMO · SENTETİK VERİ"))
    return f'<div class="lp-copy"><p>{_h(HERO_COPY)}</p><div class="chips">{chips}</div></div>'


def build_about_head() -> str:
    """"Hakkında" başlığı ve kısa tanıtım metni (sol sütun)."""
    return f'<div class="lp-sect">{_h2("HAKKINDA")}<p class="lp-lead">{_h(HOME_ABOUT)}</p></div>'


def build_stats_html() -> str:
    """Ürün kapsamı sayıları ve iki cam şekil (sağ sütun); sayılar kapsamı gösterir, kullanım iddiası taşımaz."""
    items = "".join(f'<div class="lp-stat"><b>{_h(n)}</b><span>{_h(lab)}</span><small>{_h(sub)}</small></div>'
                    for n, lab, sub in STATS)
    return (f'<div class="lp-stats-wrap"><div class="lp-stats">{items}</div><p class="lp-note">{_h(STATS_NOTE)}</p>'
            f'{build_glass("cube", "g-about")}{build_glass("cube", "g-edge")}</div>')


def build_services_head() -> str:
    """"Hizmetlerimiz" başlığı ve kısa açıklaması."""
    return f'<div class="lp-sect">{_h2("HİZMETLERİMİZ")}<p class="lp-lead">{_h(SERVICES_COPY)}</p></div>'


def build_bento_html() -> str:
    """Dört hizmetin "bento" ızgarası: çerçeveli yuvarlak kartlar, bir tanesi beyaz, bir tanesi cam şekilli süs.

    Kartlar :data:`SECTORS` sırasındadır (01 sigorta … 04 afet); sağ üstte "ÇEKİRDEK" ya da "GENİŞLEME" etiketi görünür.
    """
    cards = []
    for i, (kicker, title, text, _tag) in enumerate(SECTORS, 1):
        status = kicker.split(" · ")[0]
        white = " white" if i == 1 else ""
        cards.append(
            f'<div class="lp-bcard b{i}{white}"><span class="num">{i:02d}</span><span class="st">{_h(status)}</span>'
            f'<div class="t">{_h(tr_upper(title))}</div><p>{_h(text)}</p></div>')
    cards.append(f'<div class="lp-bcard bg" aria-hidden="true">{build_glass("pill", "g-bento")}</div>')
    return f'<div class="lp-bento">{"".join(cards)}</div>'


def build_steps_head() -> str:
    """"Nasıl çalışır" başlığı."""
    return f'<div class="lp-sect">{_h2("NASIL ÇALIŞIR")}</div>'


def build_steps_html() -> str:
    """Üç adımı çerçeveli yuvarlak kartlar olarak çizer (numara yuvarlağıyla)."""
    cards = []
    for kicker, title, text, _tag in HOW_STEPS:
        num = kicker.split()[-1]
        cards.append(f'<div class="lp-bcard"><span class="num">{_h(num)}</span>'
                     f'<div class="t">{_h(tr_upper(title))}</div><p>{_h(text)}</p></div>')
    return f'<div class="lp-steps">{"".join(cards)}</div>'


def build_offer_html(demo: bool = False) -> str:
    """Sayfa sonu çağrı bölümü: ortalanmış ince başlık, kısa metin ve arkasında hap biçimli cam şekil.

    Args:
        demo: Demo hesaplar giriş ekranında gösteriliyorsa ``True``.
    """
    text = ("Giriş ekranındaki demo hesaplarla sigortacı ve müşteri ekranlarını kendiniz gezin." if demo
            else "Hesabınız varsa giriş yapın; yoksa bize yazın.")
    return (f'<div class="lp-offer">{build_glass("pill", "g-offer")}'
            f'{_h2("UYGULAMAYI DENEYİN", "center")}<p class="lp-lead center">{_h(text)}</p></div>')


def build_page_header(page: object) -> str:
    """Sayfa başlığı: ana sayfada kahraman, diğer sayfalarda büyük ince başlık.

    Args:
        page: Sayfa kimliği (bilinmiyorsa ana sayfa kahramanı döner).

    Returns:
        HTML dizgesi.
    """
    pid = valid_page(page)
    if pid == DEFAULT_PAGE:
        return build_home_hero()
    eyebrow, bold, soft, sub = _HEADERS[pid]
    return (f'<div class="hero lp-hero"><div><div class="hero-eyebrow">{_h(eyebrow)}</div>'
            f'<div class="hero-title">{_h(bold)} <span>{_h(soft)}</span></div>'
            f'<div class="hero-sub">{_h(sub)}</div></div></div>')


def build_pillars() -> str:
    """Değer önermeleri şeridi (ana sayfa)."""
    items = "".join(f'<div class="lp-pillar"><b>{_h(t)}</b><span>{_h(s)}</span></div>' for t, s in PILLARS)
    return f'<div class="lp-pillars">{items}</div>'


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


def build_services_html() -> str:
    """Hizmetlerin ayrıntılı bölümleri: her alan için kimin için, hangi sorun, ne sunuyoruz ve hangi aşamada.

    Returns:
        HTML dizgesi (tek satır).
    """
    blocks = []
    for sid, kicker, title, who, problem, items, status in SERVICES:
        bullets = "".join(f"<li>{_h(i)}</li>" for i in items)
        core = " core" if kicker.startswith("ÇEKİRDEK") else ""
        blocks.append(
            f'<section class="lp-svc{core}" id="{_h(sid)}"><div class="lp-svc-head"><div class="n">{_h(kicker)}</div>'
            f'<div class="t">{_h(title)}</div><span class="tag">{_h(status)}</span></div>'
            f'<div class="lp-svc-body"><div><div class="lp-k">KİMLER İÇİN</div><p>{_h(who)}</p>'
            f'<div class="lp-k">NEDEN</div><p>{_h(problem)}</p></div>'
            f'<div><div class="lp-k">NE SUNUYORUZ</div><ul class="lp-list">{bullets}</ul></div></div></section>')
    return "".join(blocks)


def build_roles_html() -> str:
    """Sigortacı ve müşteri rollerinin ne gördüğünü anlatan iki sütun."""
    cols = []
    for kicker, title, items in ROLES:
        bullets = "".join(f"<li>{_h(i)}</li>" for i in items)
        cols.append(f'<div class="step"><div class="n">{_h(kicker)}</div><div class="t">{_h(title)}</div>'
                    f'<ul class="lp-list">{bullets}</ul></div>')
    return f'<div class="flow c2">{"".join(cols)}</div>'


def build_pipeline_html() -> str:
    """Uydu görüntüsünden kanıt belgesine akış çizgisi."""
    parts = []
    for i, name in enumerate(PIPELINE):
        parts.append(f'<span class="lp-node"><i>{i + 1:02d}</i>{_h(name)}</span>')
    return f'<div class="lp-pipeline">{"<span class=\"lp-arrow\">→</span>".join(parts)}</div>'


def build_faq_answer(text: str) -> str:
    """Bir SSS cevabının HTML'i."""
    return f'<p class="lp-faq">{_h(text)}</p>'


def build_status_html(demo: bool = False) -> str:
    """Prototip durumunu açıkça söyleyen not kutusu.

    Args:
        demo: Giriş ekranında demo hesapları gösteriliyorsa ``True`` (o zaman denemeye davet eklenir).
    """
    return f'<div class="note info">{STATUS_TEXT}{DEMO_TEXT if demo else ""}</div>'


def build_cta_band(demo: bool = False) -> str:
    """Sayfa sonundaki çağrı bandı (düğmeler uygulamada Streamlit düğmesi olarak çizilir).

    Args:
        demo: Demo hesaplar giriş ekranında gösteriliyorsa ``True``.
    """
    text = ("Giriş ekranındaki demo hesaplarla sigortacı ve müşteri ekranlarını kendiniz gezin." if demo
            else "Hesabınız varsa giriş yapın; yoksa bize yazın.")
    return ('<div class="lp-band"><div><div class="t">UYGULAMAYI DENEYİN</div>'
            f"<p>{_h(text)}</p></div></div>")


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


def build_contact_hints() -> str:
    """Yazarken belirtilmesi istenenler ve gizlilik notu."""
    hints = ("Ad soyad ve kurumunuz", "İlgilendiğiniz alan: sigorta, tarım, belediye ya da afet",
             "Ne yapmak istediğiniz: demo, pilot ya da yalnızca bilgi")
    items = "".join(f"<li>{_h(h)}</li>" for h in hints)
    return ('<div class="step"><div class="n">YAZARKEN</div><div class="t">Şunları belirtmeniz yeterli</div>'
            f'<ul class="lp-list">{items}</ul>'
            "<p>Bu site e-posta adresinizi ya da başka bir bilginizi toplamaz; düğme yalnızca e-posta uygulamanızı açar.</p></div>")


LANDING_BG_CSS = """
.stApp{ background:#000 !important; }
.block-container{ max-width:1240px !important; padding-top:1rem !important; }
"""

LANDING_CSS = """
.flow.c2{ grid-template-columns:repeat(2,minmax(0,1fr)); }
.flow.c4{ grid-template-columns:repeat(4,minmax(0,1fr)); }
@media (max-width:1280px){ .flow.c4{ grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width:820px){ .flow.c2, .flow.c4{ grid-template-columns:1fr; } }
/* ---- menü çubuğu (düz, siyah; yalnızca giriş düğmesi beyaz hap) ---- */
.st-key-lp_nav{ position:sticky; top:0; z-index:60; padding:12px 2px; margin:0 0 4px;
  background:rgba(0,0,0,.72); -webkit-backdrop-filter:blur(14px); backdrop-filter:blur(14px); }
.st-key-lp_nav [data-testid="stHorizontalBlock"]{ align-items:center; gap:.2rem; }
.lp-logo{ font:600 1.1rem 'Inter',sans-serif; letter-spacing:-.01em; color:#fff; white-space:nowrap; }
.lp-logo i{ font-style:normal; color:var(--red); }
@media (max-width:640px){
  .st-key-lp_nav{ position:static; }
  .st-key-lp_nav [data-testid="stHorizontalBlock"]{ flex-wrap:wrap; }
  .st-key-lp_nav [data-testid="stColumn"]{ flex:1 1 auto !important; width:auto !important; min-width:0 !important; }
  .st-key-lp_nav [data-testid="stColumn"]:first-child{ flex-basis:100% !important; }
}
.st-key-lp_nav .stButton button{ min-height:0 !important; padding:.3rem .5rem !important; background:transparent !important;
  border:none !important; box-shadow:none !important; }
.st-key-lp_nav .stButton button p{ font:400 .78rem 'Inter',sans-serif !important; letter-spacing:.02em; color:rgba(255,255,255,.62) !important;
  white-space:nowrap; }
.st-key-lp_nav .stButton button:hover p{ color:#fff !important; }
[class*="st-key-navon_"] button p{ color:#fff !important; font-weight:500 !important; }
/* ---- düğmeler: beyaz hap (birincil) ve çerçeveli hap (ikincil) ---- */
.st-key-lp_nav .st-key-lp_login button, [class*="st-key-lp_pri_"] button{ min-height:0 !important; padding:.5rem 1.25rem !important;
  background:#fff !important; border:1px solid #fff !important; border-radius:999px !important; box-shadow:none !important; }
.st-key-lp_nav .st-key-lp_login button p, [class*="st-key-lp_pri_"] button p{ color:#000 !important; font:500 .76rem 'Inter',sans-serif !important;
  letter-spacing:.02em; white-space:nowrap; }
.st-key-lp_nav .st-key-lp_login button:hover, [class*="st-key-lp_pri_"] button:hover{ background:#e4e6ea !important; }
[class*="st-key-lp_sec_"] button{ min-height:0 !important; padding:.5rem 1.25rem !important; background:transparent !important;
  border:1px solid rgba(255,255,255,.5) !important; border-radius:999px !important; box-shadow:none !important; }
[class*="st-key-lp_sec_"] button p{ color:#fff !important; font:500 .76rem 'Inter',sans-serif !important; letter-spacing:.02em;
  white-space:nowrap; }
[class*="st-key-lp_sec_"] button:hover{ background:rgba(255,255,255,.09) !important; border-color:#fff !important; }
.st-key-lp_right_btn, .st-key-lp_right_btn_how{ display:flex; justify-content:flex-end; padding-top:clamp(40px,8vw,104px); }
.st-key-lp_center_btns .stButton{ display:flex; justify-content:center; }
.st-key-lp_center_btns{ position:relative; z-index:4; }
/* ---- cam 3B şekiller (yalnızca CSS; süs) ---- */
.lp-glass{ position:absolute; z-index:3; display:block; pointer-events:none; width:var(--s,150px); height:var(--s,150px);
  border-radius:30%; transform:rotate(var(--r,-16deg));
  background:radial-gradient(90% 90% at 28% 16%, rgba(255,255,255,.34), rgba(255,255,255,0) 46%),
    radial-gradient(120% 120% at 74% 88%, rgba(120,150,255,.36), rgba(120,150,255,0) 56%),
    linear-gradient(145deg, rgba(44,48,60,.55), rgba(6,7,10,.80));
  box-shadow:inset 0 0 0 1.5px rgba(255,255,255,.22), inset 10px 12px 26px rgba(255,196,140,.30),
    inset -12px -14px 28px rgba(112,140,255,.30), 0 34px 70px rgba(0,0,0,.65);
  -webkit-backdrop-filter:blur(5px) saturate(1.5); backdrop-filter:blur(5px) saturate(1.5);
  animation:lpfloat 9s ease-in-out infinite; }
.lp-glass::before{ content:""; position:absolute; inset:0; border-radius:inherit; padding:3px;
  background:conic-gradient(from 210deg, #ffb36b, #ff5f8f, #8a7bff, #61d8ff, #ffe08a, #ffb36b);
  -webkit-mask:linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0); -webkit-mask-composite:xor;
  mask:linear-gradient(#000 0 0) content-box exclude, linear-gradient(#000 0 0); opacity:.8; }
.lp-glass::after{ content:""; position:absolute; inset:15%; border-radius:26%;
  background:linear-gradient(160deg, rgba(0,0,0,.74), rgba(0,0,0,.34)); box-shadow:inset 0 0 0 1px rgba(255,255,255,.10); }
.lp-glass.pill{ width:var(--s,240px); height:calc(var(--s,240px) * .5); border-radius:999px; }
.lp-glass.pill::after{ border-radius:999px; inset:18% 9%; }
@keyframes lpfloat{ 0%,100%{ transform:translate3d(0,0,0) rotate(var(--r,-16deg)); }
  50%{ transform:translate3d(0,-12px,0) rotate(calc(var(--r,-16deg) + 3deg)); } }
@media (prefers-reduced-motion:reduce){ .lp-glass{ animation:none; } }
/* ---- kahraman ---- */
.lp-hero2{ position:relative; padding:clamp(26px,6vw,70px) 0 clamp(10px,2vw,26px); }
.lp-eyebrow{ font:500 .7rem 'Inter',sans-serif; letter-spacing:.22em; color:var(--muted); margin-bottom:clamp(14px,2.5vw,30px); }
.lp-display{ position:relative; z-index:2; font:200 clamp(2.6rem,9vw,8rem)/.98 'Inter',sans-serif; letter-spacing:-.035em; color:#fff; }
.lp-display span{ display:block; white-space:nowrap; }
.lp-display .l2{ margin-left:1.05em; } .lp-display .l3{ margin-left:.25em; } .lp-display .l4{ margin-left:2.1em; }
.g-hero{ --s:clamp(120px,18vw,250px); --r:-14deg; top:clamp(6px,3vw,44px); left:48%; }
.lp-copy{ max-width:440px; padding:clamp(4px,1vw,14px) 0 6px; }
.lp-copy p{ margin:0 0 14px; font:300 .92rem/1.65 'Inter',sans-serif; color:rgba(255,255,255,.78); }
@media (max-width:820px){
  .g-hero{ --s:104px; left:auto; right:2%; top:0; }
  .st-key-lp_right_btn, .st-key-lp_right_btn_how{ padding-top:0; justify-content:flex-start; }
}
/* ---- bölümler ---- */
.lp-sect{ margin-top:clamp(56px,9vw,112px); }
.lp-h2{ font:200 clamp(2.1rem,5vw,3.6rem)/1.02 'Inter',sans-serif; letter-spacing:-.025em; color:#fff; }
.lp-h2.center{ text-align:center; margin:0 auto; }
.lp-lead{ margin:18px 0 18px; max-width:520px; font:300 .95rem/1.7 'Inter',sans-serif; color:rgba(255,255,255,.76); }
.lp-lead.center{ text-align:center; margin:clamp(52px,7vw,86px) auto 18px; }
.lp-stats-wrap{ position:relative; overflow:hidden; margin-top:clamp(56px,9vw,112px); padding:8px 0 46px; min-height:190px; }
.lp-stats{ position:relative; z-index:2; display:flex; flex-wrap:wrap; justify-content:flex-end; gap:clamp(18px,4vw,54px); }
.lp-stat b{ display:block; font:300 clamp(2rem,3.6vw,2.8rem)/1 'Inter',sans-serif; letter-spacing:-.02em; color:#fff; }
.lp-stat span{ display:block; margin-top:8px; font:400 .74rem 'Inter',sans-serif; letter-spacing:.12em; color:rgba(255,255,255,.62); }
.lp-stat small{ display:block; margin-top:4px; font:300 .72rem 'Inter',sans-serif; color:rgba(255,255,255,.42); }
.lp-note{ position:relative; z-index:2; margin:22px 0 0; text-align:left; max-width:340px; font:300 .72rem 'Inter',sans-serif; color:rgba(255,255,255,.4); }
.g-about{ --s:clamp(64px,7vw,96px); --r:12deg; right:26%; bottom:-2px; }
.g-edge{ --s:clamp(100px,12vw,170px); --r:-24deg; right:-4%; bottom:-30%; }
@media (max-width:820px){ .lp-stats{ justify-content:flex-start; } }
/* ---- bento kartlar ---- */
.lp-bento{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin-top:26px; }
.lp-bcard{ position:relative; display:flex; flex-direction:column; justify-content:flex-end; min-height:230px; padding:22px;
  border:1px solid rgba(255,255,255,.3); border-radius:24px; background:transparent; }
.lp-bcard .num{ position:absolute; top:18px; left:18px; width:26px; height:26px; border-radius:50%; background:#fff; color:#000;
  display:grid; place-items:center; font:600 .64rem 'Inter',sans-serif; }
.lp-bcard .st{ position:absolute; top:22px; right:22px; font:500 .6rem 'Inter',sans-serif; letter-spacing:.16em; color:rgba(255,255,255,.5); }
.lp-bcard .t{ font:500 .8rem/1.35 'Inter',sans-serif; letter-spacing:.05em; color:#fff; }
.lp-bcard p{ margin:8px 0 0; font:300 .82rem/1.55 'Inter',sans-serif; color:rgba(255,255,255,.72); }
.lp-bcard.white{ background:#fff; border-color:#fff; }
.lp-bcard.white .num{ background:#000; color:#fff; }
.lp-bcard.white .st{ color:rgba(0,0,0,.5); }
.lp-bcard.white .t{ color:#000; }
.lp-bcard.white p{ color:rgba(0,0,0,.72); }
.lp-bcard.b1{ grid-column:1; grid-row:1; } .lp-bcard.b2{ grid-column:2 / span 2; grid-row:1; }
.lp-bcard.b3{ grid-column:1; grid-row:2; } .lp-bcard.b4{ grid-column:2; grid-row:2; } .lp-bcard.bg{ grid-column:3; grid-row:2; overflow:hidden; }
.g-bento{ --s:clamp(150px,17vw,240px); --r:-32deg; left:12%; top:30%; }
@media (max-width:820px){
  .lp-bento{ grid-template-columns:1fr; }
  .lp-bcard.b1, .lp-bcard.b2, .lp-bcard.b3, .lp-bcard.b4, .lp-bcard.bg{ grid-column:1; grid-row:auto; }
  .lp-bcard.bg{ min-height:160px; }
}
.lp-steps{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin-top:26px; }
.lp-steps .lp-bcard{ min-height:200px; }
@media (max-width:820px){ .lp-steps{ grid-template-columns:1fr; } }
/* ---- çağrı bölümü ---- */
.lp-offer{ position:relative; padding:clamp(54px,9vw,120px) 0 10px; }
.lp-offer .lp-h2, .lp-offer .lp-lead{ position:relative; z-index:2; }
.g-offer{ --s:clamp(240px,50vw,640px); --r:-8deg; left:50%; margin-left:calc(var(--s) * -.5);
  top:calc(clamp(54px,9vw,120px) + 1.8rem - var(--s) * .17); z-index:1; opacity:.92; }
.g-offer.pill{ height:calc(var(--s) * .34); }
.note.info{ background:transparent; border:1px solid rgba(255,255,255,.22); border-radius:22px; margin-top:26px; }
/* ---- diğer sayfaların başlığı ---- */
.lp-hero{ border-bottom:none; padding:clamp(24px,5vw,56px) 0 6px; }
.lp-hero .hero-title{ font:200 clamp(2.2rem,6.4vw,5rem)/1.02 'Inter',sans-serif; letter-spacing:-.03em; }
.lp-hero .hero-title span{ color:rgba(255,255,255,.4); }
/* ---- hizmetler ---- */
.lp-svc{ margin:14px 0; padding:20px 22px; border-radius:22px; border:1px solid rgba(255,255,255,.22); background:transparent; }
.lp-svc.core{ border-color:rgba(255,255,255,.5); }
.lp-svc-head{ display:flex; flex-wrap:wrap; align-items:baseline; gap:6px 14px; margin-bottom:12px; }
.lp-svc-head .n{ font:600 .66rem 'Inter',sans-serif; letter-spacing:.16em; color:var(--red); }
.lp-svc-head .t{ font:500 1.15rem 'Inter',sans-serif; color:#fff; flex:1 1 280px; }
.lp-svc-head .tag{ display:inline-block; padding:2px 10px; border-radius:999px; border:1px solid var(--line2,rgba(255,255,255,.17));
  font:500 .62rem 'Inter',sans-serif; letter-spacing:.1em; color:var(--muted); white-space:nowrap; }
.lp-svc.core .lp-svc-head .tag{ color:#fff; border-color:rgba(255,255,255,.5); }
.lp-svc-body{ display:grid; grid-template-columns:minmax(0,.9fr) minmax(0,1.1fr); gap:24px; }
@media (max-width:820px){ .lp-svc-body{ grid-template-columns:1fr; } }
.lp-svc-body p{ margin:0 0 12px; color:var(--soft); font:300 .92rem/1.6 'Inter',sans-serif; }
.lp-k{ font:600 .64rem 'Inter',sans-serif; letter-spacing:.16em; color:var(--muted); margin:2px 0 4px; }
.lp-list{ margin:0 0 6px; padding-left:18px; color:var(--soft); font:300 .9rem/1.6 'Inter',sans-serif; }
.lp-list li{ margin:3px 0; }
.step .lp-list{ margin-top:8px; }
/* ---- akış ve SSS ---- */
.lp-pipeline{ display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:6px 0 4px; }
.lp-node{ padding:9px 14px; border-radius:999px; border:1px solid rgba(255,255,255,.3); background:transparent; color:var(--soft);
  font:400 .82rem 'Inter',sans-serif; white-space:nowrap; }
.lp-node i{ font-style:normal; color:var(--red); font-weight:700; margin-right:8px; font-size:.7rem; letter-spacing:.1em; }
.lp-arrow{ color:var(--dim); }
.lp-faq{ margin:0; color:var(--soft); font:300 .93rem/1.65 'Inter',sans-serif; }
/* ---- eski çağrı bandı ve iletişim ---- */
.lp-band{ margin-top:26px; padding:20px 24px; border-radius:22px; border:1px solid rgba(255,255,255,.3); }
.lp-band .t{ font:600 .72rem 'Inter',sans-serif; letter-spacing:.16em; color:#fff; }
.lp-band p{ margin:6px 0 0; color:var(--soft); font:300 .95rem/1.55 'Inter',sans-serif; }
.lp-pillars{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:18px 0 6px; }
@media (max-width:1000px){ .lp-pillars{ grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width:560px){ .lp-pillars{ grid-template-columns:1fr; } }
.lp-pillar{ padding:14px 16px; border-radius:18px; border:1px solid rgba(255,255,255,.22); }
.lp-pillar b{ display:block; font:600 .68rem 'Inter',sans-serif; letter-spacing:.14em; color:var(--red); margin-bottom:6px; }
.lp-pillar span{ color:var(--soft); font:300 .88rem/1.5 'Inter',sans-serif; }
.lp-contact{ margin-top:clamp(40px,7vw,90px); padding:26px 28px; border-radius:24px; border:1px solid rgba(255,255,255,.3); }
.lp-contact p{ margin:0 0 14px; color:rgba(255,255,255,.7); font:300 .95rem/1.6 'Inter',sans-serif; max-width:720px; }
.lp-actions{ display:flex; flex-wrap:wrap; align-items:center; gap:16px; }
.lp-cta{ display:inline-block; padding:10px 22px; border-radius:999px; background:#fff; color:#000 !important;
  font:500 .76rem 'Inter',sans-serif; letter-spacing:.04em; text-decoration:none !important; }
.lp-cta:hover{ background:#e4e6ea; }
.lp-mail{ color:var(--soft); font:400 .82rem ui-monospace,Menlo,Consolas,monospace; }
"""
