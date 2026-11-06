# ORBITAL FIELD — No-Code Çok Yönlü Uydu ve Saha Analiz Platformu

Parsel bazlı uydu analizini kod yazmadan sigorta, tarım, belediye ve afet ekiplerinin karar sürecine taşıyan
Streamlit prototipi. TÜBİTAK BİGG başvurusu için hazırlandı. **Tüm veriler sentetiktir** (gerçek bir hasar dosyasıyla karşılaştırılmadı).

Geliştirici: Elif Turut

## Demo girişleri

| Kullanıcı | Parola | Şirket / unvan | Gördükleri |
|---|---|---|---|
| `sigortaci` | `Sigortaci-Demo-2026` | Boztepe Sigorta (demo) · Hasar yöneticisi | Altı sekme; yalnızca Boztepe poliçeleri |
| `sigortaci2` | `Sigortaci2-Demo-2026` | Yaylacık Sigorta (demo) · Ekspertiz uzmanı | Altı sekme; yalnızca Yaylacık poliçeleri |
| `musteri1` · `musteri2` · `musteri3` | `Musteri1-Demo-2026` · `Musteri2-Demo-2026` · `Musteri3-Demo-2026` | — | Tüm şirketlerdeki kendi poliçeleri, parselleri ve verimlilik takibi |

Şirket adları **kurgusaldır** ("(demo)" eki bunu belirtir); gerçek bir kurumla ilgisi yoktur. Poliçelerin şirketlere dağılımı sentetiktir.

## Ekranlar

- **Web sitesi (giriş yapmadan):** üstte menü çubuğu — Ana Sayfa · Hakkımızda · Hizmetlerimiz · Nasıl çalışır · İletişim · **Giriş yap**.
  Ana sayfa siyah zeminli, ince ve büyük yazılı bir ajans tarzındadır: kademeli dev başlık ve yalnızca CSS ile çizilen cam uzay nesneleri — uydu (gövde, güneş panelleri, anten), gezegen, yörünge (görsel dosya yok),
  hakkında + ürün kapsamı sayıları (4 alan, 3 uydu indeksi, 4 karar türü; müşteri ya da kullanım sayısı değildir), numaralı hizmet ızgarası, adımlar ve çağrı bölümü.
  Hizmetlerimiz dört alanı ayrı ayrı ve hangi aşamada olduklarıyla anlatır; Nasıl çalışır akış çizgisi, beş adım ve sık sorulan soruları içerir.
  Sayfalar adres çubuğundan da açılır (`?sayfa=hakkimizda`, `hizmetler`, `nasil`, `iletisim`). Site hesap ya da veri içermez; durumunu
  (prototip, sentetik veri) açıkça yazar. Yapılmamış iş, rakam, müşteri ya da referans iddiası yoktur.
- **Sigortacı** (kendi şirketinin poliçeleri) — altı sekme:
  1. **Genel Bakış**
  2. **Sigorta:** karar tablosu, harita, kanıt belgesi. Otomatik öneri ile geçerli karar ayrı gösterilir.
  3. **Müşteri Dosyaları:** istediği müşteriyi açar; *Dosya* (poliçeler, harita, iç değerlendirme, kanıt belgesi) · *Karar ver* (onay, kısmi onay,
     ekspertiz, ret; müşteriye görünecek not ve zorunlu iç gerekçe; otomatik karara dönme) · *Müşterinin gördüğü* · *Geçmiş*.
  4. **Kalibrasyon**
  5. **Diğer Sektörler** (Tarım, Belediye, Kurtarma)
  6. **Yönetim:** *Profil* (ad, unvan, şirket, yetkiler) · *Şirket* (özet ve karar bekleyen dosyalar) · *Ekip* · *Karar kaydı* (CSV indirme) · *Sistem*.
  Karar kuralları (eşikler, muafiyet, teminat) sol menüdedir. Senaryo tohumu "Gelişmiş" altındadır; karar yalnızca varsayılan tohumda verilir.
- **Müşteri:** tek sayfa, dört sekme — Özet · Hasar durumum · Verimlilik · Belgeler. Poliçeler farklı şirketlerde olabilir; her biri şirket adıyla
  görünür. Şirketin kararı, tutarı ve müşteriye yazdığı not görünür; iç gerekçe ve kararı veren kullanıcı **hiçbir yerde** görünmez.

### Çok şirketli model ve kararlar

- Şirket bilgisi müşteriye değil **poliçeye** bağlıdır (`sirket` sütunu). Sigortacı yalnızca kendi şirketinin poliçelerini görür ve karar verir;
  başka şirketin poliçesine yazma denemesi reddedilir. Şirket, kullanıcı ve zaman damgası sunucuda oturumdan alınır.
- Otomatik motor bir **öneri** üretir; sigortacı kararı bunun üzerine yazılır ve ayrıca saklanır. Tutar aynı formülle hesaplanır
  (bedel × teminat × hasar oranı − muafiyet); elle tutar yalnızca onay ve kısmi onayda, 0 ile sigorta bedeli arasında girilebilir.
- Karar defteri **eklemelidir** (SQLite): kararı değiştirmek ya da geri almak eskisini silmez, yeni kayıt ekler. Geri alma için gerekçe zorunludur.
- Karar defteri dosyası varsayılan olarak `orbital_data/orbital.sqlite3`'tür; `ORBITAL_DB_PATH` ile başka yol verilebilir.
  **Streamlit Community Cloud'da dosya sistemi geçicidir:** uygulama yeniden başlayınca kayıtlar silinir. Gerçek kullanım için kalıcı bir veritabanı gerekir.

## Bilgisayarında çalıştırma

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## İnternette yayınlama (Streamlit Community Cloud)

1. Bu klasördeki dosyaları bir GitHub deposuna yükleyin (github.com → New repository → "uploading an existing file").
2. share.streamlit.io adresine GitHub hesabınızla girin → **Create app** → **Yup, I have an app**.
3. Depoyu seçin, **Main file path** olarak `app.py` yazın, isterseniz **App URL** alanına bir ad verin.
4. **Advanced settings** içinde Python sürümü olarak 3.12 ya da 3.13 seçin → **Deploy**.

Birkaç dakika sonra `https://<seçtiğiniz-ad>.streamlit.app` adresinden açılır; bu adres telefon ve bilgisayardan girilebilir,
tarayıcıda "Ana ekrana ekle" ile uygulama gibi de kullanılır.

### Herkese açık yayında hesaplar

Demo hesaplarının parolası bu belgede ve kaynak kodda yazar; adresi bilen herkes sigortacı olarak girebilir. Yalnızca demo için bunda sakınca yoktur
(veri sentetiktir). Gerçek müşteri ya da sigortacıyla paylaşmadan önce kendi hesaplarınızı tanımlayın:

1. `python auth.py` komutuyla her kullanıcı için bir kayıt üretin (parola düz metin olarak saklanmaz).
2. Kayıtları köşeli parantezli bir JSON listesine koyun.
3. Streamlit Cloud'da uygulamanın **Settings → Secrets** alanına şunu yapıştırın:

```toml
ORBITAL_ACCOUNTS_JSON = '''[
  {"username": "sirket", "role": "sigortaci", "display_name": "Şirket Adı", "salt": "...", "hash": "..."}
]'''
```

Örnekteki kayıt kısaltılmıştır; `python auth.py`'nin ürettiği kaydı aynen yapıştırın. Bu ayar verilince demo hesapları kapanır. Ayar bozuksa giriş kapalı kalır (demo hesaplarına geri dönülmez).

### Karşılama sayfasındaki iletişim adresi

"Bize ulaşın · Hizmet alın" düğmesi, yalnızca bir iletişim adresi tanımlıysa görünür (adres koda yazılı değildir). Streamlit Cloud'da
**Settings → Secrets** alanına ekleyin:

```toml
ORBITAL_CONTACT_EMAIL = "adresiniz@ornek.com"
```

Düğme, konusu ve gövdesi hazır bir `mailto:` bağlantısıdır (e-posta uygulamasını açar; sayfada form ya da sunucu tarafı gönderim yoktur).
Adres geçersizse ya da boşsa düğme çizilmez ve "iletişim adresi yakında yayımlanacak" yazar. Ortam değişkeni olarak da verilebilir; o zaman önceliklidir.

## Dosyalar

| Dosya | Görev |
|---|---|
| `app.py` | Arayüz (Streamlit) |
| `data_engine.py` | Sentetik veri, indeksler, karar motoru |
| `data_sources.py` | Gerçek veri adaptörleri (Sentinel-2, NASA FIRMS; canlı denenmedi) |
| `basemaps.py` | Canlı uydu harita altlıkları |
| `calibration.py` | Geriye dönük test ve eşik kalibrasyonu |
| `reports.py` | Karar kanıt paketi |
| `auth.py`, `portal.py` | Giriş, roller (sigortacı şirketi ve unvanı dahil), müşteri portalı |
| `tenancy.py` | Kurgusal demo şirketleri ve poliçe → şirket eşlemesi |
| `casework.py` | Sigortacı kararı: doğrulama, tutar, otomatik öneriyle birleştirme |
| `store.py` | Eklemeli karar defteri (SQLite) |
| `landing.py` | Giriş öncesi web sitesinin içeriği ve menüsü |
| `orbital_data/` | Çalışırken oluşur (karar defteri); depoya yüklemeyin (`.gitignore` içinde) |
| `test_*.py` | Otomatik kontroller (`python test_auth.py` gibi) |

## Sınırlar

- Gerçek Streamlit oturum davranışı, harita/Plotly çizimi ve canlı veri uç noktaları geliştirme ortamında denenmedi; arayüz sahte (stub) bir
  Streamlit ile ve önizleme görüntüleriyle sınandı. Menü çubuğu, beyaz/çerçeveli hap düğmeler ve cam şekiller `st-key-…` seçicilerine ve `backdrop-filter`/`mask-composite` desteğine dayanır; gerçek Streamlit'te ve hedef tarayıcılarda ilk yayında gözle kontrol edilmelidir.
- Giriş demo düzeyindedir: SSO, iki adımlı doğrulama, parola değiştirme, istek hızı sınırı ve bağımsız güvenlik denetimi yoktur.
- Şirketler ve poliçe dağılımı sentetiktir. Şirket bazlı kural yayını, bildirimler ve kullanıcı ekleme/çıkarma ekranı yoktur (ekip, yapılandırmadan okunur).
- Kanıt belgelerindeki bağlam çiziminde komşu parsellerin hasar renkleri başka şirketlerinki olabilir (yalnızca renk; poliçe ya da kişi bilgisi değil).
- Uygulama kendi parsel dosyanızı yüklemeyi henüz desteklemez.
