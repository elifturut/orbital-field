# NYXE · ORBITAL FIELD — No-Code Çok Yönlü Uydu ve Saha Analiz Platformu

**Nyxe** ana marka (şirket adı), **Orbital Field** ürün adıdır; arayüzde Orbital Field, Nyxe'nin alt başlığı olarak görünür.

Parsel bazlı uydu analizini kod yazmadan sigorta, tarım, belediye ve afet ekiplerinin karar sürecine taşıyan
Streamlit prototipi. TÜBİTAK BİGG başvurusu için hazırlandı. **Varsayılan veriler sentetiktir** (gerçek bir hasar dosyasıyla karşılaştırılmadı);
Analiz iş akışında isteğe bağlı olarak canlı Sentinel-2 / NASA FIRMS / Open-Meteo kaynakları seçilebilir (aşağıya bakın; canlı uç noktalara karşı henüz doğrulanmadı).

Geliştirici: Elif Turut

## Demo girişleri

| Kullanıcı | Parola | Şirket / unvan | Gördükleri |
|---|---|---|---|
| `sigortaci` | `Sigortaci-Demo-2026` | Boztepe Sigorta (demo) · Hasar yöneticisi | Yedi sekme; yalnızca Boztepe poliçeleri |
| `sigortaci2` | `Sigortaci2-Demo-2026` | Yaylacık Sigorta (demo) · Ekspertiz uzmanı | Yedi sekme; yalnızca Yaylacık poliçeleri |
| `musteri1` · `musteri2` · `musteri3` | `Musteri1-Demo-2026` · `Musteri2-Demo-2026` · `Musteri3-Demo-2026` | — | Tüm şirketlerdeki kendi poliçeleri, parselleri ve verimlilik takibi |

Şirket adları **kurgusaldır** ("(demo)" eki bunu belirtir); gerçek bir kurumla ilgisi yoktur. Poliçelerin şirketlere dağılımı sentetiktir.

## Ekranlar

- **Web sitesi (giriş yapmadan):** üstte menü çubuğu — Ana Sayfa · Hakkımızda · Hizmetlerimiz · Nasıl çalışır · İletişim · **Giriş yap**.
  Ana sayfa siyah zeminli, ince ve büyük yazılı bir ajans tarzındadır: kademeli dev başlık ve yalnızca CSS ile çizilen cam uzay nesneleri — uydu (gövde, güneş panelleri, anten), gezegen, yörünge (görsel dosya yok),
  hakkında + ürün kapsamı sayıları (4 alan, 3 uydu indeksi, 4 karar türü; müşteri ya da kullanım sayısı değildir), numaralı hizmet ızgarası, adımlar ve çağrı bölümü.
  Hizmetlerimiz dört alanı ayrı ayrı ve hangi aşamada olduklarıyla anlatır; Nasıl çalışır akış çizgisi, beş adım ve sık sorulan soruları içerir.
  Sayfalar adres çubuğundan da açılır (`?sayfa=hakkimizda`, `hizmetler`, `nasil`, `iletisim`). Site hesap ya da veri içermez; durumunu
  (prototip, sentetik veri) açıkça yazar. Yapılmamış iş, rakam, müşteri ya da referans iddiası yoktur.
- **Sigortacı** (kendi şirketinin poliçeleri) — yedi sekme:
  1. **Genel Bakış**
  2. **Analiz İş Akışı:** altı adımlı uçtan uca analiz (aşağıda).
  3. **Sigorta:** karar tablosu, harita, kanıt belgesi. Otomatik öneri ile geçerli karar ayrı gösterilir.
  4. **Müşteri Dosyaları:** istediği müşteriyi açar; *Dosya* (poliçeler, harita, iç değerlendirme, kanıt belgesi) · *Karar ver* (onay, kısmi onay,
     ekspertiz, ret; müşteriye görünecek not ve zorunlu iç gerekçe; otomatik karara dönme) · *Müşterinin gördüğü* · *Geçmiş*.
  5. **Kalibrasyon**
  6. **Diğer Sektörler** (Tarım, Belediye, Kurtarma)
  7. **Yönetim:** *Profil* (ad, unvan, şirket, yetkiler) · *Şirket* (özet ve karar bekleyen dosyalar) · *Ekip* · *Karar kaydı* (CSV indirme) · *Sistem*.
  Karar kuralları (eşikler, muafiyet, teminat) sol menüdedir. Senaryo tohumu "Gelişmiş" altındadır; karar yalnızca varsayılan tohumda verilir.
- **Müşteri:** tek sayfa, dört sekme — Özet · Hasar durumum · Verimlilik · Belgeler. Poliçeler farklı şirketlerde olabilir; her biri şirket adıyla
  görünür. Şirketin kararı, tutarı ve müşteriye yazdığı not görünür; iç gerekçe ve kararı veren kullanıcı **hiçbir yerde** görünmez.

### Analiz İş Akışı (altı adım)

Sigortacı sekmelerinden biridir; tek sayfada, yukarıdan aşağıya ilerler. Giriş ekranındaki **“Müşteri olarak dene” / “Ekspertiz olarak dene”** düğmeleri
(yalnızca demo hesapları açıkken görünür) parolasız demo girişi yapar; bu sekme **Ekspertiz** girişinde görünür, müşteri girişi müşteri portalını açar.

1. **Bölge veya parsel:** demo senaryo · haritadan çiz (çokgen/dikdörtgen) · merkez koordinat + yarıçap · GeoJSON metni. Girdi doğrulanır (geçerli koordinat, en çok 25 km’lik kutu, GeoJSON en çok 2 MB).
2. **Analiz türü:** sel · yangın · kuraklık · tarımsal gelişim.
3. **Veri kaynağı ve tarih aralığı:** *Demo (sentetik)* ya da *Canlı (Earth Search STAC/COG)*. Bulut sınırı, sigorta bedeli, ızgara çözünürlüğü; tarımda ürün ve geçmiş verim;
   kuraklıkta (varsayımsal) destek ödemesi parametreleri. Canlı seçilirse hata durumunda **sessizce demoya dönülmez**; hata açıkça gösterilir.
   “Canlı bağlantıyı sına” düğmesi STAC, Open-Meteo ve (varsa) FIRMS erişimini tek tek dener.
4. **Görüntüleri işle:** sahne seçimi, bulut/gölge maskesi (SCL), önce/sonra kompoziti, indeksler. Girdiler değişince sonuç “güncel değil” diye işaretlenir.
5. **Sonuçlar:** önce/sonra kaydırıcısı ve değişim haritası, KPI’lar, sahne listesi ve indeks zaman serisi, sınıf tablosu, **güvenilirlik göstergesi** (bulut oranı >%20 ise “Güvenilirlik düşük”),
   hava verisiyle doğrulama (Open-Meteo / ERA5 tabanlı), FIRMS yangın noktaları, sigorta ve tarım görünümü, uyarı simülasyonu (gerçek SMS/e-posta gönderilmez).
   Tarımda ayrıca rekolte tahmini (güven aralığı, hold-out hata ölçüleri). Eşik kanıtı ve saha doğrulama paneli (yanlış pozitif/negatif, ROC) her zaman görünür.
6. **Rapor:** sigorta ya da tarım raporu (PDF): parsel koordinatları, sahneler, görüntüler, tablolar, grafik, güvenilirlik, hava, yöntem ve eşikler, **ekspert kararı alanları ve imza/tarih kutuları**,
   isteğe bağlı saha fotoğrafı + elle girilen GPS. Fotoğrafın EXIF verisi atılır; yalnızca JPEG/PNG, en çok 10 MB.

Ortak motor: aynı sahne→indeks→sınıf hattı sigorta (hasar şiddeti, ödenecek tutar) ve tarım (bitki sağlığı, rekolte) görünümlerini besler.

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

`rasterio` yalnızca canlı COG okumasında, çalışma anında içe aktarılır: kurulamazsa uygulama demo kaynağıyla çalışmaya devam eder, canlı seçenek nedenini açıklayan bir hata verir.
Bu ortamda `rasterio` kurulu değildi; canlı COG okuması hiç çalıştırılmadı. `rasterio`, `reportlab` ve `pillow` ilk yayında kurulup kurulmadığı kontrol edilmelidir.

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

### Canlı veri kaynaklarını doğrulama (yayından sonra)

Canlı adaptörler (Earth Search STAC + rasterio COG okuması, Open-Meteo, NASA FIRMS) geliştirme ortamında yalnızca **sahte yanıtlarla** sınandı; bu ortamda internet çıkışı yoktu.
Yayından sonra şunu yapın: **Analiz İş Akışı → Adım 3 → Canlı → “Canlı bağlantıyı sına”.** Her kaynak için başarılı/başarısız ve süre görürsünüz. Sonuç belirleyicidir; bu belge değil.

- FIRMS için (isteğe bağlı) Streamlit Secrets'a `FIRMS_MAP_KEY = "..."` ekleyin (NASA FIRMS'ten ücretsiz alınır). Anahtar arayüzde ve hata iletilerinde gösterilmez.
- Earth Search ve Open-Meteo anahtar gerektirmez.
- Beklenmedik biçimde dar bir bölgede sahne bulunamazsa ya da bulut oranı yüksekse sonuç “güvenilirlik düşük” diye işaretlenir; bu hata değil, veri kalitesi uyarısıdır.

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
| `data_sources.py` | CDSE Statistics ve NASA FIRMS adaptörleri; veri hattı durum tablosu (canlı denenmedi) |
| `live_sources.py` | Earth Search STAC + COG okuma, Open-Meteo, FIRMS birleşimi, canlı bağlantı tanılaması (sahte yanıtlarla test edildi) |
| `analysis.py` | Ortak analiz motoru: bölge, ızgara, indeksler, sınıflama, sigorta/tarım görünümü, kuraklık destek simülasyonu |
| `demo_imagery.py` | Sentetik sahneler (sel, yangın, kuraklık, tarım) — canlı kaynakla aynı arayüz |
| `workflow.py` | Kaynak → sahne seçimi → önce/sonra → sonuç hattı |
| `workflow_page.py`, `workflow_ui.py` | Analiz İş Akışı sekmesi (adımlar, formlar) ve saf HTML/SVG kartları |
| `workflow_report.py` | PDF rapor (ReportLab; `fonts/` içindeki DejaVu yazı tipleri) |
| `yield_model.py` | Çok değişkenli rekolte tahmini (güven aralığı, hold-out); demo modeli sentetik geçmişle eğitilir |
| `thresholds.py` | Eşik kanıtı, sentetik süpürme ve gerçek saha doğrulama (yanlış pozitif/negatif, ROC) |
| `fakest.py` | Streamlit/Plotly/Folium'un sahte sürümü; testler arayüzü bununla çalıştırır |
| `domst.py`, `layout_check.py`, `test_responsive.py` | Duyarlı düzen denetimi: sayfaları Streamlit'e benzer bir iskelete döküp Chromium'da 320–820 px genişliklerde taşma, çakışma ve iframe taşması ölçer (Playwright yoksa atlanır) |
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
- Toplu parsel dosyası (Shapefile/GeoJSON dosya yükleme) henüz yoktur; tek bölge çizilebilir, koordinatla ya da GeoJSON metniyle verilebilir.

### Telefon ve tablet uyumu

Karşılama sayfaları, giriş, sigortacı sekmeleri, müşteri portalı ve analiz akışı 320–820 px genişliklerde **taklit bir Streamlit iskeletiyle** ölçüldü (`python layout_check.py`, `python test_responsive.py`):
yatay taşma, üst üste binen metin, kırpılan içerik ve iframe taşması yok. Bu gerçek Streamlit ya da gerçek cihaz değildir: Streamlit'in kendi sütun/düğme/sekme/yan menü davranışı belgelerden yaklaşık taklit edildi,
harita ve Plotly grafikleri yer tutucudur. İlk yayından sonra telefonda gözle kontrol edin (özellikle üst çubuk payı, yan menünün telefonda kapalı gelmesi ve harita üstü kartlar).
Tablolar dar ekranda kaydırılabilir kutuya ya da kart görünümüne geçer; harita altındaki Plotly/Streamlit tablolarının kendi kaydırması Streamlit'e aittir.

### Analiz sonuçlarının bilimsel durumu

- **Eşikler:** hasar sınıfları USGS FIREMON’daki dNBR aralıklarına (0,10 / 0,27 / 0,44 / 0,66) hizalıdır; bunlar literatür tabanlı **politika seçimleridir** ve Sentinel-2 ile Türkiye arazisi için bu projede
  **yerel olarak doğrulanmamıştır.** Uygulamadaki doğrulama paneli, sizin saha doğruluk (ground truth) CSV’nizle yanlış pozitif/negatif oranlarını ve ROC’u hesaplar; demo değerleri sentetiktir.
- **Güvenilirlik puanı** bir veri kalitesi sezgisidir (berraklık, karo bulutu, olay tarihine zamanlama, örnek sayısı; ağırlıklar kodda `RELIABILITY_WEIGHTS`); kalibre edilmiş bir olasılık değildir.
- **Rekolte modeli:** demo sürümü sentetik geçmişle eğitilir; gösterilen hata ve güven aralığı bu sentetik veriye aittir. Gerçek kullanımda parsel geçmişini CSV olarak yüklemeniz gerekir; tek başına NDVI rekolte tahmini için yeterli değildir.
- **Hava doğrulaması** ERA5 tabanlı yeniden-analizdir; istasyon ölçümü değildir.
- **Destek ödemesi** ve **uyarı** ekranları simülasyondur (varsayımsal parametreler; gerçek SMS/e-posta gönderilmez).
- Saha fotoğrafındaki GPS, EXIF’ten okunmaz; ekspertiz elle girer.
- Kapsam dışı / ertelenen: Landsat, gerçek zamanlı WMS/WMTS akışı, rekolte geçmişi için otomatik öznitelik çıkarımı, sonuç haritasında parsel bazlı açılır pencereler.

