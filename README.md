# ORBITAL FIELD — No-Code Çok Yönlü Uydu ve Saha Analiz Platformu

Parsel bazlı uydu analizini kod yazmadan sigorta, tarım, belediye ve afet ekiplerinin karar sürecine taşıyan
Streamlit prototipi. TÜBİTAK BİGG başvurusu için hazırlandı. **Tüm veriler sentetiktir** (gerçek bir hasar dosyasıyla karşılaştırılmadı).

Geliştirici: Elif Turut

## Demo girişleri

| Kullanıcı | Parola | Gördükleri |
|---|---|---|
| `sigortaci` | `Sigortaci-Demo-2026` | Tüm sekmeler, müşteri portalı önizlemesi |
| `musteri1` · `musteri2` · `musteri3` | `Musteri1-Demo-2026` · `Musteri2-Demo-2026` · `Musteri3-Demo-2026` | Yalnızca kendi poliçe, parsel ve verimlilik takibi |

## Ekranlar

- **Karşılama sayfası (giriş yapmadan):** biz neyiz, amaç, dört hizmet alanı, nasıl çalışır, neler sunuyoruz, güncel durum (prototip, sentetik veri)
  ve "Hizmet alın" bölümü; sağ üstteki **Giriş yap** düğmesi giriş formunu açar, formda **← Ana sayfa** ile geri dönülür.
  Sayfa hesap ya da veri içermez.
- **Sigortacı:** beş ana sekme — Genel Bakış · Sigorta · Müşteri Portalı · Kalibrasyon · Diğer Sektörler (Tarım, Belediye, Kurtarma alt sekmelerde).
  Karar kuralları (eşikler, muafiyet, teminat) sol menüdedir; Sigorta, Müşteri Portalı önizlemesi ve Kalibrasyon aynı kuralları kullanır.
  Senaryo tohumu sol menüde "Gelişmiş" altındadır.
- **Müşteri:** tek sayfa, dört sekme — Özet (harita ve poliçe tablosu) · Hasar durumum · Verimlilik · Belgeler.

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
| `auth.py`, `portal.py` | Giriş, roller, müşteri portalı |
| `landing.py` | Giriş öncesi karşılama sayfasının içeriği |
| `test_*.py` | Otomatik kontroller (`python test_auth.py` gibi) |

## Sınırlar

- Gerçek Streamlit oturum davranışı ve canlı veri uç noktaları geliştirme ortamında denenmedi.
- Giriş demo düzeyindedir: SSO, iki adımlı doğrulama ve istek hızı sınırı yoktur.
