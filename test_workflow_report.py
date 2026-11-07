"""workflow_report.py testleri: PDF içeriği (pdftotext), girdi doğrulama, fotoğraf güvenliği, işaretleme enjeksiyonu, yazı tipi yedeği."""
import dataclasses
import datetime as dt
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import analysis as an  # noqa: E402
import demo_imagery as di  # noqa: E402
import workflow as wf  # noqa: E402
import workflow_report as wr  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  [{extra}]" if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)


def raises(fn, needle=""):
    try:
        fn()
    except wr.ReportError as e:
        return needle in str(e), str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"yanlış istisna: {type(e).__name__}: {e}"
    return False, "istisna yok"


HAVE_TXT = shutil.which("pdftotext") is not None


def text_of(pdf: bytes) -> str:
    """PDF metni; satır kaydırmaları ve tablo boşlukları tek boşluğa indirilir (cümleler satır sonunda bölünebilir)."""
    if not HAVE_TXT:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf)
    try:
        raw = subprocess.run(["pdftotext", "-layout", f.name, "-"], capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(f.name)
    return re.sub(r"\s+", " ", raw)


def run_for(kind):
    scn = di.preset(kind)
    ws, we = di.default_window(scn)
    req = wf.AnalysisRequest(kind=kind, region=scn.region, start=ws, end=we, event_date=scn.event_date, max_cloud=40.0)
    return wf.run_analysis(di.DemoSource(scn), req)


def jpeg(w=200, h=120, exif=False):
    buf = io.BytesIO()
    img = Image.new("RGB", (w, h), (20, 110, 60))
    if exif:
        ex = Image.Exif()
        ex[0x010F] = "GizliMarka"
        ex[0x8825] = {1: "N", 2: (39.0, 0.0, 0.0)}
        img.save(buf, "JPEG", exif=ex)
    else:
        img.save(buf, "JPEG")
    return buf.getvalue()


WHEN = dt.datetime(2026, 11, 7, 9, 0, tzinfo=dt.timezone.utc)
print("— yardımcılar")
check("haversine: 1° enlem ≈ 111,2 km", abs(wr.haversine_m(0, 0, 1, 0) - 111_195) < 300)
check("haversine: aynı nokta 0", wr.haversine_m(39, 35, 39, 35) == 0.0)
wr._Fonts.setup()
check("Türkçe karakterli yazı tipi gömüldü", wr._Fonts.unicode_ok and wr._Fonts.regular == "DejaVu")

print("— fotoğraf güvenliği")
out = wr.prepare_photo(jpeg(exif=True))
im = Image.open(io.BytesIO(out))
check("fotoğraf JPEG olarak yeniden kodlanır, EXIF/GPS atılır", im.format == "JPEG" and not im.getexif() and b"GizliMarka" not in out)
big = Image.new("RGB", (3000, 2000), (1, 2, 3))
b = io.BytesIO()
big.save(b, "PNG")
check("büyük fotoğraf küçültülür (≤ 1600 px)", max(Image.open(io.BytesIO(wr.prepare_photo(b.getvalue()))).size) <= 1600)
for name, raw, needle in (("boş", b"", "boş"), ("görüntü değil", b"MZ\x90\x00 not an image", "okunamadı"), ("çok büyük bayt", b"x" * (wr.MAX_PHOTO_BYTES + 1), "MB")):
    ok, msg = raises(lambda raw=raw: wr.prepare_photo(raw), needle)
    check(f"fotoğraf reddi: {name}", ok, msg)
gif = io.BytesIO()
Image.new("P", (10, 10)).save(gif, "GIF")
ok, msg = raises(lambda: wr.prepare_photo(gif.getvalue()), "JPEG ve PNG")
check("fotoğraf reddi: GIF (biçim kısıtı)", ok, msg)
huge = io.BytesIO()
Image.new("L", (7000, 7000), 0).save(huge, "PNG")
ok, msg = raises(lambda: wr.prepare_photo(huge.getvalue()), "çözünürlüğü")
check("fotoğraf reddi: sıkıştırma bombası (çok piksel, az bayt)", ok and len(huge.getvalue()) < wr.MAX_PHOTO_BYTES, msg)
trunc = jpeg()[:60]
ok, msg = raises(lambda: wr.prepare_photo(trunc), "okunamadı")
check("fotoğraf reddi: kesik/bozuk dosya", ok, msg)

print("— girdi doğrulama")
yr = run_for("yangin")
for name, meta, needle in (
    ("rapor türü", wr.ReportMeta("kötü"), "Rapor türü"),
    ("fotoğraf enlemi", wr.ReportMeta("sigorta", field_photo=wr.FieldPhoto(jpeg(), 95.0, 35.0)), "konumu"),
    ("fotoğraf boylamı NaN", wr.ReportMeta("sigorta", field_photo=wr.FieldPhoto(jpeg(), 39.0, float("nan"))), "konumu"),
    ("rekolte NaN", wr.ReportMeta("tarim", yield_summary=wr.YieldSummary("Buğday", "x", float("nan"), 1, 2, .9, 4, .3, 6, True)), "sayısal"),
):
    ok, msg = raises(lambda meta=meta: wr.build_pdf(yr, meta), needle)
    check(f"build_pdf reddi: {name}", ok, msg)

print("— sigorta raporu (yangın)")
lat, lon = yr.request.region.centre
photo_in = wr.FieldPhoto(jpeg(exif=True), lat, lon, "Kuzey sınırda yanık gövde", "07.11.2026 10:30")
meta = wr.ReportMeta("sigorta", "PRS-0042", "POL-118", "Ayşe Yılmaz", "Mehmet Kaya", "EKS-3321", "Kısmi onay", "Kuzey kenarda yanık izleri doğrulandı.", photo_in, None, WHEN)
pdf = wr.build_pdf(yr, meta)
txt = text_of(pdf)
check("PDF başlığı ve boyutu makul", pdf[:5] == b"%PDF-" and 20_000 < len(pdf) < 3_000_000, len(pdf))
if HAVE_TXT:
    for needle in ("HASAR DEĞERLENDİRME RAPORU", "Parsel koordinatları", "PRS-0042", "POL-118", "Mehmet Kaya", "EKS-3321", "Kısmi onay", "İmza:", "Tarih:", "WGS84",
                   "SENTETİK DEMO", "ÖNCE", "SONRA", "Güvenilirlik puanı", "Hava durumu doğrulaması", "Yangın noktaları", "Saha fotoğrafı", "çokgenin içinde", "OF-20261107-",
                   "karar belgesi değildir", "yerel olarak doğrulanmamış", "Elif Turut", "₺"):
        check(f"PDF metni: '{needle}'", needle in txt)
    check("PDF: fotoğraf EXIF'i (marka) rapora sızmaz", "GizliMarka" not in txt and b"GizliMarka" not in pdf)
    check("PDF: ASCII dönüşümü uygulanmadı (Türkçe karakterler korunur)", "İ" in txt and "ş" in txt and "ğ" in txt)
photo_out = wr.FieldPhoto(jpeg(), lat + 0.05, lon + 0.05, "uzak", "")
txt2 = text_of(wr.build_pdf(yr, dataclasses.replace(meta, field_photo=photo_out)))
if HAVE_TXT:
    check("fotoğraf parsel dışındaysa rapor bunu söyler", "çokgenin dışında" in txt2)
nonde = dataclasses.replace(yr, is_demo=False, source_name="Sentinel-2 L2A (Earth Search)")
txt3 = text_of(wr.build_pdf(nonde, dataclasses.replace(meta, field_photo=None)))
if HAVE_TXT:
    check("demo olmayan çalıştırmada SENTETİK etiketi/uyarısı yok", "SENTETİK" not in txt3 and "Sentinel-2 L2A (Earth Search)" in txt3, txt3[:300])
    check("demo olmayan çalıştırmada FIRMS başlığında SENTETİK eki yok", "Yangın noktaları (NASA FIRMS)" in txt3 and "(NASA FIRMS) — SENTETİK" not in txt3)
    check("uzman kararı girilmezse 'Beklemede' yazar", "Beklemede" in text_of(wr.build_pdf(yr, wr.ReportMeta("sigorta", "P1", generated_at=WHEN))))

print("— işaretleme enjeksiyonu ve sınırlar")
evil = '<b>kalın</b> & <script>alert(1)</script> </para> <font color="red">k</font> \x00 ' + "A" * 3000
pdf4 = wr.build_pdf(yr, wr.ReportMeta("sigorta", "<i>P</i>&", "<x>", "<y>", "<z>", "<q>", "<w>", evil, None, None, WHEN))
t4 = text_of(pdf4)
check("kötü niyetli metin PDF'i bozmaz", pdf4[:5] == b"%PDF-")
if HAVE_TXT:
    check("etiketler düz metin olarak görünür (yorumlanmaz)", "<script>alert(1)</script>" in t4 and "<b>kalın</b>" in t4 and "<i>P</i>" in t4, t4[:200])
    check("uzun not kırpılır (1500 karakter)", t4.count("A") <= 1500 + 40 and t4.count("A") > 500)
check("_clean: kontrol karakteri ve uzunluk", "\x00" not in wr._clean("a\x00b", 50) and len(wr._clean("x" * 999, 20)) == 20)

print("— tarım raporu + rekolte")
tr = run_for("tarim")
ys = wr.YieldSummary("Buğday", "Olgunlaşma", 5.4, 4.78, 6.01, 0.9, 4.8, 0.38, 6.4, True, ("uyarı-1",))
t5 = text_of(wr.build_pdf(tr, wr.ReportMeta("tarim", "PRS-7", yield_summary=ys, generated_at=WHEN)))
if HAVE_TXT:
    check("tarım raporu başlığı ve rekolte bölümü", "TARIMSAL GELİŞİM RAPORU" in t5 and "Rekolte tahmini" in t5 and "5,40 t/ha" in t5 and "4,78 – 6,01" in t5)
    check("sentetik rekolte modeli uyarısı", "SENTETİK geçmiş veriyle eğitildi" in t5)
    check("tarım raporunda poliçe satırı yok", "Poliçe no" not in t5)
    check("tarım raporunda yangın noktası bölümü yok", "Yangın noktaları" not in t5)

print("— yazı tipi yedeği (Helvetica + ASCII)")
saved = (wr._Fonts.regular, wr._Fonts.bold, wr._Fonts.unicode_ok, wr._Fonts._done)
try:
    wr._Fonts.regular, wr._Fonts.bold, wr._Fonts.unicode_ok, wr._Fonts._done = "Helvetica", "Helvetica-Bold", False, True
    check("_t: Türkçe → ASCII", wr._t("ğüşiöçİĞ ₺ → ≥") == "gusiocIG TL -> >=")
    pdf6 = wr.build_pdf(yr, wr.ReportMeta("sigorta", "P1", generated_at=WHEN))
    check("yedek yazı tipiyle de PDF üretilir", pdf6[:5] == b"%PDF-")
finally:
    wr._Fonts.regular, wr._Fonts.bold, wr._Fonts.unicode_ok, wr._Fonts._done = saved

print("— çokgen çok köşeliyse kırpılır")
ring = tuple((35.0 + 0.004 * np.cos(t), 39.0 + 0.004 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 80, endpoint=False))
many = dataclasses.replace(yr, request=dataclasses.replace(yr.request, region=an.Region("çok köşeli", ring)))
t7 = text_of(wr.build_pdf(many, wr.ReportMeta("sigorta", "P2", generated_at=WHEN)))
if HAVE_TXT:
    check("80 köşeden ilk 30'u gösterilir ve söylenir", "80 köşesinden ilk 30" in t7)

print("— rapor no")
r1, r2 = wr._report_id(yr, wr.ReportMeta("sigorta", "A"), WHEN), wr._report_id(yr, wr.ReportMeta("sigorta", "B"), WHEN)
check("rapor no biçimli, parsele göre farklı, aynı girdide kararlı", r1.startswith("OF-20261107-") and r1 != r2 and r1 == wr._report_id(yr, wr.ReportMeta("sigorta", "A"), WHEN))

print()
if FAILS:
    print(f"BAŞARISIZ ({len(FAILS)}):")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("BAŞARISIZ: yok — tüm kontroller geçti")
