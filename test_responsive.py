"""Duyarlı düzen testleri: telefon/tablet genişliklerinde yatay taşma, metin çakışması, kırpılan içerik ve iframe taşması olmamalı.

UYARI: Gerçek Streamlit çalıştırılmaz; sayfalar ``domst`` ile Streamlit'e benzer bir iskelete dökülüp Chromium'da ölçülür
(uygulamanın kendi HTML/CSS'i birebir, Streamlit'in sütun/düğme/sekme yapısı yaklaşık). Playwright/Chromium yoksa tarayıcı testleri atlanır.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["ORBITAL_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "t.sqlite3")

import domst  # noqa: E402

domst.install()
import layout_check as lc  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  [{extra}]" if (extra and not cond) else ""))
    if not cond:
        FAILS.append(name)


print("— tarayıcısız kontroller")
import app  # noqa: E402
import landing  # noqa: E402
import workflow_ui as wui  # noqa: E402

check("karşılama CSS'inde dar ekran menü kuralı (≤900 px)", "@media (max-width:900px)" in landing.LANDING_CSS and "stHorizontalBlock\"]::after" in landing.LANDING_CSS)
check("karşılama: cam nesneler dar ekranda akışa girer", ".g-hero{ position:relative;" in landing.LANDING_CSS and ".g-offer{ position:relative;" in landing.LANDING_CSS)
check("uygulama CSS: yan dolgu duyarlı ve üst çubuk payı var", "clamp(14px,4vw,5rem)" in app.CUSTOM_CSS and "padding-top:3.9rem" in app.CUSTOM_CSS)
check("oturumsuz ekranlarda Streamlit üst çubuğu gizlenir", 'stHeader"]{display:none' in app.HIDE_SIDEBAR_CSS.replace(" ", ""))
check("harita üstü paneller: dar harita kuralı", "@media (max-width:560px)" in app.MAP_OVERLAY_CSS and "box-sizing:border-box" in app.MAP_OVERLAY_CSS)
wrapped = wui._wrap("<table><tr><td>a</td></tr></table><table></table>")
check("iş akışı: her tablo kaydırılabilir kutuya sarılır", wrapped.count('class="wf-tw"') == 2 and wrapped.count("</table></div>") == 2)
check("iş akışı: iki sütunlu ızgara taşmaz (min(310px,100%))", "minmax(min(310px,100%),1fr)" in wui._CSS)
m1 = "<html><body><p>x</p></body></html>"
fit1 = app._with_frame_fit(m1)
check("önizleme çerçevesi: betik </body> öncesine eklenir", fit1.endswith("</body></html>") and "frameElement" in fit1 and fit1.index("<script>") < fit1.index("</body>"))
check("önizleme çerçevesi: kendi sığdırması olan belgeye dokunulmaz", app._with_frame_fit("<body>frameElement</body>") == "<body>frameElement</body>")
check("önizleme çerçevesi: gövdesiz parçaya da eklenir", app._with_frame_fit("<p>x</p>").endswith("</script>"))
import reports as rep  # noqa: E402

check("kanıt belgesi betiksiz kalır (yalnızca ekran önizlemesine betik eklenir)", "<script" not in rep.PACK_CSS and "frameElement" not in rep.PACK_CSS)

if not lc.available():
    print("  (Playwright/Chromium yok: tarayıcı denetimleri atlandı)")
else:
    print("— tarayıcı denetimi (telefon ve tablet genişlikleri)")
    WIDTHS = (320, 360, 390, 414, 600, 768, 820)
    docs = lc.build_scenarios()
    check("tüm senaryolar üretildi", set(docs) >= {"landing_ana", "landing_hakkimizda", "landing_hizmetler", "landing_nasil", "landing_iletisim", "login",
                                                   "insurer_all_tabs", "insurer_workflow_result", "customer", "map_insurance", "map_customer"}, sorted(docs))
    res = lc.measure(docs, WIDTHS)
    for name, per in res.items():
        problems = [ln for ln in lc.summarize({name: per}, max_w=820)]
        check(f"{name}: dar ekranlarda taşma/çakışma yok", not problems, "; ".join(problems[:4]))
    # kontrol edilen şeyin gerçekten ölçüldüğünü göster: taşan sahte sayfa yakalanmalı
    bad = ('<!doctype html><html><body style="margin:0"><div style="width:900px;background:#222;color:#fff">geniş</div>'
           '<p style="position:absolute;top:10px;left:10px">üst</p><p style="position:absolute;top:12px;left:12px">alt</p></body></html>')
    bres = lc.measure({"kotu": bad}, (390,))
    blines = lc.summarize(bres)
    check("ölçüm aracı: taşan ve çakışan sahte sayfayı yakalar", any("yatay" in x or "taşma" in x for x in blines) and any("çakışma" in x for x in blines), blines)

print()
if FAILS:
    print(f"BAŞARISIZ ({len(FAILS)}):")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("BAŞARISIZ: yok — tüm kontroller geçti")
