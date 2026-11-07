"""Duyarlı (responsive) düzen denetimi: sayfaları ``domst`` ile HTML'e döküp Chromium'da farklı genişliklerde ölçer.

Kullanım::

    python layout_check.py                  # tüm senaryolar, özet
    python layout_check.py landing_ana 390  # bir senaryo + genişlik: ayrıntı ve ekran görüntüsü (/tmp/layout_*.png)

Denetimler (her genişlik için):
  * ``yatay``      – belge yatay kaydırma çubuğu çıkarıyor mu / görünür öğe ekran dışına taşıyor mu (kırpan bir üst öğe yoksa)
  * ``kırpık``     – ``overflow:hidden`` olup içeriği sığmayan (yazısı kesilen) öğe
  * ``çakışma``    – farklı öğelere ait iki metin parçasının üst üste binmesi
  * ``süs-metin``  – süs amaçlı cam nesnenin (``.lp-glass``) metnin üstüne binmesi
  * ``iframe``     – iframe içeriği genişliğe/yüksekliğe sığmıyor

UYARI: Gerçek Streamlit değil, taklit iskeletidir (bkz. ``domst.py``); cihazda ilk yayından sonra gözle doğrulayın.
Playwright/Chromium yoksa ``available()`` yanlış döner ve testler atlanır.
"""
import os
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

WIDTHS = (320, 360, 390, 414, 480, 600, 768, 820, 1024, 1366)

PROBE_JS = r"""
() => {
  const vw = document.documentElement.clientWidth;
  const out = {vw, docW: document.documentElement.scrollWidth, yatay: [], kirpik: [], cakisma: [], suse: [], iframe: []};
  const label = (e) => {
    let s = e.tagName.toLowerCase();
    if (e.id) s += '#' + e.id;
    const c = (typeof e.className === 'string' ? e.className : '').trim().split(/\s+/).filter(Boolean).slice(0, 3).join('.');
    if (c) s += '.' + c;
    const t = (e.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 28);
    return s + (t ? ' «' + t + '»' : '');
  };
  const visible = (e) => { const cs = getComputedStyle(e); const r = e.getBoundingClientRect();
    return cs.display !== 'none' && cs.visibility !== 'hidden' && r.width > 0 && r.height > 0; };
  const clippedByAncestor = (e) => { for (let p = e.parentElement; p && p !== document.body; p = p.parentElement) {
    const cs = getComputedStyle(p); if (/(auto|hidden|scroll|clip)/.test(cs.overflowX)) return true; } return false; };
  const all = Array.from(document.body.querySelectorAll('*')).filter((e) => !['SCRIPT', 'STYLE', 'HEAD', 'TITLE', 'META'].includes(e.tagName) && !e.closest('svg') || e.tagName === 'svg');

  // 1) yatay taşma
  const off = new Set();
  for (const e of all) {
    if (!visible(e) || clippedByAncestor(e)) continue;
    const r = e.getBoundingClientRect();
    if (r.right > vw + 1.5 || r.left < -1.5) off.add(e);
  }
  for (const e of off) {
    if (e.parentElement && off.has(e.parentElement)) continue;
    const r = e.getBoundingClientRect();
    out.yatay.push(label(e) + ` [${Math.round(r.left)}…${Math.round(r.right)} / ${vw}]`);
  }

  // 2) kırpılan içerik (süs cam nesnelerin bilerek taşan kısmı sayılmaz)
  for (const e of all) {
    const cs = getComputedStyle(e);
    if (!/(hidden|clip)/.test(cs.overflowX) || e.tagName === 'IFRAME' || e.tagName === 'DETAILS' || e.classList.contains('lp-glass') || e.closest('[data-testid="stDataFrame"]')) continue;
    if (!(e.scrollWidth > e.clientWidth + 2 && e.clientWidth > 0)) continue;
    const er = e.getBoundingClientRect();
    let worst = null;
    for (const d of e.querySelectorAll('*')) {
      if (d.closest('.lp-glass') || d.closest('svg') || !visible(d)) continue;
      const dr = d.getBoundingClientRect();
      if (dr.right > er.right + 2 || dr.left < er.left - 2) { worst = d; break; }
    }
    if (worst) out.kirpik.push(label(e) + ` [içerik ${e.scrollWidth} > kutu ${e.clientWidth}] ← ` + label(worst));
  }

  // 3) metin çakışması
  const items = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    if (!n.nodeValue.trim()) continue;
    const el = n.parentElement;
    if (!el || el.closest('script,style,svg,iframe,textarea,option') || !visible(el)) continue;
    const range = document.createRange(); range.selectNodeContents(n);
    for (const r of range.getClientRects()) {
      if (r.width < 3 || r.height < 6) continue;
      let box = {l: r.left, t: r.top, r: r.right, b: r.bottom}, hidden = false;
      for (let p = el; p && p !== document.body; p = p.parentElement) {
        const cs = getComputedStyle(p);
        if (cs.overflowX === 'visible' && cs.overflowY === 'visible') continue;
        const pr = p.getBoundingClientRect();
        box = {l: Math.max(box.l, pr.left), t: Math.max(box.t, pr.top), r: Math.min(box.r, pr.right), b: Math.min(box.b, pr.bottom)};
        if (box.r - box.l < 3 || box.b - box.t < 6) { hidden = true; break; }
      }
      if (!hidden) items.push({el, l: box.l, t: box.t, r: box.r, b: box.b});
    }
  }
  const seen = new Set();
  for (let i = 0; i < items.length && out.cakisma.length < 12; i++) {
    for (let j = i + 1; j < items.length; j++) {
      const a = items[i], c = items[j];
      if (a.el === c.el || a.el.contains(c.el) || c.el.contains(a.el)) continue;
      if (c.t > a.b) { /* sıralı değil; devam */ }
      const w = Math.min(a.r, c.r) - Math.max(a.l, c.l), h = Math.min(a.b, c.b) - Math.max(a.t, c.t);
      if (w > 3 && h > 4) {
        const area = w * h, minA = Math.min((a.r - a.l) * (a.b - a.t), (c.r - c.l) * (c.b - c.t));
        if (area > 0.2 * minA) {
          const k = label(a.el) + ' × ' + label(c.el);
          if (!seen.has(k)) { seen.add(k); out.cakisma.push(k + ` @y${Math.round(a.t + window.scrollY)}`); }
        }
      }
    }
  }

  // 4) süs nesne - metin
  for (const g of document.querySelectorAll('.lp-glass')) {
    if (!visible(g)) continue;
    const gr = g.getBoundingClientRect();
    for (const it of items) {
      if (g.contains(it.el)) continue;
      const w = Math.min(gr.right, it.r) - Math.max(gr.left, it.l), h = Math.min(gr.bottom, it.b) - Math.max(gr.top, it.t);
      if (w > 6 && h > 6) { out.suse.push(label(g) + ' ↔ ' + label(it.el) + ` @y${Math.round(it.t + window.scrollY)}`); break; }
    }
  }

  // 5) iframe
  for (const f of document.querySelectorAll('iframe')) {
    try {
      const d = f.contentDocument; if (!d || !d.documentElement) continue;
      const sw = d.documentElement.scrollWidth, sh = d.documentElement.scrollHeight;
      const bad = [];
      if (sw > f.clientWidth + 2) bad.push(`genişlik ${sw} > ${f.clientWidth}`);
      if (sh > f.clientHeight + 4) bad.push(`yükseklik ${sh} > ${f.clientHeight}`);
      if (bad.length) out.iframe.push(`iframe#${f.dataset.frame || '?'} ` + bad.join(', ') + ' «' + (d.body.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 30) + '»');
    } catch (e) { out.iframe.push('iframe okunamadı'); }
  }
  return out;
}
"""


def available() -> bool:
    """Playwright ve Chromium kullanılabilir mi."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:  # noqa: BLE001
        return False


def measure(html_docs: Dict[str, str], widths: Sequence[int] = WIDTHS, shots: Optional[Dict[str, Sequence[int]]] = None,
            shot_dir: str = "/tmp") -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Her belgeyi her genişlikte ölçer.

    Args:
        html_docs: ``ad -> HTML belgesi``.
        widths: Görünüm genişlikleri (px).
        shots: İsteğe bağlı ``ad -> [genişlikler]``; bu çiftler için tam sayfa ekran görüntüsü kaydedilir.
        shot_dir: Ekran görüntüsü klasörü.

    Returns:
        ``ad -> genişlik -> ölçüm sözlüğü``.
    """
    from playwright.sync_api import sync_playwright
    results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, doc in html_docs.items():
            results[name] = {}
            for w in widths:
                ctx = browser.new_context(viewport={"width": w, "height": 800}, device_scale_factor=1)
                page = ctx.new_page()
                page.route("**/*", lambda route: route.abort() if route.request.url.startswith("http") else route.continue_())
                page.set_content(doc, wait_until="load")
                page.wait_for_timeout(700)
                res = page.evaluate(PROBE_JS)
                results[name][w] = res
                if shots and w in shots.get(name, ()):
                    page.screenshot(path=os.path.join(shot_dir, f"layout_{name}_{w}.png"), full_page=True)
                ctx.close()
        browser.close()
    return results


def summarize(results: Dict[str, Dict[int, Dict[str, Any]]], max_w: int = 820) -> List[str]:
    """Dar genişliklerdeki (``max_w`` ve altı) sorunları düz satırlara döker."""
    lines: List[str] = []
    for name, per in results.items():
        for w, r in per.items():
            if w > max_w:
                continue
            problems = []
            if r["docW"] > r["vw"] + 1:
                problems.append(f"belge yatay kayıyor ({r['docW']} > {r['vw']})")
            for key, tag in (("yatay", "taşma"), ("kirpik", "kırpık"), ("cakisma", "çakışma"), ("suse", "süs-metin"), ("iframe", "iframe")):
                problems.extend(f"{tag}: {x}" for x in r[key][:6])
            for p_ in problems:
                lines.append(f"{name} @{w}: {p_}")
    return lines


# --------------------------------------------------------------------------- #
# Senaryolar (sahte Streamlit ile gerçek app.py)
# --------------------------------------------------------------------------- #
def build_scenarios(only: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """Uygulama ekranlarını HTML belgelerine döker.

    Args:
        only: Yalnızca bu adları üret (``None`` = hepsi).

    Returns:
        ``ad -> HTML``.
    """
    os.environ.setdefault("ORBITAL_DB_PATH", os.path.join(tempfile.mkdtemp(), "layout.sqlite3"))
    os.environ["ORBITAL_CONTACT_EMAIL"] = "iletisim@ornek.com"
    import domst
    st = domst.install()
    import app
    import auth
    import landing

    want = (lambda n: only is None or n in only)  # noqa: E731
    docs: Dict[str, str] = {}

    def run(name: str, setup: Callable[[], None], press: Sequence[str] = ()) -> None:
        if not want(name):
            return
        st.reset()
        setup()
        for k in press:
            st.pressed.add(k)
        app.main()
        docs[name] = domst.page_html(st)

    for slug, label in (("", "ana"), ("hakkimizda", "hakkimizda"), ("hizmetler", "hizmetler"), ("nasil", "nasil"), ("iletisim", "iletisim")):
        run(f"landing_{label}", lambda slug=slug: st.query_params.update({landing.QUERY_KEY: slug} if slug else {}))
    run("login", lambda: st.session_state.update({app.SESSION_VIEW: app.VIEW_LOGIN}))

    acc = [a for a in auth.demo_accounts() if a.role is auth.Role.INSURER][0]
    ins = auth.Principal(acc.username, acc.role, acc.display_name, None, None, acc.company, acc.title)
    cus_acc = [a for a in auth.demo_accounts() if a.role is auth.Role.CUSTOMER][0]
    cus = auth.Principal(cus_acc.username, cus_acc.role, cus_acc.display_name, None, None, cus_acc.company, cus_acc.title)

    def as_user(p: Any, **state: Any) -> Callable[[], None]:
        def go() -> None:
            st.session_state[app.SESSION_USER] = p.to_session()
            st.session_state[app.SESSION_SEEN] = time.time()
            st.session_state.update(state)
        return go

    # harita üstü paneller: gerçek harita yerine sahte Leaflet denetimleri (yakınlaştırma, katman düğmesi, atıf)
    def map_doc(overlay: str) -> str:
        ctl = ('<div style="position:absolute;left:10px;top:10px;width:30px;height:62px;z-index:1000;border:1px solid #888;color:#fff">+<br>−</div>'
               '<div style="position:absolute;right:10px;top:10px;width:34px;height:34px;z-index:1000;border:1px solid #888;color:#fff">☰</div>'
               '<div style="position:absolute;right:0;bottom:0;z-index:1000;color:#ccc;font:10px sans-serif;padding:2px 6px">Leaflet · © OpenStreetMap</div>')
        return ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>'
                f'<body style="margin:0;background:#334"><div style="position:relative;height:100vh;overflow:hidden">{ctl}{overlay}</div></body></html>')

    if want("map_insurance"):
        docs["map_insurance"] = map_doc(app.build_map_overlay(
            "SİGORTA · HASAR KATMANI", "Parsel bazlı", "hasar analizi", ("Hasar şiddeti", "olay sonrası 07.11.2026"),
            tuple((k, "1.234", "parsel", c) for k, c in app.DECISION_COLORS.items())))
    if want("map_customer"):
        docs["map_customer"] = map_doc(app.build_map_overlay("MÜŞTERİ · PARSELLERİM", "Parsellerim", "", ("12 parsel",)))

    run("insurer_all_tabs", as_user(ins))
    run("insurer_workflow_result", as_user(ins), press=["wf_run"])
    run("customer", as_user(cus))
    return docs


def main(argv: Sequence[str]) -> int:
    """Komut satırı girişi."""
    if not available():
        print("Playwright/Chromium yok; denetim atlandı.")
        return 0
    names = [argv[0]] if argv else None
    widths = [int(argv[1])] if len(argv) > 1 else WIDTHS
    docs = build_scenarios(names)
    shots = {n: widths for n in docs} if len(argv) > 1 else {n: (390,) for n in docs}
    res = measure(docs, widths, shots)
    lines = summarize(res)
    for ln in lines:
        print(ln)
    print(f"\n{len(lines)} sorun satırı · senaryolar: {', '.join(docs)}")
    return 1 if lines else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
