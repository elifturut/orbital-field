"""Düzen (layout) düzeneği: ``app.py`` ekranlarını Streamlit'in DOM iskeletine benzer bir HTML'e dökerek tarayıcıda ölçmeyi sağlar.

UYARI: Bu GERÇEK Streamlit DEĞİLDİR. Sütun/kapsayıcı/düğme/sekme/sekme çubuğu yapısı ve temel stilleri Streamlit'in belgelenmiş davranışına
bakarak *yaklaşık* taklit edilir (sütunlar 640 px altında alt alta dizilir, ana kapsayıcı dolgusu, ``st-key-…`` sınıfları). Özel HTML/CSS
(uygulamanın kendi yazdığı kısım: kartlar, ızgaralar, başlıklar, SVG'ler, iframe'ler) ise birebir aynıdır; taşma ve çakışma kontrolleri
asıl olarak onlar içindir. Gerçek cihazda ilk yayından sonra gözle doğrulanmalıdır.

Kullanım: ``st = domst.install()`` → ``app.main()`` → ``domst.page_html(st)``.
"""
import html as _html
import sys
from typing import Any, List, Optional

import fakest

BASE_CSS = """
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{font-family:"Source Sans Pro",system-ui,sans-serif;background:#0b0c10;color:#e6e8ec;font-size:16px;line-height:1.6}
.stApp{min-height:100vh;background:#0b0c10;color:#e6e8ec}
.block-container{padding:6rem 1rem 10rem;width:100%;max-width:none;margin:0 auto}
@media (min-width:576px){ .block-container{padding-left:5rem;padding-right:5rem} }
[data-testid="stVerticalBlock"]{display:flex;flex-direction:column;gap:1rem;width:100%}
[data-testid="stHorizontalBlock"]{display:flex;flex-direction:row;flex-wrap:wrap;gap:1rem;width:100%;align-items:flex-start}
[data-testid="stColumn"]{display:flex;flex-direction:column;min-width:0}
@media (max-width:640px){ [data-testid="stColumn"]{min-width:calc(100% - 1rem) !important} }
.stElementContainer{width:100%;min-width:0}
[data-testid="stMarkdownContainer"] p{margin:0 0 1rem}
[data-testid="stMarkdownContainer"] > :last-child{margin-bottom:0}
[data-testid="stCaptionContainer"]{font-size:.875rem;color:rgba(250,250,250,.6)}
.stButton{width:auto}
.stButton button,.stDownloadButton button,.stFormSubmitButton button{min-height:2.5rem;padding:.25rem .75rem;border-radius:.5rem;
  border:1px solid rgba(250,250,250,.2);background:#10131a;color:inherit;font:inherit;line-height:1.6;margin:0;display:inline-flex;align-items:center;justify-content:center}
.stButton button p,.stDownloadButton button p,.stFormSubmitButton button p{margin:0}
.stTabs{width:100%}
.stTabs [data-baseweb="tab-list"]{display:flex;gap:8px;overflow-x:auto}
.stTabs button[data-baseweb="tab"]{flex:none;padding:.5rem 1rem;background:none;border:0;color:inherit;white-space:nowrap;font:inherit;height:44px}
.stTabs [data-baseweb="tab-panel"]{padding-top:1rem}
.stAlert{padding:1rem;border-radius:.5rem;background:rgba(61,157,243,.16)}
[data-testid="stExpander"]{border:1px solid rgba(250,250,250,.2);border-radius:.5rem;width:100%}
[data-testid="stExpander"] summary{padding:.75rem 1rem;cursor:pointer}
[data-testid="stExpander"] .stExpDetails{padding:0 1rem 1rem}
label{display:block;font-size:.875rem;margin-bottom:.25rem}
.stwidget input,.stwidget select,.stwidget textarea,.stwidget .fakebox{width:100%;min-height:2.5rem;border-radius:.5rem;border:1px solid rgba(250,250,250,.2);
  background:#10131a;color:inherit;padding:.25rem .75rem;font:inherit}
.stwidget textarea{min-height:5rem}
.stwidget [role="radiogroup"]{display:flex;flex-direction:column;gap:.5rem}
.stwidget [role="radiogroup"].horizontal{flex-direction:row;flex-wrap:wrap;gap:.5rem 1rem}
.stwidget [role="radiogroup"] span{display:inline-block}
.stPlotlyChart,.stFolium{width:100%;border:1px dashed rgba(250,250,250,.2);border-radius:.5rem;display:flex;align-items:center;justify-content:center;color:#69717f}
[data-testid="stDataFrame"]{width:100%;overflow:auto;max-height:420px;font-size:.875rem}
[data-testid="stDataFrame"] table{border-collapse:collapse;width:max-content;min-width:100%}
[data-testid="stDataFrame"] th,[data-testid="stDataFrame"] td{padding:.3rem .6rem;border:1px solid rgba(250,250,250,.12);white-space:nowrap;text-align:left}
iframe{border:0;width:100%;display:block}
h1,h2,h3{margin:0}
"""


class Node:
    """Çok basit DOM düğümü."""

    def __init__(self, tag: str = "div", cls: str = "", attrs: str = "", style: str = ""):
        self.tag, self.cls, self.attrs, self.style = tag, cls, attrs, style
        self.kids: List[Any] = []

    def add(self, item: Any) -> None:
        self.kids.append(item)

    def render(self) -> str:
        inner = "".join(k.render() if isinstance(k, Node) else str(k) for k in self.kids)
        cls = f' class="{self.cls}"' if self.cls else ""
        style = f' style="{self.style}"' if self.style else ""
        attrs = f" {self.attrs}" if self.attrs else ""
        return f"<{self.tag}{cls}{attrs}{style}>{inner}</{self.tag}>"


def _vb(cls: str = "") -> Node:
    return Node("div", ("stVerticalBlock " + cls).strip(), 'data-testid="stVerticalBlock"')


class DomCtx:
    """``with`` bloğu ve ``col.metot(...)`` biçiminde kullanılabilen kapsayıcı tutamağı."""

    def __init__(self, st: "DomStreamlit", node: Node):
        self._st, self._node = st, node

    def __enter__(self) -> "DomCtx":
        self._st._cur.append(self._node)
        return self

    def __exit__(self, *a: Any) -> bool:
        self._st._cur.pop()
        return False

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._st, name)
        if not callable(target):
            return target

        def call(*args: Any, **kw: Any) -> Any:
            with self:
                return target(*args, **kw)

        return call


class DomStreamlit(fakest.FakeStreamlit):
    """``fakest.FakeStreamlit`` + DOM üretimi."""

    def reset(self) -> None:
        super().reset()
        self.root = _vb()
        self.side = _vb()
        self._cur: List[Node] = [self.root]
        self.sidebar = DomCtx(self, self.side)
        self.page_config: dict = {}
        self._frame_n = 0

    # ---- yardımcılar
    def _el(self, inner: str, cls: str = "") -> None:
        self._cur[-1].add(f'<div class="stElementContainer {cls}">{inner}</div>')

    def _key_cls(self, key: Optional[str]) -> str:
        return f"st-key-{key}" if key else ""

    def _label(self, label: str) -> str:
        return f"<label>{_html.escape(str(label))}</label>" if label else ""

    # ---- metin
    def markdown(self, body: str = "", unsafe_allow_html: bool = False, **kw: Any) -> None:
        super().markdown(body, unsafe_allow_html=unsafe_allow_html)
        text = str(body)
        self._el(f'<div data-testid="stMarkdownContainer">{text if unsafe_allow_html else "<p>" + _html.escape(text) + "</p>"}</div>')

    def caption(self, body: str = "", **kw: Any) -> None:
        super().caption(body)
        self._el(f'<div data-testid="stCaptionContainer">{_html.escape(str(body))}</div>')

    def write(self, *a: Any, **kw: Any) -> None:
        super().write(*a)
        self._el("".join(f"<p>{_html.escape(str(x))}</p>" for x in a))

    def subheader(self, body: str = "", **kw: Any) -> None:
        super().subheader(body)
        self._el(f"<h3>{_html.escape(str(body))}</h3>")

    def code(self, body: str = "", **kw: Any) -> None:
        super().code(body)
        self._el(f'<pre style="overflow:auto;margin:0">{_html.escape(str(body))}</pre>')

    def _alert(self, kind: str, body: str) -> None:
        self._rec(kind, body)
        self._el(f'<div class="stAlert" data-testid="stAlert">{_html.escape(str(body))}</div>')

    def info(self, body: str = "", **kw: Any) -> None:
        self._alert("info", body)

    def warning(self, body: str = "", **kw: Any) -> None:
        self._alert("warning", body)

    def error(self, body: str = "", **kw: Any) -> None:
        self._alert("error", body)

    def success(self, body: str = "", **kw: Any) -> None:
        self._alert("success", body)

    def set_page_config(self, **kw: Any) -> None:
        self.page_config = dict(kw)

    # ---- yerleşim
    def columns(self, spec: Any, **kw: Any) -> List[DomCtx]:
        weights = [1.0] * spec if isinstance(spec, int) else [float(x) for x in spec]
        total = sum(weights)
        hb = Node("div", "stHorizontalBlock", 'data-testid="stHorizontalBlock"')
        out = []
        for w in weights:
            pct = w / total * 100.0
            col = Node("div", "stColumn", 'data-testid="stColumn"', f"width:calc({pct:.4f}% - 1rem);flex:1 1 calc({pct:.4f}% - 1rem)")
            inner = _vb()
            col.add(inner)
            hb.add(col)
            out.append(DomCtx(self, inner))
        self._cur[-1].add(hb)
        return out

    def container(self, *a: Any, key: Optional[str] = None, **kw: Any) -> DomCtx:
        node = _vb(self._key_cls(key))
        self._cur[-1].add(node)
        return DomCtx(self, node)

    def form(self, key: str, **kw: Any) -> DomCtx:
        return self.container(key=key)

    def empty(self) -> DomCtx:
        return self.container()

    def spinner(self, *a: Any, **kw: Any) -> DomCtx:
        return DomCtx(self, _vb())

    def status(self, *a: Any, **kw: Any) -> DomCtx:
        return self.container()

    def tabs(self, labels: Any) -> List[DomCtx]:
        labels = list(labels)
        self._rec("tabs", labels)
        wrap = Node("div", "stTabs", 'data-testid="stTabs"')
        bar = "".join(f'<button data-baseweb="tab" role="tab" aria-selected="{"true" if i == 0 else "false"}"><div data-testid="stMarkdownContainer"><p>{_html.escape(str(t))}</p></div></button>'
                      for i, t in enumerate(labels))
        wrap.add(f'<div data-baseweb="tab-list" role="tablist">{bar}</div>')
        out = []
        for t in labels:
            panel = Node("div", "tabpanel", f'data-baseweb="tab-panel" data-tab="{_html.escape(str(t), quote=True)}"')
            inner = _vb()
            panel.add(f'<div style="margin:18px 0 6px;font:600 11px monospace;color:#6ea8fe">▼ SEKME: {_html.escape(str(t))}</div>')
            panel.add(inner)
            wrap.add(panel)
            out.append(DomCtx(self, inner))
        self._cur[-1].add(wrap)
        return out

    def expander(self, label: str = "", **kw: Any) -> DomCtx:
        self._rec("expander", label)
        det = Node("details", "stExpander", 'data-testid="stExpander" open')
        det.add(f"<summary>{_html.escape(str(label))}</summary>")
        inner = _vb()
        box = Node("div", "stExpDetails")
        box.add(inner)
        det.add(box)
        self._cur[-1].add(det)
        return DomCtx(self, inner)

    # ---- düğme ve widget'lar
    def button(self, label: str = "", key: Optional[str] = None, on_click: Any = None, args: Any = (), kwargs: Any = None, **kw: Any) -> bool:
        hit = super().button(label, key=key, on_click=on_click, args=args, kwargs=kwargs)
        self._el(f'<div data-testid="stButton" class="stButton"><button><div data-testid="stMarkdownContainer"><p>{_html.escape(str(label))}</p></div></button></div>',
                 self._key_cls(key))
        return hit

    form_submit_button = button

    def download_button(self, label: str = "", data: Any = b"", file_name: str = "", mime: str = "", key: Optional[str] = None, **kw: Any) -> bool:
        res = super().download_button(label, data=data, file_name=file_name, mime=mime, key=key)
        self._el(f'<div data-testid="stDownloadButton" class="stDownloadButton"><button><div data-testid="stMarkdownContainer"><p>{_html.escape(str(label))}</p></div></button></div>',
                 self._key_cls(key))
        return res

    def _widget(self, kind: str, label: str, body: str, key: Optional[str]) -> None:
        self._el(f'<div class="stwidget {kind}" data-testid="st{kind}">{self._label(label)}{body}</div>', self._key_cls(key))

    def selectbox(self, label: str, options: Any, index: int = 0, key: Optional[str] = None, **kw: Any) -> Any:
        res = super().selectbox(label, options, index=index, key=key)
        self._widget("Selectbox", label, f'<div class="fakebox" data-baseweb="select">{_html.escape(str(res))}</div>', key)
        return res

    def radio(self, label: str, options: Any, index: int = 0, key: Optional[str] = None, horizontal: bool = False, **kw: Any) -> Any:
        res = super().radio(label, options, index=index, key=key)
        opts = "".join(f"<label><span>◉ {_html.escape(str(o))}</span></label>" for o in options)
        self._widget("Radio", label, f'<div role="radiogroup" class="{"horizontal" if horizontal else ""}">{opts}</div>', key)
        return res

    def multiselect(self, label: str, options: Any, default: Any = None, key: Optional[str] = None, **kw: Any) -> Any:
        res = super().multiselect(label, options, default=default, key=key)
        self._widget("Multiselect", label, f'<div class="fakebox" data-baseweb="select">{_html.escape(", ".join(map(str, res)))}</div>', key)
        return res

    def slider(self, label: str, min_value: Any = 0, max_value: Any = 100, value: Any = None, step: Any = None, key: Optional[str] = None, **kw: Any) -> Any:
        res = super().slider(label, min_value=min_value, max_value=max_value, value=value, step=step, key=key)
        self._widget("Slider", label, f'<div style="height:2.5rem;display:flex;align-items:center"><div style="flex:1;height:4px;background:#333"></div><span style="margin-left:8px">{_html.escape(str(res))}</span></div>', key)
        return res

    def text_input(self, label: str, value: str = "", key: Optional[str] = None, **kw: Any) -> Any:
        res = super().text_input(label, value=value, key=key)
        self._widget("TextInput", label, f'<input value="{_html.escape(str(res or ""), quote=True)}">', key)
        return res

    def number_input(self, label: str, min_value: Any = None, max_value: Any = None, value: Any = 0, step: Any = None, key: Optional[str] = None, **kw: Any) -> Any:
        res = super().number_input(label, min_value=min_value, max_value=max_value, value=value, step=step, key=key)
        self._widget("NumberInput", label, f'<input value="{_html.escape(str(res), quote=True)}">', key)
        return res

    def text_area(self, label: str, value: str = "", key: Optional[str] = None, **kw: Any) -> Any:
        res = super().text_area(label, value=value, key=key)
        self._widget("TextArea", label, f"<textarea>{_html.escape(str(res or ''))}</textarea>", key)
        return res

    def date_input(self, label: str, value: Any = None, min_value: Any = None, max_value: Any = None, key: Optional[str] = None, **kw: Any) -> Any:
        res = super().date_input(label, value=value, min_value=min_value, max_value=max_value, key=key)
        self._widget("DateInput", label, f'<div class="fakebox">{_html.escape(str(res))}</div>', key)
        return res

    def checkbox(self, label: str, value: bool = False, key: Optional[str] = None, **kw: Any) -> Any:
        res = super().checkbox(label, value=value, key=key)
        self._widget("Checkbox", "", f"<label>{'☑' if res else '☐'} {_html.escape(str(label))}</label>", key)
        return res

    toggle = checkbox

    def file_uploader(self, label: str, key: Optional[str] = None, **kw: Any) -> Any:
        res = super().file_uploader(label, key=key)
        self._widget("FileUploader", label, '<div class="fakebox" style="min-height:5rem">Dosyayı sürükleyin veya göz atın · Browse files</div>', key)
        return res

    def progress(self, value: Any = 0, text: str = "", **kw: Any) -> Any:
        self._el(f'<div class="stProgress"><div style="height:8px;background:#222;border-radius:4px"><div style="width:{int(float(value or 0) * 100)}%;height:8px;background:#6ea8fe"></div></div><p>{_html.escape(str(text or ""))}</p></div>')
        return super().progress(value, text=text)

    # ---- tablo / grafik / bileşen
    def _table(self, data: Any) -> None:
        try:
            frame = data.data if hasattr(data, "data") and hasattr(data.data, "to_html") else data
            body = frame.to_html(index=False, border=0) if hasattr(frame, "to_html") else f"<pre>{_html.escape(str(frame))}</pre>"
        except Exception:  # noqa: BLE001
            body = "<pre>(tablo)</pre>"
        self._el(f'<div data-testid="stDataFrame">{body}</div>')

    def dataframe(self, data: Any = None, **kw: Any) -> None:
        super().dataframe(data)
        self._table(data)

    def table(self, data: Any = None, **kw: Any) -> None:
        super().table(data)
        self._table(data)

    def plotly_chart(self, fig: Any, **kw: Any) -> None:
        super().plotly_chart(fig)
        self._el('<div class="stPlotlyChart" style="height:400px">(grafik)</div>')

    def pyplot(self, fig: Any, **kw: Any) -> None:
        super().pyplot(fig)
        self._el('<div class="stPlotlyChart" style="height:400px">(grafik)</div>')

    def image(self, img: Any, **kw: Any) -> None:
        super().image(img, **kw)
        self._el('<div class="stFolium" style="height:240px">(görsel)</div>')

    def _frame(self, markup: str, height: Any = None, **kw: Any) -> None:
        super()._frame(markup, height, **kw)
        self._frame_n += 1
        h = int(height) if height else 150
        self._el(f'<iframe data-frame="{self._frame_n}" height="{h}" srcdoc="{_html.escape(markup, quote=True)}"></iframe>')

    def map_placeholder(self, height: int = 480) -> None:
        self._el(f'<div class="stFolium" style="height:{height}px">(harita)</div>')


def install() -> DomStreamlit:
    """``fakest.install`` ile aynı sahte modülleri kurar, ancak DOM üreten sürümüyle (tekrar çağrılırsa aynı örneği döndürür)."""
    existing = sys.modules.get("streamlit")
    if isinstance(existing, DomStreamlit):
        return existing
    real = fakest.FakeStreamlit
    fakest.FakeStreamlit = DomStreamlit  # type: ignore[misc]
    try:
        st = fakest.install()
    finally:
        fakest.FakeStreamlit = real  # type: ignore[misc]
    st.components = type(st.components)(v1=type(st.components.v1)(html=st._frame))

    def st_folium(fmap: Any, **kw: Any) -> Any:
        st.folium_calls.append((fmap, kw))
        st.map_placeholder()
        return st.map_return.get(kw.get("key"), {})

    sys.modules["streamlit_folium"].st_folium = st_folium
    return st


def page_html(st: DomStreamlit, sidebar: bool = False) -> str:
    """Çizilmiş ekranı tek bir HTML belgesine çevirir (Streamlit iskeleti içinde).

    Args:
        st: ``install()`` ile kurulmuş sahte Streamlit.
        sidebar: Yan menü içeriği de eklensin mi (dar ekranda yan menü kapalı gelir; varsayılan eklemez).
    """
    side = f'<aside data-testid="stSidebar" style="width:21rem;padding:1rem">{st.side.render()}</aside>' if sidebar else ""
    return ('<!doctype html><html lang="tr"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<style>{BASE_CSS}</style></head><body><div class=\"stApp\">{side}"
            f'<main data-testid="stMain"><div data-testid="stMainBlockContainer" class="block-container">{st.root.render()}</div></main></div></body></html>')
