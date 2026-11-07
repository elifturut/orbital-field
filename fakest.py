"""Sahte Streamlit/Folium/Plotly düzeneği: app.py'yi içe aktarıp ekran işlevlerini çalıştırır (gerçek Streamlit DEĞİLDİR; arayüz mantığını sınar, gerçek çizimi değil).

Kayıt tutar: markdown (HTML), uyarılar, tablolar, indirme düğmeleri, harita/grafik çağrıları, widget değerleri.
"""
import sys
import types
from unittest.mock import MagicMock


class Rerun(Exception):
    pass


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value


class Ctx:
    """`with` bloğu ve `c1.slider(...)` biçiminde kullanılabilen kapsayıcı."""

    def __init__(self, fake):
        self._fake = fake

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(self._fake, name)


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.__version__ = "1.50.0"
        self.reset()
        self.column_config = MagicMock()
        self.sidebar = Ctx(self)
        errors = types.ModuleType("streamlit.errors")

        class StreamlitAPIException(Exception):
            pass

        errors.StreamlitAPIException = StreamlitAPIException
        self.errors = errors
        self.StreamlitAPIException = StreamlitAPIException
        self.components = types.SimpleNamespace(v1=types.SimpleNamespace(html=self._frame))

    def reset(self):
        self.session_state = SessionState()
        self.query_params = {}
        self.secrets = {}
        self.pressed = set()          # bu çalıştırmada "tıklanmış" düğme anahtarları
        self.out = []                 # (tür, içerik)
        self.widgets = {}             # anahtar -> son değer
        self.downloads = []
        self.figures = []
        self.tables = []
        self.folium_calls = []
        self.map_return = {}
        self.frames = []              # components.v1.html çağrıları: (markup, height)

    # ---- kayıt
    def _rec(self, kind, content=""):
        self.out.append((kind, content))

    def html(self):
        return "\n".join(str(c) for k, c in self.out if k == "markdown")

    def texts(self, kind):
        return [c for k, c in self.out if k == kind]

    def _frame(self, markup, height=None, **kw):
        self.frames.append((markup, height))
        self._rec("frame", (markup, height))

    def frame_html(self):
        return "\n".join(m for m, _ in self.frames)

    # ---- metin
    def markdown(self, body="", unsafe_allow_html=False, **kw):
        self._rec("markdown", body)

    def caption(self, body="", **kw):
        self._rec("caption", body)

    def write(self, *a, **kw):
        self._rec("write", a)

    def info(self, body="", **kw):
        self._rec("info", body)

    def warning(self, body="", **kw):
        self._rec("warning", body)

    def error(self, body="", **kw):
        self._rec("error", body)

    def success(self, body="", **kw):
        self._rec("success", body)

    def code(self, body="", **kw):
        self._rec("code", body)

    def subheader(self, body="", **kw):
        self._rec("subheader", body)

    def set_page_config(self, **kw):
        pass

    def toast(self, body="", **kw):
        self._rec("toast", body)

    # ---- yerleşim
    def columns(self, spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        return [Ctx(self) for _ in range(n)]

    def tabs(self, labels):
        self._rec("tabs", list(labels))
        return [Ctx(self) for _ in labels]

    def expander(self, label="", **kw):
        self._rec("expander", label)
        return Ctx(self)

    def container(self, *a, **kw):
        return Ctx(self)

    def form(self, key, **kw):
        return Ctx(self)

    def empty(self):
        return Ctx(self)

    def spinner(self, *a, **kw):
        return Ctx(self)

    def progress(self, *a, **kw):
        return Ctx(self)

    def status(self, *a, **kw):
        return Ctx(self)

    # ---- widget'lar
    def _value(self, key, default):
        if key is None:
            return default
        if key not in self.session_state:
            self.session_state[key] = default
        self.widgets[key] = self.session_state[key]
        return self.session_state[key]

    def button(self, label="", key=None, on_click=None, args=(), kwargs=None, **kw):
        hit = key is not None and key in self.pressed
        if hit:
            self.pressed.discard(key)
            if on_click:
                on_click(*(args or ()), **(kwargs or {}))
        return hit

    form_submit_button = button

    def download_button(self, label="", data=b"", file_name="", mime="", key=None, **kw):
        self.downloads.append({"label": label, "data": data, "file_name": file_name, "mime": mime, "key": key})
        return False

    def selectbox(self, label, options, index=0, key=None, **kw):
        opts = list(options)
        if key is not None and key in self.session_state and self.session_state[key] in opts:
            return self._value(key, None)
        default = opts[index] if opts and index is not None else None
        if key is not None:
            self.session_state[key] = default
            self.widgets[key] = default
        return default

    def radio(self, label, options, index=0, key=None, **kw):
        return self.selectbox(label, options, index=index, key=key)

    def multiselect(self, label, options, default=None, key=None, **kw):
        return self._value(key, list(default or []))

    def slider(self, label, min_value=0, max_value=100, value=None, step=None, key=None, **kw):
        default = min_value if value is None else value
        return self._value(key, default)

    def select_slider(self, label, options=(), value=None, key=None, **kw):
        opts = list(options)
        return self._value(key, value if value is not None else (opts[0] if opts else None))

    def number_input(self, label, min_value=None, max_value=None, value=0, step=None, key=None, **kw):
        return self._value(key, value)

    def text_input(self, label, value="", key=None, **kw):
        return self._value(key, value)

    def text_area(self, label, value="", key=None, **kw):
        return self._value(key, value)

    def checkbox(self, label, value=False, key=None, **kw):
        return self._value(key, value)

    def toggle(self, label, value=False, key=None, **kw):
        return self._value(key, value)

    def date_input(self, label, value=None, min_value=None, max_value=None, key=None, **kw):
        return self._value(key, value)

    def file_uploader(self, label, key=None, **kw):
        return self._value(key, None)

    def rerun(self):
        raise Rerun()

    # ---- tablo/grafik
    def dataframe(self, data=None, **kw):
        self.tables.append(data)

    def table(self, data=None, **kw):
        self.tables.append(data)

    def plotly_chart(self, fig, **kw):
        self.figures.append(fig)

    def pyplot(self, fig, **kw):
        self.figures.append(fig)

    def image(self, img, **kw):
        self._rec("image", kw)

    def metric(self, label, value, **kw):
        self._rec("metric", (label, value))

    def cache_data(self, *a, **kw):
        if len(a) == 1 and callable(a[0]) and not kw:
            return a[0]
        return lambda f: f

    cache_resource = cache_data


def install():
    """Sahte modülleri sys.modules'a yerleştirir ve FakeStreamlit örneğini döndürür."""
    st = FakeStreamlit()
    sys.modules["streamlit"] = st
    sys.modules["streamlit.errors"] = st.errors
    folium = MagicMock(name="folium")
    plugins = MagicMock(name="folium.plugins")
    raster = MagicMock(name="folium.raster_layers")
    folium.plugins = plugins
    folium.raster_layers = raster
    sys.modules.update({"folium": folium, "folium.plugins": plugins, "folium.raster_layers": raster})
    plotly = MagicMock(name="plotly")
    go = MagicMock(name="plotly.graph_objects")
    plotly.graph_objects = go
    sys.modules.update({"plotly": plotly, "plotly.graph_objects": go})
    sf = types.ModuleType("streamlit_folium")

    def st_folium(fmap, **kw):
        st.folium_calls.append((fmap, kw))
        return st.map_return.get(kw.get("key"), {})

    sf.st_folium = st_folium
    sys.modules["streamlit_folium"] = sf
    return st
