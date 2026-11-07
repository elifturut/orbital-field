"""
app.py — No-Code Çok Yönlü Uydu ve Saha Analiz Platformu (B2B SaaS prototipi)
=============================================================================

Dört sektöre (Sigorta, Tarım, Belediye, Afet/Kurtarma) tek arayüzden hizmet veren,
sade, NASA esinli koyu temalı (Inter tipografi, ince çizgiler, tek kırmızı vurgu) Streamlit uygulaması.

* Arayüz: ``st.markdown`` ile enjekte edilen özel CSS (uzay siyahı ``#0b0c10``, koyu gri paneller,
  ince çizgiler, sakin mavi vurgu; NASA kırmızısı yalnızca uyarılarda), üzerine gelince hafifçe
  parlayan HTML/CSS KPI kartları.
* Haritalar: ``folium`` + ``streamlit-folium``; altlık seçenekleri ``basemaps.py``'de (koyu CARTO, Esri uydu,
  günlük NASA VIIRS "canlı uydu", EOX Sentinel-2 mozaik). Karoları tarayıcı çeker, sunucu istek yapmaz.
* Grafikler: ``plotly.graph_objects`` — hepsi ``plotly_dark`` şablonu ve tamamen saydam arka plan.
* Veri: ``data_engine.py`` içindeki Gauss dağılımlı sentetik simülatörler (dış veri API'si yok).

* Giriş ve roller: ``auth.py`` (PBKDF2 parola özeti, deneme sınırı, boşta kalma zaman aşımı, rol → izin).
  **Sigortacı** tüm sekmeleri ve müşteri portalı önizlemesini; **müşteri** yalnızca kendi poliçe ve hasar
  durumunu görür (``portal.py``: satır ve sütun süzmesi, sade dilli gerekçe). Demo hesapları ekranda yazar;
  gerçek hesaplar ``ORBITAL_ACCOUNTS_JSON`` ortam değişkeniyle ya da ``st.secrets`` ile verilir (bu durumda demo hesapları kapanır).

Çalıştırma::

    pip install -r requirements.txt
    streamlit run app.py

Her sekme ayrı bir fonksiyondur: ``render_insurance_tab``, ``render_casefiles_tab``,
``render_calibration_tab``, ``render_agriculture_tab``, ``render_municipality_tab`` ve ``render_rescue_tab``;
müşteri tek sayfa görür (``render_customer_page``).

Geliştirici: Elif Turut
"""

from __future__ import annotations

import datetime as dt
import html
import logging
import os
import re
import time
from dataclasses import astuple, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium import plugins
from folium.raster_layers import ImageOverlay
from streamlit.errors import StreamlitAPIException
from streamlit_folium import st_folium

import analysis as an_core
import auth
import basemaps as bm
import calibration as cal
import casework as cw
import data_engine as de
import data_sources as src
import landing
import portal as pt
import reports as rep
import store as stm
import tenancy
import workflow_page as wpage
from data_engine import ClaimRules, format_tr

LOGGER = logging.getLogger("cok_yonlu_platform")

# --------------------------------------------------------------------------- #
# Renkler ve sabitler
# --------------------------------------------------------------------------- #
COLOR_BG = "#0b0c10"
COLOR_ACCENT = "#6ea8fe"        # sakin mavi vurgu
COLOR_AMBER = "#f2b84b"         # dikkat
COLOR_DANGER = "#fc3d21"        # NASA kırmızısı — yalnızca kritik / uyarı
COLOR_OK = "#4cc38a"
COLOR_MUTED = "#8b93a1"
FONT_FAMILY = "Inter, 'Helvetica Neue', Arial, sans-serif"
GRID_COLOR = "rgba(255,255,255,0.06)"

BASEMAP_OPTIONS: Tuple[str, ...] = bm.LABELS

#: dNBR şiddet sınıfı renkleri (0=yanmamış … 4=yüksek).
SEVERITY_COLORS: Tuple[str, ...] = ("#2f8f83", "#dcc060", "#ec9a4d", "#e0452f", "#9a2f7a")
DECISION_COLORS: Dict[str, str] = {
    "ONAY": "#4cc38a", "KISMİ ONAY": "#6ea8fe", "EKSPERTİZ": "#f2b84b", "RET": "#fc3d21",
}
DECISION_ICONS: Dict[str, str] = {
    "ONAY": "✓ ONAY", "KISMİ ONAY": "◐ KISMİ ONAY", "EKSPERTİZ": "○ EKSPERTİZ", "RET": "✕ RET",
}
RISK_STOPS: Tuple[Tuple[float, str], ...] = (
    (0.0, "#2f8f83"), (35.0, "#dcc060"), (60.0, "#ec9a4d"), (80.0, "#e0452f"), (100.0, "#9a2f7a"),
)
HEALTH_COLORS: Dict[str, str] = {
    "Sağlıklı": "#4cc38a", "İzlemede": "#e3c35a", "Stresli": "#ec9a4d", "Kritik": "#fc3d21",
}
NDVI_STOPS: Tuple[Tuple[float, str], ...] = (
    (0.0, "#7a4b2a"), (0.25, "#c9a227"), (0.5, "#8ac926"), (0.75, "#2a9d4a"), (0.9, "#0b6e4f"),
)
STRESS_STOPS: Tuple[Tuple[float, str], ...] = (
    (0.0, "#2f8f83"), (0.3, "#dcc060"), (0.55, "#ec9a4d"), (1.0, "#e0452f"),
)
RISK_COLORS: Dict[str, str] = {
    "Düşük": "#4cc38a", "Orta": "#f2b84b", "Yüksek": "#f5821f", "Kritik": "#fc3d21",
}
ROUTE_COLORS: Dict[str, str] = {"Açık": "#4cc38a", "Riskli": "#f2b84b", "Kapalı": "#fc3d21"}
UNIT_COLORS: Dict[str, str] = {
    "İtfaiye": "#f0705a", "AFAD": "#6ea8fe", "Orman Yangın": "#5fc98a",
    "112 Sağlık": "#f1f3f5", "Jandarma": "#a78bfa",
}
CROP_COLORS: Dict[str, str] = {c.name: c.colour for c in de.CROPS}

SECTORS: Tuple[str, ...] = ("SİGORTA", "TARIM", "BELEDİYE", "KURTARMA")


@dataclass(frozen=True)
class MarketContext:
    """Türkiye devlet destekli tarım sigortası 2025 özeti.

    Değerler Tarım ve Orman Bakanlığı verilerine dayanan bir haberden alınmıştır; uygulama bu
    rakamları hesaplamaz, yalnızca pazar bağlamı olarak gösterir.

    Attributes:
        year: Veri yılı.
        policies: Toplam poliçe sayısı.
        insured_area_decare: Sigortalı bitkisel üretim alanı (dekar).
        paid_claims_tl: Toplam hasar ödemesi (TL).
        state_support_tl: Devlet prim desteği (TL).
        source: Kaynak açıklaması.
    """

    year: int = 2025
    policies: int = 3_807_514
    insured_area_decare: int = 44_329_788
    paid_claims_tl: float = 34_044_934_324.0
    state_support_tl: float = 24_675_882_205.0
    source: str = "Tarım ve Orman Bakanlığı verilerine dayanan haber (alomaliye.com, 27.03.2026)"


MARKET = MarketContext()


# --------------------------------------------------------------------------- #
# Özel CSS (sade, NASA esinli koyu tema) — st.markdown ile enjekte edilir
# --------------------------------------------------------------------------- #
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500;600;700&display=swap');

:root{
  --bg:#0b0c10; --surface:#10131a; --surface2:#151a22;
  --line:rgba(255,255,255,.09); --line2:rgba(255,255,255,.17);
  --text:#e6e8ec; --soft:#cfd4dc; --muted:#8b93a1; --dim:#69717f;
  --accent:#6ea8fe; --red:#fc3d21; --amber:#f2b84b; --ok:#4cc38a;
  --radius:12px;
}

/* ---------- Genel zemin ve tipografi ---------- */
.stApp{ background:var(--bg); color:var(--text); }
.stApp, .stApp p, .stApp label, .stApp li, .stApp button, .stApp input, .stApp textarea{
  font-family:'Inter','Helvetica Neue',Arial,system-ui,sans-serif; font-feature-settings:"tnum" 1,"cv11" 1;
}
[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer{ visibility:hidden; }
.block-container{ padding-top:1.6rem; padding-bottom:2.5rem; max-width:1440px; }
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-track{ background:var(--bg); }
::-webkit-scrollbar-thumb{ background:#232832; border-radius:8px; border:2px solid var(--bg); }
::-webkit-scrollbar-thumb:hover{ background:#3a4150; }
[data-testid="stCaptionContainer"]{ color:var(--muted); }
[data-testid="stWidgetLabel"] p{
  font:500 .72rem 'Inter',sans-serif; letter-spacing:.08em; text-transform:uppercase; color:var(--muted);
}

/* ---------- Yan menü ---------- */
section[data-testid="stSidebar"]{ background:#0d0f14; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] .block-container{ padding-top:1rem; }
.brand{ display:flex; align-items:center; gap:12px; padding:4px 2px 16px; border-bottom:1px solid var(--line); margin-bottom:6px; }
.brand-mark{ position:relative; flex:none; width:30px; height:30px; border-radius:50%; border:1.5px solid var(--text); }
.brand-mark::after{ content:""; position:absolute; top:-3px; right:1px; width:7px; height:7px; border-radius:50%; background:var(--red); }
.brand-name{ font:700 .8rem 'Inter',sans-serif; letter-spacing:.2em; color:#fff; }
.brand-sub{ font:500 .6rem 'Inter',sans-serif; letter-spacing:.16em; color:var(--muted); margin-top:3px; }
.side-label{ display:flex; align-items:center; gap:10px; margin:22px 0 6px; font:600 .64rem 'Inter',sans-serif; letter-spacing:.18em; color:var(--muted); }
.side-label b{ color:var(--dim); font-weight:600; }
.side-label::after{ content:""; flex:1; height:1px; background:var(--line); }
.sys-card{
  margin-top:20px; padding:12px 14px; border-radius:10px; border:1px solid var(--line); background:var(--surface);
  font:400 .72rem/1.85 ui-monospace,'SF Mono',Menlo,Consolas,monospace; color:var(--muted);
}
.sys-card b{ color:var(--soft); font-weight:500; display:inline-block; min-width:72px; }

/* ---------- Başlık (hero) ---------- */
.hero{
  display:flex; justify-content:space-between; align-items:flex-end; gap:18px; flex-wrap:wrap;
  padding:6px 0 22px; border-bottom:1px solid var(--line);
}
.hero-eyebrow{ font:600 .66rem 'Inter',sans-serif; letter-spacing:.22em; color:var(--red); }
.hero-title{ font:600 2rem/1.15 'Inter',sans-serif; letter-spacing:-.015em; margin:.5rem 0 .45rem; color:#fff; }
.hero-title span{ color:var(--muted); font-weight:400; }
.hero-sub{ color:var(--muted); font-size:.98rem; line-height:1.55; max-width:760px; }
.chips{ display:flex; gap:8px; flex-wrap:wrap; }
.chip{
  font:500 .68rem 'Inter',sans-serif; letter-spacing:.08em; padding:5px 11px; border-radius:999px;
  border:1px solid var(--line2); color:var(--muted); white-space:nowrap; text-transform:uppercase;
}
.chip.live::before{ content:""; display:inline-block; width:6px; height:6px; border-radius:50%; background:var(--ok); margin-right:8px; vertical-align:1px; }

/* ---------- KPI kartları ---------- */
.kpi-grid{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:18px 0 6px; }
@media (max-width:1100px){ .kpi-grid{ grid-template-columns:repeat(2,minmax(0,1fr)); } }
.kpi{
  position:relative; padding:16px 18px 15px; border-radius:var(--radius); border:1px solid var(--line);
  background:var(--surface); transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}
.kpi::before{ content:""; position:absolute; left:18px; top:-1px; width:26px; height:2px; background:var(--kpi,var(--line2)); }
.kpi:hover{
  transform:translateY(-1px); border-color:rgba(110,168,254,.45);
  box-shadow:0 0 0 1px rgba(110,168,254,.10), 0 10px 28px rgba(0,0,0,.35), 0 0 22px rgba(110,168,254,.10);
}
.kpi.alert{ border-color:rgba(252,61,33,.42); }
.kpi.alert:hover{ border-color:rgba(252,61,33,.7); box-shadow:0 0 0 1px rgba(252,61,33,.12), 0 10px 28px rgba(0,0,0,.35), 0 0 22px rgba(252,61,33,.12); }
.kpi-label{ font:500 .66rem 'Inter',sans-serif; letter-spacing:.1em; color:var(--muted); text-transform:uppercase; }
.kpi-value{
  margin-top:.55rem; font:600 clamp(1.15rem,1.65vw,1.6rem)/1.15 'Inter',sans-serif; letter-spacing:-.01em;
  color:var(--kpi,#fff); overflow-wrap:anywhere;
}
.kpi-sub{ margin-top:.4rem; font:400 .78rem 'Inter',sans-serif; color:var(--dim); }
.kpi-sub b{ color:var(--soft); font-weight:500; }

/* ---------- Bölüm başlıkları, özet, not, altbilgi ---------- */
.sec-title{ display:flex; align-items:center; gap:14px; margin:30px 0 12px; font:600 .7rem 'Inter',sans-serif; letter-spacing:.16em; color:var(--soft); }
.sec-title::after{ content:""; order:1; flex:1; height:1px; background:var(--line); }
.sec-title small{ order:2; font:400 .72rem 'Inter',sans-serif; letter-spacing:0; color:var(--dim); }
.brief{
  margin-top:14px; padding:12px 16px; border-left:2px solid var(--line2); background:var(--surface);
  border-radius:0 10px 10px 0; font:400 .88rem/1.75 'Inter',sans-serif; color:#b4bcc8;
}
.brief .k{ color:var(--muted); font:600 .66rem 'Inter',sans-serif; letter-spacing:.12em; margin-right:10px; }
.note{
  margin-top:12px; padding:10px 14px; border-radius:10px; border:1px solid rgba(242,184,75,.35);
  background:rgba(242,184,75,.06); color:#efcf8e; font-size:.92rem;
}
.note.err{ border-color:rgba(252,61,33,.5); background:rgba(252,61,33,.07); color:#ff9c8c; }
.note.info{ border-color:var(--line2); background:var(--surface); color:var(--soft); }
.note.ok{ border-color:rgba(76,195,138,.4); background:rgba(76,195,138,.06); color:#9be2bd; }
.footer{
  margin-top:38px; padding:16px 0 4px; border-top:1px solid var(--line); display:flex; justify-content:space-between;
  gap:12px; flex-wrap:wrap; font:400 .74rem 'Inter',sans-serif; color:var(--dim);
}
.footer b{ color:var(--muted); font-weight:500; }

/* ---------- Sekmeler ---------- */
.stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--line); }
.stTabs button[data-baseweb="tab"]{ background:transparent; color:var(--muted); padding:10px 18px; border-radius:8px 8px 0 0; }
.stTabs button[data-baseweb="tab"] p{ font:600 .76rem 'Inter',sans-serif; letter-spacing:.14em; }
.stTabs button[data-baseweb="tab"]:hover{ color:var(--text); }
.stTabs button[data-baseweb="tab"][aria-selected="true"]{ color:#fff; }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--red); height:2px; }
.stTabs [data-baseweb="tab-border"]{ background:transparent; }

/* ---------- Girdi bileşenleri ---------- */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"]{
  background-color:var(--surface) !important; border:1px solid var(--line2) !important; border-radius:8px !important; color:var(--text) !important;
}
div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover{ border-color:var(--accent) !important; }
[data-baseweb="popover"] ul, [role="listbox"]{ background:var(--surface2) !important; border:1px solid var(--line2); border-radius:10px; }
[role="option"]:hover{ background:rgba(110,168,254,.10) !important; }
div[data-baseweb="slider"] div[role="slider"]{ background:var(--accent) !important; border-color:var(--accent) !important; box-shadow:none; }
.stRadio [role="radiogroup"] label{ padding:5px 10px; border-radius:8px; border:1px solid transparent; transition:all .15s; }
.stRadio [role="radiogroup"] label:hover{ border-color:var(--line); background:rgba(255,255,255,.03); }
[data-testid="stExpander"]{ border:1px solid var(--line); border-radius:10px; background:var(--surface); }
div[data-testid="stVerticalBlockBorderWrapper"]{ border-color:var(--line) !important; border-radius:var(--radius) !important; background:var(--surface); }

/* ---------- Düğmeler ---------- */
.stDownloadButton button, [data-testid="stDownloadButton"] button, .stButton button{
  width:100%; color:var(--text); background:transparent; border:1px solid var(--line2); border-radius:8px;
  padding:.55rem 1rem; transition:all .15s;
}
.stDownloadButton button p, [data-testid="stDownloadButton"] button p, .stButton button p{ color:inherit; font:600 .74rem 'Inter',sans-serif; letter-spacing:.1em; }
.stDownloadButton button:hover, [data-testid="stDownloadButton"] button:hover, .stButton button:hover{
  border-color:var(--accent); color:#fff; background:rgba(110,168,254,.08);
}

/* ---------- Grafik, tablo, harita ve açıklama kartları ---------- */
.stPlotlyChart, [data-testid="stPlotlyChart"]{ border:1px solid var(--line); border-radius:var(--radius); padding:6px 8px; overflow:hidden; background:var(--surface); }
[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:10px; overflow:hidden; background:var(--surface); }
iframe{ border-radius:var(--radius); }
.legend{
  display:flex; flex-wrap:wrap; align-items:center; gap:6px 18px; margin:8px 0 2px; padding:8px 14px;
  border:1px solid var(--line); border-radius:10px; background:var(--surface); font:400 .74rem 'Inter',sans-serif; color:var(--muted);
}
.legend b{ color:var(--soft); font:600 .64rem 'Inter',sans-serif; letter-spacing:.14em; }
.legend i{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:7px; vertical-align:-1px; }
.legend i.dot{ border-radius:50%; }
.map-cap{ margin:0 0 8px; font:600 .66rem 'Inter',sans-serif; letter-spacing:.14em; color:var(--muted); }
.map-cap b{ color:#fff; font-weight:600; }

.kpi-grid.c3{ grid-template-columns:repeat(3,minmax(0,1fr)); }
.kpi-grid.c5{ grid-template-columns:repeat(5,minmax(0,1fr)); }
.kpi-grid.c6{ grid-template-columns:repeat(6,minmax(0,1fr)); }
@media (max-width:1280px){ .kpi-grid.c5, .kpi-grid.c6{ grid-template-columns:repeat(3,minmax(0,1fr)); } }
@media (max-width:820px){ .kpi-grid, .kpi-grid.c3, .kpi-grid.c5, .kpi-grid.c6{ grid-template-columns:repeat(1,minmax(0,1fr)); } }

/* ---------- Genel Bakış: akış ve aşama kartları ---------- */
.flow{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:6px 0 4px; }
@media (max-width:1100px){ .flow{ grid-template-columns:1fr; } }
.step{ padding:16px 18px 17px; border:1px solid var(--line); border-radius:var(--radius); background:var(--surface); }
.step .n{ font:600 .64rem 'Inter',sans-serif; letter-spacing:.16em; color:var(--muted); }
.step .t{ margin:.55rem 0 .4rem; font:600 1.02rem 'Inter',sans-serif; color:#fff; }
.step p{ margin:0; font:400 .86rem/1.6 'Inter',sans-serif; color:var(--muted); }
.step .tag{ display:inline-block; margin-top:.85rem; padding:2px 9px; border-radius:999px; border:1px solid var(--line2);
  font:500 .62rem 'Inter',sans-serif; letter-spacing:.1em; color:var(--muted); }
.step.core{ border-color:rgba(110,168,254,.38); }
.step.core .n, .step.core .tag{ color:var(--accent); border-color:rgba(110,168,254,.38); }

/* ---------- Giriş, kullanıcı kartı ve müşteri portalı ---------- */
.who{ margin:12px 0 8px; padding:11px 14px; border:1px solid var(--line); border-radius:10px; background:var(--surface); }
.who .r{ font:600 .6rem 'Inter',sans-serif; letter-spacing:.18em; color:var(--accent); }
.who .n{ margin-top:4px; font:600 .9rem 'Inter',sans-serif; color:#fff; overflow-wrap:anywhere; }
.who .u{ margin-top:2px; font:400 .72rem ui-monospace,Menlo,Consolas,monospace; color:var(--muted); }
.who .c{ margin-top:8px; padding-top:7px; border-top:1px solid var(--line); font:500 .76rem 'Inter',sans-serif; color:var(--soft);
  overflow-wrap:anywhere; }
.who .c i{ display:block; margin-top:1px; font:400 .7rem 'Inter',sans-serif; font-style:normal; color:var(--muted); }
.profile{ display:flex; gap:18px; align-items:center; padding:18px 20px; margin-bottom:10px; }
.profile .av{ flex:none; width:58px; height:58px; border-radius:16px; display:flex; align-items:center; justify-content:center;
  font:700 1.15rem 'Inter',sans-serif; letter-spacing:.04em; color:#0b0c10; background:var(--accent); }
.profile .nm{ font:600 1.2rem 'Inter',sans-serif; color:#fff; overflow-wrap:anywhere; }
.profile .tt{ margin-top:2px; font:400 .88rem 'Inter',sans-serif; color:var(--muted); }
.profile .co{ display:inline-block; margin-top:8px; padding:3px 11px; border:1px solid rgba(110,168,254,.38); border-radius:999px;
  font:600 .68rem 'Inter',sans-serif; letter-spacing:.12em; color:var(--accent); }
.verdict .by{ margin-top:12px; padding-top:10px; border-top:1px solid var(--line); font:400 .82rem/1.6 'Inter',sans-serif;
  color:var(--muted); }
.verdict .by b{ color:var(--soft); font-weight:600; }
.verdict .cnote{ margin-top:10px; padding:10px 12px; border-left:2px solid var(--accent); background:var(--surface2);
  border-radius:6px; font:400 .9rem/1.6 'Inter',sans-serif; color:var(--text); overflow-wrap:anywhere; }
.login-head{ padding:34px 0 14px; }
.login-head .hero-title{ font-size:1.7rem; }
.demo-accounts{ margin-top:14px; padding:12px 14px; border:1px dashed var(--line2); border-radius:10px; color:var(--muted);
  font:400 .8rem/1.9 'Inter',sans-serif; }
.demo-accounts b{ color:var(--soft); font-weight:600; }
.demo-accounts code{ font:500 .76rem ui-monospace,Menlo,Consolas,monospace; color:var(--text); background:var(--surface2);
  padding:1px 6px; border-radius:5px; }
.verdict{ padding:16px 18px; border:1px solid var(--line); border-left:3px solid var(--vc,var(--line2)); border-radius:var(--radius);
  background:var(--surface); }
.verdict .st{ font:600 .64rem 'Inter',sans-serif; letter-spacing:.16em; color:var(--vc,var(--muted)); text-transform:uppercase; }
.verdict .hl{ margin:.5rem 0 .4rem; font:600 1.2rem/1.3 'Inter',sans-serif; color:#fff; }
.verdict p{ margin:0; font:400 .92rem/1.65 'Inter',sans-serif; color:var(--muted); }
ol.tracker{ list-style:none; margin:14px 0 0; padding:0; }
ol.tracker li{ display:flex; gap:14px; padding:10px 2px; border-bottom:1px solid var(--line); }
ol.tracker .dot{ flex:none; width:22px; height:22px; border-radius:50%; border:1.5px solid var(--dim); text-align:center;
  font:600 .68rem/19px 'Inter',sans-serif; color:var(--dim); }
ol.tracker .done .dot{ background:var(--ok); border-color:var(--ok); color:#0b0c10; }
ol.tracker .current .dot{ border-color:var(--accent); color:var(--accent); box-shadow:0 0 0 3px rgba(110,168,254,.16); }
ol.tracker .closed .dot{ background:var(--muted); border-color:var(--muted); color:#0b0c10; }
ol.tracker b{ display:block; font:600 .9rem 'Inter',sans-serif; color:#fff; }
ol.tracker .d{ display:block; margin-top:2px; font:400 .82rem/1.5 'Inter',sans-serif; color:var(--muted); }
ol.tracker .waiting b{ color:var(--muted); }

/* ---------- Cam (glass) katmanı: harita odaklı, hap gezinme, ince tipografi ---------- */
:root{ --glass:rgba(18,22,29,.66); --edge:rgba(255,255,255,.085); --edge2:rgba(255,255,255,.17); --r-lg:20px; }
.stApp{
  background:
    radial-gradient(1100px 520px at 88% -8%, rgba(96,140,120,.13), transparent 62%),
    radial-gradient(900px 480px at -8% 108%, rgba(110,168,254,.08), transparent 58%),
    #090b0f;
}
section[data-testid="stSidebar"]{ background:rgba(10,12,16,.88); -webkit-backdrop-filter:blur(18px); backdrop-filter:blur(18px); }
.kpi, .step, .brief, .legend, .sys-card, .who, .note, .demo-accounts, .verdict, .profile{
  background:linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.012)), var(--glass);
  -webkit-backdrop-filter:blur(16px) saturate(1.15); backdrop-filter:blur(16px) saturate(1.15);
  border:1px solid var(--edge); border-radius:var(--r-lg);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.05), 0 8px 30px rgba(0,0,0,.28);
}
.legend{ border-radius:999px; padding:8px 18px; }
.brief{ border-left:1px solid var(--edge); }
.kpi::before{ left:20px; }
.kpi-value{ font-weight:300; font-size:clamp(1.4rem,2.1vw,2.05rem); letter-spacing:-.02em; }
.kpi-value .dim, .mo-stat .v .dim{ opacity:.45; }
.kpi-value.long{ font-size:clamp(1.05rem,1.55vw,1.5rem); white-space:nowrap; }
.hero{ border-bottom:none; padding:2px 0 8px; }
.hero-title{ font-weight:300; font-size:clamp(1.9rem,3vw,2.7rem); letter-spacing:-.025em; }
.hero-title span{ font-weight:300; color:rgba(255,255,255,.38); }
.chip{ background:rgba(255,255,255,.04); -webkit-backdrop-filter:blur(10px); backdrop-filter:blur(10px); border-color:var(--edge2); }
.sec-title{ margin-top:34px; }
.stTabs [data-baseweb="tab-list"]{
  gap:4px; padding:5px; width:fit-content; max-width:100%; overflow-x:auto; margin-bottom:10px;
  border:1px solid var(--edge); border-radius:999px; background:var(--glass);
  -webkit-backdrop-filter:blur(16px); backdrop-filter:blur(16px);
}
.stTabs button[data-baseweb="tab"]{ padding:8px 18px; border-radius:999px; }
.stTabs button[data-baseweb="tab"][aria-selected="true"]{ background:rgba(255,255,255,.14); box-shadow:inset 0 1px 0 rgba(255,255,255,.14); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{ display:none; }
.stDownloadButton button, [data-testid="stDownloadButton"] button, .stButton button, .stFormSubmitButton button,
[data-testid="stFormSubmitButton"] button{
  width:100%; border-radius:999px; background:rgba(255,255,255,.05); border:1px solid var(--edge2);
  -webkit-backdrop-filter:blur(10px); backdrop-filter:blur(10px);
}
.stFormSubmitButton button p, [data-testid="stFormSubmitButton"] button p{ color:inherit; font:600 .74rem 'Inter',sans-serif; letter-spacing:.1em; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"]{ border-radius:14px !important; }
[data-testid="stExpander"], [data-testid="stDataFrame"], .stPlotlyChart, [data-testid="stPlotlyChart"],
div[data-testid="stVerticalBlockBorderWrapper"]{ border-radius:var(--r-lg) !important; }
[data-testid="stExpander"], div[data-testid="stVerticalBlockBorderWrapper"]{
  background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.01)), var(--glass);
  border:1px solid var(--edge) !important;
}
iframe{ border-radius:var(--r-lg); }
ol.tracker li{ border-bottom-color:var(--edge); }
"""


# --------------------------------------------------------------------------- #
# Küçük yardımcılar
# --------------------------------------------------------------------------- #
def _h(value: object) -> str:
    """Değeri HTML için kaçışlar.

    Args:
        value: Herhangi bir değer.

    Returns:
        Güvenli HTML metni.
    """
    return html.escape(str(value), quote=True)


def _compact(markup: str) -> str:
    """Çok satırlı HTML'i tek satıra indirger.

    Markdown, 4+ boşlukla girintili satırları kod bloğu sayar; bu işlem bunu önler.

    Args:
        markup: Çok satırlı HTML.

    Returns:
        Tek satırlık HTML.
    """
    return " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())


def render_html(markup: str) -> None:
    """HTML parçasını Streamlit'e güvenle basar.

    Args:
        markup: Basılacak HTML.
    """
    st.markdown(_compact(markup), unsafe_allow_html=True)


def inject_css() -> None:
    """Özel CSS temasını sayfaya enjekte eder."""
    st.markdown(f"<style>{CUSTOM_CSS}{landing.LANDING_CSS}</style>", unsafe_allow_html=True)


def _supports_width_stretch() -> bool:
    """Kurulu Streamlit sürümü ``width="stretch"`` API'sini destekliyor mu (≥ 1.50)?

    Returns:
        Destekliyorsa ``True``.
    """
    try:
        major, minor = (int(part) for part in st.__version__.split(".")[:2])
    except (ValueError, AttributeError):
        return False
    return (major, minor) >= (1, 50)


def stretch(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Bir Streamlit öğesini kap genişliğine yayar; sürüm farklarına dayanıklıdır.

    Yeni sürümlerde ``width="stretch"``, eskilerinde ``use_container_width=True``
    kullanılır; biri reddedilirse diğeri denenir.

    Args:
        function: ``st.plotly_chart``, ``st.dataframe``, ``st.download_button`` vb.
        *args: Konumsal argümanlar.
        **kwargs: Anahtar argümanlar.

    Returns:
        Çağrılan fonksiyonun dönüş değeri.

    Raises:
        RuntimeError: İki yöntem de reddedilirse.
    """
    attempts: List[Dict[str, Any]] = [{"width": "stretch"}, {"use_container_width": True}]
    if not _supports_width_stretch():
        attempts.reverse()
    last_error: Optional[Exception] = None
    for extra in attempts:
        try:
            return function(*args, **kwargs, **extra)
        except (TypeError, StreamlitAPIException) as exc:
            last_error = exc
    raise RuntimeError("Streamlit öğesi genişliğe yayılamadı.") from last_error


def panel() -> Any:
    """Kenarlıklı kontrol paneli kabı döndürür (eski sürümlerde sade kap).

    Returns:
        Bir ``with`` bloğunda kullanılabilir Streamlit kabı.
    """
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def ramp_color(value: float, stops: Sequence[Tuple[float, str]]) -> str:
    """Değeri renk duraklarına göre doğrusal (RGB) enterpolasyonla renge çevirir.

    Args:
        value: Girdi değeri.
        stops: ``(değer, "#rrggbb")`` çiftleri (artan değer sırasıyla).

    Returns:
        ``"#rrggbb"`` renk dizgesi.
    """
    xs = np.array([s[0] for s in stops], dtype=float)
    rgb = np.array([[int(s[1][i : i + 2], 16) for i in (1, 3, 5)] for s in stops], dtype=float)
    out = [int(round(float(np.interp(value, xs, rgb[:, k])))) for k in range(3)]
    return "#{:02x}{:02x}{:02x}".format(*out)


def hex_rgba(color: str, alpha: float) -> str:
    """``#rrggbb`` rengini ``rgba(r,g,b,a)`` dizgesine çevirir.

    Args:
        color: Hex renk.
        alpha: Saydamlık (0–1).

    Returns:
        CSS ``rgba`` dizgesi.
    """
    c = color.lstrip("#")
    return f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{alpha})"


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Tabloyu Excel (TR) uyumlu CSV baytlarına çevirir (UTF-8 BOM, ``;`` ayracı, ``,`` ondalık).

    Args:
        frame: Dışa aktarılacak tablo.

    Returns:
        CSV baytları.
    """
    return frame.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


# --------------------------------------------------------------------------- #
# Ayarlar (yan menü → sekmeler)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AppSettings:
    """Yan menüden gelen genel ayarlar.

    Attributes:
        seed: Sentetik veri tohumu.
        basemap: Seçili harita altlığı ve (tarihli katmanlar için) görüntü günü.
    """

    seed: int
    basemap: bm.BasemapChoice


def get_settings() -> AppSettings:
    """Genel ayarları ``st.session_state``'ten okur.

    Returns:
        ``AppSettings`` örneği (henüz yan menü çizilmediyse varsayılanlarla).
    """
    return AppSettings(
        seed=int(st.session_state.get("seed", 2026)),
        basemap=bm.make_choice(st.session_state.get("basemap", BASEMAP_OPTIONS[0]),
                               st.session_state.get("basemap_date")),
    )


# --------------------------------------------------------------------------- #
# HTML bileşenleri
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Kpi:
    """Tek bir KPI kartının içeriği.

    Attributes:
        label: Kart etiketi.
        value: Büyük değer metni.
        sub: Alt açıklama (HTML içerebilir; çağıran kaçışlamalıdır).
        color: Vurgu rengi (boş = nötr beyaz; yalnızca dikkat/uyarı kartlarında kullanılır).
        alert: ``True`` ise kart kırmızı çerçeve alır.
    """

    label: str
    value: str
    sub: str = ""
    color: str = ""
    alert: bool = False


def dim_decimals(text: str) -> str:
    """Metni HTML için kaçışlar ve içindeki ondalık kısmı (``,8`` / ``,229``) soluk gösterir.

    Büyük, ince rakamlarda tam kısım öne çıkar, ondalık geri çekilir (ör. ``%70<soluk>,8</soluk>``).

    Args:
        text: Değer metni (Türkçe sayı biçimi: ondalık ayracı virgül).

    Returns:
        Kaçışlanmış HTML.
    """
    return re.sub(r"(,\d+)(?!\d)", r'<span class="dim">\1</span>', _h(text))


def build_kpi_grid(items: Sequence[Kpi], columns: int = 4) -> str:
    """Üzerine gelince parlayan KPI kartı ızgarasının HTML'ini üretir.

    Args:
        items: KPI listesi.
        columns: Sütun sayısı (3, 4, 5 ya da 6).

    Returns:
        HTML dizgesi.
    """
    cls = {3: " c3", 5: " c5", 6: " c6"}.get(columns, "")
    cards = []
    for k in items:
        sub = f'<div class="kpi-sub">{k.sub}</div>' if k.sub else ""
        alert = " alert" if k.alert else ""
        style = f' style="--kpi:{k.color}"' if k.color else ""
        cards.append(
            f'<div class="kpi{alert}"{style}>'
            f'<div class="kpi-label">{_h(k.label)}</div>'
            f'<div class="kpi-value{" long" if len(k.value) > 12 else ""}">{dim_decimals(k.value)}</div>{sub}</div>'
        )
    return f'<div class="kpi-grid{cls}">{"".join(cards)}</div>'


def build_brand_html() -> str:
    """Yan menü marka bloğunun HTML'ini üretir.

    Returns:
        HTML dizgesi.
    """
    return """
    <div class="brand">
      <div class="brand-mark"></div>
      <div><div class="brand-name">ORBITAL FIELD</div>
      <div class="brand-sub">EARTH OBSERVATION PLATFORM</div></div>
    </div>"""


def build_hero_html(eyebrow: str, title: str, accent: str, subtitle: str, chips: Sequence[str]) -> str:
    """Sektör sekmesi başlık (hero) bloğunu üretir.

    Args:
        eyebrow: Üst küçük etiket.
        title: Başlığın düz kısmı.
        accent: Başlığın soluk (gri) vurgulu kısmı.
        subtitle: Alt açıklama.
        chips: Sağdaki hap etiketleri.

    Returns:
        HTML dizgesi.
    """
    chip_html = "".join(f'<span class="chip">{_h(c)}</span>' for c in chips)
    return f"""
    <div class="hero">
      <div><div class="hero-eyebrow">{_h(eyebrow)}</div>
      <div class="hero-title">{_h(title)} <span>{_h(accent)}</span></div>
      <div class="hero-sub">{_h(subtitle)}</div></div>
      <div class="chips">{chip_html}</div>
    </div>"""


def build_section_title(title: str, hint: str = "") -> str:
    """Bölüm başlığı HTML'i.

    Args:
        title: Başlık.
        hint: Sağdaki küçük ipucu.

    Returns:
        HTML dizgesi.
    """
    small = f"<small>{_h(hint)}</small>" if hint else ""
    return f'<div class="sec-title">{_h(title)}{small}</div>'


def build_legend_html(title: str, items: Sequence[Tuple[str, str]], dot: bool = False) -> str:
    """Harita/grafik altı renk açıklaması (lejant) HTML'i.

    Args:
        title: Lejant başlığı.
        items: ``(renk, etiket)`` çiftleri.
        dot: ``True`` ise yuvarlak işaretçi.

    Returns:
        HTML dizgesi.
    """
    cls = ' class="dot"' if dot else ""
    body = "".join(f'<span><i{cls} style="background:{c}"></i>{_h(t)}</span>' for c, t in items)
    return f'<div class="legend"><b>{_h(title)}</b>{body}</div>'


def build_step_cards(items: Sequence[Tuple[str, str, str, str]], core: int = 0) -> str:
    """Numaralı adım/aşama kartları ızgarasının HTML'ini üretir.

    Args:
        items: ``(üst etiket, başlık, açıklama, etiket)`` dörtlüleri.
        core: İlk kaç kartın "çekirdek" (mavi çerçeveli) vurgusu alacağı.

    Returns:
        HTML dizgesi.
    """
    cards = []
    for i, (kicker, title, text, tag) in enumerate(items):
        tag_html = f'<span class="tag">{_h(tag)}</span>' if tag else ""
        cards.append(
            f'<div class="step{" core" if i < core else ""}"><div class="n">{_h(kicker)}</div>'
            f'<div class="t">{_h(title)}</div><p>{_h(text)}</p>{tag_html}</div>'
        )
    return f'<div class="flow">{"".join(cards)}</div>'


def build_footer_html() -> str:
    """Sayfa altbilgisinin HTML'ini üretir.

    Returns:
        HTML dizgesi.
    """
    now = dt.datetime.now(de.TR_TZ).strftime("%d.%m.%Y %H:%M")
    return f"""
    <div class="footer">
      <span><b>ORBITAL FIELD</b> · Çok Yönlü Uydu ve Saha Analiz Platformu · Prototip v1.0</span>
      <span>Varsayılan veriler sentetiktir · {now} (UTC+3)</span>
      <span>Geliştirici: <b>Elif Turut</b></span>
    </div>"""


def render_brief(lines: Sequence[str]) -> None:
    """Sade, numaralı özet kutusunu basar.

    Args:
        lines: Gösterilecek satırlar (HTML kaçışlanır).
    """
    body = "<br>".join(f'<span class="k">{i:02d}</span>{_h(x)}' for i, x in enumerate(lines, start=1))
    render_html(f'<div class="brief">{body}</div>')


# --------------------------------------------------------------------------- #
# Önbellekli veri yükleyiciler (yalnızca basit argümanlarla anahtarlanır)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_insurance(seed: int) -> de.InsuranceDataset:
    """Sigorta veri setini üretir ve önbelleğe alır.

    Args:
        seed: Veri tohumu.

    Returns:
        ``InsuranceDataset``.
    """
    return de.InsuranceSimulator(seed=seed).simulate()


# --------------------------------------------------------------------------- #
# Şirket verisi ve karar defteri
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CompanyData:
    """Bir sigortacının çalıştığı şirkete ait karar verisi.

    Attributes:
        ds: Sigorta veri seti (TÜM şirketlerin geometrisi; yalnızca konum hesabı için).
        auto: Tam tablo, yalnızca otomatik motor kararları (kanıt belgesi ve motor doğrulaması bunu kullanır).
        final: Tam tablo, sigortacı kararları uygulanmış (geometriyle aynı sıra).
        mine: ``final`` içinden YALNIZCA sigortacının şirketinin poliçeleri. İndeksi geometri konumudur.
        company: Şirket kimliği.
        error: Karar defteri okunamadıysa nedeni (boş = sorun yok); kararlar o zaman yalnızca otomatik gösterilir.
    """

    ds: de.InsuranceDataset
    auto: pd.DataFrame
    final: pd.DataFrame
    mine: pd.DataFrame
    company: str
    error: str


def read_decisions(company: Optional[str], seed: int) -> Tuple[Dict[str, cw.Decision], str]:
    """Karar defterinden geçerli kararları okur.

    Args:
        company: Sigortacı için şirket kimliği (yalnızca o şirketin kayıtları). ``None`` ise müşteri ekranı için tüm
            şirketlerin kararları, İÇ NOT VE KARARI VEREN BİLGİSİ BOŞALTILMIŞ olarak döner.
        seed: Veri tohumu.

    Returns:
        ``({poliçe: karar}, hata_iletisi)``; defter açılamazsa boş sözlük ve neden.
    """
    try:
        db = stm.shared_store()
        return (db.effective(company, seed) if company else db.public_effective(seed)), ""
    except (stm.StoreError, cw.CaseworkError) as exc:
        LOGGER.warning("Karar defteri okunamadı: %s", exc)
        return {}, str(exc)


def company_data(principal: Optional[auth.Principal], seed: int, rules: ClaimRules) -> CompanyData:
    """Sigortacının şirketi için veri, otomatik kararlar ve sigortacı kararlarını hazırlar.

    Args:
        principal: Oturum açmış sigortacı.
        seed: Veri tohumu.
        rules: Otomatik karar kuralları.

    Returns:
        ``CompanyData``.

    Raises:
        PermissionError: Oturum sigortacı değilse ya da geçerli bir şirkete bağlı değilse.
    """
    auth.require(principal, auth.PERM_INSURANCE)
    assert principal is not None
    if not tenancy.valid_company(principal.company):
        raise PermissionError("Hesabınız geçerli bir sigorta şirketine bağlı değil.")
    company = str(principal.company)
    ds, auto = evaluate_for(seed, rules)
    decisions, error = read_decisions(company, seed)
    final = cw.apply_decisions(auto, decisions)
    mine = final[final["sirket"] == company]
    return CompanyData(ds=ds, auto=auto, final=final, mine=mine, company=company, error=error)


@st.cache_data(show_spinner=False)
def load_agriculture(seed: int, obs_doy: int) -> de.AgricultureDataset:
    """Tarım veri setini üretir ve önbelleğe alır.

    Args:
        seed: Veri tohumu.
        obs_doy: Gözlem günü (yılın günü).

    Returns:
        ``AgricultureDataset``.
    """
    return de.AgricultureSimulator(seed=seed).simulate(obs_doy)


@st.cache_data(show_spinner=False)
def load_municipality(seed: int, t0: int, t1: int) -> de.MunicipalityDataset:
    """Belediye veri setini üretir ve önbelleğe alır.

    Args:
        seed: Veri tohumu.
        t0: Başlangıç yılı.
        t1: Karşılaştırma yılı.

    Returns:
        ``MunicipalityDataset``.
    """
    return de.MunicipalitySimulator(seed=seed).simulate(t0, t1)


@st.cache_data(show_spinner=False)
def load_rescue_scenario(seed: int) -> de.RescueScenario:
    """Afet senaryosunu (sabit kısım) üretir ve önbelleğe alır.

    Args:
        seed: Veri tohumu.

    Returns:
        ``RescueScenario``.
    """
    return de.RescueSimulator(seed=seed).build_scenario()


@st.cache_data(show_spinner=False)
def load_rescue_snapshot(seed: int, t_hours: int) -> de.RescueSnapshot:
    """Belirli saatin kriz görüntüsünü üretir ve önbelleğe alır.

    Args:
        seed: Veri tohumu.
        t_hours: Başlangıçtan geçen saat.

    Returns:
        ``RescueSnapshot``.
    """
    return de.RescueSimulator(seed=seed).snapshot(load_rescue_scenario(seed), t_hours)


@st.cache_data(show_spinner=False)
def load_rescue_timeline(seed: int) -> pd.DataFrame:
    """Tüm saatler için yangın/nüfus KPI zaman serisini hesaplar.

    Args:
        seed: Veri tohumu.

    Returns:
        ``saat``, ``aktif``, ``frp``, ``etkilenen``, ``kritik``, ``yanan_ha`` sütunlu tablo.
    """
    sim = de.RescueSimulator(seed=seed)
    sc = load_rescue_scenario(seed)
    rows = []
    for t in range(0, sc.horizon_h + 1):
        k = sim.snapshot(sc, t).kpis
        rows.append({"saat": t, "aktif": k["active_hotspots"], "frp": k["frp_total_mw"],
                     "etkilenen": k["affected_pop"], "kritik": k["critical_zones"],
                     "yanan_ha": k["burned_ha"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plotly stili — plotly_dark + tamamen saydam arka plan
# --------------------------------------------------------------------------- #
def style_figure(fig: go.Figure, height: int = 380, title: str = "", legend_top: bool = True) -> go.Figure:
    """Şekle ortak sade koyu stili uygular (``plotly_dark``, saydam zemin).

    Args:
        fig: Biçimlendirilecek şekil.
        height: Piksel yüksekliği.
        title: Grafik başlığı.
        legend_top: ``True`` ise lejant grafiğin üstünde yatay durur.

    Returns:
        Aynı ``go.Figure`` nesnesi.
    """
    legend = (
        dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0,
             bgcolor="rgba(0,0,0,0)", font=dict(size=11))
        if legend_top else dict(bgcolor="rgba(0,0,0,0)")
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=52, r=28, t=64 if title else 40, b=48),
        title=dict(text=title, x=0.01, xanchor="left",
                   font=dict(family=FONT_FAMILY, size=12, color="#cfd4dc")) if title else None,
        font=dict(family=FONT_FAMILY, size=13, color="#b4bcc8"),
        legend=legend,
        hoverlabel=dict(bgcolor="#151a22", bordercolor="rgba(255,255,255,0.22)", font=dict(family=FONT_FAMILY, size=13)),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor="rgba(255,255,255,0.16)")
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor="rgba(255,255,255,0.16)")
    return fig


def show_figure(fig: go.Figure, key: str) -> None:
    """Plotly şeklini kap genişliğinde çizer.

    Args:
        fig: Çizilecek şekil.
        key: Benzersiz widget anahtarı.
    """
    stretch(st.plotly_chart, fig, key=key, config={"displaylogo": False, "responsive": True})


def show_table(frame: pd.DataFrame, config: Optional[Dict[str, Any]] = None, height: int = 360) -> None:
    """Tabloyu ``st.dataframe`` ile kap genişliğinde gösterir.

    Args:
        frame: Gösterilecek tablo.
        config: ``st.column_config`` sözlüğü.
        height: Piksel yüksekliği.
    """
    stretch(st.dataframe, frame, hide_index=True, column_config=config or {}, height=height)


def show_html_frame(markup: str, height: int = 780) -> None:
    """Bağımsız bir HTML belgesini yalıtılmış çerçevede (iframe) gösterir.

    Belge kendi (açık) temasını taşır; uygulamanın koyu CSS'iyle karışmaması için çerçeve kullanılır.
    Kurulu Streamlit çerçeve bileşenini sunmuyorsa kısa bir bilgi notu gösterilir.

    Args:
        markup: Tam HTML belgesi.
        height: Piksel yüksekliği.
    """
    frame = getattr(getattr(getattr(st, "components", None), "v1", None), "html", None)
    if frame is None:
        render_html('<div class="note info">Bu Streamlit sürümünde belge önizlemesi yok; '
                    "belgeyi indirip tarayıcıda açın.</div>")
        return
    frame(markup, height=height, scrolling=True)


# --------------------------------------------------------------------------- #
# Folium yardımcıları
# --------------------------------------------------------------------------- #
POPUP_CSS = """
<style>
.leaflet-popup-content-wrapper{background:#10131a;color:#cfd4dc;border:1px solid rgba(255,255,255,.16);
  border-radius:10px;box-shadow:0 10px 28px rgba(0,0,0,.55)}
.leaflet-popup-tip{background:#10131a;border:1px solid rgba(255,255,255,.16)}
.leaflet-popup-close-button{color:#8b93a1 !important}
.leaflet-tooltip{background:#10131a;color:#e6e8ec;border:1px solid rgba(255,255,255,.18);border-radius:6px;
  font:12px Inter,'Segoe UI',sans-serif;box-shadow:none}
.leaflet-control-layers{background-color:#10131a !important;color:#e6e8ec !important;border-color:#232832 !important}
.leaflet-bar{border:none !important;box-shadow:none !important;background:transparent !important}
.leaflet-bar a{width:34px !important;height:34px !important;line-height:34px !important;border-radius:50% !important;margin-bottom:8px;background-color:rgba(16,20,27,.68) !important;-webkit-backdrop-filter:blur(10px);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,.14) !important;color:#e6e8ec !important}
.leaflet-bar a:hover{background-color:rgba(255,255,255,.16) !important}
.leaflet-control-layers{border-radius:18px !important;border:1px solid rgba(255,255,255,.14) !important;background-color:rgba(16,20,27,.72) !important;-webkit-backdrop-filter:blur(10px);backdrop-filter:blur(10px)}
.leaflet-top,.leaflet-bottom{z-index:1200 !important}
.leaflet-control-attribution{background:rgba(10,12,16,.55) !important;color:#8b93a1 !important;border-radius:10px 0 0 0;font-size:10px}
.leaflet-control-attribution a{color:#a9b3c2 !important}
.leaflet-control-scale-line{background:rgba(10,12,16,.55) !important;color:#cfd4dc !important;border-color:#8b93a1 !important}
.pp{font:12.5px/1.5 Inter,'Segoe UI',system-ui,sans-serif;min-width:210px}
.pp h4{margin:0 0 8px;font:600 13px Inter,'Segoe UI',sans-serif;letter-spacing:.02em;color:#fff}
.pp table{border-collapse:collapse;width:100%}
.pp td{padding:2px 0;vertical-align:top}.pp td:first-child{color:#8b93a1;padding-right:12px;white-space:nowrap}
.pp td:last-child{text-align:right;color:#e6e8ec}
.pp .bar{height:4px;border-radius:3px;background:#232832;margin:2px 0 7px;overflow:hidden}
.pp .bar i{display:block;height:100%;border-radius:3px}
.pp .tag{display:inline-block;padding:1px 8px;border-radius:999px;font-size:10.5px;color:#0b0c10;font-weight:600}
</style>
"""


def popup_html(title: str, rows: Sequence[Tuple[str, str]], bars: Sequence[Tuple[str, float, str]] = (),
               tag: Optional[Tuple[str, str]] = None) -> str:
    """Koyu temalı harita pop-up HTML'i üretir (tüm değerler kaçışlanır).

    Args:
        title: Pop-up başlığı.
        rows: ``(etiket, değer)`` satırları.
        bars: ``(etiket, 0–1 oran, renk)`` mini çubukları.
        tag: İsteğe bağlı ``(metin, renk)`` durum etiketi.

    Returns:
        HTML dizgesi.
    """
    tag_html = f'<span class="tag" style="background:{tag[1]}">{_h(tag[0])}</span>' if tag else ""
    bar_html = "".join(
        f'<div style="font-size:11px;color:#8b93a1">{_h(lbl)}</div>'
        f'<div class="bar"><i style="width:{max(0.0, min(1.0, frac)) * 100:.0f}%;background:{col}"></i></div>'
        for lbl, frac, col in bars
    )
    body = "".join(f"<tr><td>{_h(a)}</td><td>{_h(b)}</td></tr>" for a, b in rows)
    return f'<div class="pp"><h4>{_h(title)} {tag_html}</h4>{bar_html}<table>{body}</table></div>'


#: Uydu altlıklarında (VIIRS, Esri, EOX) görüntüyü hafif karartıp doygunluğu düşürür; üstteki paneller okunur kalır.
SATELLITE_TONE_CSS = "<style>.leaflet-tile-pane{filter:brightness(.8) saturate(.88) contrast(1.06)}</style>"

MAP_OVERLAY_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500;600;700&display=swap');
.mo{position:fixed;inset:0;z-index:1100;pointer-events:none;font-family:Inter,'Segoe UI',system-ui,sans-serif;color:#e6e8ec}
.mo::before,.mo::after{content:"";position:absolute;left:0;right:0;height:170px}
.mo::before{top:0;background:linear-gradient(180deg,rgba(6,8,11,.74),rgba(6,8,11,0))}
.mo::after{bottom:0;background:linear-gradient(0deg,rgba(6,8,11,.68),rgba(6,8,11,0))}
.mo-title{position:absolute;left:64px;top:14px;max-width:min(56%,560px);z-index:1}
.mo-eyebrow{font:600 10px Inter,sans-serif;letter-spacing:.2em;color:#fc3d21}
.mo-h{margin-top:6px;font:300 clamp(20px,3vw,34px)/1.1 Inter,sans-serif;letter-spacing:-.02em;color:#fff;text-shadow:0 2px 18px rgba(0,0,0,.6)}
.mo-h span{color:rgba(255,255,255,.52)}
.mo-chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}
.mo-chip{pointer-events:auto;font:500 10.5px Inter,sans-serif;letter-spacing:.06em;padding:5px 11px;border-radius:999px;color:#cfd4dc;
  background:rgba(16,20,27,.62);-webkit-backdrop-filter:blur(10px);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,.14)}
.mo-stats{position:absolute;left:18px;bottom:34px;display:flex;gap:10px;flex-wrap:wrap;max-width:calc(100% - 36px);z-index:1}
.mo-stat{pointer-events:auto;min-width:112px;padding:11px 15px 12px;border-radius:18px;background:rgba(16,20,27,.64);
  -webkit-backdrop-filter:blur(14px) saturate(1.2);backdrop-filter:blur(14px) saturate(1.2);border:1px solid rgba(255,255,255,.1);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.07),0 10px 30px rgba(0,0,0,.35)}
.mo-stat .l{font:500 9.5px Inter,sans-serif;letter-spacing:.12em;color:#9aa3b1;text-transform:uppercase}
.mo-stat .l i{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--c);margin-right:7px;vertical-align:0}
.mo-stat .v{margin-top:6px;font:300 26px/1.05 Inter,sans-serif;letter-spacing:-.02em;color:#fff}
.mo-stat .s{margin-top:3px;font:400 10.5px Inter,sans-serif;color:#8b93a1}
</style>"""


def build_map_overlay(eyebrow: str, title: str, accent: str = "", chips: Sequence[str] = (),
                      stats: Sequence[Tuple[str, str, str, str]] = ()) -> str:
    """Haritanın içine gömülen yüzen paneller: sol üstte ince büyük başlık + hap etiketler, sol altta cam özet kartları.

    Paneller harita iframe'inin içindedir (Streamlit bileşenlerinin üstüne taşmaz); üst ve alt kenarda okunurluk için
    hafif karartma vardır. Yalnızca kartların kendisi fare olaylarını alır, harita sürüklenebilir kalır.

    Args:
        eyebrow: Başlık üstü küçük etiket.
        title: Başlığın ana kısmı.
        accent: Başlığın soluk devamı.
        chips: Başlık altındaki hap etiketler.
        stats: ``(etiket, değer, alt metin, renk)`` özet kartları; renk boş değilse etiketin yanında nokta olur.

    Returns:
        ``<style>`` + panel HTML'i (``folium.Element`` içine verilir; tüm metin kaçışlanır).
    """
    chip_html = f'<div class="mo-chips">{"".join(f"<span class=mo-chip>{_h(c)}</span>" for c in chips)}</div>' if chips else ""
    head = f"{_h(title)} <span>{_h(accent)}</span>" if accent else _h(title)
    cards = []
    for label, value, sub, color in stats:
        dot = "<i></i>" if color else ""
        small = f'<div class="s">{_h(sub)}</div>' if sub else ""
        cards.append(f'<div class="mo-stat" style="--c:{_h(color or "#8b93a1")}"><div class="l">{dot}{_h(label)}</div>'
                     f'<div class="v">{dim_decimals(value)}</div>{small}</div>')
    stat_html = f'<div class="mo-stats">{"".join(cards)}</div>' if cards else ""
    return (f'{MAP_OVERLAY_CSS}<div class="mo"><div class="mo-title"><div class="mo-eyebrow">{_h(eyebrow)}</div>'
            f'<div class="mo-h">{head}</div>{chip_html}</div>{stat_html}</div>')


def make_base_map(centre: Tuple[float, float], zoom: int, basemap: Optional[bm.BasemapChoice] = None,
                  overlay: str = "") -> folium.Map:
    """Koyu temalı temel haritayı kurar; tüm altlıkları katman denetimine ekler, seçili olanı açar.

    Karoları tarayıcı çeker: kapalı katmanlar için ağ isteği yapılmaz. Tarihli katmanın (NASA VIIRS) günü
    ``basemap.day`` ile verilir ve hem katman adında hem köşedeki kaynak notunda görünür.

    Args:
        centre: ``(enlem, boylam)`` merkez.
        zoom: Başlangıç yakınlaştırma düzeyi.
        basemap: Seçili altlık; ``None`` ise koyu harita.
        overlay: ``build_map_overlay`` çıktısı (boşsa yüzen panel yok).

    Returns:
        ``folium.Map`` nesnesi.
    """
    choice = basemap or bm.make_choice(None)
    fmap = folium.Map(location=list(centre), zoom_start=zoom, tiles=None, control_scale=True,
                      prefer_canvas=True, zoom_control=True)
    for spec in bm.BASEMAPS:
        folium.TileLayer(
            tiles=bm.tile_url(spec, choice.day), attr=bm.attribution(spec, choice.day),
            name=bm.layer_name(spec, choice.day), subdomains=spec.subdomains, max_zoom=spec.max_zoom,
            max_native_zoom=spec.max_native_zoom, show=spec.key == choice.key,
        ).add_to(fmap)
    plugins.Fullscreen(position="topleft", title="Tam ekran", title_cancel="Çık").add_to(fmap)
    plugins.MousePosition(position="bottomright", separator=" | ", num_digits=4, prefix="Konum:").add_to(fmap)
    fmap.get_root().header.add_child(folium.Element(POPUP_CSS))
    if choice.is_satellite:
        fmap.get_root().header.add_child(folium.Element(SATELLITE_TONE_CSS))
    if overlay:
        fmap.get_root().html.add_child(folium.Element(overlay))
    return fmap


def show_map(fmap: folium.Map, key: str, height: int = 540) -> None:
    """Folium haritasını Streamlit'e basar; haritadaki etkileşimler sayfayı yeniden çalıştırmaz.

    Args:
        fmap: Çizilecek harita.
        key: Benzersiz widget anahtarı.
        height: Piksel yüksekliği.
    """
    try:
        st_folium(fmap, height=height, use_container_width=True, returned_objects=[], key=key)
    except TypeError:
        st_folium(fmap, height=height, returned_objects=[], key=key)


def fit_bounds(fmap: folium.Map, lats: Sequence[float], lons: Sequence[float], pad: float = 0.002) -> None:
    """Haritayı verilen koordinat kümesini kapsayacak şekilde ayarlar.

    Args:
        fmap: Harita.
        lats: Enlemler.
        lons: Boylamlar.
        pad: Kenar payı (derece).
    """
    fmap.fit_bounds([[min(lats) - pad, min(lons) - pad], [max(lats) + pad, max(lons) + pad]])


def draw_region_map(current: Optional[an_core.Region], key: str) -> Optional[Dict[str, Any]]:
    """Haritada çokgen / dikdörtgen çizdirir ve çizilen son çokgenin GeoJSON geometrisini döndürür.

    Çizim araçları sol üsttedir. Seçili bölge varsa haritada gösterilir ve harita ona odaklanır; yoksa Türkiye görünür.

    Args:
        current: Şu anki bölge (yoksa ``None``).
        key: Benzersiz widget anahtarı (anahtar değişince çizimler sıfırlanır).

    Returns:
        ``Polygon`` geometrisi ya da henüz çizim yoksa ``None``.
    """
    centre, zoom = (current.centre, 14) if current is not None else ((39.0, 35.0), 6)
    fmap = make_base_map(centre, zoom, get_settings().basemap)
    if current is not None:
        folium.GeoJson(current.geojson(), name="Seçili bölge",
                       style_function=lambda _f: {"color": COLOR_ACCENT, "weight": 2, "fillOpacity": 0.12}).add_to(fmap)
    shape = {"shapeOptions": {"color": COLOR_ACCENT, "weight": 2}}
    plugins.Draw(export=False, position="topleft",
                 draw_options={"polyline": False, "circle": False, "marker": False, "circlemarker": False,
                               "polygon": {"allowIntersection": False, **shape}, "rectangle": shape},
                 edit_options={"edit": False, "remove": True}).add_to(fmap)
    try:
        out = st_folium(fmap, height=460, use_container_width=True, returned_objects=["all_drawings"], key=key)
    except TypeError:
        out = st_folium(fmap, height=460, returned_objects=["all_drawings"], key=key)
    for feat in reversed((out or {}).get("all_drawings") or []):
        geom = feat.get("geometry") if isinstance(feat, dict) else None
        if isinstance(geom, dict) and geom.get("type") == "Polygon":
            return geom
    return None


# =========================================================================== #
# SEKME 0 — GENEL BAKIŞ (odak, çalışma akışı, yol haritası, veri hattı)
# =========================================================================== #
def build_pipeline_frame() -> pd.DataFrame:
    """Veri hattı kaynaklarını (``data_sources.PIPELINE``) gösterim tablosuna çevirir.

    Returns:
        Türkçe sütun adlı tablo.
    """
    return pd.DataFrame([
        {"Kaynak": r.name, "Ürün": r.product, "Platformda kullanım": r.use,
         "Bu prototipte": r.prototype, "Canlı adaptör": r.adapter, "Erişim": r.access}
        for r in src.PIPELINE
    ])


def render_overview_tab() -> None:
    """GENEL BAKIŞ sekmesini çizer: pazar bağlamı, çalışma akışı, yol haritası, veri hattı ve şeffaflık notu."""
    auth.require(session_principal(), auth.PERM_OVERVIEW)
    render_html(build_hero_html(
        "PROJE ÖZETİ", "UYDU ANALİZİ,", "KARAR TABLOSUNA.",
        "Parsel bazlı uydu indekslerini sigortacının ve tarım ekibinin kendi kurallarına bağlayan, kod yazmadan "
        "kullanılan analiz platformu.",
        ["DEMO · SENTETİK VERİ", "PROTOTİP v1.0"]))
    render_html(
        '<div class="note info"><b>Varsayılan veriler sentetiktir.</b> Gerçek Sentinel-2 (Earth Search), NASA FIRMS ve Open-Meteo '
        "bağlantıları <b>Analiz iş akışı</b> sekmesinde “Canlı” kaynak seçeneğiyle denenebilir. Bu adaptörler sahte yanıtlarla "
        "çevrimdışı test edildi; canlı uç noktalara karşı henüz doğrulanmadı (sekmedeki “Canlı bağlantıyı sına” düğmesi bunu yapar).</div>")

    render_html(build_section_title("PAZAR BAĞLAMI · TÜRKİYE TARIM SİGORTASI", f"{MARKET.year} · devlet destekli"))
    render_html(build_kpi_grid([
        Kpi("Poliçe", f"{format_tr(MARKET.policies / 1e6, 2)} M", "devlet destekli tarım sigortası"),
        Kpi("Sigortalı alan", f"{format_tr(MARKET.insured_area_decare / 1e6, 1)} M dekar", "bitkisel üretim"),
        Kpi("Hasar ödemesi", f"₺{format_tr(MARKET.paid_claims_tl / 1e9, 1)} mlr", "yıllık toplam"),
        Kpi("Devlet prim desteği", f"₺{format_tr(MARKET.state_support_tl / 1e9, 1)} mlr", "yıllık toplam"),
    ]))
    st.caption(f"Kaynak: {MARKET.source}. Bu rakamlar uygulamada hesaplanmaz, yalnızca bağlam olarak gösterilir.")

    render_html(build_section_title("ÇALIŞMA AKIŞI", "her sektörde aynı çekirdek"))
    render_html(build_step_cards([
        ("ADIM 01", "Parsel ve olay", "Parsel sınırları ve olay tarihi seçilir. Prototipte sentetik parsel ızgarası, "
         "canlı modda müşterinin parsel dosyası kullanılır.", ""),
        ("ADIM 02", "İndeks ve sınıf", "NDVI, NBR ve olay öncesi/sonrası dNBR hesaplanır; parseller USGS dNBR "
         "şiddet sınıflarına ayrılır.", ""),
        ("ADIM 03", "Kural, karar ve kanıt", "Muafiyet, teminat, avans ve eşikler arayüzden girilir; her poliçe için "
         "onay, kısmi onay, ekspertiz ya da ret çıkar. Kurallar geçmiş kararlara karşı sınanır, her karar için "
         "kanıt paketi üretilir.", ""),
    ]))

    render_html(build_section_title("YOL HARİTASI", "çekirdek pazar önce"))
    render_html(build_step_cards([
        ("AŞAMA 1 · ÇEKİRDEK", "Tarım sigortası ve izleme", "Parsel hasar doğrulama, fenoloji, su stresi ve rekolte. "
         "Pilot hedefi: bir sigortacıyla gerçek parsel verisi üzerinde çalışmak.", "SİGORTA · TARIM"),
        ("AŞAMA 2 · GENİŞLEME", "Belediye", "İki tarihli görüntüden kaçak yapılaşma ve yeşil alan kaybı; "
         "mahalle bazlı önceliklendirme.", "BELEDİYE"),
        ("AŞAMA 3 · GENİŞLEME", "Afet ve kurtarma", "VIIRS/MODIS termal anomali, tahliye rotaları ve saha "
         "ekibi koordinasyonu.", "KURTARMA"),
    ], core=1))

    render_html(build_section_title("VERİ HATTI", "gerçek veriye geçiş"))
    with st.expander("Veri kaynakları ve canlı moda geçiş"):
        show_table(build_pipeline_frame(), height=232)
        cred = src.load_credentials()
        cdse_state = "tanımlı" if cred.has_cdse else "tanımlı değil"
        firms_state = "tanımlı" if cred.has_firms else "tanımlı değil"
        render_html(
            '<div class="note info">Canlı moda geçmek için ortam değişkenleri: <b>CDSE_CLIENT_ID</b>, '
            "<b>CDSE_CLIENT_SECRET</b>, <b>FIRMS_MAP_KEY</b>. Şu an: "
            f"CDSE <b>{cdse_state}</b> · FIRMS <b>{firms_state}</b>. "
            "Değerler ekranda gösterilmez.</div>")

    render_html(build_section_title("NELER GERÇEK, NELER DEĞİL"))
    render_brief([
        "İndeks formülleri (NDVI, NBR, dNBR) ve sınıflama eşikleri gerçektir; girdi bantları ve parseller sentetiktir.",
        "Karar kuralları örnek eşiklerdir; bir sigortacının poliçe şartlarıyla henüz doğrulanmadı.",
        "Geriye dönük test ve kanıt paketi gerçek algoritmalardır; Kalibrasyon sekmesindeki örnek geçmiş ise "
        "sentetiktir ve doğruluk kanıtı değildir.",
        "Canlı veri adaptörleri çevrimdışı sahte oturumla test edildi; gerçek hesap anahtarıyla denenmedi.",
        "Giriş ve rol ayrımı demo düzeyindedir: parolalar PBKDF2 ile özetlenir ve rol kapıları sunucuda uygulanır; "
        "ancak kurumsal kimlik sağlayıcı, iki adımlı doğrulama ve oturum kayıt defteri yoktur. Müşteri ekranındaki "
        "otomatik kararlar varsayılan kurallarla hesaplanır, sigortacının yayımladığı kurallar değildir; sigortacı kararı "
        "verilen poliçelerde ise o karar olduğu gibi gösterilir.",
        "Sigorta şirketleri kurgusaldır ve poliçelerin şirketlere dağılımı sentetiktir (poliçe numarasından türetilir); "
        "bir müşterinin poliçeleri birden çok şirkette olabilir. Sigortacı yalnızca kendi şirketinin poliçelerini görür; bu "
        "ayrım sunucuda uygulanır, ancak çok kiracılı bir üretim sisteminin denetimlerinden geçmemiştir.",
        "Sigortacı kararları sunucudaki yerel bir SQLite dosyasında tutulur. Streamlit Community Cloud'da dosya sistemi "
        "kalıcı değildir: uygulama yeniden başlayınca kararlar silinir. Kalıcı kullanım için barındırılan bir veritabanı gerekir.",
        "Harita altlığı olarak seçilen NASA VIIRS görüntüsü gerçek ve günceldir (yaklaşık 250 m); üzerindeki "
        "parsel, hasar ve rotalar sentetiktir. Altlık yalnızca görsel bağlamdır, hiçbir hesaba girmez.",
    ])


# =========================================================================== #
# SEKME 1 — SİGORTA (Risk ve Hasar)
# =========================================================================== #
INS_MODES: Tuple[str, ...] = ("Hasar şiddeti (dNBR)", "Tazminat kararı", "Poliçe risk skoru")
#: Sol menüdeki dört eşik kaydırıcısının ``session_state`` anahtarları (ClaimRules alanı → anahtar).
INS_STATE_KEYS: Dict[str, str] = {
    "approve_dnbr": "ins_approve", "partial_dnbr": "ins_partial",
    "review_dnbr": "ins_review", "min_confidence": "ins_conf",
}


def effective_rules(approve: float, partial: float, review: float, confidence: float,
                    deductible: float, coverage: float) -> Tuple[ClaimRules, bool]:
    """Kaydırıcı değerlerinden tutarlı (ret < kısmi < tam) ``ClaimRules`` kurar.

    Args:
        approve: Tam onay eşiği.
        partial: Kısmi onay eşiği.
        review: Ret eşiği.
        confidence: Asgari gözlem güveni.
        deductible: Muafiyet (%).
        coverage: Teminat oranı (%).

    Returns:
        ``(kurallar, eşikler otomatik düzeltildi mi)``.
    """
    a, p, r, adjusted = cal.consistent_thresholds(float(approve), float(partial), float(review))
    rules = ClaimRules(approve_dnbr=a, partial_dnbr=p, review_dnbr=r, min_confidence=float(confidence),
                       deductible_pct=float(deductible), coverage_pct=float(coverage))
    return rules, adjusted


def current_rules_checked() -> Tuple[ClaimRules, bool]:
    """Kenar çubuğundaki karar kuralı kaydırıcılarının güncel değerlerinden ``ClaimRules`` üretir.

    Returns:
        ``(kurallar, eşikler otomatik düzeltildi mi)``; kaydırıcılar henüz çizilmediyse ``ClaimRules`` varsayılanları.
    """
    d, state = ClaimRules(), st.session_state
    return effective_rules(
        state.get("ins_approve", d.approve_dnbr), state.get("ins_partial", d.partial_dnbr),
        state.get("ins_review", d.review_dnbr), state.get("ins_conf", d.min_confidence),
        state.get("ins_ded", d.deductible_pct), state.get("ins_cover", d.coverage_pct))


def current_rules() -> ClaimRules:
    """Kenar çubuğundaki karar kurallarının güncel halini döndürür (``current_rules_checked`` yalnızca kurallar)."""
    return current_rules_checked()[0]


def _threshold_tuple(rules: ClaimRules) -> Tuple[float, float, float, float]:
    """Kalibre edilen dört eşiği, ``INS_STATE_KEYS`` sırasıyla ve 2 ondalığa yuvarlı döndürür."""
    return tuple(round(float(getattr(rules, field)), 2) for field in INS_STATE_KEYS)  # type: ignore[return-value]


def apply_calibrated_rules(values: Tuple[float, float, float, float], lines: Tuple[str, ...]) -> None:
    """``on_click`` geri çağrısı: önerilen eşikleri sol menüdeki kural kaydırıcılarına yazar.

    Geri çağrılar betik yeniden çalışmadan ÖNCE işlendiği için, başka sekmedeki
    kaydırıcıların ``session_state`` değerlerini değiştirmek güvenlidir.

    Args:
        values: ``(tam onay, kısmi onay, ret, güven)`` eşikleri.
        lines: Kanıt paketine yazılacak kalibrasyon bilgisi satırları.
    """
    for key, value in zip(INS_STATE_KEYS.values(), values):
        st.session_state[key] = round(float(value), 2)
    st.session_state["cal_applied"] = {"rules": tuple(round(float(v), 2) for v in values), "lines": tuple(lines)}


def reset_rules() -> None:
    """``on_click`` geri çağrısı: dört eşiği varsayılana döndürür ve uygulanmış kalibrasyonu unutur."""
    base = ClaimRules()
    for field, key in INS_STATE_KEYS.items():
        st.session_state[key] = round(float(getattr(base, field)), 2)
    st.session_state.pop("cal_applied", None)


def calibration_lines_for(rules: ClaimRules) -> Tuple[str, ...]:
    """Sol menüdeki kurallar hâlâ uygulanmış kalibrasyonla aynıysa kanıt paketi satırlarını döndürür.

    Args:
        rules: Geçerli kurallar.

    Returns:
        Satırlar; kaydırıcılar sonradan değiştirildiyse boş.
    """
    applied = st.session_state.get("cal_applied")
    if applied and _threshold_tuple(rules) == tuple(applied["rules"]):
        return tuple(applied["lines"])
    return ()


# --------------------------------------------------------------------------- #
# Karar kanıt paketi (reports.py)
# --------------------------------------------------------------------------- #
def evidence_context(ds: de.InsuranceDataset, rules: ClaimRules, seed: int,
                     calibration_lines: Tuple[str, ...]) -> rep.EvidenceContext:
    """Kanıt paketi için ortak bağlamı kurar (bu prototipte veri sentetiktir).

    Args:
        ds: Sigorta veri seti.
        rules: Geçerli kurallar.
        seed: Veri tohumu.
        calibration_lines: Kalibrasyon bilgisi satırları.

    Returns:
        ``EvidenceContext``.
    """
    return rep.EvidenceContext(
        rules=rules, event_label=ds.event_label, event_date=ds.event_date,
        generated_at=dt.datetime.now(de.TR_TZ).replace(second=0, microsecond=0),
        severity_colors=SEVERITY_COLORS, synthetic=True,
        provenance=f"Sentetik demo verisi (Sentinel-2 L2A benzetimi) · tohum {seed}",
        calibration_lines=calibration_lines)


@st.cache_data(show_spinner="Kanıt paketi hazırlanıyor…")
def build_evidence_zip(seed: int, rules_tuple: Tuple[float, ...], police_nos: Tuple[str, ...],
                       calibration_lines: Tuple[str, ...]) -> bytes:
    """Seçili poliçeler için ZIP kanıt paketini üretir ve önbelleğe alır.

    Args:
        seed: Veri tohumu.
        rules_tuple: ``astuple(ClaimRules)``.
        police_nos: Pakete girecek poliçeler.
        calibration_lines: Kalibrasyon bilgisi satırları.

    Returns:
        ZIP baytları.
    """
    ds = load_insurance(seed)
    rules = ClaimRules(*rules_tuple)
    claims = de.InsuranceSimulator.evaluate_claims(ds.parcels, rules)
    return rep.build_pack_zip(claims, police_nos, evidence_context(ds, rules, seed, calibration_lines), ds.geometry)


def _prepare_zip(key: Tuple[Any, ...]) -> None:
    """``on_click`` geri çağrısı: toplu paketin hazırlanmasını ister (indirme düğmesi bu bayrakla görünür)."""
    st.session_state["ins_zip_ready"] = key


def render_evidence_section(ds: de.InsuranceDataset, claims: pd.DataFrame, view: pd.DataFrame,
                            rules: ClaimRules, seed: int) -> None:
    """Sigorta sekmesindeki KARAR KANIT PAKETİ bölümünü çizer: seçim, önizleme, tekli ve toplu indirme.

    Args:
        ds: Sigorta veri seti.
        claims: Tam tablo, YALNIZCA otomatik motor kararlarıyla (belge motoru kararı bağımsız doğrular; sigortacı
            kararı bu tabloda yoktur).
        view: Filtrelenmiş/sıralanmış görünüm (paket kapsamı; yalnızca sigortacının kendi şirketinin poliçeleri).
        rules: Geçerli kurallar.
        seed: Veri tohumu.

    Raises:
        PermissionError: Oturum sigortacı değilse (kanıt paketi iç ölçüm ve eşik içerir).
    """
    auth.require(session_principal(), auth.PERM_EVIDENCE)
    render_html(build_section_title("KARAR KANIT PAKETİ", "her kararın nasıl çıktığını belgeleyin"))
    if view.empty:
        render_html('<div class="note info">Filtreye uyan poliçe yok; kanıt paketi için filtreyi genişletin.</div>')
        return
    lines = calibration_lines_for(rules)
    ctx = evidence_context(ds, rules, seed, lines)
    options = list(view["police_no"])
    if st.session_state.get("ins_pack_pick") not in options:
        st.session_state["ins_pack_pick"] = options[0]
    p1, p2 = st.columns([3, 1])
    pick = p1.selectbox("Poliçe", options, key="ins_pack_pick")
    preview = p2.checkbox("Belgeyi önizle", value=True, key="ins_pack_preview")
    name, document = rep.build_claim_document(claims, str(pick), ctx, ds.geometry)
    if preview:
        show_html_frame(document, height=1180)
    d1, d2 = st.columns(2)
    d1.download_button("↓  BU KARAR İÇİN KANIT BELGESİ (HTML)", data=document.encode("utf-8"), file_name=name,
                       mime="text/html", key="ins_pack_one")
    scope = tuple(options[:rep.MAX_PACK_FILES])
    zip_key = (seed, astuple(rules), scope, lines)
    if st.session_state.get("ins_zip_ready") == zip_key:
        d2.download_button(f"↓  TOPLU PAKET · {len(scope)} KARAR (ZIP)",
                           data=build_evidence_zip(seed, astuple(rules), scope, lines),
                           file_name="karar_kanit_paketi.zip", mime="application/zip", key="ins_pack_zip")
    else:
        d2.button(f"Toplu paketi hazırla · {len(scope)} karar (ZIP)", on_click=_prepare_zip, args=(zip_key,),
                  key="ins_zip_prepare")
    cut = " Tablo filtreniz çok geniş: ilk %d karar pakete alındı." % rep.MAX_PACK_FILES if len(options) > rep.MAX_PACK_FILES else ""
    manual = int((view["karar_kaynagi"] == cw.SOURCE_INSURER).sum())
    render_brief([
        "Belge, otomatik motorun önerisini belgeler. "
        + (f"Listedeki {manual} poliçede sigortacı kararı geçerlidir; o kararlar belgede değil, Müşteri dosyaları → Geçmiş "
           "sekmesinde kayıtlıdır." if manual else "Sigortacı kararı verilen poliçelerin kararı Müşteri dosyaları sekmesinde kayıtlıdır."),
        "Belgedeki kural izi ve tutar hesabı motordan bağımsız yeniden yürütülür; motor çıktısıyla uyuşmazsa belge üretilmez.",
        "Toplu paket, yukarıdaki tablo filtresindeki tüm kararları içerir: her karar için HTML, bir özet CSV'si ve "
        "SHA-256 özetli manifest." + cut,
        "SHA-256 özeti içeriğin sonradan değişmediğini gösterir; imza ya da kimlik doğrulama değildir. "
        "Belgeler demo verisinden üretilmiştir ve bunu görünür biçimde belirtir.",
    ])


def build_insurance_map(ds: de.InsuranceDataset, claims: pd.DataFrame, mode: str, basemap: bm.BasemapChoice) -> folium.Map:
    """Parsel bazlı GeoJSON hasar katmanını içeren haritayı kurar.

    Args:
        ds: Sigorta veri seti (geometri için).
        claims: Karar sütunlu parsel tablosu. Tüm parselleri ya da bir şirketin altkümesini içerebilir; İNDEKS, satırın
            geometri konumudur (``evaluate_claims`` çıktısının varsayılan indeksi böyledir ve süzmede korunur).
        mode: Renklendirme modu (``INS_MODES`` öğelerinden biri).
        basemap: Seçili harita altlığı ve görüntü günü.

    Returns:
        Hazır ``folium.Map``.
    """
    frame = claims.copy()
    if mode == INS_MODES[0]:
        frame["renk"] = [SEVERITY_COLORS[int(i)] for i in frame["siddet_no"]]
    elif mode == INS_MODES[1]:
        frame["renk"] = frame["karar"].map(DECISION_COLORS)
    else:
        frame["renk"] = [ramp_color(float(v), RISK_STOPS) for v in frame["risk_skoru"]]
    frame["dnbr_txt"] = [format_tr(v, 3) for v in frame["dnbr"]]
    frame["bedel_txt"] = ["₺" + format_tr(v, 0) for v in frame["sigorta_bedeli_tl"]]
    frame["odeme_txt"] = ["₺" + format_tr(v, 0) for v in frame["odenecek_tl"]]
    frame["risk_txt"] = [format_tr(v, 1) for v in frame["risk_skoru"]]
    geojson = ds.geojson(frame, positions=[int(i) for i in claims.index])

    counts = claims["karar"].value_counts()
    overlay = build_map_overlay(
        "SİGORTA · HASAR KATMANI", "Parsel bazlı", "hasar analizi", (mode, f"olay sonrası {ds.event_date:%d.%m.%Y}"),
        tuple((k, str(int(counts.get(k, 0))), "parsel", c) for k, c in DECISION_COLORS.items()))
    fmap = make_base_map(ds.centre, 12, basemap, overlay)
    folium.GeoJson(
        geojson,
        name="Parsel hasar katmanı (dNBR)",
        style_function=lambda f: {"fillColor": f["properties"]["renk"], "color": "#0b0c10",
                                  "weight": 0.8, "fillOpacity": 0.74},
        highlight_function=lambda f: {"weight": 2.6, "color": COLOR_ACCENT, "fillOpacity": 0.92},
        tooltip=folium.GeoJsonTooltip(
            fields=["police_no", "dnbr_txt", "siddet", "karar"],
            aliases=["Poliçe", "dNBR", "Şiddet", "Karar"], sticky=True, localize=True),
        popup=folium.GeoJsonPopup(
            fields=["police_no", "sigortali", "varlik_turu", "bedel_txt", "dnbr_txt", "siddet",
                    "risk_txt", "karar", "gerekce", "odeme_txt"],
            aliases=["Poliçe no", "Sigortalı", "Varlık", "Sigorta bedeli", "dNBR", "Hasar şiddeti",
                     "Risk skoru", "Otomatik karar", "Gerekçe", "Ödenecek tutar"]),
    ).add_to(fmap)
    folium.LayerControl(collapsed=True, position="topright").add_to(fmap)
    fit_bounds(fmap, list(ds.geometry.corners_lat.ravel()), list(ds.geometry.corners_lon.ravel()), 0.003)
    return fmap


def build_risk_chart(claims: pd.DataFrame, top_n: int) -> go.Figure:
    """dNBR tabanlı poliçe risk skorlaması çubuk grafiğini üretir.

    Çubuk = risk skoru (renk: dNBR şiddet sınıfı); elmas işaretler = ikincil eksende dNBR.

    Args:
        claims: Karar sütunlu parsel tablosu.
        top_n: Gösterilecek en riskli poliçe sayısı.

    Returns:
        ``go.Figure``.
    """
    top = claims.nlargest(top_n, "risk_skoru").reset_index(drop=True)
    custom = np.stack([top["dnbr"], top["siddet"], top["karar"], top["varlik_turu"]], axis=-1)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top["police_no"], y=top["risk_skoru"], name="Risk skoru",
        marker=dict(color=[SEVERITY_COLORS[int(i)] for i in top["siddet_no"]],
                    line=dict(color="rgba(11,12,16,0.9)", width=1)),
        customdata=custom,
        hovertemplate=("<b>%{x}</b><br>Risk skoru: %{y:.1f}<br>dNBR: %{customdata[0]}<br>"
                       "Şiddet: %{customdata[1]}<br>Karar: %{customdata[2]}<br>Varlık: %{customdata[3]}<extra></extra>"),
    ))
    fig.add_trace(go.Scatter(
        x=top["police_no"], y=top["dnbr"], name="dNBR", yaxis="y2", mode="markers",
        marker=dict(symbol="diamond", size=9, color="#ffffff", line=dict(color=COLOR_ACCENT, width=1.5)),
        hovertemplate="<b>%{x}</b><br>dNBR: %{y:.3f}<extra></extra>",
    ))
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y", y0=75, y1=75,
                  line=dict(color=COLOR_DANGER, width=1.2, dash="dot"))
    fig.add_annotation(xref="paper", x=1, yref="y", y=75, text="ÇOK YÜKSEK RİSK ≥ 75", showarrow=False,
                       xanchor="right", yanchor="bottom", font=dict(size=10, color=COLOR_DANGER))
    style_figure(fig, 400, f"POLİÇE RİSK SKORU · dNBR TABANLI · EN RİSKLİ {top_n}")
    fig.update_xaxes(tickangle=-60, tickfont=dict(size=9))
    fig.update_yaxes(title_text="Risk skoru (0–100)", range=[0, 105])
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="dNBR", range=[-0.1, 1.0],
                                  showgrid=False, linecolor="rgba(255,255,255,0.16)"), bargap=0.25)
    return fig


def build_severity_chart(claims: pd.DataFrame) -> go.Figure:
    """USGS dNBR şiddet sınıflarına göre parsel sayısı ve alan dağılımı grafiği.

    Args:
        claims: Karar sütunlu parsel tablosu.

    Returns:
        ``go.Figure``.
    """
    counts = np.bincount(claims["siddet_no"].to_numpy(dtype=int), minlength=len(de.SEVERITY_LABELS))
    area = np.bincount(claims["siddet_no"].to_numpy(dtype=int), weights=claims["alan_ha"].to_numpy(),
                       minlength=len(de.SEVERITY_LABELS))
    fig = go.Figure(go.Bar(
        x=list(de.SEVERITY_LABELS), y=counts, marker=dict(color=list(SEVERITY_COLORS)),
        text=[f"{int(c)} parsel" for c in counts], textposition="outside", cliponaxis=False,
        customdata=area, hovertemplate="<b>%{x}</b><br>%{y} parsel<br>%{customdata:.0f} ha<extra></extra>",
    ))
    style_figure(fig, 290, "dNBR ŞİDDET SINIFI DAĞILIMI (USGS)", legend_top=False)
    fig.update_yaxes(title_text="Parsel sayısı", range=[0, float(counts.max()) * 1.2 + 1])
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_layout(showlegend=False, bargap=0.3)
    return fig


def render_insurance_tab() -> None:
    """SİGORTA sekmesini çizer: KPI'lar, GeoJSON hasar haritası; altında kararlar, grafikler ve kanıt paketi sekmeleri.

    Karar kuralları kenar çubuğundaki kaydırıcılardan gelir (``render_rule_controls``).
    """
    principal = session_principal()
    auth.require(principal, auth.PERM_INSURANCE)
    cfg = get_settings()
    rules, adjusted = current_rules_checked()
    cd = company_data(principal, cfg.seed, rules)
    ds, claims = cd.ds, cd.mine                 # claims: yalnızca bu şirketin poliçeleri (indeks = geometri konumu)
    render_html(build_hero_html(
        "ÇEKİRDEK PAZAR · SİGORTA", "RİSK VE", "HASAR ANALİZİ",
        "Şirketinizin poliçeleri için parsel bazlı dNBR hasar katmanı, risk skoru ve otomatik tazminat önerisi. "
        "Karar kurallarını sol menüden değiştirin; her şey anında yeniden hesaplanır. Bir müşteriye özel karar vermek "
        "için Müşteri dosyaları sekmesini kullanın.",
        [tenancy.company_name(cd.company).upper(), "DEMO · SENTETİK VERİ", f"OLAY SONRASI: {ds.event_date:%d.%m.%Y}"]))
    if adjusted:
        render_html('<div class="note info">Eşikler tutarlı olacak şekilde otomatik düzeltildi '
                    '(ret &lt; kısmi onay &lt; tam onay).</div>')
    if cd.error:
        render_html(f'<div class="note err"><b>Karar defteri okunamadı:</b> {_h(cd.error)} Sigortacı kararları '
                    "gösterilemiyor; yalnızca otomatik kararlar listelenir.</div>")
    s = cw.summarize_final(claims)

    render_html(build_kpi_grid([
        Kpi("Poliçe / Parsel", format_tr(s["policy_count"]), f"{format_tr(claims['alan_ha'].sum())} ha toplam alan"),
        Kpi("Hasarlı Alan", f"{format_tr(s['damaged_ha'])} ha", f"Ortalama dNBR <b>{format_tr(s['mean_dnbr'], 3)}</b>",
            COLOR_AMBER),
        Kpi("Onaylanan Tazminat", f"₺{format_tr(s['paid_tl'] / 1e6, 1)} M",
            f"Bekleyen <b>₺{format_tr(s['pending_tl'] / 1e6, 1)} M</b>"),
        Kpi("Otomatik Karar Oranı", f"%{format_tr(s['automation_pct'], 1)}",
            f"Ekspertize düşen: <b>{format_tr(s['review'])}</b> · sigortacı kararı: <b>{format_tr(s['manual'])}</b>"),
        Kpi("Karar Dağılımı",
            f"{int(s['approved'])} · {int(s['partial'])} · {int(s['review'])} · {int(s['rejected'])}",
            "onay · kısmi · ekspertiz · ret"),
    ], columns=5))

    mode = st.radio("Harita renklendirmesi", list(INS_MODES), horizontal=True, key="ins_mode")
    show_map(build_insurance_map(ds, claims, mode, cfg.basemap), key="ins_map", height=660)
    if mode == INS_MODES[0]:
        render_html(build_legend_html("dNBR", list(zip(SEVERITY_COLORS, de.SEVERITY_LABELS))))
    elif mode == INS_MODES[1]:
        render_html(build_legend_html("KARAR", [(c, k) for k, c in DECISION_COLORS.items()]))
    else:
        render_html(build_legend_html("RİSK", [(ramp_color(v, RISK_STOPS), t) for v, t in
                                                ((10, "Düşük"), (45, "Orta"), (70, "Yüksek"), (95, "Çok yüksek"))]))

    tab_table, tab_charts, tab_pack = st.tabs(list(INS_SUBTABS))
    with tab_table:
        view = render_decision_table(claims, s, rules)
    with tab_charts:
        top_n = st.slider("Grafikte gösterilecek poliçe", 10, 40, 20, 5, key="ins_topn")
        chart_a, chart_b = st.columns(2)
        with chart_a:
            show_figure(build_risk_chart(claims, int(top_n)), key="ins_risk_chart")
        with chart_b:
            show_figure(build_severity_chart(claims), key="ins_sev_chart")
    with tab_pack:
        render_evidence_section(ds, cd.auto, view, rules, cfg.seed)


#: Sigorta sekmesindeki alt sekmeler.
INS_SUBTABS: Tuple[str, ...] = ("Kararlar", "Grafikler", "Kanıt paketi")


def render_decision_table(claims: pd.DataFrame, s: Dict[str, Any], rules: ClaimRules) -> pd.DataFrame:
    """Sigorta sekmesinin "Kararlar" alt sekmesi: filtre, sıralama, tablo, CSV ve özet.

    Args:
        claims: Tüm karar tablosu.
        s: ``summarize_claims`` çıktısı.
        rules: Geçerli kurallar.

    Returns:
        Filtrelenmiş ve sıralanmış görünüm (kanıt paketinin kapsamı).
    """
    f1, f2, f3 = st.columns([2, 2, 1])
    picked = f1.multiselect("Karar filtresi", list(DECISION_ICONS), default=list(DECISION_ICONS), key="ins_filter")
    query = f2.text_input("Poliçe no / sigortalı ara", key="ins_query")
    sort_by = f3.selectbox("Sırala", ["Risk skoru ↓", "dNBR ↓", "Ödenecek ↓", "Poliçe no"], key="ins_sort")
    view = claims[claims["karar"].isin(picked)]
    if query.strip():
        q = query.strip().lower()
        view = view[view["police_no"].str.lower().str.contains(q, regex=False)
                    | view["sigortali"].str.lower().str.contains(q, regex=False)]
    sort_map = {"Risk skoru ↓": ("risk_skoru", False), "dNBR ↓": ("dnbr", False),
                "Ödenecek ↓": ("odenecek_tl", False), "Poliçe no": ("police_no", True)}
    col, asc = sort_map[sort_by]
    view = view.sort_values(col, ascending=asc)
    cols = ["police_no", "sigortali", "varlik_turu", "alan_ha", "dnbr", "siddet", "risk_skoru", "guven",
            "otomatik_karar", "karar", "karar_kaynagi", "gerekce", "hasar_yuzdesi", "odenecek_tl", "bekleyen_tl"]
    shown = view[cols].copy()
    shown["karar"] = shown["karar"].map(DECISION_ICONS)
    shown["otomatik_karar"] = shown["otomatik_karar"].map(DECISION_ICONS)
    cc = st.column_config
    show_table(shown, {
        "police_no": cc.TextColumn("Poliçe No"), "sigortali": cc.TextColumn("Sigortalı"),
        "varlik_turu": cc.TextColumn("Varlık"), "alan_ha": cc.NumberColumn("Alan (ha)", format="%.1f"),
        "dnbr": cc.NumberColumn("dNBR", format="%.3f"), "siddet": cc.TextColumn("Şiddet"),
        "risk_skoru": cc.ProgressColumn("Risk skoru", min_value=0, max_value=100, format="%.0f"),
        "guven": cc.NumberColumn("Güven", format="%.2f"), "otomatik_karar": cc.TextColumn("Otomatik öneri"),
        "karar": cc.TextColumn("Geçerli karar"), "karar_kaynagi": cc.TextColumn("Karar kaynağı"),
        "gerekce": cc.TextColumn("Gerekçe"),
        "hasar_yuzdesi": cc.ProgressColumn("Hasar %", min_value=0, max_value=100, format="%.0f"),
        "odenecek_tl": cc.NumberColumn("Ödenecek (₺)", format="%.0f"),
        "bekleyen_tl": cc.NumberColumn("Bekleyen (₺)", format="%.0f"),
    }, height=380)
    st.download_button("↓  TAZMİNAT KARARLARI (CSV)", data=csv_bytes(view[cols]),
                       file_name="tazminat_kararlari.csv", mime="text/csv", key="ins_download")
    render_brief([
        f"{len(view)} / {len(claims)} poliçe listeleniyor · geçerli karar: {int(s['approved'])} ONAY, "
        f"{int(s['partial'])} KISMİ ONAY (otomatik avans %{rules.advance_pct:.0f}), {int(s['review'])} EKSPERTİZ, "
        f"{int(s['rejected'])} RET · bunların {int(s['manual'])} tanesi sigortacı kararı",
        f"toplam ödeme ₺{format_tr(s['paid_tl'])} · bekleyen ₺{format_tr(s['pending_tl'])} · "
        f"muafiyet %{rules.deductible_pct:.1f} · teminat %{rules.coverage_pct:.0f}",
    ])
    return view


# =========================================================================== #
# SEKME 1b — KALİBRASYON (geriye dönük test)
# =========================================================================== #
CAL_SOURCES: Tuple[str, ...] = ("Örnek geçmiş (sentetik)", "CSV yükle")


@st.cache_data(show_spinner=False)
def load_sample_history(seed: int) -> cal.HistoryTable:
    """Sentetik örnek geçmişi üretir ve gerçek CSV okuma yolundan geçirir (önbellekli).

    Args:
        seed: Veri tohumu.

    Returns:
        ``HistoryTable``.
    """
    return cal.parse_history(cal.history_csv(cal.sample_history(seed)))


@st.cache_data(show_spinner=False)
def parse_uploaded_history(raw: bytes) -> cal.HistoryTable:
    """Yüklenen CSV'yi doğrular (önbellekli).

    Args:
        raw: Dosya baytları.

    Returns:
        ``HistoryTable``.

    Raises:
        CalibrationError: Dosya okunamazsa ya da geçersizse.
    """
    return cal.parse_history(raw)


@st.cache_data(show_spinner="Eşikler aranıyor…")
def run_calibration(history: pd.DataFrame, rules_tuple: Tuple[float, ...], limit: float,
                    test_fraction: float, seed: int) -> cal.CalibrationResult:
    """Kalibrasyonu çalıştırır (önbellekli).

    Args:
        history: ``HistoryTable.frame``.
        rules_tuple: ``astuple(ClaimRules)`` (mevcut kurallar).
        limit: Azami kritik hata oranı (0–1).
        test_fraction: Test payı.
        seed: Bölme tohumu.

    Returns:
        ``CalibrationResult``.
    """
    return cal.calibrate(history, ClaimRules(*rules_tuple), limit, test_fraction, seed)


def pct_text(value: float, decimals: int = 1) -> str:
    """0–1 oranını ``%xx,x`` biçiminde yazar."""
    return "%" + format_tr(100.0 * value, decimals)


def money_m(value: float, signed: bool = False) -> str:
    """TL tutarını milyon olarak yazar (ör. ``₺3,9 M``); ``signed`` ise işareti başa koyar (``−₺3,9 M``)."""
    sign = ("+" if value > 0 else "−" if value < 0 else "") if signed else ""
    return f"{sign}₺{format_tr(abs(value) / 1e6 if signed else value / 1e6, 1)} M"


def build_confusion_chart(metrics: cal.Metrics, title: str) -> go.Figure:
    """Karışıklık matrisi ısı haritası: satır = uzman kararı, sütun = platform kararı.

    Renk, satır payını (duyarlılık) gösterir; hücre yazısı kayıt sayısıdır. Kırmızı çerçeveli
    hücreler kritik hatadır (haksız ödeme riski ve haksız ret).

    Args:
        metrics: Ölçümler.
        title: Grafik başlığı.

    Returns:
        ``go.Figure``.
    """
    labels = list(de.DECISION_LABELS)
    counts = metrics.confusion.astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    fig = go.Figure(go.Heatmap(
        z=share.tolist(), x=labels, y=labels, text=metrics.confusion.tolist(), texttemplate="%{text}",
        colorscale=[[0.0, "#10131a"], [1.0, COLOR_ACCENT]], showscale=False, zmin=0.0, zmax=1.0, xgap=3, ygap=3,
        hovertemplate="Uzman: %{y}<br>Platform: %{x}<br>%{text} kayıt · satırın %{z:.0%}<extra></extra>"))
    for row, col in ((0, 2), (0, 3), (2, 0), (3, 0)):
        fig.add_shape(type="rect", xref="x", yref="y", x0=col - 0.5, x1=col + 0.5, y0=row - 0.5, y1=row + 0.5,
                      line=dict(color=COLOR_DANGER, width=2), fillcolor="rgba(0,0,0,0)")
    style_figure(fig, 330, title, legend_top=False)
    fig.update_xaxes(title_text="Platform kararı", tickfont=dict(size=10))
    fig.update_yaxes(title_text="Uzman kararı", autorange="reversed", tickfont=dict(size=10))
    fig.update_layout(showlegend=False)
    return fig


def build_sensitivity_chart(curve: pd.DataFrame, key: str, current: float, tuned: float) -> go.Figure:
    """Tek eşiğin değişimine karşı tam uyum ve kritik hata eğrisi (diğer eşikler önerilen değerde).

    Args:
        curve: ``sensitivity_curves`` çıktısından bir tablo.
        key: ``ClaimRules`` alan adı.
        current: Mevcut eşik değeri.
        tuned: Önerilen eşik değeri.

    Returns:
        ``go.Figure``.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["deger"], y=curve["uyum_pct"], name="Tam uyum %", mode="lines",
                             line=dict(color=COLOR_ACCENT, width=2),
                             hovertemplate="eşik %{x:.2f} → uyum %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=curve["deger"], y=curve["kritik_pct"], name="Kritik hata %", mode="lines",
                             line=dict(color=COLOR_DANGER, width=2),
                             hovertemplate="eşik %{x:.2f} → kritik hata %{y:.1f}%<extra></extra>"))
    for value, colour, dash in ((current, "#cfd4dc", "dot"), (tuned, COLOR_ACCENT, "dash")):
        fig.add_shape(type="line", xref="x", yref="paper", x0=value, x1=value, y0=0, y1=1,
                      line=dict(color=colour, width=1.2, dash=dash))
    style_figure(fig, 250, cal.PARAM_LABELS[key].upper())
    fig.update_yaxes(range=[0, 100], title_text="%")
    return fig


def metric_kpis(m: cal.Metrics, limit: float) -> List[Kpi]:
    """Bir kural setinin geçmişe karşı ölçümlerini KPI kartlarına çevirir.

    Args:
        m: Ölçümler.
        limit: Azami kritik hata oranı (0–1); aşılırsa kart uyarı rengine döner.

    Returns:
        Beş KPI.
    """
    return [
        Kpi("Tam uyum", pct_text(m.agreement), f"%95 aralık: {pct_text(m.agreement_low, 0)}–{pct_text(m.agreement_high, 0)}"
            f" · n={format_tr(m.n)}"),
        Kpi("Haksız ödeme riski", format_tr(m.false_approval), "uzman RET dedi, platform öder",
            COLOR_AMBER if m.false_approval else ""),
        Kpi("Haksız ret", format_tr(m.wrongful_denial), "uzman ödedi, platform reddeder",
            COLOR_AMBER if m.wrongful_denial else ""),
        Kpi("Kritik hata", pct_text(m.critical_rate), f"sınır {pct_text(limit)}", alert=m.critical_rate > limit),
        Kpi("Otomasyon", pct_text(m.automation), f"ekspertize düşen {pct_text(m.expert_share)}"),
    ]


def _select_history(seed: int) -> Tuple[Optional[cal.HistoryTable], str, bool]:
    """Geçmiş veri kaynağını seçtirir ve doğrulanmış tabloyu döndürür.

    Args:
        seed: Veri tohumu (örnek geçmiş ve şablon için).

    Returns:
        ``(tablo | None, kaynak etiketi, sentetik mi)``.
    """
    with panel():
        source = st.radio("Veri kaynağı", list(CAL_SOURCES), horizontal=True, key="cal_source")
        if source == CAL_SOURCES[0]:
            table = load_sample_history(seed)
            render_html(
                '<div class="note info"><b>Örnek geçmiş SENTETİKTİR.</b> Platformdan farklı eşiklerle karar veren '
                "varsayımsal bir eksperin, gürültülü okumalarla verdiği kararlardan oluşur. Mekanizmayı gösterir; "
                "gerçek dünya doğruluğunu KANITLAMAZ. Anlamlı sonuç için bir sigortacının geçmiş kararları gerekir.</div>")
            return table, "örnek geçmiş (sentetik)", True
        up_col, tpl_col = st.columns([3, 1])
        uploaded = up_col.file_uploader("Geçmiş karar dosyası (CSV)", type=["csv"], key="cal_upload")
        tpl_col.download_button("↓  ŞABLON (CSV)", data=cal.template_csv(seed), file_name="gecmis_kararlar_sablon.csv",
                                mime="text/csv", key="cal_template")
        render_html(
            '<div class="note info">Zorunlu sütunlar: <b>dnbr</b>, <b>guven</b> (0–1), <b>gercek_karar</b> '
            "(ONAY · KISMİ ONAY · EKSPERTİZ · RET). İsteğe bağlı: <b>police_aktif</b>, <b>sigorta_bedeli_tl</b>, "
            "<b>gercek_odeme_tl</b> (ikisi birlikte varsa finansal karşılaştırma yapılır). ; veya , ayraçlı; "
            "ondalık için nokta ya da virgül; binlik ayırıcı kullanmayın. Dosya bu oturumun dışına gönderilmez.</div>")
        if uploaded is None:
            return None, "", False
        try:
            return parse_uploaded_history(uploaded.getvalue()), f"yüklenen dosya ({uploaded.name})", False
        except cal.CalibrationError as exc:
            render_html(f'<div class="note err"><b>Dosya okunamadı:</b> {_h(exc)}</div>')
            return None, "", False


def render_calibration_tab() -> None:
    """KALİBRASYON sekmesini çizer: geçmiş karar yükleme, mevcut kuralların sınaması, kalibrasyon önerisi."""
    auth.require(session_principal(), auth.PERM_CALIBRATION)
    cfg = get_settings()
    rules = current_rules()
    render_html(build_hero_html(
        "ÇEKİRDEK PAZAR · SİGORTA", "GERİYE DÖNÜK TEST VE", "KALİBRASYON",
        "Kurallarınızı geçmiş kararlarınıza karşı sınayın: uyumu, kritik hatayı ve finansal sapmayı ölçün; "
        "eşikleri yalnızca eğitim kayıtlarında arayıp hiç görülmemiş test kayıtlarında doğrulayın.",
        ["EĞİTİM / TEST AYRIMI", "WILSON %95 ARALIK", "TEK KARAR MOTORU"]))

    render_html(build_section_title("GEÇMİŞ KARARLAR", "uzman kararlarını platformla karşılaştırın"))
    table, label, synthetic = _select_history(cfg.seed)
    if table is None:
        return
    for warning in table.warnings:
        render_html(f'<div class="note info">{_h(warning)}</div>')
    frame = table.frame

    render_html(build_section_title("AYARLAR", "mevcut kurallar: sol menüdeki karar kuralları"))
    with panel():
        s1, s2, s3 = st.columns(3)
        limit_pct = s1.slider("Azami kritik hata (%)", 0.0, 10.0, 2.0, 0.5, key="cal_limit")
        test_pct = s2.slider("Test payı (%)", 10, 50, 30, 5, key="cal_test")
        with s3:
            render_html(
                '<div class="brief" style="margin-top:0">'
                f'<span class="k">01</span>tam onay ≥ {format_tr(rules.approve_dnbr, 2)} · kısmi ≥ {format_tr(rules.partial_dnbr, 2)}<br>'
                f'<span class="k">02</span>ret &lt; {format_tr(rules.review_dnbr, 2)} · güven ≥ {format_tr(rules.min_confidence, 2)}</div>')
    limit = float(limit_pct) / 100.0

    before_all = cal.backtest(frame, rules)
    dis = cal.disagreements(frame, rules)
    enough = len(frame) >= cal.MIN_ROWS_CALIBRATE
    res = run_calibration(frame, astuple(rules), limit, float(test_pct) / 100.0, cfg.seed) if enough else None
    not_enough = (f'<div class="note">Kalibrasyon için en az {cal.MIN_ROWS_CALIBRATE} geçerli kayıt gerekir '
                  f"(bulunan: {len(frame)}). Mevcut kuralların sınaması “Sınama” sekmesinde.</div>")

    tab_test, tab_rec, tab_charts, tab_dis = st.tabs(list(CAL_SUBTABS))
    with tab_test:
        render_html(build_section_title("MEVCUT KURALLARIN SINAMASI", f"{format_tr(len(frame))} geçmiş karar · {label}"))
        render_html(build_kpi_grid(metric_kpis(before_all, limit), columns=5))
    with tab_rec:
        if res is None:
            render_html(not_enough)
        else:
            render_calibration_result(res, rules, limit, label, synthetic)
    with tab_charts:
        if res is None:
            render_html(not_enough)
        else:
            render_calibration_charts(res, rules)
    with tab_dis:
        render_html(build_section_title("PLATFORM İLE UZMANIN AYRIŞTIĞI KAYITLAR", "mevcut kurallar · tüm geçmiş"))
        if dis.empty:
            render_html('<div class="note ok">Mevcut kurallar tüm geçmiş kararları birebir yeniden üretiyor.</div>')
        else:
            shown = dis.head(200).copy()
            shown["police_aktif"] = np.where(shown["police_aktif"], "Evet", "Hayır")
            cc = st.column_config
            show_table(shown, {
                "tur": cc.TextColumn("Sapma türü"), "dnbr": cc.NumberColumn("dNBR", format="%.3f"),
                "guven": cc.NumberColumn("Güven", format="%.2f"), "police_aktif": cc.TextColumn("Poliçe aktif"),
                "gercek_karar": cc.TextColumn("Uzman kararı"), "platform_karar": cc.TextColumn("Platform kararı"),
            }, height=300)
            st.download_button("↓  TÜM SAPMALAR (CSV)", data=cal.history_csv(dis), file_name="sapmalar.csv",
                               mime="text/csv", key="cal_dis_download")
            st.caption(f"{format_tr(len(dis))} sapma; tabloda ilk {format_tr(len(shown))} kayıt gösteriliyor, "
                       "kritik hatalar üstte.")

    notes = list(res.notes) if res is not None else []
    with st.expander("Yöntem ve sınırlar"):
        render_brief([
            "Geriye dönük test ve üretim kararı aynı işlevi (decide_codes) kullanır; test edilen kural ile uygulanan kural ayrışamaz.",
            "Eşikler yalnızca eğitim kayıtlarında aranır; sonuçlar hiç görülmemiş test kayıtlarında raporlanır. "
            "Muafiyet, teminat ve avans poliçe şartıdır ve kalibre edilmez.",
            "Kalibrasyon, geçmiş kararların kuralıyla (dNBR + güven) açıklanabildiği ölçüde anlamlıdır; "
            "uzmanlar başka bilgilere de bakmış olabilir." + (" " + " ".join(notes) if notes else ""),
        ])


#: Kalibrasyon sekmesindeki alt sekmeler.
CAL_SUBTABS: Tuple[str, ...] = ("Sınama", "Öneri", "Grafikler", "Ayrışmalar")


def render_calibration_result(res: cal.CalibrationResult, rules: ClaimRules, limit: float, label: str,
                       synthetic: bool) -> None:
    """Kalibrasyon sonuçlarını çizer: test KPI'ları, değişiklik tablosu, uygula düğmesi ve grafikler.

    Args:
        res: Kalibrasyon sonucu.
        rules: Mevcut kurallar.
        limit: Azami kritik hata oranı (0–1).
        label: Veri kaynağı etiketi.
        synthetic: Kaynak sentetik mi.
    """
    tb, ta = res.test_before, res.test_after
    delta_pts = 100.0 * (ta.agreement - tb.agreement)
    fin = res.financial
    render_html(build_section_title("KALİBRASYON ÖNERİSİ",
                                    f"test bölümü · {format_tr(res.n_test)} kayıt (eğitim {format_tr(res.n_train)})"))
    render_html(build_kpi_grid([
        Kpi("Test uyumu · mevcut", pct_text(tb.agreement),
            f"%95 aralık: {pct_text(tb.agreement_low, 0)}–{pct_text(tb.agreement_high, 0)}"),
        Kpi("Test uyumu · önerilen", pct_text(ta.agreement),
            f"{'+' if delta_pts >= 0 else '−'}{format_tr(abs(delta_pts), 1)} puan · kazanılan {res.test_wins} / "
            f"kaybedilen {res.test_losses} kayıt · p={format_tr(res.p_value, 2)}",
            COLOR_OK if res.significant else ""),
        Kpi("Kritik hata · önerilen", pct_text(ta.critical_rate), f"mevcut {pct_text(tb.critical_rate)} · sınır {pct_text(limit)}",
            alert=ta.critical_rate > limit),
        Kpi("Otomasyon · önerilen", pct_text(ta.automation), f"mevcut {pct_text(tb.automation)}"),
        (Kpi("Finansal sapma · önerilen", money_m(fin.tuned_gap_tl, signed=True),
             f"mevcut {money_m(fin.base_gap_tl, signed=True)} · uzman {money_m(fin.human_tl)}")
         if fin is not None else Kpi("Finansal sapma", "—", "bedel ve ödeme sütunları yok")),
    ], columns=5))
    if fin is not None:
        st.caption("Finansal sapma = platformun ödeyeceği + ekspertize bırakacağı toplam − uzmanların ödediği toplam "
                   "(test bölümü). p değeri: yalnızca bir kuralın doğru bildiği kayıtlar üzerinde iki yönlü işaret testi.")
    if res.recommended and not res.significant:
        render_html('<div class="note">Test bölümündeki iyileşme istatistiksel olarak <b>belirgin değil</b> '
                    f"(p={format_tr(res.p_value, 2)}). Öneri uygulanabilir ancak daha çok kayıtla doğrulanmadan "
                    "güvenilmemelidir.</div>")

    tuned = res.tuned_rules
    cc = st.column_config
    show_table(res.changes, {
        "parametre": cc.TextColumn("Parametre"), "mevcut": cc.NumberColumn("Mevcut", format="%.2f"),
        "onerilen": cc.NumberColumn("Önerilen", format="%.2f"), "fark": cc.NumberColumn("Fark", format="%.2f"),
    }, height=200)

    source_text = "SENTETİK örnek geçmiş" if synthetic else label
    lines = (
        f"Eşikler {format_tr(res.n_train + res.n_test)} geçmiş karar üzerinde kalibre edildi "
        f"({format_tr(res.n_train)} eğitim / {format_tr(res.n_test)} test; kaynak: {source_text}).",
        f"Test bölümünde tam uyum {pct_text(tb.agreement)} → {pct_text(ta.agreement)} "
        f"(%95 aralık {pct_text(ta.agreement_low, 0)}–{pct_text(ta.agreement_high, 0)}); "
        f"kritik hata {pct_text(tb.critical_rate)} → {pct_text(ta.critical_rate)}.",
    )
    b1, b2, b3 = st.columns([3, 1, 3])
    b1.button("✓  ÖNERİLEN KURALLARI UYGULA", on_click=apply_calibrated_rules,
              args=(_threshold_tuple(tuned), lines), key="cal_apply", disabled=not res.recommended)
    b2.button("↺  Sıfırla", on_click=reset_rules, key="cal_reset")
    if calibration_lines_for(rules):
        render_html('<div class="note ok">Önerilen eşikler sol menüdeki karar kurallarına uygulandı; '
                    "kanıt paketlerine kalibrasyon bilgisi eklenir. Mevcut kurallar artık önerilenle aynı olduğundan "
                    "yukarıdaki iki sütun birbirine eşittir.</div>")
    elif not res.recommended:
        render_html('<div class="note">Öneri test bölümünde güvenilir bir iyileşme göstermediği için uygulanamaz '
                    "(uyum artmadı ya da kritik hata sınırı aşıldı).</div>")


def render_calibration_charts(res: cal.CalibrationResult, rules: ClaimRules) -> None:
    """Kalibrasyon grafikleri: karışıklık matrisleri ve eşik duyarlılığı.

    Args:
        res: Kalibrasyon sonucu.
        rules: Mevcut kurallar.
    """
    tb, ta, tuned = res.test_before, res.test_after, res.tuned_rules
    render_html(build_section_title("KARIŞIKLIK MATRİSİ", "satır: uzman · sütun: platform · kırmızı çerçeve: kritik hata"))
    g1, g2 = st.columns(2)
    with g1:
        show_figure(build_confusion_chart(tb, "MEVCUT KURALLAR · TEST BÖLÜMÜ"), key="cal_conf_before")
    with g2:
        show_figure(build_confusion_chart(ta, "ÖNERİLEN KURALLAR · TEST BÖLÜMÜ"), key="cal_conf_after")

    render_html(build_section_title("EŞİK DUYARLILIĞI", "dikey çizgi: noktalı = mevcut · kesikli = önerilen (eğitim bölümü)"))
    keys = list(cal.BOUNDS)
    for row_keys in (keys[:2], keys[2:]):
        cols = st.columns(2)
        for col, key in zip(cols, row_keys):
            with col:
                show_figure(build_sensitivity_chart(res.sensitivity[key], key, float(getattr(rules, key)),
                                                    float(getattr(tuned, key))), key=f"cal_sens_{key}")


# =========================================================================== #
# SEKME 2 — TARIM (Fenoloji ve Rekolte)
# =========================================================================== #
AGRI_YEAR: int = 2026
AGRI_METRICS: Tuple[str, ...] = ("NDVI", "EVI", "Su stresi", "Verim endeksi")
BASE_YIELD: Dict[str, float] = {c.name: c.base_yield for c in de.CROPS}


def field_color(row: pd.Series, metric: str) -> str:
    """Tarla satırı için seçilen ölçüte göre dolgu rengini hesaplar.

    Args:
        row: Tarla tablosundan bir satır.
        metric: ``AGRI_METRICS`` öğelerinden biri.

    Returns:
        ``#rrggbb`` renk.
    """
    if metric == "NDVI":
        return ramp_color(float(row["ndvi"]), NDVI_STOPS)
    if metric == "EVI":
        return ramp_color(float(row["evi"]) * 1.25, NDVI_STOPS)
    if metric == "Su stresi":
        return ramp_color(float(row["su_stresi"]), STRESS_STOPS)
    ratio = float(row["rekolte_t_ha"]) / BASE_YIELD[str(row["bitki"])]
    return ramp_color(ratio * 0.8, NDVI_STOPS)


def build_agriculture_map(ds: de.AgricultureDataset, frame: pd.DataFrame, metric: str, focus: str,
                          basemap: bm.BasemapChoice) -> folium.Map:
    """Tarla sınırlarını ve tıklanınca NDVI/EVI gösteren pop-up'ları içeren haritayı kurar.

    Args:
        ds: Tarım veri seti (geometri için).
        frame: Gösterilecek tarlalar (``ds.fields`` süzülmüş hali; indeks = geometri sırası).
        metric: Renklendirme ölçütü.
        focus: Vurgulanacak tarla kimliği ("" ise yok).
        basemap: Seçili harita altlığı ve görüntü günü.

    Returns:
        Hazır ``folium.Map``.
    """
    geo = ds.geometry
    overlay = build_map_overlay("TARIM · TARLA SINIRLARI", "Tarla", "izleme",
                                (metric, f"{len(frame)} tarla", f"odak {focus}" if focus else "tıklayınca NDVI/EVI"))
    fmap = make_base_map(ds.centre, 12, basemap, overlay)
    group = folium.FeatureGroup(name="Tarla sınırları", show=True)
    for i, row in frame.iterrows():
        idx = int(i)
        coords = list(zip(geo.corners_lat[idx].tolist(), geo.corners_lon[idx].tolist()))
        is_focus = row["tarla_id"] == focus
        popup = popup_html(
            f"{row['tarla_id']} · {row['bitki']}",
            [("Alan", f"{format_tr(row['alan_ha'], 1)} ha"),
             ("NDVI / beklenen", f"{format_tr(row['ndvi'], 3)} / {format_tr(row['ndvi_beklenen'], 3)}"),
             ("EVI", format_tr(row["evi"], 3)),
             ("Su stresi indeksi", format_tr(row["su_stresi"], 2)),
             ("Sulama kalitesi", format_tr(row["sulama_kalitesi"], 2)),
             ("Rekolte", f"{format_tr(row['rekolte_t_ha'], 1)} t/ha"),
             ("Toplam / gelir", f"{format_tr(row['rekolte_toplam_t'])} t · ₺{format_tr(row['gelir_tahmini_tl'] / 1e6, 1)} M")],
            bars=[("NDVI", float(row["ndvi"]), ramp_color(float(row["ndvi"]), NDVI_STOPS)),
                  ("EVI", float(row["evi"]), ramp_color(float(row["evi"]) * 1.25, NDVI_STOPS)),
                  ("Su stresi", float(row["su_stresi"]), ramp_color(float(row["su_stresi"]), STRESS_STOPS))],
            tag=(str(row["saglik"]), HEALTH_COLORS[str(row["saglik"])]),
        )
        folium.Polygon(
            locations=coords, color=COLOR_ACCENT if is_focus else "#0b0c10", weight=3.5 if is_focus else 1.4,
            fill=True, fill_color=field_color(row, metric), fill_opacity=0.72,
            tooltip=f"{row['tarla_id']} · {row['bitki']} · NDVI {format_tr(row['ndvi'], 2)}",
            popup=folium.Popup(popup, max_width=300),
        ).add_to(group)
        folium.Marker(
            location=[float(row["enlem"]), float(row["boylam"])],
            icon=folium.DivIcon(
                icon_size=(70, 18), icon_anchor=(35, 9),
                html=(f'<div style="font:700 11px \'Segoe UI\',sans-serif;color:#fff;text-align:center;'
                      f'text-shadow:0 0 4px #000,0 0 8px #000;pointer-events:none">{_h(row["tarla_id"])}</div>')),
        ).add_to(group)
    group.add_to(fmap)
    folium.LayerControl(collapsed=True, position="topright").add_to(fmap)
    fit_bounds(fmap, list(geo.corners_lat.ravel()), list(geo.corners_lon.ravel()), 0.004)
    return fmap


def build_phenology_chart(ds: de.AgricultureDataset, crops: Sequence[str], focus: str, obs: dt.date) -> go.Figure:
    """Yıllık fenoloji eğrilerini (çoklu çizgi) ve su stresi dönemlerini çizer.

    Args:
        ds: Tarım veri seti.
        crops: Gösterilecek bitki türleri.
        focus: Kesikli çizgiyle eklenecek tarla kimliği ("" ise yok).
        obs: Gözlem tarihi (dikey işaret).

    Returns:
        ``go.Figure``.
    """
    fig = go.Figure()
    for p in ds.stress_periods:
        fig.add_shape(type="rect", xref="x", yref="paper", x0=p.start.isoformat(), x1=p.end.isoformat(),
                      y0=0, y1=1, fillcolor="rgba(252,61,33,0.13)", line=dict(width=0), layer="below")
        mid = p.start + (p.end - p.start) / 2
        fig.add_annotation(x=mid.isoformat(), xref="x", y=1, yref="paper", yanchor="top", showarrow=False,
                           text=f"SU STRESİ · {p.days} gün", font=dict(size=10, color=COLOR_DANGER))
    for crop in crops:
        fig.add_trace(go.Scatter(
            x=ds.crop_curves.index, y=ds.crop_curves[crop], name=crop, mode="lines",
            line=dict(color=CROP_COLORS[crop], width=2.6, shape="spline", smoothing=0.6),
            hovertemplate=f"<b>{crop}</b><br>%{{x|%d.%m.%Y}}<br>NDVI: %{{y:.2f}}<extra></extra>"))
    if focus:
        fig.add_trace(go.Scatter(
            x=ds.field_curves.index, y=ds.field_curves[focus], name=f"{focus} (seçili tarla)", mode="lines",
            line=dict(color="#ffffff", width=2.2, dash="dash"),
            hovertemplate=f"<b>{focus}</b><br>%{{x|%d.%m.%Y}}<br>NDVI: %{{y:.2f}}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=ds.climate["tarih"], y=ds.climate["sicaklik_c"], name="Sıcaklık (°C)", yaxis="y2", mode="lines",
        line=dict(color=COLOR_AMBER, width=1, dash="dot"), opacity=0.65,
        hovertemplate="%{x|%d.%m.%Y}<br>Sıcaklık: %{y:.1f} °C<extra></extra>"))
    fig.add_shape(type="line", xref="x", yref="paper", x0=obs.isoformat(), x1=obs.isoformat(), y0=0, y1=1,
                  line=dict(color=COLOR_ACCENT, width=1.4, dash="dash"))
    fig.add_annotation(x=obs.isoformat(), xref="x", y=0, yref="paper", yanchor="bottom", xanchor="left",
                       showarrow=False, text=" GÖZLEM", font=dict(size=10, color=COLOR_ACCENT))
    style_figure(fig, 440, "YILLIK FENOLOJİ EĞRİSİ (NDVI) · SU STRESİ DÖNEMLERİ")
    fig.update_yaxes(title_text="NDVI", range=[0, 1])
    fig.update_xaxes(tickformat="%b", dtick="M1")
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Sıcaklık (°C)", range=[-15, 40],
                                  showgrid=False, linecolor="rgba(242,184,75,0.4)"), hovermode="x unified")
    return fig


def build_yield_chart(frame: pd.DataFrame) -> go.Figure:
    """Tarla bazında rekolte verim endeksi (beklenen = %100) çubuk grafiği.

    Args:
        frame: Gösterilecek tarlalar.

    Returns:
        ``go.Figure``.
    """
    data = frame.copy()
    data["endeks"] = 100.0 * data["rekolte_t_ha"] / data["bitki"].map(BASE_YIELD)
    data = data.sort_values("endeks", ascending=False)
    custom = np.stack([data["bitki"], data["rekolte_t_ha"], data["rekolte_toplam_t"]], axis=-1)
    fig = go.Figure(go.Bar(
        x=data["tarla_id"], y=data["endeks"], marker=dict(color=[CROP_COLORS[b] for b in data["bitki"]],
                                                           line=dict(color="rgba(11,12,16,0.9)", width=1)),
        customdata=custom, text=[f"{v:.0f}" for v in data["endeks"]], textposition="outside", cliponaxis=False,
        hovertemplate=("<b>%{x}</b> · %{customdata[0]}<br>Verim endeksi: %{y:.0f}%<br>"
                       "Rekolte: %{customdata[1]:.1f} t/ha · %{customdata[2]:.0f} t<extra></extra>")))
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y", y0=100, y1=100,
                  line=dict(color=COLOR_ACCENT, width=1.2, dash="dot"))
    style_figure(fig, 420, "REKOLTE VERİM ENDEKSİ · BEKLENEN = %100", legend_top=False)
    fig.update_yaxes(title_text="Verim endeksi (%)", range=[0, max(120.0, float(data["endeks"].max()) * 1.18)])
    fig.update_layout(showlegend=False, bargap=0.3)
    return fig


def render_agriculture_tab() -> None:
    """TARIM sekmesini çizer: kontrol paneli, KPI'lar, tarla haritası, fenoloji ve rekolte grafikleri."""
    auth.require(session_principal(), auth.PERM_SECTORS)
    cfg = get_settings()
    render_html(build_hero_html(
        "ÇEKİRDEK PAZAR · TARIM", "FENOLOJİ VE", "REKOLTE İZLEME",
        "Tarla sınırları, NDVI/EVI bitki sağlığı, yıllık fenoloji eğrileri, su stresi dönemleri ve rekolte tahmini.",
        ["DEMO · SENTETİK VERİ", "SENTINEL-2 · NDVI/EVI", f"SEZON {AGRI_YEAR}"]))

    render_html(build_section_title("GÖZLEM KONTROLLERİ", "tarihi kaydırın — NDVI/EVI ve sağlık sınıfları güncellenir"))
    with panel():
        c1, c2, c3 = st.columns([2, 1, 1])
        obs = c1.slider("Gözlem tarihi", min_value=dt.date(AGRI_YEAR, 3, 1), max_value=dt.date(AGRI_YEAR, 10, 15),
                        value=dt.date(AGRI_YEAR, 7, 9), step=dt.timedelta(days=1), format="DD.MM.YYYY", key="agr_date")
        crop_pick = c2.selectbox("Bitki türü", ["Tümü"] + [c.name for c in de.CROPS], key="agr_crop")
        metric = c3.radio("Harita rengi", list(AGRI_METRICS), horizontal=True, key="agr_metric")

    ds = load_agriculture(cfg.seed, obs.timetuple().tm_yday)
    frame = ds.fields if crop_pick == "Tümü" else ds.fields[ds.fields["bitki"] == crop_pick]
    with panel():
        focus_pick = st.selectbox("Fenoloji grafiğinde vurgulanacak tarla", ["— yok —"] + frame["tarla_id"].tolist(),
                                  key="agr_focus")
    focus = "" if focus_pick.startswith("—") else focus_pick

    s = de.AgricultureSimulator.summarize(ds)
    area = float(frame["alan_ha"].sum())
    mean_ndvi = float(np.average(frame["ndvi"], weights=frame["alan_ha"])) if area else 0.0
    mean_evi = float(np.average(frame["evi"], weights=frame["alan_ha"])) if area else 0.0
    stressed = float(frame.loc[frame["saglik"].isin(["Stresli", "Kritik"]), "alan_ha"].sum())
    render_html(build_kpi_grid([
        Kpi("Toplam Tarla Alanı", f"{format_tr(area)} ha", f"{len(frame)} tarla · gözlem {obs:%d.%m.%Y}"),
        Kpi("Ortalama NDVI", format_tr(mean_ndvi, 2), f"EVI <b>{format_tr(mean_evi, 2)}</b>"),
        Kpi("Stresli / Kritik Alan", f"%{format_tr(100.0 * stressed / max(area, 1e-9), 1)}",
            f"<b>{format_tr(stressed)}</b> ha", COLOR_AMBER, alert=stressed / max(area, 1e-9) > 0.4),
        Kpi("Rekolte Tahmini", f"{format_tr(frame['rekolte_toplam_t'].sum())} t",
            f"~<b>₺{format_tr(frame['gelir_tahmini_tl'].sum() / 1e6, 1)} M</b> gelir"),
        Kpi("Su Stresi (Yıllık)", f"{int(s['stress_days'])} gün", f"<b>{len(ds.stress_periods)}</b> ayrı dönem", COLOR_AMBER),
    ], columns=5))

    left, right = st.columns([3, 2])
    with left:
        show_map(build_agriculture_map(ds, frame, metric, focus, cfg.basemap), key="agr_map", height=520)
        if metric == "Su stresi":
            render_html(build_legend_html("SU STRESİ", [(ramp_color(v, STRESS_STOPS), t) for v, t in
                                                         ((0.05, "Düşük"), (0.3, "Orta"), (0.55, "Yüksek"), (0.95, "Aşırı"))]))
        else:
            render_html(build_legend_html(metric.upper(), [(ramp_color(v, NDVI_STOPS), t) for v, t in
                                                           ((0.05, "Çıplak"), (0.3, "Zayıf"), (0.55, "Orta"), (0.8, "Yoğun"))]))
    with right:
        show_figure(build_yield_chart(frame), key="agr_yield")
        render_html(build_legend_html("BİTKİ", [(CROP_COLORS[c], c) for c in sorted(frame["bitki"].unique())]))

    crops = [c.name for c in de.CROPS if crop_pick in ("Tümü", c.name)]
    show_figure(build_phenology_chart(ds, crops, focus, obs), key="agr_pheno")
    render_brief([
        f"{p.start:%d.%m} → {p.end:%d.%m}: {p.days} gün su stresi · tepe indeks {format_tr(p.peak_index, 2)} · "
        f"ort. sıcaklık {format_tr(p.mean_temp_c, 1)} °C" for p in ds.stress_periods
    ] or ["Bu sezon belirgin su stresi dönemi tespit edilmedi."])

    render_html(build_section_title("TARLA ENVANTERİ", "gözlem tarihindeki durum"))
    cols = ["tarla_id", "bitki", "alan_ha", "ndvi", "evi", "ndvi_beklenen", "saglik", "su_stresi",
            "sulama_kalitesi", "rekolte_t_ha", "rekolte_toplam_t", "gelir_tahmini_tl"]
    cc = st.column_config
    show_table(frame[cols], {
        "tarla_id": cc.TextColumn("Tarla"), "bitki": cc.TextColumn("Bitki"),
        "alan_ha": cc.NumberColumn("Alan (ha)", format="%.1f"), "ndvi": cc.NumberColumn("NDVI", format="%.3f"),
        "evi": cc.NumberColumn("EVI", format="%.3f"), "ndvi_beklenen": cc.NumberColumn("Beklenen NDVI", format="%.3f"),
        "saglik": cc.TextColumn("Sağlık"),
        "su_stresi": cc.ProgressColumn("Su stresi", min_value=0.0, max_value=1.0, format="%.2f"),
        "sulama_kalitesi": cc.ProgressColumn("Sulama", min_value=0.0, max_value=1.0, format="%.2f"),
        "rekolte_t_ha": cc.NumberColumn("Rekolte (t/ha)", format="%.1f"),
        "rekolte_toplam_t": cc.NumberColumn("Toplam (t)", format="%.0f"),
        "gelir_tahmini_tl": cc.NumberColumn("Gelir tahmini (₺)", format="%.0f"),
    }, height=330)
    st.download_button("↓  TARLA ENVANTERİ (CSV)", data=csv_bytes(frame[cols]), file_name="tarla_envanteri.csv",
                       mime="text/csv", key="agr_download")


# =========================================================================== #
# SEKME 3 — BELEDİYE (Şehircilik ve İmar)
# =========================================================================== #
NEIGHBOURHOOD_COLORS: Tuple[str, ...] = (
    "#6ea8fe", "#4cc3c3", "#4cc38a", "#e3c35a", "#ec9a4d", "#e5738f", "#a78bfa", "#8fb6d9",
)
PRIORITY_ORDER: Tuple[str, ...] = ("Yüksek", "Orta", "Düşük")


def build_imagery_map(ds: de.MunicipalityDataset, year: int, rgba: np.ndarray, sites: pd.DataFrame,
                      mark_sites: bool, show_rects: bool, basemap: bm.BasemapChoice) -> folium.Map:
    """Uydu görüntüsü simülasyonunu ``ImageOverlay`` ile gösteren haritayı kurar.

    ``mark_sites=True`` ise kaçak yapılaşma noktaları dolu kırmızı ``CircleMarker`` olarak,
    ``False`` ise (T0 haritası) yalnızca içi boş kesikli halka olarak işaretlenir.

    Args:
        ds: Belediye veri seti.
        year: Haritanın yılı (başlık/katman adı için).
        rgba: Uydu görüntüsü simülasyonu (``uint8`` RGBA).
        sites: Gösterilecek kaçak yapı noktaları.
        mark_sites: Dolu kırmızı işaretler mi, boş halkalar mı.
        show_rects: Mahalle sınırları çizilsin mi.
        basemap: Seçili harita altlığı ve görüntü günü.

    Returns:
        Hazır ``folium.Map``.
    """
    if mark_sites:
        overlay = build_map_overlay(f"BELEDİYE · T1 · {year}", "Kaçak", "yapılaşma", ("kırmızı dolu halka = tespit",))
    else:
        overlay = build_map_overlay(f"BELEDİYE · T0 · {year}", "Uydu görüntüsü", "simülasyonu", ("kesikli halka = T0 hâli",))
    fmap = make_base_map(ds.centre, 13, basemap, overlay)
    bounds = [list(ds.bounds[0]), list(ds.bounds[1])]
    ImageOverlay(image=rgba, bounds=bounds, opacity=0.96, name=f"Uydu görüntüsü simülasyonu {year}",
                 interactive=False, cross_origin=False, zindex=3).add_to(fmap)
    if show_rects:
        rect_group = folium.FeatureGroup(name="Mahalle sınırları")
        for name, sw, ne in ds.neighbourhood_rects:
            folium.Rectangle(bounds=[list(sw), list(ne)], color=COLOR_ACCENT, weight=1, opacity=0.7,
                             fill=False, dash_array="4 6", tooltip=name).add_to(rect_group)
            folium.Marker(
                location=[(sw[0] + ne[0]) / 2.0, (sw[1] + ne[1]) / 2.0],
                icon=folium.DivIcon(icon_size=(110, 16), icon_anchor=(55, 8), html=(
                    f'<div style="font:700 10px \'Segoe UI\',sans-serif;letter-spacing:.14em;color:#e6e8ec;'
                    f'text-align:center;text-shadow:0 0 4px #000,0 0 8px #000;pointer-events:none">'
                    f'{_h(name.upper())}</div>'))).add_to(rect_group)
        rect_group.add_to(fmap)
    group = folium.FeatureGroup(name="Kaçak yapılaşma" if mark_sites else "T1'de kaçak yapılaşacak alanlar")
    max_area = float(sites["kacak_alan_m2"].max()) if len(sites) else 1.0
    for _, r in sites.iterrows():
        radius = 6.0 + 9.0 * float(np.sqrt(r["kacak_alan_m2"] / max_area))
        if mark_sites:
            folium.CircleMarker(
                location=[float(r["enlem"]), float(r["boylam"])], radius=radius, color="#fc3d21", weight=2.2,
                fill=True, fill_color="#fc3d21", fill_opacity=0.5,
                tooltip=f"{r['nokta_id']} · {format_tr(r['kacak_alan_m2'])} m² · {r['oncelik']}",
                popup=folium.Popup(popup_html(
                    f"KAÇAK YAPILAŞMA · {r['nokta_id']}",
                    [("Mahalle", str(r["mahalle"])), ("Tespit edilen alan", f"{format_tr(r['kacak_alan_m2'])} m²"),
                     ("Önceki arazi (T0)", str(r["onceki_arazi"])), ("Tespit güveni", format_tr(r["guven"], 2)),
                     ("Koordinat", f"{format_tr(r['enlem'], 4)}, {format_tr(r['boylam'], 4)}")],
                    tag=(str(r["oncelik"]), "#fc3d21" if r["oncelik"] == "Yüksek" else COLOR_AMBER)), max_width=300),
            ).add_to(group)
        else:
            folium.CircleMarker(
                location=[float(r["enlem"]), float(r["boylam"])], radius=radius, color="#fc3d21", weight=1.4,
                opacity=0.75, fill=False, dash_array="3 5",
                tooltip=f"{r['nokta_id']} · T0'da henüz yapılaşma yok").add_to(group)
    group.add_to(fmap)
    folium.LayerControl(collapsed=True, position="topright").add_to(fmap)
    fmap.fit_bounds(bounds)
    return fmap


def build_stacked_area(stats: pd.DataFrame, value_col: str, title: str, unit: str, t0: int, t1: int,
                       hover_col: Optional[str] = None, hover_unit: str = "") -> go.Figure:
    """Mahalle bazlı yığılmış alan (area) grafiği üretir.

    Args:
        stats: ``MunicipalityDataset.stats`` tablosu.
        value_col: Yığılan değer sütunu.
        title: Grafik başlığı.
        unit: Y ekseni birimi.
        t0: Başlangıç yılı (işaretlenir).
        t1: Karşılaştırma yılı (işaretlenir).
        hover_col: Üzerine gelince ek gösterilecek sütun (ör. yoğunluk).
        hover_unit: ``hover_col`` birimi.

    Returns:
        ``go.Figure``.
    """
    pivot = stats.pivot(index="yil", columns="mahalle", values=value_col)
    extra = stats.pivot(index="yil", columns="mahalle", values=hover_col) if hover_col else None
    fig = go.Figure()
    for i, name in enumerate(de.NEIGHBOURHOODS):
        color = NEIGHBOURHOOD_COLORS[i % len(NEIGHBOURHOOD_COLORS)]
        hover = f"<b>{name}</b><br>%{{x}}: %{{y:.1f}} {unit}"
        trace = dict(x=pivot.index, y=pivot[name], name=name, mode="lines", stackgroup="mahalle",
                     line=dict(width=1.2, color=color), fillcolor=hex_rgba(color, 0.42))
        if extra is not None:
            trace["customdata"] = extra[name]
            hover += f"<br>Yoğunluk: %{{customdata:.1f}} {hover_unit}"
        fig.add_trace(go.Scatter(hovertemplate=hover + "<extra></extra>", **trace))
    for year, label in ((t0, "T0"), (t1, "T1")):
        fig.add_shape(type="line", xref="x", yref="paper", x0=year, x1=year, y0=0, y1=1,
                      line=dict(color="#ffffff", width=1.2, dash="dash"))
        fig.add_annotation(x=year, xref="x", y=1, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"{label} · {year}", font=dict(size=10, color="#ffffff"))
    style_figure(fig, 420, title)
    fig.update_xaxes(dtick=1, title_text="Yıl")
    fig.update_yaxes(title_text=unit)
    return fig


def build_neighbourhood_table(ds: de.MunicipalityDataset) -> pd.DataFrame:
    """T0 → T1 mahalle karşılaştırma özet tablosunu üretir.

    Args:
        ds: Belediye veri seti.

    Returns:
        Mahalle başına yeşil alan, altyapı yoğunluğu ve kaçak yapı özeti.
    """
    a = ds.stats[ds.stats["yil"] == ds.t0].set_index("mahalle")
    b = ds.stats[ds.stats["yil"] == ds.t1].set_index("mahalle")
    illegal = ds.sites.groupby("mahalle").agg(kacak_nokta=("nokta_id", "count"), kacak_m2=("kacak_alan_m2", "sum"))
    out = pd.DataFrame({
        "mahalle": a.index,
        "yesil_t0": a["yesil_ha"].to_numpy(), "yesil_t1": b["yesil_ha"].to_numpy(),
        "yesil_degisim_pct": (100.0 * (b["yesil_ha"] - a["yesil_ha"]) / a["yesil_ha"].clip(lower=1e-9)).to_numpy(),
        "altyapi_t0": a["altyapi_yogunlugu"].to_numpy(), "altyapi_t1": b["altyapi_yogunlugu"].to_numpy(),
    })
    out = out.merge(illegal, how="left", left_on="mahalle", right_index=True).fillna({"kacak_nokta": 0, "kacak_m2": 0})
    out["kacak_nokta"] = out["kacak_nokta"].astype(int)
    return out.round(1)


def render_municipality_tab() -> None:
    """BELEDİYE sekmesini çizer: T0/T1 uydu simülasyonu, kaçak yapı haritası ve mahalle alan grafikleri."""
    auth.require(session_principal(), auth.PERM_SECTORS)
    cfg = get_settings()
    render_html(build_hero_html(
        "GENİŞLEME · BELEDİYE", "ŞEHİRCİLİK VE", "İMAR DENETİMİ",
        "T0 / T1 uydu görüntüsü simülasyonu, kaçak yapılaşma tespiti, mahalle bazlı altyapı yoğunluğu ve yeşil alan azalışı.",
        ["DEMO · SENTETİK VERİ", "ARAZİ ÖRTÜSÜ DEĞİŞİMİ", f"{de.MunicipalitySimulator.YEAR_MIN}–{de.MunicipalitySimulator.YEAR_MAX}"]))

    render_html(build_section_title("DÖNEM KARŞILAŞTIRMASI", "T0 ve T1 yıllarını seçin"))
    years = list(range(de.MunicipalitySimulator.YEAR_MIN, de.MunicipalitySimulator.YEAR_MAX + 1))
    with panel():
        c1, c2, c3 = st.columns([2, 2, 1])
        t0, t1 = c1.select_slider("Karşılaştırma dönemi (T0 → T1)", options=years, value=(2019, 2026), key="mun_years")
        pick = c2.multiselect("Öncelik filtresi", list(PRIORITY_ORDER), default=list(PRIORITY_ORDER), key="mun_priority")
        show_rects = c3.checkbox("Mahalle sınırları", value=True, key="mun_rects")
    if t0 >= t1:
        t0, t1 = (t1 - 1, t1) if t1 > years[0] else (years[0], years[0] + 1)
        render_html(f'<div class="note info">T0 yılı T1 yılından küçük olmalıdır; dönem {t0}–{t1} olarak düzeltildi.</div>')

    ds = load_municipality(cfg.seed, int(t0), int(t1))
    sites = ds.sites[ds.sites["oncelik"].isin(pick)].reset_index(drop=True)
    s = de.MunicipalitySimulator.summarize(ds)
    render_html(build_kpi_grid([
        Kpi("Kaçak Yapılaşma Noktası", format_tr(s["site_count"]), f"Yüksek öncelik: <b>{int(s['high_priority'])}</b>",
            COLOR_DANGER, alert=s["high_priority"] > 0),
        Kpi("Kaçak Yapı Alanı", f"{format_tr(s['illegal_m2'])} m²", f"≈ <b>{format_tr(s['illegal_m2'] / 10000.0, 1)}</b> ha",
            COLOR_AMBER),
        Kpi("Yapılaşma Artışı", f"+{format_tr(s['built_growth_ha'])} ha", f"{ds.t0} → {ds.t1}"),
        Kpi("Yeşil Alan Kaybı", f"−{format_tr(s['green_loss_ha'])} ha", f"<b>%{format_tr(s['green_loss_pct'], 1)}</b> azalış"),
        Kpi("En Riskli Mahalle", str(s["worst_neighbourhood"]), f"Ort. altyapı yoğunluğu <b>{format_tr(s['mean_infra'], 1)}</b> km/km²"),
    ], columns=5))

    left, right = st.columns(2)
    with left:
        show_map(build_imagery_map(ds, ds.t0, ds.rgba_t0, sites, False, show_rects, cfg.basemap), "mun_map_t0", 500)
    with right:
        show_map(build_imagery_map(ds, ds.t1, ds.rgba_t1, sites, True, show_rects, cfg.basemap), "mun_map_t1", 500)
    render_html(build_legend_html("ARAZİ ÖRTÜSÜ", [("#1a4062", "Su"), ("#2e6836", "Yeşil alan"), ("#9e886a", "Açık arazi"),
                                                   ("#b2b2b8", "Yapılaşma")])
                + build_legend_html("İŞARET", [("#fc3d21", "Kaçak yapılaşma (T1, dolu halka)"),
                                               ("#8b93a1", "T0 haritasındaki kesikli halka = aynı alanın T0 hâli")], dot=True))

    c_a, c_b = st.columns(2)
    with c_a:
        show_figure(build_stacked_area(ds.stats, "altyapi_km", "MAHALLE BAZLI ALTYAPI YOĞUNLUĞU · YOL/ŞEBEKE UZUNLUĞU",
                                       "km", ds.t0, ds.t1, "altyapi_yogunlugu", "km/km²"), key="mun_infra")
    with c_b:
        show_figure(build_stacked_area(ds.stats, "yesil_ha", "YEŞİL ALAN AZALIŞI · MAHALLE BAZLI", "ha", ds.t0, ds.t1,
                                       "yesil_pct", "% yeşil"), key="mun_green")

    render_html(build_section_title("MAHALLE KARŞILAŞTIRMASI", f"{ds.t0} → {ds.t1}"))
    cc = st.column_config
    show_table(build_neighbourhood_table(ds), {
        "mahalle": cc.TextColumn("Mahalle"), "yesil_t0": cc.NumberColumn(f"Yeşil {ds.t0} (ha)", format="%.1f"),
        "yesil_t1": cc.NumberColumn(f"Yeşil {ds.t1} (ha)", format="%.1f"),
        "yesil_degisim_pct": cc.NumberColumn("Yeşil değişim (%)", format="%.1f"),
        "altyapi_t0": cc.NumberColumn(f"Altyapı {ds.t0} (km/km²)", format="%.1f"),
        "altyapi_t1": cc.NumberColumn(f"Altyapı {ds.t1} (km/km²)", format="%.1f"),
        "kacak_nokta": cc.NumberColumn("Kaçak nokta", format="%d"), "kacak_m2": cc.NumberColumn("Kaçak alan (m²)", format="%.0f"),
    }, height=330)

    render_html(build_section_title("KAÇAK YAPILAŞMA DENETİM LİSTESİ", f"{len(sites)} nokta"))
    cols = ["nokta_id", "mahalle", "kacak_alan_m2", "onceki_arazi", "oncelik", "guven", "enlem", "boylam"]
    show_table(sites[cols], {
        "nokta_id": cc.TextColumn("Nokta"), "mahalle": cc.TextColumn("Mahalle"),
        "kacak_alan_m2": cc.NumberColumn("Alan (m²)", format="%.0f"), "onceki_arazi": cc.TextColumn("Önceki arazi"),
        "oncelik": cc.TextColumn("Öncelik"), "guven": cc.ProgressColumn("Güven", min_value=0.0, max_value=1.0, format="%.2f"),
        "enlem": cc.NumberColumn("Enlem", format="%.5f"), "boylam": cc.NumberColumn("Boylam", format="%.5f"),
    }, height=320)
    st.download_button("↓  DENETİM LİSTESİ (CSV)", data=csv_bytes(sites[cols]), file_name="kacak_yapilasma_listesi.csv",
                       mime="text/csv", key="mun_download")


# =========================================================================== #
# SEKME 4 — KURTARMA (Afet ve Kriz Yönetimi)
# =========================================================================== #
RES_LAYERS: Tuple[str, ...] = (
    "Isı haritası (VIIRS/MODIS)", "Termal noktalar", "Tahliye rotaları", "Saha ekipleri",
    "Yerleşimler", "Toplanma alanları", bm.FIRE_LABEL,
)
#: Varsayılan açık katmanlar: gerçek yangın katmanı sentetik senaryoyu karıştırmasın diye kapalı başlar.
RES_DEFAULT_LAYERS: Tuple[str, ...] = tuple(name for name in RES_LAYERS if name != bm.FIRE_LABEL)
HEAT_GRADIENT: Dict[float, str] = {0.2: "#4a0d67", 0.4: "#d00000", 0.65: "#ff7b00", 0.85: "#ffd60a", 1.0: "#ffffff"}


def _step_time(delta: int) -> None:
    """Simülasyon saatini ``delta`` kadar ilerletir/geri alır (düğme geri çağrısı).

    Args:
        delta: Saat farkı (negatif = geri).
    """
    st.session_state["res_t"] = int(np.clip(int(st.session_state.get("res_t", 18)) + delta, 0, 72))


def _div_pill(text: str, color: str, glow: bool = True) -> str:
    """Harita üzerinde kullanılan küçük etiket (hap) HTML'i.

    Args:
        text: Etiket metni.
        color: Arka plan rengi.
        glow: Parlama gölgesi eklensin mi.

    Returns:
        HTML dizgesi.
    """
    shadow = f"box-shadow:0 0 10px {color};" if glow else ""
    return (f'<div style="display:inline-block;padding:2px 8px;border-radius:999px;background:{color};'
            f'color:#0b0c10;font:700 10px \'Segoe UI\',sans-serif;white-space:nowrap;{shadow}">{_h(text)}</div>')


def build_rescue_map(sc: de.RescueScenario, snap: de.RescueSnapshot, layers: Sequence[str], basemap: bm.BasemapChoice) -> folium.Map:
    """Termal anomali ısı haritası, tahliye rotaları ve saha ekiplerini içeren dinamik haritayı kurar.

    Args:
        sc: Sabit senaryo.
        snap: Anlık kriz görüntüsü.
        layers: Açık katman adları (``RES_LAYERS`` alt kümesi).
        basemap: Seçili harita altlığı ve görüntü günü.

    Returns:
        Hazır ``folium.Map``.
    """
    overlay = build_map_overlay("AFET · CANLI OPERASYON", f"T+{snap.t_hours:.0f}", "saat",
                                (f"{len(snap.heat_points)} aktif anomali", f"{len(layers)} katman açık"))
    fmap = make_base_map(sc.centre, 11, basemap, overlay)
    if RES_LAYERS[0] in layers and snap.heat_points:
        heat = folium.FeatureGroup(name=RES_LAYERS[0])
        plugins.HeatMap(snap.heat_points, radius=22, blur=26, min_opacity=0.35, max_zoom=14,
                        gradient=HEAT_GRADIENT).add_to(heat)
        heat.add_to(fmap)
    if RES_LAYERS[1] in layers:
        pts = folium.FeatureGroup(name=RES_LAYERS[1])
        for _, r in snap.active.iterrows():
            folium.CircleMarker(
                location=[float(r["enlem"]), float(r["boylam"])], radius=3, weight=0.6,
                color="#ffd60a" if r["kaynak"].startswith("VIIRS") else "#ff7b00", fill=True, fill_opacity=0.9,
                tooltip=(f"{r['kaynak']} · FRP {format_tr(r['frp_mw'], 1)} MW · {format_tr(r['parlaklik_k'], 0)} K "
                         f"· güven: {r['guven']}")).add_to(pts)
        pts.add_to(fmap)
    if RES_LAYERS[2] in layers:
        routes = folium.FeatureGroup(name=RES_LAYERS[2])
        for _, r in snap.routes.iterrows():
            path = sc.route_paths[r["yerlesim"]]
            dash = {"Açık": None, "Riskli": "10 8", "Kapalı": "2 8"}[r["durum"]]
            folium.PolyLine(
                locations=path.tolist(), color=ROUTE_COLORS[r["durum"]], weight=5 if r["durum"] == "Açık" else 4,
                opacity=0.92, dash_array=dash,
                tooltip=(f"{r['yerlesim']} → {r['toplanma']} · {format_tr(r['uzunluk_km'], 1)} km · "
                         f"{int(r['sure_dk'])} dk · {r['durum']}")).add_to(routes)
        routes.add_to(fmap)
    if RES_LAYERS[4] in layers:
        sett = folium.FeatureGroup(name=RES_LAYERS[4])
        for _, r in snap.settlements.iterrows():
            folium.CircleMarker(
                location=[float(r["enlem"]), float(r["boylam"])], radius=6 + 10 * float(np.sqrt(r["nufus"] / 9000.0)),
                color="#ffffff", weight=1.2, fill=True, fill_color=RISK_COLORS[r["risk"]], fill_opacity=0.88,
                tooltip=f"{r['yerlesim']} · {r['risk']} risk · {format_tr(r['nufus'])} kişi",
                popup=folium.Popup(popup_html(
                    str(r["yerlesim"]),
                    [("Nüfus", format_tr(r["nufus"])), ("Etkilenen nüfus", format_tr(r["etkilenen_nufus"])),
                     ("En yakın anomali", f"{format_tr(r['en_yakin_km'], 1)} km"),
                     ("Rüzgâr altı faktörü", format_tr(r["ruzgar_alti"], 2))],
                    tag=(f"{r['risk']} risk", RISK_COLORS[r["risk"]])), max_width=280)).add_to(sett)
        sett.add_to(fmap)
    if RES_LAYERS[5] in layers:
        assembly = folium.FeatureGroup(name=RES_LAYERS[5])
        for _, r in sc.assembly.iterrows():
            folium.Marker(
                location=[float(r["enlem"]), float(r["boylam"])],
                tooltip=f"{r['ad']} · kapasite {format_tr(r['kapasite'])} kişi",
                icon=folium.DivIcon(icon_size=(26, 26), icon_anchor=(13, 13), html=(
                    '<div style="width:26px;height:26px;border-radius:7px;background:#4cc38a;color:#0b0c10;'
                    'font:800 14px \'Segoe UI\',sans-serif;display:flex;align-items:center;justify-content:center;'
                    'box-shadow:0 0 0 3px rgba(76,195,138,.25)">T</div>'))).add_to(assembly)
        assembly.add_to(fmap)
    if RES_LAYERS[3] in layers:
        teams = folium.FeatureGroup(name=RES_LAYERS[3])
        targets = snap.settlements.set_index("yerlesim")[["enlem", "boylam"]]
        for _, r in snap.teams.iterrows():
            color = UNIT_COLORS.get(str(r["birim"]), COLOR_ACCENT)
            if r["hedef"] in targets.index and r["durum"] != "Konuşlanıyor":
                tgt = targets.loc[r["hedef"]]
                folium.PolyLine([[float(r["enlem"]), float(r["boylam"])], [float(tgt["enlem"]), float(tgt["boylam"])]],
                                color=color, weight=1.4, opacity=0.7, dash_array="2 6").add_to(teams)
            folium.Marker(
                location=[float(r["enlem"]), float(r["boylam"])],
                tooltip=f"{r['ekip']} · {r['birim']} · {r['durum']} · hedef: {r['hedef']}",
                icon=folium.DivIcon(icon_size=(70, 18), icon_anchor=(35, 9), html=_div_pill(str(r["ekip"]), color))
            ).add_to(teams)
        teams.add_to(fmap)
    if bm.FIRE_LABEL in layers:
        for options in bm.fire_wms_options(basemap.day):
            folium.WmsTileLayer(**options).add_to(fmap)
    folium.LayerControl(collapsed=True, position="topright").add_to(fmap)
    return fmap


def build_fire_timeline(tl: pd.DataFrame, t: int) -> go.Figure:
    """Aktif termal anomali ve etkilenen nüfusun zaman serisi grafiği.

    Args:
        tl: ``load_rescue_timeline`` çıktısı.
        t: Mevcut simülasyon saati (dikey işaret).

    Returns:
        ``go.Figure``.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tl["saat"], y=tl["aktif"], name="Aktif termal anomali", mode="lines", fill="tozeroy",
                             line=dict(color="#ff7b00", width=2), fillcolor="rgba(255,123,0,0.25)",
                             hovertemplate="T+%{x} sa<br>Aktif anomali: %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=tl["saat"], y=tl["etkilenen"], name="Etkilenen nüfus", mode="lines", yaxis="y2",
                             line=dict(color=COLOR_DANGER, width=2.2),
                             hovertemplate="T+%{x} sa<br>Etkilenen nüfus: %{y:,.0f}<extra></extra>"))
    fig.add_shape(type="line", xref="x", yref="paper", x0=t, x1=t, y0=0, y1=1,
                  line=dict(color=COLOR_ACCENT, width=1.6, dash="dash"))
    fig.add_annotation(x=t, xref="x", y=1, yref="paper", yanchor="bottom", showarrow=False,
                       text=f"T+{t} sa", font=dict(size=11, color=COLOR_ACCENT))
    style_figure(fig, 330, "OLAY ZAMAN ÇİZELGESİ")
    fig.update_xaxes(title_text="Başlangıçtan geçen saat", dtick=12)
    fig.update_yaxes(title_text="Aktif anomali (piksel)")
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Etkilenen nüfus", showgrid=False,
                                  linecolor="rgba(252,61,33,0.4)"))
    return fig


def build_population_chart(snap: de.RescueSnapshot) -> go.Figure:
    """Yerleşim bazında etkilenen nüfus (risk sınıfı renkli) yatay çubuk grafiği.

    Args:
        snap: Anlık kriz görüntüsü.

    Returns:
        ``go.Figure``.
    """
    data = snap.settlements.sort_values("etkilenen_nufus", ascending=True)
    fig = go.Figure(go.Bar(
        y=data["yerlesim"], x=data["etkilenen_nufus"], orientation="h",
        marker=dict(color=[RISK_COLORS[r] for r in data["risk"]], line=dict(color="rgba(11,12,16,0.9)", width=1)),
        customdata=np.stack([data["nufus"], data["risk"], data["en_yakin_km"]], axis=-1),
        text=[f"{r}" for r in data["risk"]], textposition="auto",
        hovertemplate=("<b>%{y}</b><br>Etkilenen: %{x:,.0f} / %{customdata[0]:,.0f} kişi<br>"
                       "Risk: %{customdata[1]}<br>En yakın anomali: %{customdata[2]} km<extra></extra>")))
    style_figure(fig, 330, "YERLEŞİM BAZLI ETKİLENEN NÜFUS", legend_top=False)
    fig.update_xaxes(title_text="Etkilenen nüfus (kişi)")
    fig.update_layout(showlegend=False, bargap=0.25)
    return fig


def render_rescue_tab() -> None:
    """KURTARMA sekmesini çizer: zaman kontrolü, acil durum KPI paneli, dinamik harita ve operasyon tabloları."""
    auth.require(session_principal(), auth.PERM_SECTORS)
    cfg = get_settings()
    sc = load_rescue_scenario(cfg.seed)
    render_html(build_hero_html(
        "GENİŞLEME · AFET", "AFET VE KRİZ", "KOMUTA PANELİ",
        "VIIRS/MODIS benzeri termal anomaliler, etkilenen nüfus, tahliye rotaları ve saha ekibi konumları.",
        ["DEMO · SENTETİK VERİ", "VIIRS 375 m · MODIS 1 km", f"RÜZGÂR {sc.wind_bearing_deg:.0f}° (KD)"]))

    st.session_state.setdefault("res_t", 18)
    render_html(build_section_title("OLAY ZAMANI", "kaydırıcıyı oynatın ya da düğmelerle ilerletin"))
    with panel():
        c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
        t = int(c1.slider("Olay başlangıcından geçen süre (saat)", 0, int(sc.horizon_h), key="res_t"))
        c2.button("◀  −6 SA", on_click=_step_time, args=(-6,), key="res_b_back")
        c3.button("+1 SA  ▶", on_click=_step_time, args=(1,), key="res_b_fwd")
        c4.button("+6 SA  ▶▶", on_click=_step_time, args=(6,), key="res_b_fwd6")
        layers = st.multiselect("Katmanlar", list(RES_LAYERS), default=list(RES_DEFAULT_LAYERS), key="res_layers")
        if bm.FIRE_LABEL in layers:
            render_html(
                f'<div class="note info"><b>Gerçek veri · {cfg.basemap.day.isoformat()} (UTC).</b> {_h(bm.FIRE_NOTE)} '
                "Günü kenar çubuğundaki “Görüntü günü” alanından değiştirebilirsiniz.</div>")

    snap = load_rescue_snapshot(cfg.seed, t)
    k = snap.kpis
    need_evac = float(snap.settlements.loc[snap.settlements["risk"].isin(["Kritik", "Yüksek"]), "nufus"].sum())
    render_html(build_section_title("ACİL DURUM KPI PANELİ", f"{snap.timestamp:%d.%m.%Y %H:%M} · T+{t} saat"))
    render_html(build_kpi_grid([
        Kpi("Etkilenen Nüfus", format_tr(k["affected_pop"]),
            f"toplamın <b>%{format_tr(100 * k['affected_pop'] / max(k['total_pop'], 1), 1)}</b>'i · {format_tr(k['total_pop'])} kişi",
            COLOR_DANGER, alert=k["affected_pop"] > 0),
        Kpi("Kritik Risk Bölgesi", f"{int(k['critical_zones'])}", f"Yüksek risk: <b>{int(k['high_zones'])}</b> bölge",
            COLOR_DANGER, alert=k["critical_zones"] > 0),
        Kpi("Aktif Termal Anomali", format_tr(k["active_hotspots"]), f"FRP toplamı <b>{format_tr(k['frp_total_mw'])}</b> MW",
            COLOR_AMBER),
        Kpi("Yanan Alan (kümülatif)", f"{format_tr(k['burned_ha'])} ha", f"T+{t} saatte", COLOR_AMBER),
    ], columns=4))
    render_html(build_kpi_grid([
        Kpi("Tahliye Gereken Nüfus", format_tr(need_evac), "kritik + yüksek riskli yerleşimler"),
        Kpi("Tahliye Rotaları", f"{int(k['routes_open'])} · {int(k['routes_risky'])} · {int(k['routes_closed'])}",
            "açık · riskli · kapalı"),
        Kpi("Saha Ekipleri", f"{int(k['teams_on_site'])} / {int(k['teams_total'])}", "sahada / toplam ekip"),
        Kpi("Rüzgâr", f"{sc.wind_bearing_deg:.0f}° KD", "yangın bu yöne ilerliyor"),
    ], columns=4))

    left, right = st.columns([3, 2])
    with left:
        show_map(build_rescue_map(sc, snap, layers, cfg.basemap), key="res_map", height=640)
        render_html(
            build_legend_html("RİSK", [(c, r) for r, c in RISK_COLORS.items()], dot=True)
            + build_legend_html("ROTA", [(c, r) for r, c in ROUTE_COLORS.items()])
            + build_legend_html("EKİP", [(c, u) for u, c in UNIT_COLORS.items()], dot=True))
    with right:
        show_figure(build_fire_timeline(load_rescue_timeline(cfg.seed), t), key="res_timeline")
        show_figure(build_population_chart(snap), key="res_population")

    render_html(build_section_title("OPERASYON TABLOLARI", "yerleşim riski · tahliye rotaları · saha ekipleri"))
    tab_a, tab_b, tab_c = st.tabs(["Yerleşim riski", "Tahliye rotaları", "Saha ekipleri"])
    cc = st.column_config
    with tab_a:
        view = snap.settlements.sort_values("etkilenen_nufus", ascending=False)[
            ["yerlesim", "risk", "nufus", "etkilenen_nufus", "en_yakin_km", "ruzgar_alti"]]
        show_table(view, {
            "yerlesim": cc.TextColumn("Yerleşim"), "risk": cc.TextColumn("Risk sınıfı"),
            "nufus": cc.NumberColumn("Nüfus", format="%d"), "etkilenen_nufus": cc.NumberColumn("Etkilenen", format="%d"),
            "en_yakin_km": cc.NumberColumn("En yakın anomali (km)", format="%.1f"),
            "ruzgar_alti": cc.ProgressColumn("Rüzgâr altı faktörü", min_value=0.0, max_value=1.0, format="%.2f"),
        }, height=330)
        st.download_button("↓  YERLEŞİM RİSK TABLOSU (CSV)", data=csv_bytes(view), file_name=f"yerlesim_riski_T{t}.csv",
                           mime="text/csv", key="res_download")
    with tab_b:
        if snap.routes.empty:
            render_html('<div class="note info">Bu saatte tahliye gerektiren yerleşim bulunmuyor.</div>')
        else:
            show_table(snap.routes, {
                "yerlesim": cc.TextColumn("Yerleşim"), "toplanma": cc.TextColumn("Toplanma alanı"),
                "uzunluk_km": cc.NumberColumn("Uzunluk (km)", format="%.1f"), "sure_dk": cc.NumberColumn("Süre (dk)", format="%d"),
                "durum": cc.TextColumn("Rota durumu"), "min_yangin_km": cc.NumberColumn("Yangına min. uzaklık (km)", format="%.2f"),
                "nufus": cc.NumberColumn("Nüfus", format="%d"),
            }, height=300)
    with tab_c:
        show_table(snap.teams[["ekip", "birim", "durum", "hedef", "ilerleme_pct", "enlem", "boylam"]], {
            "ekip": cc.TextColumn("Ekip"), "birim": cc.TextColumn("Birim"), "durum": cc.TextColumn("Durum"),
            "hedef": cc.TextColumn("Hedef yerleşim"),
            "ilerleme_pct": cc.ProgressColumn("Etap ilerlemesi", min_value=0, max_value=100, format="%.0f"),
            "enlem": cc.NumberColumn("Enlem", format="%.5f"), "boylam": cc.NumberColumn("Boylam", format="%.5f"),
        }, height=330)


# =========================================================================== #
# GİRİŞ VE ROLLER
# =========================================================================== #
ACCOUNTS_ENV = "ORBITAL_ACCOUNTS_JSON"
SESSION_USER, SESSION_SEEN = "auth_user", "auth_seen"
SESSION_LOGIN, SESSION_NOTICE = "auth_login_state", "auth_notice"
#: Oturumsuz ziyaretçinin gördüğü ekran: karşılama sayfası ya da giriş formu.
SESSION_VIEW, VIEW_LANDING, VIEW_LOGIN = "ui_view", "landing", "login"
#: Karşılama sitesinde açık sayfa (``landing.PAGES`` kimliklerinden biri).
SESSION_PAGE = "ui_page"
#: Oturum kapalıyken yan menü gösterilmez (boş bir şerit kalmasın).
HIDE_SIDEBAR_CSS = ('<style>section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], '
                    '[data-testid="collapsedControl"]{display:none !important}</style>')
#: Müşteri oturumu her zaman bu tohum ve varsayılan kurallarla çalışır (müşteri hiçbir ayarı değiştiremez).
DEFAULT_SEED = 2026
LOGIN_MESSAGES: Dict[str, str] = {
    "invalid": "Kullanıcı adı ya da parola hatalı.",
    "empty": "Kullanıcı adını ve parolayı girin.",
}


def load_accounts() -> Tuple[Tuple[auth.Account, ...], bool, str]:
    """Geçerli hesapları yükler: ``ORBITAL_ACCOUNTS_JSON`` varsa onu, yoksa demo hesaplarını.

    Ortam değişkeni verilmişse ama bozuksa demo hesaplarına DÜŞÜLMEZ (kapalı kalır).

    Returns:
        ``(hesaplar, demo_mu, yapılandırma_hatası)``.
    """
    raw = os.environ.get(ACCOUNTS_ENV, "").strip()
    if not raw:
        # Barındırma hizmetleri (ör. Streamlit Community Cloud) gizli ayarları ``st.secrets`` ile verir.
        try:
            value = st.secrets.get(ACCOUNTS_ENV, "")
        except Exception:      # secrets.toml yok (yerel çalıştırma) ya da okunamadı
            value = ""
        if value and not isinstance(value, str):
            return (), False, f"{ACCOUNTS_ENV} bir JSON metni olmalı (üç tırnak içinde)."
        raw = value.strip() if isinstance(value, str) else ""
    if not raw:
        return auth.demo_accounts(), True, ""
    try:
        return auth.parse_accounts(raw), False, ""
    except ValueError as exc:
        return (), False, str(exc)


def session_principal() -> Optional[auth.Principal]:
    """Oturumdaki kullanıcıyı döndürür; oturum yoksa ya da kayıt bozuksa ``None``.

    Oturum durumu sunucu tarafındadır; tarayıcı bu kaydı değiştiremez.
    """
    return auth.Principal.from_session(st.session_state.get(SESSION_USER))


def logout(notice: str = "Çıkış yapıldı.") -> None:
    """Oturumu kapatır: giriş sayacı dışında tüm oturum durumunu siler (sonraki kullanıcıya veri kalmaz).

    Args:
        notice: Giriş ekranında bir kez gösterilecek ileti.
    """
    keep = st.session_state.get(SESSION_LOGIN)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    if isinstance(keep, auth.LoginState):
        st.session_state[SESSION_LOGIN] = keep
    st.session_state[SESSION_VIEW] = VIEW_LOGIN       # ileti (ör. "Çıkış yapıldı") giriş ekranında görünsün
    if notice:
        st.session_state[SESSION_NOTICE] = notice


def gate(now: Optional[float] = None) -> Optional[auth.Principal]:
    """Her çalıştırmada oturumu doğrular; yoksa giriş ekranını çizer.

    Args:
        now: Şimdiki an (saniye); ``None`` ise ``time.time()``.

    Returns:
        Oturum açmış kullanıcı ya da (giriş ekranı çizildiyse) ``None``.
    """
    now = time.time() if now is None else now
    raw = st.session_state.get(SESSION_USER)
    principal = auth.Principal.from_session(raw)
    if raw is not None and principal is None:
        logout("Oturum doğrulanamadı; yeniden giriş yapın.")
    elif principal is not None and auth.session_expired(st.session_state.get(SESSION_SEEN), now):
        logout("Uzun süre işlem yapılmadığı için oturumunuz kapatıldı.")
        principal = None
    if principal is None:
        if st.session_state.get(SESSION_VIEW) == VIEW_LOGIN:
            render_login(now)
        else:
            render_landing()
        return None
    st.session_state[SESSION_SEEN] = now
    return principal


def build_demo_accounts_html() -> str:
    """Demo hesaplarının (herkese açık) giriş ipucunu üretir."""
    rows = "".join(
        f"<div><b>{_h(a.display_name)}</b>"
        + (f" · {_h(tenancy.company_name(a.company))}" if a.company else "")
        + f" · <code>{_h(a.username)}</code> / <code>{_h(auth.DEMO_PASSWORDS[a.username])}</code></div>"
        for a in auth.demo_accounts())
    return ('<div class="demo-accounts"><b>DEMO HESAPLARI</b> · herkese açık, yalnızca sentetik veri için<br>'
            f"{rows}</div>")


def show_login() -> None:
    """``on_click`` geri çağrısı: giriş formunu açar."""
    st.session_state[SESSION_VIEW] = VIEW_LOGIN


def show_landing() -> None:
    """``on_click`` geri çağrısı: karşılama sayfasına döner."""
    st.session_state[SESSION_VIEW] = VIEW_LANDING


def read_query_page() -> str:
    """Adres çubuğundaki ``?sayfa=`` değerini okur (yoksa ya da okunamazsa boş metin)."""
    try:
        return str(st.query_params.get(landing.QUERY_KEY, ""))
    except Exception:            # eski sürüm ya da çalışma ortamı sorgu parametresi vermiyor
        return ""


def write_query_page(page: str) -> None:
    """Açık sayfayı adres çubuğuna yazar (paylaşılabilir bağlantı); ana sayfada parametre kaldırılır."""
    try:
        if page == landing.DEFAULT_PAGE:
            if landing.QUERY_KEY in st.query_params:
                del st.query_params[landing.QUERY_KEY]
        else:
            st.query_params[landing.QUERY_KEY] = page
    except Exception:            # adres çubuğu yazılamıyorsa gezinme yine de çalışır
        LOGGER.debug("Sorgu parametresi yazılamadı", exc_info=True)


def show_page(page: str) -> None:
    """``on_click`` geri çağrısı: karşılama sitesinde sayfayı değiştirir (bilinmeyen sayfa ana sayfaya düşer)."""
    pid = landing.valid_page(page)
    st.session_state[SESSION_PAGE] = pid
    st.session_state[SESSION_VIEW] = VIEW_LANDING
    write_query_page(pid)


def current_page() -> str:
    """Açık karşılama sayfası: oturumdaki değer, yoksa adres çubuğundaki ``?sayfa=``, o da yoksa ana sayfa."""
    page = st.session_state.get(SESSION_PAGE)
    if page is None:
        page = st.session_state[SESSION_PAGE] = landing.valid_page(read_query_page())
    return landing.valid_page(page)


def load_contact_email() -> str:
    """İletişim adresini ``ORBITAL_CONTACT_EMAIL`` ortam değişkeninden ya da ``st.secrets``'tan okur ve doğrular.

    Returns:
        Geçerli adres; tanımsız ya da geçersizse boş metin (iletişim düğmesi çizilmez).
    """
    value: Any = os.environ.get(landing.CONTACT_ENV, "")
    if not str(value).strip():
        try:
            value = st.secrets.get(landing.CONTACT_ENV, "")
        except Exception:      # secrets.toml yok ya da okunamadı
            value = ""
    return landing.valid_email(value)


def read_secret(name: str) -> str:
    """Bir gizli ayarı ortam değişkeninden ya da ``st.secrets``'tan okur (yoksa boş metin; değer günlüğe/ekrana yazılmaz).

    Args:
        name: Ayar adı (örn. ``FIRMS_MAP_KEY``).
    """
    value: Any = os.environ.get(name, "")
    if not str(value).strip():
        try:
            value = st.secrets.get(name, "")
        except Exception:      # secrets.toml yok ya da okunamadı
            value = ""
    return str(value or "").strip()


def render_site_nav(current: str) -> None:
    """Karşılama sitesinin üst menü çubuğu: marka, beş sayfa düğmesi ve "Giriş yap".

    Etkin sayfanın düğmesi ``navon_`` ile başlayan anahtar alır (CSS onu vurgular).

    Args:
        current: Açık sayfanın kimliği.
    """
    with st.container(key="lp_nav"):
        cols = st.columns([2.2] + [1.0] * len(landing.PAGES) + [1.05])
        with cols[0]:
            render_html(landing.build_nav_brand())
        for col, (pid, label) in zip(cols[1:-1], landing.PAGES):
            with col:
                st.button(label, key=("navon_" if pid == current else "nav_") + pid, on_click=show_page, args=(pid,))
        with cols[-1]:
            st.button("GİRİŞ YAP", key="lp_login", on_click=show_login)


def render_site_cta(is_demo: bool, where: str, contact: bool = True) -> None:
    """Sayfa sonu çağrı bandı ve düğmeleri (demoyu dene / giriş yap, iletişim).

    Args:
        is_demo: Demo hesaplar gösteriliyorsa ``True``.
        where: Düğme anahtarlarını sayfaya özgü kılan ek (aynı anahtar iki kez çizilemez).
        contact: İletişim sayfasına giden düğme de çizilsin mi.
    """
    render_html(landing.build_cta_band(is_demo))
    left, mid, _ = st.columns([1, 1, 2.4])
    with left:
        st.button("DEMOYU DENE" if is_demo else "GİRİŞ YAP", key=f"lp_pri_login_{where}", on_click=show_login)
    if contact:
        with mid:
            st.button("İLETİŞİM →", key=f"lp_sec_contact_{where}", on_click=show_page, args=("iletisim",))


def render_site_home(is_demo: bool, email: str) -> None:
    """Ana sayfa (siyah, ince yazılı ajans tarzı): kahraman, hakkında + kapsam, hizmet ızgarası, adımlar, çağrı, iletişim.

    Başlık ve kartlar HTML, düğmeler gerçek Streamlit düğmesidir (HTML içine gömülemezler); düğmeler anahtar önekiyle
    (``lp_pri_`` beyaz hap, ``lp_sec_`` çerçeveli hap) biçimlenir.
    """
    try_label = "DEMOYU DENE" if is_demo else "GİRİŞ YAP"
    # kahraman: kısa metin ve iki düğme sağ sütunda
    _, right = st.columns([1.25, 1])
    with right:
        render_html(landing.build_hero_copy())
        b1, b2 = st.columns(2)
        with b1:
            st.button(try_label, key="lp_pri_login_hero", on_click=show_login)
        with b2:
            st.button("HİZMETLERİMİZ →", key="lp_sec_services_hero", on_click=show_page, args=("hizmetler",))
    # hakkında + ürün kapsamı
    about_l, about_r = st.columns([1, 1.15])
    with about_l:
        render_html(landing.build_about_head())
        st.button("HAKKIMIZDA →", key="lp_sec_about", on_click=show_page, args=("hakkimizda",))
    with about_r:
        render_html(landing.build_stats_html())
    # hizmetler
    head_l, head_r = st.columns([3, 1])
    with head_l:
        render_html(landing.build_services_head())
    with head_r:
        with st.container(key="lp_right_btn"):
            st.button("TÜM HİZMETLER →", key="lp_pri_services", on_click=show_page, args=("hizmetler",))
    render_html(landing.build_bento_html())
    # nasıl çalışır
    step_l, step_r = st.columns([3, 1])
    with step_l:
        render_html(landing.build_steps_head())
    with step_r:
        with st.container(key="lp_right_btn_how"):
            st.button("AYRINTILI ANLATIM →", key="lp_sec_how", on_click=show_page, args=("nasil",))
    render_html(landing.build_steps_html())
    render_html(landing.build_status_html(is_demo))
    # çağrı ve iletişim
    render_html(landing.build_offer_html(is_demo))
    with st.container(key="lp_center_btns"):
        _, c1, c2, _ = st.columns([1.2, 1, 1, 1.2])
        with c1:
            st.button(try_label, key="lp_pri_login_offer", on_click=show_login)
        with c2:
            st.button("İLETİŞİM →", key="lp_sec_contact_offer", on_click=show_page, args=("iletisim",))
    render_html(landing.build_contact_html(email, is_demo))


def render_site_about(is_demo: bool) -> None:
    """Hakkımızda: biz neyiz, ilkeler, bulunduğumuz yer (yol haritası) ve geliştirici."""
    render_html(build_section_title("BİZ NEYİZ", "ORBITAL FIELD"))
    render_html(landing.build_about_html())
    render_html(build_section_title("İLKELERİMİZ", "dört söz"))
    render_html(landing.build_cards(landing.PRINCIPLES, columns=4))
    render_html(build_section_title("NEREDEYİZ", "yapılanlar ve planlananlar"))
    render_html(landing.build_cards(landing.ROADMAP, columns=3))
    render_html(build_section_title("GELİŞTİRİCİ"))
    render_html(landing.build_cards(landing.TEAM, columns=2))
    render_html(landing.build_status_html(is_demo))
    render_site_cta(is_demo, "about")


def render_site_services(is_demo: bool) -> None:
    """Hizmetlerimiz: dört alanın ayrıntısı, ortak özellikler ve sigortacı/müşteri rolleri."""
    render_html(build_section_title("ALANLAR", "her birinde ne sunduğumuz ve hangi aşamada olduğumuz"))
    render_html(landing.build_services_html())
    render_html(build_section_title("ORTAK ÖZELLİKLER", "dört alanda da aynı çekirdek"))
    render_html(landing.build_cards(landing.FEATURES, columns=3))
    render_html(build_section_title("KİM NE GÖRÜR", "sigortacı ve müşteri"))
    render_html(landing.build_roles_html())
    render_html(landing.build_status_html(is_demo))
    render_site_cta(is_demo, "services")


def render_site_how(is_demo: bool) -> None:
    """Nasıl çalışır: akış çizgisi, beş adım ve sık sorulan sorular."""
    render_html(build_section_title("AKIŞ", "uydu görüntüsünden kanıt belgesine"))
    render_html(landing.build_pipeline_html())
    render_html(build_section_title("ADIM ADIM"))
    render_html(landing.build_cards(landing.HOW_DETAIL, columns=3))
    render_html(build_section_title("SIKÇA SORULANLAR", "dürüst cevaplar"))
    for question, answer in landing.FAQ:
        with st.expander(question, expanded=False):
            render_html(landing.build_faq_answer(answer))
    render_html(landing.build_status_html(is_demo))
    render_site_cta(is_demo, "how")


def render_site_contact(is_demo: bool, email: str) -> None:
    """İletişim: e-posta düğmesi (adres tanımlıysa), yazarken belirtilecekler ve demo daveti."""
    render_html(build_section_title("HİZMET ALIN", "bize ulaşın"))
    render_html(landing.build_contact_html(email, is_demo))
    render_html(build_section_title("YAZARKEN"))
    render_html(landing.build_contact_hints())
    render_html(landing.build_status_html(is_demo))
    render_site_cta(is_demo, "contact", contact=False)


def render_landing() -> None:
    """Giriş öncesi herkese açık web sitesi: üst menü çubuğu ve beş sayfa (ana sayfa, hakkımızda, hizmetler, ...).

    Hesap, müşteri ya da veri içermez; yalnızca sabit tanıtım metni ve (tanımlıysa) iletişim adresi gösterilir.
    """
    _, is_demo, _ = load_accounts()
    email = load_contact_email()
    render_html(HIDE_SIDEBAR_CSS)
    render_html(f"<style>{landing.LANDING_BG_CSS}</style>")
    page = current_page()
    render_site_nav(page)
    render_html(landing.build_page_header(page))
    if page == "hakkimizda":
        render_site_about(is_demo)
    elif page == "hizmetler":
        render_site_services(is_demo)
    elif page == "nasil":
        render_site_how(is_demo)
    elif page == "iletisim":
        render_site_contact(is_demo, email)
    else:
        render_site_home(is_demo, email)
    render_html(build_footer_html())


def render_login(now: float) -> None:
    """Giriş ekranını çizer ve gönderilen formu doğrular.

    Hata iletileri hep aynıdır (kullanıcı adının var olup olmadığı sızmaz); art arda hatalı denemede
    oturum geçici kilitlenir. Parola, ``session_state``'e yazılmaz (formsuz anahtar yok, gönderimde temizlenir).

    Args:
        now: Şimdiki an (saniye).
    """
    accounts, is_demo, config_error = load_accounts()
    state = st.session_state.get(SESSION_LOGIN)
    if not isinstance(state, auth.LoginState):
        state = st.session_state[SESSION_LOGIN] = auth.LoginState()
    render_html(HIDE_SIDEBAR_CSS)
    _, mid, _ = st.columns([1, 1.25, 1])
    with mid:
        render_html(build_brand_html())
        st.button("← Ana sayfa", on_click=show_landing, key="login_back")
        render_html('<div class="login-head"><div class="hero-eyebrow">GİRİŞ</div>'
                    '<div class="hero-title">HESABINIZA <span>GİRİŞ YAPIN</span></div>'
                    '<div class="hero-sub">Sigortacılar kendi şirketlerinin poliçelerini ve müşteri dosyalarını yönetir; '
                    "müşteriler tüm sigorta şirketlerindeki kendi poliçe ve hasar durumlarını görür.</div></div>")
        notice = st.session_state.pop(SESSION_NOTICE, "")
        if notice:
            render_html(f'<div class="note info">{_h(notice)}</div>')
        if config_error:
            render_html(f'<div class="note err"><b>Hesap yapılandırması geçersiz</b> ({_h(ACCOUNTS_ENV)}): '
                        f"{_h(config_error)} Giriş kapalı.</div>")
            return
        if is_demo:
            render_html('<div class="side-label"><b>HIZLI DENEME</b> · demo hesabıyla tek tıkla</div>')
            q1, q2 = st.columns(2)
            for col, role, label, key in ((q1, auth.Role.CUSTOMER, "MÜŞTERİ OLARAK DENE", "login_try_customer"),
                                          (q2, auth.Role.INSURER, "EKSPERTİZ OLARAK DENE", "login_try_insurer")):
                if col.button(label, key=key):
                    demo = auth.quick_demo_account(accounts, role)
                    quick = auth.authenticate(accounts, demo.username, auth.DEMO_PASSWORDS[demo.username], state, now) if demo else None
                    if quick is not None and quick.code == "ok" and quick.principal is not None:
                        st.session_state[SESSION_USER] = quick.principal.to_session()
                        st.session_state[SESSION_SEEN] = now
                        st.rerun()
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("Kullanıcı adı", max_chars=auth.MAX_USERNAME)
            password = st.text_input("Parola", type="password", max_chars=auth.MAX_PASSWORD)
            submitted = st.form_submit_button("GİRİŞ YAP")
        if submitted:
            result = auth.authenticate(accounts, username, password, state, now)
            if result.code == "ok" and result.principal is not None:
                st.session_state[SESSION_USER] = result.principal.to_session()
                st.session_state[SESSION_SEEN] = now
                st.rerun()
            elif result.code == "locked":
                render_html('<div class="note err">Çok fazla hatalı deneme. '
                            f"{int(result.retry_after) + 1} saniye sonra yeniden deneyin.</div>")
            else:
                render_html(f'<div class="note err">{_h(LOGIN_MESSAGES.get(result.code, LOGIN_MESSAGES["invalid"]))}</div>')
        elif state.locked_until > now:
            render_html('<div class="note err">Çok fazla hatalı deneme. '
                        f"{int(state.locked_until - now) + 1} saniye sonra yeniden deneyin.</div>")
        if is_demo:
            render_html(build_demo_accounts_html())
            st.caption("Bu bir demodur: hesaplar kaynak kodunda yer alır. Gerçek kullanımda "
                       f"{ACCOUNTS_ENV} ile kendi hesaplarınızı verin; demo hesapları kapanır.")


def build_user_card_html(principal: auth.Principal) -> str:
    """Yan menüdeki kullanıcı kartı (rol, ad, kullanıcı adı; sigortacıda şirket ve unvan)."""
    badge = {auth.Role.INSURER: "SİGORTACI", auth.Role.CUSTOMER: "MÜŞTERİ"}[principal.role]   # str.upper() 'i'yi bozar
    company = ""
    if principal.role is auth.Role.INSURER:
        title = f"<i>{_h(principal.title)}</i>" if principal.title else ""
        company = f'<div class="c">{_h(tenancy.company_name(principal.company))}{title}</div>'
    return (f'<div class="who"><div class="r">{badge}</div>'
            f'<div class="n">{_h(principal.display_name)}</div><div class="u">@{_h(principal.username)}</div>{company}</div>')


def build_profile_html(principal: auth.Principal) -> str:
    """Sigortacı profil kartı: baş harfler, ad, unvan ve bağlı olduğu şirket."""
    initials = "".join(w[0] for w in principal.display_name.split()[:2]).upper() or principal.username[:1].upper()
    title = principal.title or "Sigortacı"
    return (f'<div class="profile"><div class="av">{_h(initials)}</div><div><div class="nm">{_h(principal.display_name)}</div>'
            f'<div class="tt">{_h(title)} · @{_h(principal.username)}</div>'
            f'<span class="co">{_h(tenancy.company_name(principal.company))}</span></div></div>')


# =========================================================================== #
# MÜŞTERİ PORTALI (müşteri: kendi verisi · sigortacı: önizleme + iç değerlendirme)
# =========================================================================== #
PORTAL_TABLE_COLUMNS: Tuple[str, ...] = (
    "police_no", "sirket_adi", "varlik_turu", "alan_ha", "sigorta_bedeli_tl", "siddet", "hasar_yuzdesi", "durum",
    "odenecek_tl", "bekleyen_tl",
)
#: Sigortacının iç değerlendirme tablosu: müşterinin ASLA görmediği alanlar.
INTERNAL_TABLE_COLUMNS: Tuple[str, ...] = (
    "police_no", "risk_skoru", "risk_sinifi", "dnbr", "nbr_once", "nbr_sonra", "ndvi_once", "ndvi_sonra", "guven",
    "gerekce",
)


def evaluate_for(seed: int, rules: ClaimRules) -> Tuple[de.InsuranceDataset, pd.DataFrame]:
    """Veri setini ve kurallara göre karar tablosunu üretir.

    Args:
        seed: Veri tohumu.
        rules: Karar kuralları.

    Returns:
        ``(veri seti, karar tablosu)``.
    """
    ds = load_insurance(seed)
    return ds, de.InsuranceSimulator.evaluate_claims(ds.parcels, rules)


def build_verdict_html(row: pd.Series, color: str) -> str:
    """Seçili poliçenin durum kartı: durum etiketi, başlık, sade dilli gerekçe, şirket ve (varsa) şirket notu.

    Hiçbir sayı/eşik/iç not içermez; şirket notu yalnızca sigortacının "müşteriye görünecek" olarak yazdığı metindir.
    """
    by_insurer = str(row["karar_kaynagi"]) != "Otomatik"
    who = "sigorta şirketinizin yetkilisi" if by_insurer else "uydu analizi (otomatik)"
    note = str(row["musteri_notu"]).strip() if by_insurer else ""
    note_html = f'<div class="cnote"><b>{_h(row["sirket_adi"])} notu:</b> {_h(note)}</div>' if note else ""
    return (f'<div class="verdict" style="--vc:{_h(color)}"><div class="st">{_h(row["durum"])}</div>'
            f'<div class="hl">{_h(pt.HEADLINES[str(row["karar"])])}</div><p>{_h(row["aciklama"])}</p>{note_html}'
            f'<div class="by">Sigorta şirketi: <b>{_h(row["sirket_adi"])}</b> · kararı veren: <b>{_h(who)}</b></div></div>')


def build_company_chips_html(policies: pd.DataFrame) -> str:
    """Müşterinin poliçelerinin hangi şirketlerde olduğunu hap etiketleriyle özetler."""
    counts = policies["sirket_adi"].value_counts()
    chips = "".join(f'<span class="chip">{_h(name)} · {int(n)} poliçe</span>' for name, n in counts.items())
    return f'<div class="chips" style="margin:2px 0 14px">{chips}</div>'



def build_tracker_html(steps: Sequence[pt.Step]) -> str:
    """Hasar takip adımlarının dikey zaman çizelgesi."""
    marks = {pt.STEP_DONE: "✓", pt.STEP_CURRENT: "●", pt.STEP_WAITING: "○", pt.STEP_CLOSED: "■"}
    items = "".join(
        f'<li class="{_h(s.state)}"><span class="dot">{marks[s.state]}</span><div><b>{_h(s.label)}</b>'
        f'<span class="d">{_h(s.detail)}</span></div></li>' for s in steps)
    return f'<ol class="tracker">{items}</ol>'


def build_customer_map(ds: de.InsuranceDataset, view: pt.CustomerView, basemap: bm.BasemapChoice,
                       eyebrow: str = "MÜŞTERİ PORTALI") -> folium.Map:
    """Yalnızca müşterinin kendi parsellerini içeren harita (komşu parsel ve iç alan yok).

    Args:
        ds: Sigorta veri seti (yalnızca geometri için).
        view: ``portal.build_view`` çıktısı.
        basemap: Seçili harita altlığı.
        eyebrow: Haritanın üst etiketi.

    Returns:
        Hazır ``folium.Map``.
    """
    geojson = pt.own_geojson(ds.geometry, view, SEVERITY_COLORS)
    pos = list(view.positions)
    lats, lons = ds.geometry.corners_lat[pos].ravel(), ds.geometry.corners_lon[pos].ravel()
    overlay = build_map_overlay(eyebrow, "Parsellerim", "", (f"{len(view.policies)} parsel",))
    fmap = make_base_map((float(lats.mean()), float(lons.mean())), 15, basemap, overlay)
    folium.GeoJson(
        geojson,
        name="Parsellerim",
        style_function=lambda f: {"fillColor": f["properties"]["renk"], "color": "#0b0c10", "weight": 1.2,
                                  "fillOpacity": 0.78},
        highlight_function=lambda f: {"weight": 2.8, "color": COLOR_ACCENT, "fillOpacity": 0.92},
        tooltip=folium.GeoJsonTooltip(fields=["police_no", "durum"], aliases=["Poliçe", "Durum"], sticky=True,
                                      localize=True),
        popup=folium.GeoJsonPopup(
            fields=["police_no", "sirket", "varlik_turu", "alan_txt", "bedel_txt", "siddet", "durum", "odeme_txt",
                    "bekleyen_txt"],
            aliases=["Poliçe no", "Sigorta şirketi", "Varlık", "Alan", "Sigorta bedeli", "Hasar sınıfı", "Durum",
                     "Ödenecek", "Bekleyen"]),
    ).add_to(fmap)
    folium.LayerControl(collapsed=True, position="topright").add_to(fmap)
    fit_bounds(fmap, list(lats), list(lons), 0.0015)
    return fmap


#: Sezon seyri etiketi → renk (müşteri ekranı; ham ölçüm değil).
TRACK_COLORS: Dict[str, str] = {
    "Beklenen seyirde": COLOR_OK, "Hafif geride": COLOR_AMBER, "Geride": "#ec9a4d", "Belirgin geride": COLOR_DANGER,
    de.NOT_TRACKED: COLOR_MUTED,
}


def _signed_pct(value: float) -> str:
    """İşaretli yüzde metni: ``+%5``, ``−%7``, ``%0``."""
    sign = "+" if value >= 0.5 else "−" if value <= -0.5 else ""
    return f"{sign}%{format_tr(abs(value), 0)}"


def build_growth_chart(pview: pt.ProductivityView, police_no: str) -> go.Figure:
    """Seçili parselin bitki gelişim indeksini (beklenen = 100) ve olay sonrası gözlemi çizer.

    Yalnızca müşterinin kendi parselinin göreli indeksi çizilir; ham uydu ölçümü ve başka parsel yoktur.

    Args:
        pview: ``portal.build_productivity`` çıktısı.
        police_no: Takip edilen parselin poliçe numarası.

    Returns:
        ``go.Figure``.
    """
    values = pview.index[police_no]
    dates, last = pview.dates, pview.event_index
    pre_x, pre_y = list(dates[:last]), list(values[:last])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pre_x, y=pre_y, name="Bitki gelişimi", mode="lines+markers",
        line=dict(color=COLOR_ACCENT, width=2.6, shape="spline", smoothing=0.5), marker=dict(size=5),
        hovertemplate="%{x|%d.%m.%Y}<br>Beklenene göre: %{y:.0f}%<extra></extra>"))
    if pre_x:
        fig.add_trace(go.Scatter(
            x=[pre_x[-1], dates[last]], y=[pre_y[-1], float(values[last])], name="Olay etkisi", mode="lines",
            line=dict(color=COLOR_DANGER, width=2.2, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[dates[last]], y=[float(values[last])], name="Olay sonrası gözlem", mode="markers",
        marker=dict(color=COLOR_DANGER, size=11, symbol="diamond", line=dict(color="#ffffff", width=1.2)),
        hovertemplate="%{x|%d.%m.%Y}<br>Beklenene göre: %{y:.0f}%<extra></extra>"))
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y", y0=100, y1=100,
                  line=dict(color=COLOR_MUTED, width=1.2, dash="dot"))
    fig.add_annotation(xref="paper", x=0.0, yref="y", y=100, yanchor="bottom", xanchor="left", showarrow=False,
                       text="beklenen = 100", font=dict(size=10, color=COLOR_MUTED))
    fig.add_shape(type="line", xref="x", yref="paper", x0=dates[last].isoformat(), x1=dates[last].isoformat(),
                  y0=0, y1=1, line=dict(color=COLOR_DANGER, width=1.2, dash="dot"))
    fig.add_annotation(x=dates[last].isoformat(), xref="x", y=1, yref="paper", yanchor="top", xanchor="right",
                       showarrow=False, text="OLAY ", font=dict(size=10, color=COLOR_DANGER))
    style_figure(fig, 380, f"BİTKİ GELİŞİMİ · {police_no} · BEKLENEN = 100")
    top = max(130.0, float(np.nanmax(values)) * 1.1)
    fig.update_yaxes(title_text="Beklenene göre (%)", range=[0, top])
    fig.update_xaxes(tickformat="%b", dtick="M1")
    fig.update_layout(hovermode="x unified")
    return fig


def build_parcel_yield_chart(pview: pt.ProductivityView) -> go.Figure:
    """Takip edilen parsellerin tahmini verimini beklenen verime oranla (beklenen = %100) çubuk olarak çizer.

    Args:
        pview: ``portal.build_productivity`` çıktısı (en az bir takip edilen parsel olmalı).

    Returns:
        ``go.Figure``.
    """
    data = pview.table[pview.table["seyir"] != de.NOT_TRACKED].copy()
    data["endeks"] = 100.0 * data["tahmini_t_ha"] / data["beklenen_t_ha"]
    custom = np.stack([data["varlik_turu"], data["tahmini_t_ha"], data["beklenen_t_ha"], data["olay_kaybi_pct"],
                       data["seyir"]], axis=-1)
    fig = go.Figure(go.Bar(
        x=data["police_no"], y=data["endeks"], customdata=custom,
        marker=dict(color=[TRACK_COLORS[de.track_label(float(v))] for v in data["endeks"]],
                    line=dict(color="rgba(11,12,16,0.9)", width=1)),
        text=[f"{v:.0f}" for v in data["endeks"]], textposition="outside", cliponaxis=False,
        hovertemplate=("<b>%{x}</b> · %{customdata[0]}<br>Tahmini verim endeksi: %{y:.0f}%<br>"
                       "Tahmini: %{customdata[1]:.1f} ton/ha · beklenen: %{customdata[2]:.1f} ton/ha<br>"
                       "Olay kaybı: %{customdata[3]:.0f}% · seyir: %{customdata[4]}<extra></extra>")))
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y", y0=100, y1=100,
                  line=dict(color=COLOR_ACCENT, width=1.2, dash="dot"))
    style_figure(fig, 380, "TAHMİNİ VERİM ENDEKSİ · BEKLENEN = %100", legend_top=False)
    fig.update_yaxes(title_text="Beklenen verime göre (%)", range=[0, max(120.0, float(data["endeks"].max()) * 1.2)])
    fig.update_xaxes(tickangle=-30, type="category")
    fig.update_layout(showlegend=False, bargap=0.35)
    return fig


def render_productivity_section(pview: pt.ProductivityView, police_no: str, prefix: str) -> None:
    """VERİMLİLİK TAKİBİ bölümü: özet kartlar, seçili parselin gelişim eğrisi, parsel karşılaştırması ve tablo.

    Yalnızca ``portal.ProductivityView`` (müşterinin kendi satırları) kullanılır.

    Args:
        pview: Müşterinin verimlilik görünümü.
        police_no: Yukarıdaki seçicide seçili poliçe (parsel).
        prefix: Widget anahtarı öneki (``cp`` müşteri, ``pv`` önizleme).
    """
    render_html(build_section_title("VERİMLİLİK TAKİBİ", f"seçili poliçe · {police_no} · beklenen düzey = 100"))
    if pview.tracked == 0:
        render_html('<div class="note info">Poliçelerinizdeki varlık türleri için bitki verimi takibi yapılmıyor.</div>')
        return
    assert pview.mean_score is not None and pview.mean_loss is not None and pview.forecast_index is not None

    table = pview.table
    row = table[table["police_no"] == police_no].iloc[0]
    tracked_here = police_no in pview.index
    if tracked_here:
        render_html(build_kpi_grid([
            Kpi("Tahmini verim", f"{format_tr(float(row['tahmini_t_ha']), 1)} ton/ha",
                f"beklenen {format_tr(float(row['beklenen_t_ha']), 1)} ton/ha"),
            Kpi("Tahmini toplam üretim", f"{format_tr(float(row['tahmini_toplam_t']), 1)} ton",
                f"beklenen {format_tr(float(row['beklenen_toplam_t']), 1)} ton"),
            Kpi("Beklenene göre", _signed_pct(float(row["fark_pct"])),
                f"sezon seyri %{format_tr(float(row['verim_skoru']), 0)} · olay kaybı %{format_tr(float(row['olay_kaybi_pct']), 0)}",
                TRACK_COLORS[de.track_label(100.0 + float(row["fark_pct"]))]),
        ], columns=3))
    left, right = st.columns([3, 2])
    with left:
        if tracked_here:
            show_figure(build_growth_chart(pview, police_no), key=f"{prefix}_growth")
        else:
            render_html(f'<div class="note info">{_h(pt.productivity_text(row))}</div>')
    with right:
        show_figure(build_parcel_yield_chart(pview), key=f"{prefix}_yield")
    if tracked_here:
        render_html(f'<div class="note info"><b>{_h(police_no)}</b> · {_h(row["varlik_turu"])} — '
                    f'{_h(pt.productivity_text(row))}</div>')
    cc = st.column_config
    with st.expander("Parsellerimin verimlilik tablosu"):
        show_table(table[list(pt.PRODUCTIVITY_COLUMNS)], {
            "police_no": cc.TextColumn("Poliçe No"), "parsel_id": cc.TextColumn("Parsel"),
            "varlik_turu": cc.TextColumn("Varlık"), "alan_ha": cc.NumberColumn("Alan (ha)", format="%.1f"),
            "verim_skoru": cc.NumberColumn("Sezon seyri %", format="%.0f"),
            "olay_kaybi_pct": cc.NumberColumn("Olay kaybı %", format="%.0f"),
            "beklenen_t_ha": cc.NumberColumn("Beklenen (t/ha)", format="%.1f"),
            "tahmini_t_ha": cc.NumberColumn("Tahmini (t/ha)", format="%.1f"),
            "fark_pct": cc.NumberColumn("Fark %", format="%.0f"),
            "beklenen_toplam_t": cc.NumberColumn("Beklenen (t)", format="%.1f"),
            "tahmini_toplam_t": cc.NumberColumn("Tahmini (t)", format="%.1f"),
            "seyir": cc.TextColumn("Seyir"),
        }, height=min(360, 76 + 35 * len(table)))
    seyir = de.track_label(pview.mean_score)
    render_html(f'<div class="note">Tüm parselleriniz: {pview.tracked} / {len(pview.table)} parsel takipte · '
                f"ortalama sezon seyri %{format_tr(pview.mean_score, 0)} ({_h(seyir.lower())}) · "
                f"olay etkisi −%{format_tr(pview.mean_loss, 0)} · "
                f"tahmini verim endeksi %{format_tr(pview.forecast_index, 0)} (alan ağırlıklı). "
                "Veriler sentetik demodur; beklenen verimler <b>örnek referans değerlerdir</b> (gerçek kayıtlarla "
                "kalibre edilmedi). Tahmin, olay tarihine kadarki bitki gelişimi ile hasar oranından basit bir modelle "
                "hesaplanır ve gerçek hasadı garanti etmez.</div>")


#: Müşteri ekranındaki sekmeler (sırayla): özet+harita, seçili poliçenin hasarı, verimlilik, belgeler.
PORTAL_TABS: Tuple[str, ...] = ("Özet", "Hasar durumum", "Verimlilik", "Belgeler")
#: Karar kodu → özet kartında görünen kısa ad.
PORTAL_STATUS_TEXT: Tuple[Tuple[str, str], ...] = (
    ("ONAY", "onay"), ("KISMİ ONAY", "kısmi onay"), ("EKSPERTİZ", "ekspertiz"), ("RET", "ret"),
)


def render_portal_view(ds: de.InsuranceDataset, claims: pd.DataFrame, rules: ClaimRules, name: str,
                       basemap: bm.BasemapChoice, prefix: str, seed: int,
                       company: Optional[str] = None) -> pt.CustomerView:
    """Bir müşterinin göreceği ekranı çizer (müşteri oturumu ve sigortacı önizlemesi aynı kodu kullanır).

    Yalnızca ``portal.build_view`` çıktısı kullanılır: başka müşterinin satırı, risk skoru ve ham ölçüm bu
    fonksiyona hiç girmez.

    Args:
        ds: Sigorta veri seti.
        claims: Tüm karar tablosu (satır süzmesi ``build_view``'da yapılır).
        rules: Kararı üreten kurallar.
        name: Müşteri adı (``sigortali``).
        basemap: Seçili harita altlığı.
        prefix: Widget anahtarı öneki (``cp`` müşteri, ``pv`` önizleme).
        seed: Veri tohumu (verimlilik takibi için).
        company: Verilirse yalnızca bu şirketin poliçeleri (sigortacı önizlemesi); ``None`` ise müşterinin tüm
            şirketlerdeki poliçeleri (müşteri oturumu).

    Returns:
        Çizilen ``CustomerView``.
    """
    view = pt.build_view(claims, name, rules, company)
    cnt = view.counts
    options = list(view.policies["police_no"])
    company_of = dict(zip(view.policies["police_no"], view.policies["sirket_adi"]))
    pick_key = f"{prefix}_pick"
    if st.session_state.get(pick_key) not in options:
        st.session_state[pick_key] = options[0]
    pick = st.selectbox("Seçili poliçe", options, key=pick_key, format_func=lambda p: f"{p} · {company_of.get(p, '—')}")
    idx = options.index(pick)
    row, pos = view.policies.iloc[idx], view.positions[idx]
    decision = str(row["karar"])

    tab_summary, tab_claim, tab_prod, tab_docs = st.tabs(list(PORTAL_TABS))

    with tab_summary:
        status = " · ".join(f"{cnt[key]} {text}" for key, text in PORTAL_STATUS_TEXT if cnt[key])
        render_html(build_kpi_grid([
            Kpi("Poliçelerim" if company is None else "Poliçeler", format_tr(len(view.policies)),
                f"{view.policies['sirket_adi'].nunique()} şirkette · " + (status or "durum bilgisi yok")),
            Kpi("Toplam sigorta bedeli", f"₺{format_tr(view.total_insured, 0)}", "tüm poliçeleriniz"),
            Kpi("Ödenecek tutar", f"₺{format_tr(view.total_paid, 0)}", "onaylanan / avans", COLOR_OK),
            Kpi("Bekleyen tutar", f"₺{format_tr(view.total_pending, 0)}", "ekspertiz sonrası belirlenir", COLOR_AMBER),
        ], columns=4))
        render_html(build_company_chips_html(view.policies))
        show_map(build_customer_map(ds, view, basemap), key=f"{prefix}_map", height=430)
        render_html(build_legend_html("HASAR SINIFI", list(zip(SEVERITY_COLORS, de.SEVERITY_LABELS))))
        render_html(build_section_title("POLİÇELERİM", f"müşteri no {view.number} · yalnızca size ait poliçeler"))
        cc = st.column_config
        show_table(view.policies[list(PORTAL_TABLE_COLUMNS)], {
            "police_no": cc.TextColumn("Poliçe No"), "sirket_adi": cc.TextColumn("Sigorta şirketi"),
            "varlik_turu": cc.TextColumn("Varlık"),
            "alan_ha": cc.NumberColumn("Alan (ha)", format="%.1f"),
            "sigorta_bedeli_tl": cc.NumberColumn("Sigorta bedeli (₺)", format="%.0f"),
            "siddet": cc.TextColumn("Hasar sınıfı"),
            "hasar_yuzdesi": cc.ProgressColumn("Hasar oranı %", min_value=0, max_value=100, format="%.0f"),
            "durum": cc.TextColumn("Durum"), "odenecek_tl": cc.NumberColumn("Ödenecek (₺)", format="%.0f"),
            "bekleyen_tl": cc.NumberColumn("Bekleyen (₺)", format="%.0f"),
        }, height=min(360, 76 + 35 * len(view.policies)))

    with tab_claim:
        steps = pt.steps_for(decision, ds.event_date, float(row["odenecek_tl"]), float(row["bekleyen_tl"]),
                             rules.advance_pct, insurer=str(row["karar_kaynagi"]) != "Otomatik")
        render_html(build_section_title("HASAR DURUMU", f"seçili poliçe · {pick}"))
        render_html(build_kpi_grid([
            Kpi("Hasar oranı", f"%{format_tr(float(row['hasar_yuzdesi']), 0)}", _h(str(row["siddet"]))),
            Kpi("Ödenecek", f"₺{format_tr(float(row['odenecek_tl']), 0)}", color=COLOR_OK),
            Kpi("Bekleyen", f"₺{format_tr(float(row['bekleyen_tl']), 0)}", color=COLOR_AMBER),
        ], columns=3))
        left, right = st.columns([3, 2])
        with left:
            render_html(build_verdict_html(row, DECISION_COLORS[decision]))
        with right:
            render_html(build_tracker_html(steps))

    with tab_prod:
        prod = de.simulate_productivity(ds.parcels, ds.event_date, claims["hasar_yuzdesi"].to_numpy(), seed)
        pview = pt.build_productivity(prod, view)   # tüm parsellerin verisi burada süzülür; sonrası yalnızca kendi satırları
        del prod
        render_productivity_section(pview, str(pick), prefix)

    with tab_docs:
        render_html(build_section_title("KARAR BİLDİRİMİ", f"seçili poliçe · {pick}"))
        notice = pt.build_notice_html(claims.iloc[pos], name, rules, ds.event_label, ds.event_date,
                                      dt.datetime.now(de.TR_TZ), ds.geometry, pos, SEVERITY_COLORS)
        d1, d2 = st.columns([1, 2])
        d1.download_button("↓  KARAR BİLDİRİMİ (HTML)", data=notice.encode("utf-8"),
                           file_name=pt.notice_filename(str(pick), decision), mime="text/html", key=f"{prefix}_notice")
        if d2.checkbox("Bildirimi burada önizle", value=False, key=f"{prefix}_notice_preview"):
            show_html_frame(notice, height=1100)
        render_html('<div class="note info">Bu ekrandaki bilgiler sentetik demo verisinden üretilmiştir; gerçek bir hasar '
                    "kaydı değildir. İtiraz ve belge yükleme bu prototipte yoktur.</div>")
    return view


def render_customer_page() -> None:
    """Müşterinin tek sayfası: yalnızca kendi poliçe ve hasar durumu (varsayılan tohum ve kurallar).

    Raises:
        PermissionError: Oturum bir müşteri oturumu değilse.
    """
    principal = session_principal()
    auth.require(principal, auth.PERM_PORTAL_OWN)
    assert principal is not None
    rules = ClaimRules()
    ds, auto = evaluate_for(DEFAULT_SEED, rules)
    name = pt.resolve_customer(auto, principal)     # eşleme OTOMATİK tablodan: sigortacı kararları müşteri sıralamasını değiştirmez
    decisions, store_error = read_decisions(None, DEFAULT_SEED)      # tüm şirketler; iç not ve veren bilgisi boşaltılmış
    claims = cw.apply_decisions(auto, decisions)
    if name is None:
        render_html(build_hero_html("MÜŞTERİ PORTALI", "HASAR", "TAKİBİM",
                                    "Poliçelerinizin hasar durumu, ödeme adımları ve parsellerinizin verimlilik takibi.",
                                    ["DEMO · SENTETİK VERİ"]))
        render_html('<div class="note">Hesabınıza bağlı poliçe bulunamadı. Lütfen sigortacınızla iletişime geçin.</div>')
        return
    n_companies = int(claims.loc[claims["sigortali"] == name, "sirket"].nunique())
    render_html(build_hero_html(
        "MÜŞTERİ PORTALI", "HASAR", "TAKİBİM",
        "Tüm sigorta şirketlerindeki poliçelerinizin hasar durumu, ödeme adımları ve parsellerinizin verimlilik takibi.",
        ["DEMO · SENTETİK VERİ", f"MÜŞTERİ NO: {pt.customer_no(name)}", f"{n_companies} SİGORTA ŞİRKETİ",
         f"OLAY SONRASI: {ds.event_date:%d.%m.%Y}"]))
    if store_error:
        render_html('<div class="note info">Sigorta şirketlerinin verdiği kararlar şu an alınamıyor; yalnızca otomatik '
                    "uydu analizi sonuçları gösteriliyor.</div>")
    render_portal_view(ds, claims, rules, name, get_settings().basemap, "cp", DEFAULT_SEED)


# =========================================================================== #
# SEKME — MÜŞTERİ DOSYALARI (sigortacı: müşteri aç, incele, karar ver)
# =========================================================================== #
CASE_TABS: Tuple[str, ...] = ("Dosya", "Karar ver", "Müşterinin gördüğü", "Geçmiş")
#: Kararı kaydedince bir kez gösterilen ileti ve form anahtarlarını yenileyen sayaç (``session_state`` anahtarları).
CASE_FLASH, CASE_GEN = "cf_flash", "cf_gen"
CASE_MANUAL_MODES: Tuple[str, ...] = ("Hesaplanan tutar", "Tutarı elle gir")
#: Karar seçeneklerinin sırası (arayüzde): onay en önde.
CASE_CHOICES: Tuple[str, ...] = ("ONAY", "KISMİ ONAY", "EKSPERTİZ", "RET")


def now_iso() -> str:
    """Şimdiki an (Türkiye saati), saniye hassasiyetinde ISO 8601 metni."""
    return dt.datetime.now(de.TR_TZ).replace(microsecond=0).isoformat()


def read_history(company: str, seed: int, police_no: Optional[str] = None) -> Tuple[List[cw.Decision], str]:
    """Şirketin karar kaydını (yeniden eskiye) okur; iç not içerir, yalnızca sigortacı ekranları içindir.

    Args:
        company: Şirket kimliği.
        seed: Veri tohumu.
        police_no: İsteğe bağlı tek poliçe.

    Returns:
        ``(kayıtlar, hata_iletisi)``.
    """
    try:
        return stm.shared_store().history(company, seed, police_no), ""
    except (stm.StoreError, cw.CaseworkError) as exc:
        LOGGER.warning("Karar kaydı okunamadı: %s", exc)
        return [], str(exc)


def build_decision_preview_frame(auto_row: pd.Series, rules: ClaimRules) -> pd.DataFrame:
    """Dört kararın bu poliçe için vereceği tutarlar (kaydetmeden önce karşılaştırma)."""
    rows = []
    for label in CASE_CHOICES:
        d = cw.plan_decision(auto_row, label, rules, None, "", "önizleme")
        rows.append({"karar": DECISION_ICONS[label], "odenecek_tl": d.paid_tl, "bekleyen_tl": d.pending_tl})
    return pd.DataFrame(rows)


def save_decision(principal: Optional[auth.Principal], cd: CompanyData, police_no: str, decision: str,
                  manual_tl: Optional[float], public_note: str, internal_note: str, rules: ClaimRules, seed: int,
                  revert: bool = False) -> str:
    """Sigortacı kararını (ya da geri almayı) doğrulayıp karar defterine yazar.

    Şirket, kullanıcı ve zaman damgası SUNUCUDA oturumdan alınır; arayüzden gelen yalnızca poliçe, karar, tutar ve notlardır.
    Poliçe sigortacının şirketine ait değilse işlem reddedilir.

    Args:
        principal: Oturum açmış sigortacı.
        cd: Şirket verisi (poliçe sahipliği buradan denetlenir).
        police_no: Poliçe numarası.
        decision: Karar (geri almada yok sayılır).
        manual_tl: İsteğe bağlı elle tutar.
        public_note: Müşteriye görünecek not.
        internal_note: İç gerekçe.
        rules: Tutarı hesaplayan kurallar.
        seed: Veri tohumu.
        revert: ``True`` ise otomatik karara dönülür.

    Returns:
        Hata iletisi; başarılıysa boş metin.

    Raises:
        PermissionError: Karar verme izni yoksa.
    """
    auth.require(principal, auth.PERM_DECIDE)
    assert principal is not None
    if seed != DEFAULT_SEED:
        return f"Karar yalnızca varsayılan veri tohumunda ({DEFAULT_SEED}) verilebilir; müşteriler yalnızca bu veriyi görür."
    hit = cd.mine.index[cd.mine["police_no"] == police_no]
    if len(hit) != 1 or cd.company != principal.company:
        return "Bu poliçe şirketinize ait değil; işlem reddedildi."
    auto_row = cd.auto.loc[hit[0]]
    try:
        if revert:
            note = cw.clean_text(internal_note, "İç gerekçe", min_len=cw.INTERNAL_MIN)
            record = cw.Decision(
                company=cd.company, police_no=police_no, seed=seed, action=cw.ACTION_REVERT, decision="", paid_tl=0.0,
                pending_tl=0.0, public_note="", internal_note=note, decided_by=principal.username,
                decided_by_name=principal.display_name[:120], decided_at=now_iso(),
                auto_decision=str(auto_row["karar"]), auto_paid_tl=float(auto_row["odenecek_tl"]))
        else:
            draft = cw.plan_decision(auto_row, decision, rules, manual_tl, public_note, internal_note)
            record = cw.Decision(
                company=cd.company, police_no=police_no, seed=seed, action=cw.ACTION_DECIDE, decision=draft.decision,
                paid_tl=draft.paid_tl, pending_tl=draft.pending_tl, public_note=draft.public_note,
                internal_note=draft.internal_note, decided_by=principal.username,
                decided_by_name=principal.display_name[:120], decided_at=now_iso(),
                auto_decision=str(auto_row["karar"]), auto_paid_tl=float(auto_row["odenecek_tl"]))
        stm.shared_store().add(record)
    except (cw.CaseworkError, stm.StoreError) as exc:
        return str(exc)
    return ""


def render_case_file(cd: CompanyData, own: pd.DataFrame, name: str, pick: str, rules: ClaimRules,
                     cfg: AppSettings) -> None:
    """"Dosya" alt sekmesi: müşterinin (bu şirketteki) poliçeleri, haritası, iç değerlendirme ve kanıt belgesi."""
    s = cw.summarize_final(own)
    render_html(build_kpi_grid([
        Kpi("Poliçe", format_tr(len(own)), f"{format_tr(own['alan_ha'].sum(), 1)} ha · {_h(pt.customer_no(name))}"),
        Kpi("Toplam sigorta bedeli", f"₺{format_tr(own['sigorta_bedeli_tl'].sum(), 0)}", "bu şirketteki poliçeler"),
        Kpi("Ödenecek", f"₺{format_tr(s['paid_tl'], 0)}", "geçerli kararlara göre", COLOR_OK),
        Kpi("Bekleyen", f"₺{format_tr(s['pending_tl'], 0)}",
            f"{int(s['review'])} ekspertiz · {int(s['manual'])} sigortacı kararı", COLOR_AMBER),
    ], columns=4))
    cc = st.column_config
    cols = ["police_no", "varlik_turu", "alan_ha", "sigorta_bedeli_tl", "siddet", "otomatik_karar", "karar",
            "karar_kaynagi", "odenecek_tl", "bekleyen_tl"]
    table = own[cols].copy()
    table["karar"] = table["karar"].map(DECISION_ICONS)
    table["otomatik_karar"] = table["otomatik_karar"].map(DECISION_ICONS)
    show_table(table, {
        "police_no": cc.TextColumn("Poliçe No"), "varlik_turu": cc.TextColumn("Varlık"),
        "alan_ha": cc.NumberColumn("Alan (ha)", format="%.1f"),
        "sigorta_bedeli_tl": cc.NumberColumn("Sigorta bedeli (₺)", format="%.0f"), "siddet": cc.TextColumn("Şiddet"),
        "otomatik_karar": cc.TextColumn("Otomatik öneri"), "karar": cc.TextColumn("Geçerli karar"),
        "karar_kaynagi": cc.TextColumn("Kaynak"), "odenecek_tl": cc.NumberColumn("Ödenecek (₺)", format="%.0f"),
        "bekleyen_tl": cc.NumberColumn("Bekleyen (₺)", format="%.0f"),
    }, height=min(340, 76 + 35 * len(table)))

    view = pt.build_view(cd.final, name, rules, cd.company)
    show_map(build_customer_map(cd.ds, view, cfg.basemap, "MÜŞTERİ DOSYASI"), key="cf_map", height=380)
    render_html(build_legend_html("HASAR SINIFI", list(zip(SEVERITY_COLORS, de.SEVERITY_LABELS))))

    render_html(build_section_title("İÇ DEĞERLENDİRME", "müşteri bunları görmez"))
    internal = own[list(INTERNAL_TABLE_COLUMNS)].reset_index(drop=True)
    show_table(internal, {
        "police_no": cc.TextColumn("Poliçe No"),
        "risk_skoru": cc.ProgressColumn("Risk skoru", min_value=0, max_value=100, format="%.0f"),
        "risk_sinifi": cc.TextColumn("Risk sınıfı"), "dnbr": cc.NumberColumn("dNBR", format="%.3f"),
        "nbr_once": cc.NumberColumn("NBR önce", format="%.3f"), "nbr_sonra": cc.NumberColumn("NBR sonra", format="%.3f"),
        "ndvi_once": cc.NumberColumn("NDVI önce", format="%.3f"), "ndvi_sonra": cc.NumberColumn("NDVI sonra", format="%.3f"),
        "guven": cc.NumberColumn("Güven", format="%.2f"), "gerekce": cc.TextColumn("Gerekçe (iç)"),
    }, height=min(340, 76 + 35 * len(internal)))

    render_html(build_section_title("KANIT BELGESİ", f"seçili poliçe · {pick}"))
    if st.checkbox("Bu poliçenin kanıt belgesini hazırla", value=False, key="cf_evidence"):
        auth.require(session_principal(), auth.PERM_EVIDENCE)
        ctx = evidence_context(cd.ds, rules, cfg.seed, calibration_lines_for(rules))
        fname, document = rep.build_claim_document(cd.auto, pick, ctx, cd.ds.geometry)
        st.download_button("↓  KANIT BELGESİ (HTML)", data=document.encode("utf-8"), file_name=fname,
                           mime="text/html", key="cf_evidence_dl")
        show_html_frame(document, height=1180)
        render_html('<div class="note info">Belge otomatik motorun önerisini belgeler. Sigortacı kararı verildiyse '
                    "o karar Geçmiş sekmesinde kayıtlıdır.</div>")


def render_case_decision(principal: auth.Principal, cd: CompanyData, own: pd.DataFrame, name: str, pick: str,
                         rules: ClaimRules, cfg: AppSettings) -> None:
    """"Karar ver" alt sekmesi: otomatik öneri, geçerli karar ve karar formu (kaydet / otomatiğe dön)."""
    auth.require(principal, auth.PERM_DECIDE)
    row = own[own["police_no"] == pick].iloc[0]
    auto_row = cd.auto.loc[row.name]
    by_insurer = str(row["karar_kaynagi"]) == cw.SOURCE_INSURER
    render_html(build_section_title("KARAR", f"{pick} · {name}"))
    render_html(build_kpi_grid([
        Kpi("Otomatik öneri", DECISION_ICONS[str(auto_row["karar"])], _h(str(auto_row["gerekce"]))),
        Kpi("Geçerli karar", DECISION_ICONS[str(row["karar"])], "sigortacı kararı" if by_insurer else "otomatik öneri"),
        Kpi("Hasar oranı", f"%{format_tr(float(row['hasar_yuzdesi']), 0)}", _h(str(row["siddet"]))),
        Kpi("Ödenecek · bekleyen", f"₺{format_tr(float(row['odenecek_tl']), 0)}",
            f"bekleyen <b>₺{format_tr(float(row['bekleyen_tl']), 0)}</b>"),
    ], columns=4))
    if not bool(row["police_aktif"]):
        render_html('<div class="note err">Bu poliçe olay tarihinde <b>pasif / dönem dışı</b> görünüyor. Onay vermeden önce '
                    "poliçe kaydını kontrol edin.</div>")
    last, last_err = read_history(cd.company, cfg.seed, pick)
    if by_insurer and last:
        d = last[0]
        render_html(f'<div class="note info"><b>Son karar:</b> {_h(d.decided_at.replace("T", " ")[:16])} · '
                    f'{_h(d.decided_by_name)} (@{_h(d.decided_by)}) · iç gerekçe: {_h(d.internal_note)}</div>')
    if cfg.seed != DEFAULT_SEED:
        render_html(f'<div class="note info">Karar vermek için sol menüdeki veri tohumunu varsayılan değerde ({DEFAULT_SEED}) '
                    "tutun: müşteriler yalnızca bu veriyi görür, başka tohumdaki kararlar uygulanmaz.</div>")
        return
    if cd.error or last_err:
        render_html(f'<div class="note err"><b>Karar defteri kullanılamıyor:</b> {_h(cd.error or last_err)} '
                    "Karar kaydedilemez.</div>")
        return

    render_html(build_section_title("KARARA GÖRE TUTARLAR", "kaydetmeden önce karşılaştırın"))
    cc = st.column_config
    show_table(build_decision_preview_frame(auto_row, rules), {
        "karar": cc.TextColumn("Karar"), "odenecek_tl": cc.NumberColumn("Ödenecek (₺)", format="%.0f"),
        "bekleyen_tl": cc.NumberColumn("Bekleyen (₺)", format="%.0f"),
    }, height=190)

    gen = int(st.session_state.get(CASE_GEN, 0))
    tag = f"{pick}_{gen}"
    render_html(build_section_title("KARARI KAYDET", "müşteri kararı, tutarı ve müşteri notunu görür"))
    with st.form(f"cf_form_{tag}", clear_on_submit=False):
        decision = st.radio("Karar", list(CASE_CHOICES), index=CASE_CHOICES.index(str(row["karar"])),
                            format_func=lambda k: DECISION_ICONS[k], horizontal=True, key=f"cf_dec_{tag}")
        mode = st.radio("Tutar", list(CASE_MANUAL_MODES), horizontal=True, key=f"cf_mode_{tag}")
        manual = st.number_input("Elle tutar (₺) — yalnızca ONAY ve KISMİ ONAY için", min_value=0.0,
                                 max_value=float(row["sigorta_bedeli_tl"]), value=0.0, step=100.0, key=f"cf_amt_{tag}")
        public_note = st.text_area("Müşteriye görünecek not (isteğe bağlı)", max_chars=cw.NOTE_MAX, key=f"cf_pub_{tag}",
                                   height=80)
        internal_note = st.text_area(f"İç gerekçe (zorunlu, en az {cw.INTERNAL_MIN} karakter — müşteri görmez)",
                                     max_chars=cw.NOTE_MAX, key=f"cf_int_{tag}", height=80)
        submitted = st.form_submit_button("KARARI KAYDET")
    if submitted:
        manual_tl = float(manual) if mode == CASE_MANUAL_MODES[1] else None
        err = save_decision(principal, cd, pick, str(decision), manual_tl, str(public_note), str(internal_note), rules,
                            cfg.seed)
        if err:
            render_html(f'<div class="note err">{_h(err)}</div>')
        else:
            st.session_state[CASE_FLASH] = f"{pick} için karar kaydedildi: {DECISION_ICONS[str(decision)]}."
            st.session_state[CASE_GEN] = gen + 1
            st.rerun()

    if by_insurer:
        render_html(build_section_title("OTOMATİK ÖNERİYE DÖN", "sigortacı kararını geri alır; kayıt silinmez"))
        with st.form(f"cf_revert_{tag}", clear_on_submit=False):
            reason = st.text_area(f"Geri alma gerekçesi (zorunlu, en az {cw.INTERNAL_MIN} karakter)",
                                  max_chars=cw.NOTE_MAX, key=f"cf_rev_{tag}", height=70)
            reverted = st.form_submit_button("OTOMATİK KARARA DÖN")
        if reverted:
            err = save_decision(principal, cd, pick, "", None, "", str(reason), rules, cfg.seed, revert=True)
            if err:
                render_html(f'<div class="note err">{_h(err)}</div>')
            else:
                st.session_state[CASE_FLASH] = f"{pick}: otomatik karara dönüldü."
                st.session_state[CASE_GEN] = gen + 1
                st.rerun()
    render_brief([
        "Karar, tutar ve müşteri notu müşterinin ekranında ve karar bildiriminde görünür; iç gerekçe yalnızca sizde kalır.",
        "Kayıt eklemelidir: bir kararı değiştirmek ya da geri almak eskisini silmez, yeni bir kayıt ekler (Geçmiş sekmesi).",
        "Tutar, otomatik motorla aynı formülle hesaplanır (bedel × teminat × hasar oranı − muafiyet); elle tutar 0 ile "
        "sigorta bedeli arasında olmalıdır.",
    ])


def render_case_history(cd: CompanyData, own: pd.DataFrame, cfg: AppSettings) -> None:
    """"Geçmiş" alt sekmesi: seçili müşterinin (bu şirketteki) poliçeleri için karar kaydı."""
    records, err = read_history(cd.company, cfg.seed)
    if err:
        render_html(f'<div class="note err">Karar kaydı okunamadı: {_h(err)}</div>')
        return
    mine = set(own["police_no"])
    shown = [d for d in records if d.police_no in mine]
    render_html(build_section_title("KARAR GEÇMİŞİ", f"{len(shown)} kayıt · yeniden eskiye"))
    if not shown:
        render_html('<div class="note info">Bu müşterinin poliçeleri için henüz sigortacı kararı kaydı yok; '
                    "geçerli kararlar otomatik önerilerdir.</div>")
        return
    cc = st.column_config
    show_table(cw.log_frame(shown), {
        "no": cc.NumberColumn("No", format="%d"), "zaman": cc.TextColumn("Zaman"), "poliçe": cc.TextColumn("Poliçe"),
        "işlem": cc.TextColumn("İşlem"), "karar": cc.TextColumn("Karar"),
        "ödenecek_tl": cc.NumberColumn("Ödenecek (₺)", format="%.0f"),
        "bekleyen_tl": cc.NumberColumn("Bekleyen (₺)", format="%.0f"), "otomatik_karar": cc.TextColumn("Otomatik öneri"),
        "müşteri_notu": cc.TextColumn("Müşteri notu"), "iç_gerekçe": cc.TextColumn("İç gerekçe"),
        "veren": cc.TextColumn("Veren"),
    }, height=min(420, 76 + 35 * len(shown)))


def render_casefiles_tab() -> None:
    """MÜŞTERİ DOSYALARI sekmesi (sigortacı): şirketin müşterilerinden birini aç, incele, karar ver.

    Yalnızca sigortacının şirketinin poliçeleri listelenir; bir müşterinin başka şirketlerdeki poliçeleri görünmez.

    Raises:
        PermissionError: Oturum sigortacı değilse.
    """
    principal = session_principal()
    auth.require(principal, auth.PERM_DECIDE)
    auth.require(principal, auth.PERM_PORTAL_ANY)
    assert principal is not None
    cfg = get_settings()
    rules = current_rules()
    cd = company_data(principal, cfg.seed, rules)
    render_html(build_hero_html(
        "SİGORTACI · MÜŞTERİ DOSYALARI", "MÜŞTERİ", "DOSYALARI",
        "Şirketinizin müşterilerinden birini açın; poliçelerini, uydu kanıtlarını ve müşterinin gördüğü ekranı inceleyin, "
        "kararı siz verin ya da otomatik öneriyi değiştirin.",
        [tenancy.company_name(cd.company).upper(), "DEMO · SENTETİK VERİ"]))
    flash = st.session_state.pop(CASE_FLASH, "")
    if flash:
        render_html(f'<div class="note ok">{_h(flash)}</div>')
    if cd.error:
        render_html(f'<div class="note err"><b>Karar defteri okunamadı:</b> {_h(cd.error)} Karar verilemez; yalnızca '
                    "otomatik öneriler görünür.</div>")

    customers = cw.customer_list(cd.mine)
    f1, f2 = st.columns([3, 2])
    query = f1.text_input("Müşteri ara (ad ya da müşteri no)", key="cf_query")
    only_pending = f2.checkbox("Yalnızca ekspertiz bekleyenler", key="cf_pending")
    shown = customers
    if query.strip():
        q = query.strip().lower()
        shown = shown[shown["musteri"].str.lower().str.contains(q, regex=False)
                      | shown["musteri_no"].str.lower().str.contains(q, regex=False)]
    if only_pending:
        shown = shown[shown["ekspertiz"] > 0]
    render_html(build_section_title("MÜŞTERİLER", f"{len(shown)} / {len(customers)} · yalnızca bu şirketteki poliçeler"))
    cc = st.column_config
    show_table(shown, {
        "musteri": cc.TextColumn("Müşteri"), "musteri_no": cc.TextColumn("Müşteri no"),
        "police_sayisi": cc.NumberColumn("Poliçe", format="%d"), "ekspertiz": cc.NumberColumn("Ekspertiz bekleyen", format="%d"),
        "sigortaci_karari": cc.NumberColumn("Sigortacı kararı", format="%d"),
        "bedel_tl": cc.NumberColumn("Sigorta bedeli (₺)", format="%.0f"),
        "odenecek_tl": cc.NumberColumn("Ödenecek (₺)", format="%.0f"),
        "bekleyen_tl": cc.NumberColumn("Bekleyen (₺)", format="%.0f"),
    }, height=min(320, 76 + 35 * max(len(shown), 1)))
    if shown.empty:
        render_html('<div class="note info">Filtreye uyan müşteri yok.</div>')
        return
    names = shown["musteri"].tolist()
    info = {r.musteri: f"{r.musteri} · {r.musteri_no} · {r.police_sayisi} poliçe"
            + (f" · {r.ekspertiz} ekspertiz" if r.ekspertiz else "") for r in shown.itertuples()}
    if st.session_state.get("cf_customer") not in names:
        st.session_state["cf_customer"] = names[0]
    name = st.selectbox("Müşteri dosyası aç", names, key="cf_customer", format_func=lambda n: info.get(n, n))
    own = cd.mine[cd.mine["sigortali"] == name]         # yalnızca bu şirketin poliçeleri; indeks = geometri konumu
    options = list(own["police_no"])
    icon_of = dict(zip(own["police_no"], own["karar"]))
    if st.session_state.get("cf_policy") not in options:
        st.session_state["cf_policy"] = options[0]
    pick = str(st.selectbox("Poliçe", options, key="cf_policy",
                            format_func=lambda p: f"{p} · {DECISION_ICONS.get(icon_of.get(p, ''), '—')}"))

    tab_file, tab_decide, tab_view, tab_hist = st.tabs(list(CASE_TABS))
    with tab_file:
        render_case_file(cd, own, name, pick, rules, cfg)
    with tab_decide:
        render_case_decision(principal, cd, own, name, pick, rules, cfg)
    with tab_view:
        st.caption("Müşterinin kendi hesabıyla gördüğü ekran; yalnızca sizin şirketinizdeki poliçeler gösterilir "
                   "(müşterinin başka şirketlerdeki poliçelerini siz görmezsiniz). Müşteri oturumu otomatik kararları "
                   "varsayılan kurallarla hesaplar; bu önizleme sol menüdeki kurallarınızı kullanır.")
        with st.expander("Kim neyi görür?"):
            show_table(build_visibility_frame(), height=min(520, 76 + 35 * len(pt.VISIBILITY)))
        with panel():
            render_portal_view(cd.ds, cd.final, rules, name, cfg.basemap, "pv", cfg.seed, company=cd.company)
    with tab_hist:
        render_case_history(cd, own, cfg)


def build_visibility_frame() -> pd.DataFrame:
    """Rol görünürlük tablosu (kim neyi görür) — ``portal.VISIBILITY``'den."""
    mark = {True: "✓", False: "—"}
    return pd.DataFrame([{"Bilgi": label, "Müşteri": mark[cust], "Sigortacı": mark[ins]}
                         for label, cust, ins in pt.VISIBILITY])


# =========================================================================== #
# SEKME — YÖNETİM (sigortacı: profil, şirket, ekip, karar kaydı, sistem)
# =========================================================================== #
ADMIN_TABS: Tuple[str, ...] = ("Profil", "Şirket", "Ekip", "Karar kaydı", "Sistem")
PERMISSION_TEXT: Tuple[Tuple[str, str], ...] = (
    (auth.PERM_INSURANCE, "Şirketinizin poliçeleri, risk skorları ve otomatik kararlar"),
    (auth.PERM_DECIDE, "Müşteri dosyası açma ve tazminat kararı verme / geri alma"),
    (auth.PERM_PORTAL_ANY, "Müşterinin gördüğü ekranı önizleme (yalnızca kendi şirketinizin poliçeleri)"),
    (auth.PERM_EVIDENCE, "Karar kanıt paketi"),
    (auth.PERM_CALIBRATION, "Geriye dönük test ve kalibrasyon"),
    (auth.PERM_ADMIN, "Şirket yönetimi: profil, ekip, karar kaydı"),
    (auth.PERM_OVERVIEW, "Platform genel bakışı"),
    (auth.PERM_SECTORS, "Tarım, belediye ve kurtarma modülleri"),
)
RULE_LABELS: Tuple[Tuple[str, str], ...] = (
    ("approve_dnbr", "Tam onay eşiği (dNBR ≥)"), ("partial_dnbr", "Kısmi onay eşiği (dNBR ≥)"),
    ("review_dnbr", "Ret eşiği (dNBR <)"), ("min_confidence", "Asgari gözlem güveni"),
    ("deductible_pct", "Muafiyet (%)"), ("coverage_pct", "Teminat oranı (%)"), ("advance_pct", "Kısmi onay avansı (%)"),
)


def render_admin_profile(principal: auth.Principal) -> None:
    """"Profil" alt sekmesi: kişi, unvan, şirket ve yetkiler."""
    render_html(build_profile_html(principal))
    company = tenancy.company_name(principal.company)
    info = pd.DataFrame([
        {"Bilgi": "Ad", "Değer": principal.display_name}, {"Bilgi": "Kullanıcı adı", "Değer": principal.username},
        {"Bilgi": "Unvan", "Değer": principal.title or "—"}, {"Bilgi": "Rol", "Değer": auth.ROLE_LABELS[principal.role]},
        {"Bilgi": "Sigorta şirketi", "Değer": company},
        {"Bilgi": "Oturum süresi", "Değer": f"{int(auth.IDLE_TIMEOUT_S // 60)} dk işlem yapılmazsa kapanır"},
    ])
    show_table(info, height=76 + 35 * len(info))
    render_html(build_section_title("YETKİLERİM", "bu hesabın yapabildikleri"))
    perms = pd.DataFrame([{"Yetki": text} for perm, text in PERMISSION_TEXT if auth.allowed(principal, perm)])
    show_table(perms, height=76 + 35 * len(perms))
    render_brief([
        "Hesaplar yapılandırmayla yönetilir (ORBITAL_ACCOUNTS_JSON); bu prototipte parola değiştirme ve hesap ekleme ekranı yoktur.",
        "Yetkiler role bağlıdır; unvan yalnızca görüntü içindir ve yetki vermez.",
    ])


def render_admin_company(cd: CompanyData) -> None:
    """"Şirket" alt sekmesi: portföy özeti ve karar bekleyen dosyalar."""
    mine = cd.mine
    s = cw.summarize_final(mine)
    render_html(build_kpi_grid([
        Kpi("Müşteri", format_tr(mine["sigortali"].nunique()), "bu şirkette poliçesi olan"),
        Kpi("Poliçe", format_tr(len(mine)), f"{format_tr(mine['alan_ha'].sum(), 1)} ha"),
        Kpi("Sigorta bedeli", f"₺{format_tr(mine['sigorta_bedeli_tl'].sum() / 1e6, 1)} M", "toplam"),
        Kpi("Ödenecek", f"₺{format_tr(s['paid_tl'] / 1e6, 2)} M", "geçerli kararlara göre", COLOR_OK),
        Kpi("Bekleyen", f"₺{format_tr(s['pending_tl'] / 1e6, 2)} M", f"{int(s['review'])} dosya ekspertizde", COLOR_AMBER),
    ], columns=5))
    render_html(build_section_title("KARAR BEKLEYEN DOSYALAR", "ekspertiz durumundaki poliçeler · en yüksek tutar önde"))
    queue = mine[mine["karar"] == "EKSPERTİZ"].sort_values("bekleyen_tl", ascending=False).head(15)
    cc = st.column_config
    if queue.empty:
        render_html('<div class="note ok">Karar bekleyen dosya yok.</div>')
    else:
        show_table(queue[["police_no", "sigortali", "varlik_turu", "alan_ha", "siddet", "bekleyen_tl"]], {
            "police_no": cc.TextColumn("Poliçe No"), "sigortali": cc.TextColumn("Müşteri"),
            "varlik_turu": cc.TextColumn("Varlık"), "alan_ha": cc.NumberColumn("Alan (ha)", format="%.1f"),
            "siddet": cc.TextColumn("Şiddet"), "bekleyen_tl": cc.NumberColumn("Bekleyen (₺)", format="%.0f"),
        }, height=min(420, 76 + 35 * len(queue)))
        st.caption("Dosyayı Müşteri dosyaları sekmesinden açıp karar verebilirsiniz.")
    render_html(build_section_title("VARLIK TÜRÜNE GÖRE PORTFÖY"))
    by_asset = mine.groupby("varlik_turu").agg(
        police=("police_no", "size"), alan=("alan_ha", "sum"), bedel=("sigorta_bedeli_tl", "sum"),
        odenecek=("odenecek_tl", "sum")).reset_index().sort_values("bedel", ascending=False)
    show_table(by_asset, {
        "varlik_turu": cc.TextColumn("Varlık türü"), "police": cc.NumberColumn("Poliçe", format="%d"),
        "alan": cc.NumberColumn("Alan (ha)", format="%.1f"), "bedel": cc.NumberColumn("Sigorta bedeli (₺)", format="%.0f"),
        "odenecek": cc.NumberColumn("Ödenecek (₺)", format="%.0f"),
    }, height=min(380, 76 + 35 * len(by_asset)))
    render_brief([
        "Şirket adı ve poliçelerin şirketlere dağılımı kurgusal/sentetiktir; poliçe numarasından türetilir.",
        "Aynı müşterinin başka şirketlerdeki poliçeleri bu sayılara girmez ve sizin tarafınızdan görülemez.",
    ])


def render_admin_team(principal: auth.Principal, cd: CompanyData) -> None:
    """"Ekip" alt sekmesi: aynı şirkete bağlı sigortacı hesapları (yapılandırmadan, yalnızca okunur)."""
    accounts, is_demo, config_error = load_accounts()
    if config_error:
        render_html(f'<div class="note err">Hesap yapılandırması okunamadı: {_h(config_error)}</div>')
        return
    team = [a for a in accounts if a.role is auth.Role.INSURER and a.company == cd.company]
    render_html(build_section_title("EKİP", f"{tenancy.company_name(cd.company)} · {len(team)} hesap"))
    rows = [{"Ad": a.display_name, "Kullanıcı adı": a.username, "Unvan": a.title or "—",
             "": "siz" if a.username == principal.username else ""} for a in team]
    frame = pd.DataFrame(rows, columns=["Ad", "Kullanıcı adı", "Unvan", ""])
    show_table(frame, height=76 + 35 * max(len(frame), 1))
    notes = ["Ekip, hesap yapılandırmasından okunur; parola özetleri ve diğer şirketlerin hesapları gösterilmez.",
             "Hesap ekleme/çıkarma yapılandırmayla yapılır (ORBITAL_ACCOUNTS_JSON); bu ekran yalnızca okur."]
    if is_demo:
        notes.append("Şu an dahili demo hesapları etkin; gerçek kullanımda kendi hesaplarınızı tanımlayın.")
    render_brief(notes)


def render_admin_log(cd: CompanyData, cfg: AppSettings) -> None:
    """"Karar kaydı" alt sekmesi: şirketin tüm karar/geri alma kayıtları ve CSV dışa aktarma."""
    records, err = read_history(cd.company, cfg.seed)
    if err:
        render_html(f'<div class="note err">Karar kaydı okunamadı: {_h(err)}</div>')
        return
    decisions = sum(d.action == cw.ACTION_DECIDE for d in records)
    render_html(build_kpi_grid([
        Kpi("Kayıt", format_tr(len(records)), f"veri tohumu {cfg.seed}"),
        Kpi("Karar", format_tr(decisions), "ekleme"),
        Kpi("Geri alma", format_tr(len(records) - decisions), "otomatiğe dönüş"),
    ], columns=3))
    query = st.text_input("Poliçe / karar ara", key="adm_log_query")
    shown = records
    if query.strip():
        q = query.strip().lower()
        shown = [d for d in records if q in d.police_no.lower() or q in d.decision.lower() or q in d.decided_by.lower()]
    cc = st.column_config
    if not shown:
        render_html('<div class="note info">Gösterilecek kayıt yok.</div>')
    else:
        show_table(cw.log_frame(shown), {
            "no": cc.NumberColumn("No", format="%d"), "zaman": cc.TextColumn("Zaman"), "poliçe": cc.TextColumn("Poliçe"),
            "işlem": cc.TextColumn("İşlem"), "karar": cc.TextColumn("Karar"),
            "ödenecek_tl": cc.NumberColumn("Ödenecek (₺)", format="%.0f"),
            "bekleyen_tl": cc.NumberColumn("Bekleyen (₺)", format="%.0f"), "otomatik_karar": cc.TextColumn("Otomatik öneri"),
            "müşteri_notu": cc.TextColumn("Müşteri notu"), "iç_gerekçe": cc.TextColumn("İç gerekçe"),
            "veren": cc.TextColumn("Veren"),
        }, height=min(460, 76 + 35 * len(shown)))
        st.download_button("↓  KARAR KAYDI (CSV)", data=csv_bytes(cw.log_frame(shown, safe=True)),
                           file_name="karar_kaydi.csv", mime="text/csv", key="adm_log_csv")
    render_brief([
        "Kayıt eklemelidir: hiçbir satır değiştirilmez ya da silinmez; kararı değiştirmek yeni kayıt ekler.",
        "CSV dışa aktarımında formül olarak okunabilecek hücreler (= + - @) başına kesme işareti konarak korunur.",
        "Kayıtlar sunucudaki SQLite dosyasındadır; Streamlit Community Cloud'da uygulama yeniden başlayınca silinir.",
    ])


def render_admin_system(cd: CompanyData, rules: ClaimRules, cfg: AppSettings) -> None:
    """"Sistem" alt sekmesi: karar defteri durumu, geçerli kurallar ve henüz olmayanlar."""
    try:
        db = stm.shared_store()
        where = ("ORBITAL_DB_PATH ile yapılandırıldı" if os.environ.get(stm.DB_ENV, "").strip()
                 else "varsayılan klasör (orbital_data/)")
        store_rows = [{"Bilgi": "Karar defteri", "Değer": "açık"}, {"Bilgi": "Dosya", "Değer": os.path.basename(db.path)},
                      {"Bilgi": "Konum", "Değer": where}]
    except stm.StoreError as exc:
        store_rows = [{"Bilgi": "Karar defteri", "Değer": f"açılamadı: {exc}"}]
    show_table(pd.DataFrame(store_rows), height=76 + 35 * len(store_rows))
    render_html(build_section_title("GEÇERLİ KARAR KURALLARI", "sol menüden değişir · yalnızca bu oturum"))
    rule_rows = [{"Kural": label, "Değer": float(getattr(rules, field))} for field, label in RULE_LABELS]
    cc = st.column_config
    show_table(pd.DataFrame(rule_rows), {"Değer": cc.NumberColumn("Değer", format="%.2f")}, height=76 + 35 * len(rule_rows))
    render_brief([
        "Kurallar oturuma özgüdür: kaydedilmez ve müşterilere yayımlanmaz. Müşteri ekranındaki otomatik kararlar "
        "varsayılan kurallarla hesaplanır.",
        "Bu prototipte henüz yok: parola değiştirme, şirket kurallarını kalıcı yayımlama, e-posta/SMS bildirimi, "
        "kurumsal kimlik sağlayıcı ve iki adımlı doğrulama.",
    ])


def render_admin_tab() -> None:
    """YÖNETİM sekmesi (sigortacı): profil, şirket portföyü, ekip, karar kaydı ve sistem durumu.

    Raises:
        PermissionError: Oturum sigortacı değilse.
    """
    principal = session_principal()
    auth.require(principal, auth.PERM_ADMIN)
    assert principal is not None
    cfg = get_settings()
    rules = current_rules()
    cd = company_data(principal, cfg.seed, rules)
    render_html(build_hero_html(
        "SİGORTACI · YÖNETİM", "ŞİRKET", "YÖNETİMİ",
        "Profilinizi, şirketinizin portföyünü ve ekibini, verilen kararların kaydını ve sistem durumunu görün.",
        [tenancy.company_name(cd.company).upper(), "DEMO · SENTETİK VERİ"]))
    tab_profile, tab_company, tab_team, tab_log, tab_system = st.tabs(list(ADMIN_TABS))
    with tab_profile:
        render_admin_profile(principal)
    with tab_company:
        render_admin_company(cd)
    with tab_team:
        render_admin_team(principal, cd)
    with tab_log:
        render_admin_log(cd, cfg)
    with tab_system:
        render_admin_system(cd, rules, cfg)



# =========================================================================== #
# Yan menü ve ana akış
# =========================================================================== #
def render_rule_controls() -> None:
    """Sigortacı için karar kuralı kaydırıcıları (sol menü). Sigorta, Müşteri portalı ve Kalibrasyon aynı kuralları kullanır."""
    render_html('<div class="side-label"><b>01</b> KARAR KURALLARI</div>')
    st.slider("Tam onay eşiği (dNBR ≥)", *cal.BOUNDS["approve_dnbr"], 0.44, 0.01, key="ins_approve")
    st.slider("Kısmi onay eşiği (dNBR ≥)", *cal.BOUNDS["partial_dnbr"], 0.27, 0.01, key="ins_partial")
    st.slider("Ret eşiği (dNBR <)", *cal.BOUNDS["review_dnbr"], 0.10, 0.01, key="ins_review")
    st.slider("Min. gözlem güveni", *cal.BOUNDS["min_confidence"], 0.80, 0.01, key="ins_conf")
    st.slider("Muafiyet (%)", 0.0, 10.0, 2.0, 0.5, key="ins_ded")
    st.slider("Teminat oranı (%)", 50, 100, 90, 5, key="ins_cover")


def render_basemap_controls() -> None:
    """Harita altlığı seçimi, (gerekirse) görüntü günü ve açıklamalar; her iki rolde ortaktır."""
    chosen = bm.by_label(st.radio("Harita altlığı", list(BASEMAP_OPTIONS), key="basemap"))
    fire_on = bm.FIRE_LABEL in st.session_state.get("res_layers", ())
    if chosen.dated or fire_on:
        today = bm.utc_today()
        st.date_input("Görüntü günü (UTC)", value=bm.default_date(today),
                      min_value=chosen.first_day or bm.GIBS_FIRST_DAY, max_value=today, key="basemap_date")
    st.caption(bm.sidebar_note(get_settings().basemap))
    if fire_on:
        st.caption("Görüntü günü, Kurtarma haritasındaki gerçek yangın katmanı için de geçerlidir.")


def render_sidebar(principal: auth.Principal) -> None:
    """Sol menüyü çizer: marka, kullanıcı kartı ve çıkış, ardından role göre ayarlar.

    Sigortacı karar kurallarını, harita altlığını, "Gelişmiş" altındaki senaryo tohumunu ve sistem kartını görür;
    müşteri yalnızca harita altlığını (kural, tohum ve sürüm bilgisi yok).

    Args:
        principal: Oturum açmış kullanıcı.
    """
    with st.sidebar:
        render_html(build_brand_html())
        render_html(build_user_card_html(principal))
        st.button("Çıkış yap", on_click=logout, key="auth_logout")
        if principal.role is auth.Role.INSURER:
            render_rule_controls()
            render_html('<div class="side-label"><b>02</b> HARİTA</div>')
        else:
            render_html('<div class="side-label"><b>01</b> HARİTA</div>')
        render_basemap_controls()
        if principal.role is auth.Role.INSURER:
            with st.expander("Gelişmiş"):
                st.number_input("Veri tohumu (seed)", min_value=1, max_value=99999, value=2026, step=1, key="seed")
                st.caption("Aynı tohum her zaman aynı sentetik senaryoyu üretir.")
            render_html(f"""
            <div class="sys-card">
              <b>VERİ</b> Demo · sentetik<br><b>ADAPTÖR</b> Sentinel-2 · FIRMS<br><b>MOTOR</b> numpy {_h(np.__version__)}<br>
              <b>HARİTA</b> folium {_h(getattr(folium, '__version__', '?'))}<br>
              <b>ARAYÜZ</b> streamlit {_h(getattr(st, '__version__', '?'))}<br><b>SEKTÖR</b> 4 modül<br>
              <b>DURUM</b> ÇEVRİMİÇİ
            </div>""")
        else:
            render_html('<div class="sys-card"><b>VERİ</b> Demo · sentetik<br><b>DURUM</b> ÇEVRİMİÇİ</div>')


def safe_render(renderer: Callable[[], None], name: str) -> None:
    """Bir sekme çizicisini çalıştırır; hata olursa uygulamayı düşürmek yerine uyarı kartı gösterir.

    Yetki hatası (``PermissionError``) iletisiyle gösterilir. Diğer hatalarda sigortacı ayrıntıyı görür; müşteri
    yalnızca genel bir ileti görür (ayrıntı günlüğe yazılır).

    Args:
        renderer: Argümansız sekme çizici fonksiyonu.
        name: Hata mesajında gösterilecek sekme adı.
    """
    try:
        renderer()
    except Exception as exc:  # noqa: BLE001 — arayüz sınırında bilinçli geniş yakalama
        LOGGER.exception("%s sekmesi çizilemedi", name)
        principal = session_principal()
        detail = _h(exc) if isinstance(exc, PermissionError) or (principal and principal.role is auth.Role.INSURER) \
            else "Ayrıntı günlüğe yazıldı."
        render_html(f'<div class="note err"><b>{_h(name)}</b> sekmesi çizilemedi: {detail}</div>')


#: "Diğer sektörler" sekmesinin alt sekmeleri: etiket → (hata iletisinde görünen ad, çizici).
SECTOR_RENDERERS: Dict[str, Tuple[str, Callable[[], None]]] = {
    "Tarım": ("Tarım", render_agriculture_tab),
    "Belediye": ("Belediye", render_municipality_tab),
    "Kurtarma": ("Kurtarma", render_rescue_tab),
}


def render_sectors_tab() -> None:
    """DİĞER SEKTÖRLER sekmesini çizer: tarım, belediye ve kurtarma modülleri alt sekmelerde durur."""
    auth.require(session_principal(), auth.PERM_SECTORS)
    render_html(build_section_title("DİĞER SEKTÖRLER", "aynı uydu çekirdeği, üç genişleme alanı"))
    labels = list(SECTOR_RENDERERS)
    for tab, label in zip(st.tabs(labels), labels):
        title, renderer = SECTOR_RENDERERS[label]
        with tab:
            safe_render(renderer, title)


def render_workflow_page() -> None:
    """ANALİZ İŞ AKIŞI sekmesi: altı adımlı analiz sayfasına uygulamanın yardımcılarını verir."""
    auth.require(session_principal(), auth.PERM_SECTORS)
    wpage.render_workflow_tab(wpage.Kit(
        render_html=render_html, show_frame=show_html_frame, panel=panel, hero=build_hero_html, section_title=build_section_title,
        kpi_grid=build_kpi_grid, kpi=Kpi, rules=current_rules, draw_map=draw_region_map, read_secret=read_secret, csv_bytes=csv_bytes,
        show_table=show_table, html_escape=_h))


#: Sigortacı sekme etiketi → (hata iletisinde görünen ad, çizici). Etiketler ``auth.INSURER_TABS`` ile birebir aynıdır.
TAB_RENDERERS: Dict[str, Tuple[str, Callable[[], None]]] = {
    "GENEL BAKIŞ": ("Genel Bakış", render_overview_tab),
    "ANALİZ İŞ AKIŞI": ("Analiz iş akışı", render_workflow_page),
    "SİGORTA": ("Sigorta", render_insurance_tab),
    "MÜŞTERİ DOSYALARI": ("Müşteri dosyaları", render_casefiles_tab),
    "KALİBRASYON": ("Kalibrasyon", render_calibration_tab),
    "DİĞER SEKTÖRLER": ("Diğer sektörler", render_sectors_tab),
    "YÖNETİM": ("Yönetim", render_admin_tab),
}


def main() -> None:
    """Giriş noktası: sayfa ayarı, tema, oturum kapısı; sonra role göre sekmeler (sigortacı) ya da tek sayfa (müşteri)."""
    st.set_page_config(page_title="Çok Yönlü Uydu ve Saha Analiz Platformu", page_icon="🛰️", layout="wide",
                       initial_sidebar_state="expanded")
    inject_css()
    principal = gate()
    if principal is None:
        return
    render_sidebar(principal)
    if principal.role is auth.Role.CUSTOMER:
        safe_render(render_customer_page, "Hasar takibi")
    else:
        labels = auth.tabs_for(principal)
        for tab, label in zip(st.tabs(list(labels)), labels):
            title, renderer = TAB_RENDERERS[label]
            with tab:
                safe_render(renderer, title)
    render_html(build_footer_html())


if __name__ == "__main__":
    main()
