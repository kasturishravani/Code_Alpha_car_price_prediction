import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="AUTOVAL", page_icon="🚗", layout="centered")

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&display=swap');

header {visibility:hidden;}
footer {visibility:hidden;}

.stApp {
    background-image: url("https://images.unsplash.com/photo-1503376780353-7e6692767b70?q=80&w=2000");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.65);
    z-index: 0;
    pointer-events: none;
}
.main .block-container {
    max-width: 820px;
    padding-top: 1.5rem;
    position: relative;
    z-index: 1;
}

.title {
    text-align: center;
    color: #00f2fe;
    font-size: 3.8rem;
    font-family: 'Orbitron', sans-serif;
    font-weight: 900;
    letter-spacing: 0.12em;
    text-shadow: 0 0 24px #00f2fe, 0 0 60px rgba(0,242,254,0.3);
    margin-bottom: 0.1rem;
}
.subtitle {
    text-align: center;
    color: rgba(255,255,255,0.75);
    font-size: 15px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 1.8rem;
}
.divider { border: none; border-top: 1px solid rgba(0,242,254,0.2); margin: 1rem 0 1.5rem; }

label, .stSelectbox label, .stTextInput label, .stNumberInput label, .stSlider label, p, span {
    color: white !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.04em !important;
}
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
    background: rgba(0,0,0,0.65) !important;
    border: 1.5px solid rgba(0,242,254,0.45) !important;
    border-radius: 10px !important;
}
div[data-baseweb="input"] > div:focus-within, div[data-baseweb="select"] > div:focus-within {
    border-color: #00f2fe !important;
    box-shadow: 0 0 0 2px rgba(0,242,254,0.15) !important;
}
input { color: white !important; }

div.stButton { display: flex; justify-content: center; }
div.stButton button {
    width: 260px; height: 52px; border-radius: 12px;
    border: 2px solid #00f2fe;
    background: rgba(0,242,254,0.07);
    color: #00f2fe;
    font-size: 16px;
    font-family: 'Orbitron', sans-serif;
    font-weight: 700;
    letter-spacing: 0.08em;
    transition: all 0.2s;
}
div.stButton button:hover {
    background: #00f2fe; color: #000;
    box-shadow: 0 0 24px rgba(0,242,254,0.5);
    transform: translateY(-1px);
}

/* Car image */
.car-img-wrap {
    display: flex; justify-content: center; margin: 1rem 0 1.5rem;
}
.car-img-wrap img {
    border-radius: 16px;
    border: 2px solid rgba(0,242,254,0.5);
    box-shadow: 0 0 40px rgba(0,242,254,0.25);
    max-width: 400px; width: 100%; object-fit: cover;
    background: rgba(0,0,0,0.4);
}

/* Result card */
.result-card {
    background: rgba(0,0,0,0.90);
    border: 2px solid #00f2fe;
    border-radius: 20px;
    text-align: center;
    padding: 30px 24px 26px;
    margin-top: 1.5rem;
    box-shadow: 0 0 50px rgba(0,242,254,0.18), inset 0 0 60px rgba(0,242,254,0.03);
    position: relative; overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00f2fe, transparent);
}
.result-car-name {
    font-family: 'Orbitron', sans-serif;
    color: #00f2fe;
    font-size: 1.75rem; font-weight: 900;
    letter-spacing: 0.1em; text-transform: uppercase;
    text-shadow: 0 0 20px rgba(0,242,254,0.6);
    margin-bottom: 6px;
}
.result-meta {
    font-family: 'Rajdhani', sans-serif;
    color: rgba(255,255,255,0.5);
    font-size: 13px; letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 22px;
}
.result-price-label {
    font-family: 'Rajdhani', sans-serif;
    color: rgba(255,255,255,0.4); font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.22em; margin-bottom: 6px;
}
.result-price {
    font-family: 'Orbitron', sans-serif;
    color: #ffffff; font-size: 3.2rem; font-weight: 900;
    text-shadow: 0 0 30px rgba(0,242,254,0.55), 0 0 70px rgba(0,242,254,0.15);
    line-height: 1; margin-bottom: 6px;
}
.result-price-sub {
    font-family: 'Rajdhani', sans-serif;
    color: #00f2fe; font-size: 13px; letter-spacing: 0.14em; margin-bottom: 22px; opacity: 0.8;
}
.result-divider { border: none; border-top: 1px solid rgba(0,242,254,0.15); margin: 16px 0; }
.result-details { display: flex; justify-content: center; gap: 28px; flex-wrap: wrap; }
.result-detail-item { text-align: center; }
.result-detail-label {
    font-family: 'Rajdhani', sans-serif;
    color: rgba(255,255,255,0.35); font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.18em; margin-bottom: 3px;
}
.result-detail-val {
    font-family: 'Rajdhani', sans-serif;
    color: white; font-size: 15px; font-weight: 700;
}
.badge-row { display: flex; justify-content: center; gap: 8px; margin-top: 16px; flex-wrap: wrap; }
.badge {
    font-family: 'Rajdhani', sans-serif; font-size: 11px; font-weight: 700;
    padding: 4px 12px; border: 1px solid rgba(0,242,254,0.35); border-radius: 99px;
    color: #00f2fe; letter-spacing: 0.1em; text-transform: uppercase;
    background: rgba(0,242,254,0.06);
}
.section-header {
    font-family: 'Rajdhani', sans-serif;
    color: rgba(0,242,254,0.7); font-size: 11px; font-weight: 700;
    letter-spacing: 0.22em; text-transform: uppercase;
    margin: 1.5rem 0 0.6rem;
    display: flex; align-items: center; gap: 8px;
}
.section-header::after { content: ''; flex: 1; height: 1px; background: rgba(0,242,254,0.15); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CAR IMAGES — Wikipedia / open CDN URLs
# ─────────────────────────────────────────────
CAR_IMAGES = {
    # ── Maruti / Suzuki ──────────────────────────────────────────────────────
    "swift":         "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/2018_Suzuki_Swift_SZ-T_1.0_Front.jpg/1280px-2018_Suzuki_Swift_SZ-T_1.0_Front.jpg",
    "alto 800":      "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Maruti_Suzuki_Alto_800_in_India.jpg/1280px-Maruti_Suzuki_Alto_800_in_India.jpg",
    "alto k10":      "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Maruti_Suzuki_Alto_800_in_India.jpg/1280px-Maruti_Suzuki_Alto_800_in_India.jpg",
    "800":           "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Maruti_800_AC_-_ZC_series.jpg/1280px-Maruti_800_AC_-_ZC_series.jpg",
    "wagon r":       "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Suzuki_Wagon_R_Stingray_%28second_generation%2C_first_facelift%29%2C_front_8.22.19.jpg/1280px-Suzuki_Wagon_R_Stingray_%28second_generation%2C_first_facelift%29%2C_front_8.22.19.jpg",
    "dzire":         "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Maruti_Suzuki_Dzire_Tour_S_CNG_%28third_generation%2C_facelift%29%2C_front_10.24.21.jpg/1280px-Maruti_Suzuki_Dzire_Tour_S_CNG_%28third_generation%2C_facelift%29%2C_front_10.24.21.jpg",
    "ciaz":          "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Maruti_Suzuki_Ciaz_facelift_%28India%29%2C_front_8.17.19.jpg/1280px-Maruti_Suzuki_Ciaz_facelift_%28India%29%2C_front_8.17.19.jpg",
    "ertiga":        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/2019_Suzuki_Ertiga_GL_1.5_%28Philippines%29_front_view.jpg/1280px-2019_Suzuki_Ertiga_GL_1.5_%28Philippines%29_front_view.jpg",
    "baleno":        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/2019_Suzuki_Baleno_SZ5_BOOSTERJET_1.0_Front.jpg/1280px-2019_Suzuki_Baleno_SZ5_BOOSTERJET_1.0_Front.jpg",
    "ignis":         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/2017_Suzuki_Ignis_SZ5_ALLGRIP_1.2_Front.jpg/1280px-2017_Suzuki_Ignis_SZ5_ALLGRIP_1.2_Front.jpg",
    "ritz":          "https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Maruti_Suzuki_Ritz.jpg/1280px-Maruti_Suzuki_Ritz.jpg",
    "sx4":           "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Suzuki_SX4_facelift_front_20100906.jpg/1280px-Suzuki_SX4_facelift_front_20100906.jpg",
    "vitara brezza": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Maruti_Suzuki_Vitara_Brezza_facelift.jpg/1280px-Maruti_Suzuki_Vitara_Brezza_facelift.jpg",
    "s cross":       "https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/2016_Suzuki_S-Cross_SZ5_DDIS_1.6_Front.jpg/1280px-2016_Suzuki_S-Cross_SZ5_DDIS_1.6_Front.jpg",
    "omni":          "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Maruti_Suzuki_Omni_in_India.jpg/1280px-Maruti_Suzuki_Omni_in_India.jpg",

    # ── Toyota ───────────────────────────────────────────────────────────────
    "innova":        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/64/Toyota_Innova_Crysta_%28facelift%2C_India%29_front_quarter.jpg/1280px-Toyota_Innova_Crysta_%28facelift%2C_India%29_front_quarter.jpg",
    "fortuner":      "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/2021_Toyota_Fortuner_GR_Sport_%28Thailand%29_front_quarter.jpg/1280px-2021_Toyota_Fortuner_GR_Sport_%28Thailand%29_front_quarter.jpg",
    "corolla altis": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/2017_Toyota_Corolla_Altis_%281.8G%2C_facelift%2C_Thailand%29_front.jpg/1280px-2017_Toyota_Corolla_Altis_%281.8G%2C_facelift%2C_Thailand%29_front.jpg",
    "corolla":       "https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/2017_Toyota_Corolla_Altis_%281.8G%2C_facelift%2C_Thailand%29_front.jpg/1280px-2017_Toyota_Corolla_Altis_%281.8G%2C_facelift%2C_Thailand%29_front.jpg",
    "etios g":       "https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Toyota_Etios_2014_%28India%29.jpg/1280px-Toyota_Etios_2014_%28India%29.jpg",
    "etios liva":    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Toyota_Etios_Liva_facelift_%28India%29%2C_front_8.3.14.jpg/1280px-Toyota_Etios_Liva_facelift_%28India%29%2C_front_8.3.14.jpg",
    "etios cross":   "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/Toyota_Etios_Liva_facelift_%28India%29%2C_front_8.3.14.jpg/1280px-Toyota_Etios_Liva_facelift_%28India%29%2C_front_8.3.14.jpg",
    "etios gd":      "https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Toyota_Etios_2014_%28India%29.jpg/1280px-Toyota_Etios_2014_%28India%29.jpg",
    "camry":         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/2021_Toyota_Camry_%28AXVH71R%29_Ascent_Sport_sedan_%282021-10-06%29_01.jpg/1280px-2021_Toyota_Camry_%28AXVH71R%29_Ascent_Sport_sedan_%282021-10-06%29_01.jpg",
    "land cruiser":  "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/2016_Toyota_Land_Cruiser_%28VDJ200R%29_GXL_wagon_%282016-08-16%29_01.jpg/1280px-2016_Toyota_Land_Cruiser_%28VDJ200R%29_GXL_wagon_%282016-08-16%29_01.jpg",

    # ── Hyundai ──────────────────────────────────────────────────────────────
    "creta":         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/2020_Hyundai_Creta_%28GS%2C_facelift%29_1.5_MPI_Style_%28Indonesia%29_front.jpg/1280px-2020_Hyundai_Creta_%28GS%2C_facelift%29_1.5_MPI_Style_%28Indonesia%29_front.jpg",
    "i20":           "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/2021_Hyundai_i20_Premium_1.0T-GDi_Front.jpg/1280px-2021_Hyundai_i20_Premium_1.0T-GDi_Front.jpg",
    "i10":           "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Hyundai_Grand_i10_Nios_Sportz_AMT_2020.jpg/1280px-Hyundai_Grand_i10_Nios_Sportz_AMT_2020.jpg",
    "grand i10":     "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Hyundai_Grand_i10_Nios_Sportz_AMT_2020.jpg/1280px-Hyundai_Grand_i10_Nios_Sportz_AMT_2020.jpg",
    "eon":           "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Hyundai_Eon_Era_Plus_front_view.jpg/1280px-Hyundai_Eon_Era_Plus_front_view.jpg",
    "verna":         "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Hyundai_Verna_2020.jpg/1280px-Hyundai_Verna_2020.jpg",
    "xcent":         "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Hyundai_Grand_i10_Nios_Sportz_AMT_2020.jpg/1280px-Hyundai_Grand_i10_Nios_Sportz_AMT_2020.jpg",
    "elantra":       "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/2021_Hyundai_Elantra_SEL_front_7.1.21.jpg/1280px-2021_Hyundai_Elantra_SEL_front_7.1.21.jpg",

    # ── Honda Cars ───────────────────────────────────────────────────────────
    "city":          "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/2020_Honda_City_RS_%28GN2%2C_hybrid%29_front_quarter.jpg/1280px-2020_Honda_City_RS_%28GN2%2C_hybrid%29_front_quarter.jpg",
    "brio":          "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/Honda_Brio_1.2_V_AT_%28second_generation%2C_facelift%2C_Indonesia%29%2C_front_8.21.20.jpg/1280px-Honda_Brio_1.2_V_AT_%28second_generation%2C_facelift%2C_Indonesia%29%2C_front_8.21.20.jpg",
    "amaze":         "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Honda_Amaze_2018_facelift_India.jpg/1280px-Honda_Amaze_2018_facelift_India.jpg",
    "jazz":          "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/2020_Honda_Jazz_EX_%28facelift%2C_Indonesia%29%2C_front_1.21.21.jpg/1280px-2020_Honda_Jazz_EX_%28facelift%2C_Indonesia%29%2C_front_1.21.21.jpg",

    # ── Royal Enfield ─────────────────────────────────────────────────────────
    "royal enfield classic 350":  "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Royal_Enfield_Classic_350.jpg/1280px-Royal_Enfield_Classic_350.jpg",
    "royal enfield classic 500":  "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Royal_Enfield_Classic_500.jpg/1280px-Royal_Enfield_Classic_500.jpg",
    "royal enfield thunder 350":  "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Royal_Enfield_Thunderbird_350.jpg/1280px-Royal_Enfield_Thunderbird_350.jpg",
    "royal enfield thunder 500":  "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Royal_Enfield_Thunderbird_350.jpg/1280px-Royal_Enfield_Thunderbird_350.jpg",
    "royal enfield bullet 350":   "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Royal_Enfield_Classic_350.jpg/1280px-Royal_Enfield_Classic_350.jpg",

    # ── KTM ──────────────────────────────────────────────────────────────────
    "ktm rc200":     "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/KTM_RC_200.jpg/1280px-KTM_RC_200.jpg",
    "ktm rc390":     "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/KTM_RC_390.jpg/1280px-KTM_RC_390.jpg",
    "ktm 390 duke":  "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/2020_KTM_390_Duke_in_Orange.jpg/1280px-2020_KTM_390_Duke_in_Orange.jpg",
    "ktm 390 duke ": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/2020_KTM_390_Duke_in_Orange.jpg/1280px-2020_KTM_390_Duke_in_Orange.jpg",

    # ── Bajaj ─────────────────────────────────────────────────────────────────
    "bajaj dominar 400":        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Bajaj_Dominar_400.jpg/1280px-Bajaj_Dominar_400.jpg",
    "bajaj pulsar rs200":       "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Bajaj_Pulsar_RS200.jpg/1280px-Bajaj_Pulsar_RS200.jpg",
    "bajaj pulsar ns 200":      "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Bajaj_Pulsar_NS200.jpg/1280px-Bajaj_Pulsar_NS200.jpg",
    "bajaj pulsar  ns 200":     "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Bajaj_Pulsar_NS200.jpg/1280px-Bajaj_Pulsar_NS200.jpg",
    "bajaj pulsar 150":         "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Bajaj_Pulsar_150.jpg/1280px-Bajaj_Pulsar_150.jpg",
    "bajaj pulsar 135 ls":      "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Bajaj_Pulsar_150.jpg/1280px-Bajaj_Pulsar_150.jpg",
    "bajaj pulsar 220 f":       "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Bajaj_Pulsar_150.jpg/1280px-Bajaj_Pulsar_150.jpg",
    "bajaj avenger 220":        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Bajaj_Avenger_220.jpg/1280px-Bajaj_Avenger_220.jpg",
    "bajaj avenger 220 dtsi":   "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Bajaj_Avenger_220.jpg/1280px-Bajaj_Avenger_220.jpg",
    "bajaj avenger street 220": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Bajaj_Avenger_220.jpg/1280px-Bajaj_Avenger_220.jpg",
    "bajaj avenger 150":        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Bajaj_Avenger_220.jpg/1280px-Bajaj_Avenger_220.jpg",
    "bajaj avenger 150 street": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Bajaj_Avenger_220.jpg/1280px-Bajaj_Avenger_220.jpg",
    "bajaj discover 100":       "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Bajaj_Pulsar_150.jpg/1280px-Bajaj_Pulsar_150.jpg",
    "bajaj discover 125":       "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Bajaj_Pulsar_150.jpg/1280px-Bajaj_Pulsar_150.jpg",
    "bajaj  ct 100":            "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Bajaj_Pulsar_150.jpg/1280px-Bajaj_Pulsar_150.jpg",

    # ── Hero ──────────────────────────────────────────────────────────────────
    "hero splender plus":     "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Hero_Splendor_Plus.jpg/1280px-Hero_Splendor_Plus.jpg",
    "hero splender ismart":   "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Hero_Splendor_Plus.jpg/1280px-Hero_Splendor_Plus.jpg",
    "hero super splendor":    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Hero_Splendor_Plus.jpg/1280px-Hero_Splendor_Plus.jpg",
    "hero passion pro":       "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Hero_Splendor_Plus.jpg/1280px-Hero_Splendor_Plus.jpg",
    "hero passion x pro":     "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Hero_Splendor_Plus.jpg/1280px-Hero_Splendor_Plus.jpg",
    "hero honda passion pro": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Hero_Splendor_Plus.jpg/1280px-Hero_Splendor_Plus.jpg",
    "hero glamour":           "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Hero_Splendor_Plus.jpg/1280px-Hero_Splendor_Plus.jpg",
    "hero extreme":           "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Hero_Splendor_Plus.jpg/1280px-Hero_Splendor_Plus.jpg",
    "hero hunk":              "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Hero_Splendor_Plus.jpg/1280px-Hero_Splendor_Plus.jpg",
    "hero  cbz xtreme":       "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Hero_Splendor_Plus.jpg/1280px-Hero_Splendor_Plus.jpg",
    "hero honda cbz extreme": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Hero_Splendor_Plus.jpg/1280px-Hero_Splendor_Plus.jpg",
    "hero  ignitor disc":     "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Hero_Splendor_Plus.jpg/1280px-Hero_Splendor_Plus.jpg",

    # ── Honda Bikes ────────────────────────────────────────────────────────────
    "honda cb hornet 160r": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Honda_CB_Hornet_160R.jpg/1280px-Honda_CB_Hornet_160R.jpg",
    "honda cb shine":       "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Honda_CB_Shine.jpg/1280px-Honda_CB_Shine.jpg",
    "honda cb unicorn":     "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Honda_CB_Shine.jpg/1280px-Honda_CB_Shine.jpg",
    "honda cb trigger":     "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Honda_CB_Shine.jpg/1280px-Honda_CB_Shine.jpg",
    "honda cb twister":     "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Honda_CB_Shine.jpg/1280px-Honda_CB_Shine.jpg",
    "honda cbr 150":        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Honda_CB_Hornet_160R.jpg/1280px-Honda_CB_Hornet_160R.jpg",
    "honda karizma":        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Honda_CB_Hornet_160R.jpg/1280px-Honda_CB_Hornet_160R.jpg",
    "honda activa 125":     "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Honda_Activa_125.jpg/1280px-Honda_Activa_125.jpg",
    "honda activa 4g":      "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Honda_Activa_125.jpg/1280px-Honda_Activa_125.jpg",
    "activa 3g":            "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Honda_Activa_125.jpg/1280px-Honda_Activa_125.jpg",
    "activa 4g":            "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Honda_Activa_125.jpg/1280px-Honda_Activa_125.jpg",
    "honda dream yuga ":    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Honda_CB_Shine.jpg/1280px-Honda_CB_Shine.jpg",

    # ── TVS ────────────────────────────────────────────────────────────────────
    "tvs apache rtr 160": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/TVS_Apache_RTR_160.jpg/1280px-TVS_Apache_RTR_160.jpg",
    "tvs apache rtr 180": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/TVS_Apache_RTR_160.jpg/1280px-TVS_Apache_RTR_160.jpg",
    "tvs jupyter":        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/TVS_Jupiter.jpg/1280px-TVS_Jupiter.jpg",
    "tvs wego":           "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/TVS_Jupiter.jpg/1280px-TVS_Jupiter.jpg",
    "tvs sport ":         "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/TVS_Apache_RTR_160.jpg/1280px-TVS_Apache_RTR_160.jpg",

    # ── Yamaha ─────────────────────────────────────────────────────────────────
    "yamaha fz 16":      "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Yamaha_FZ-16.jpg/1280px-Yamaha_FZ-16.jpg",
    "yamaha fz s ":      "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Yamaha_FZ-16.jpg/1280px-Yamaha_FZ-16.jpg",
    "yamaha fz s v 2.0": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Yamaha_FZ-16.jpg/1280px-Yamaha_FZ-16.jpg",
    "yamaha fz  v 2.0":  "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Yamaha_FZ-16.jpg/1280px-Yamaha_FZ-16.jpg",
    "yamaha fazer ":     "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Yamaha_FZ-16.jpg/1280px-Yamaha_FZ-16.jpg",

    # ── Others ─────────────────────────────────────────────────────────────────
    "um renegade mojave":  "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Bajaj_Avenger_220.jpg/1280px-Bajaj_Avenger_220.jpg",
    "hyosung gt250r":      "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/KTM_RC_200.jpg/1280px-KTM_RC_200.jpg",
    "mahindra mojo xt300": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Bajaj_Dominar_400.jpg/1280px-Bajaj_Dominar_400.jpg",
    "suzuki access 125":   "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Honda_Activa_125.jpg/1280px-Honda_Activa_125.jpg",
}
DEFAULT_IMAGE = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/2018_Maruti_Suzuki_Swift_ZXi_1.2_Front.jpg/1280px-2018_Maruti_Suzuki_Swift_ZXi_1.2_Front.jpg"
def get_car_image(car_name: str) -> str:
    key = car_name.lower().strip()
    if key in CAR_IMAGES:
        return CAR_IMAGES[key]
    for k, v in CAR_IMAGES.items():
        if k in key or key in k:
            return v
    return DEFAULT_IMAGE

# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv("dataset/car_data.csv")
    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)
    df = df[df["Selling_Price"] > 0]
    le_name  = LabelEncoder()
    le_fuel  = LabelEncoder()
    le_sell  = LabelEncoder()
    le_trans = LabelEncoder()
    df["Car_Name_enc"]  = le_name.fit_transform(df["Car_Name"].str.lower().str.strip())
    df["Fuel_enc"]      = le_fuel.fit_transform(df["Fuel_Type"])
    df["Seller_enc"]    = le_sell.fit_transform(df["Seller_Type"])
    df["Trans_enc"]     = le_trans.fit_transform(df["Transmission"])
    features = ["Car_Name_enc","Year","Present_Price","Kms_Driven","Fuel_enc","Seller_enc","Trans_enc","Owner"]
    X, y = df[features], df["Selling_Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    return model, le_name, le_fuel, le_sell, le_trans, df

model, le_name, le_fuel, le_sell, le_trans, df = train_model()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("<div class='title'>AUTOVAL</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI Powered Car Price Predictor</div>", unsafe_allow_html=True)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUTS
# ─────────────────────────────────────────────
all_cars = sorted(df["Car_Name"].str.strip().unique().tolist())

st.markdown("<div class='section-header'>Vehicle Details</div>", unsafe_allow_html=True)
car_name = st.selectbox("Car Brand & Model", all_cars)

col1, col2 = st.columns(2)
with col1:
    year = st.number_input("Manufacturing Year", min_value=2000, max_value=2025, value=2018)
with col2:
    present_price = st.number_input("Present Price (Lakhs)", min_value=0.5, max_value=100.0, value=8.0, step=0.1)

col3, col4 = st.columns(2)
with col3:
    km = st.number_input("Kms Driven", min_value=0, max_value=500000, value=30000, step=500)
with col4:
    owner = st.slider("Previous Owners", 0, 5, 0)

st.markdown("<div class='section-header'>Specifications</div>", unsafe_allow_html=True)
col5, col6, col7 = st.columns(3)
with col5:
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
with col6:
    seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
with col7:
    trans = st.selectbox("Transmission", ["Manual", "Automatic"])

# ─────────────────────────────────────────────
# CAR IMAGE  (st.image — no external HTML needed)
# ─────────────────────────────────────────────
img_url = get_car_image(car_name)
col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
with col_img2:
    st.image(img_url, use_container_width=True)

# ─────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚗  Calculate Value"):
    with st.spinner("Running AI model..."):
        time.sleep(1.2)

    try:
        car_key = car_name.lower().strip()
        if car_key not in le_name.classes_:
            matches = [c for c in le_name.classes_ if car_key[:4] in c]
            car_key = matches[0] if matches else le_name.classes_[0]
        car_enc   = le_name.transform([car_key])[0]
        fuel_enc  = le_fuel.transform([fuel])[0]
        sell_enc  = le_sell.transform([seller])[0]
        trans_enc = le_trans.transform([trans])[0]
        X_in = np.array([[car_enc, year, present_price, km, fuel_enc, sell_enc, trans_enc, owner]])
        prediction = float(model.predict(X_in)[0])
        prediction = max(0.1, round(prediction, 2))
    except Exception:
        prediction = round(present_price * 0.65, 2)

    age_years   = 2024 - year
    depreciation = round(((present_price - prediction) / present_price) * 100, 1) if present_price > 0 else 0

    # ── RESULT CARD ──
    st.markdown(f"""
    <div class="result-card">
      <div class="result-car-name">{car_name.upper()}</div>
      <div class="result-meta">{year} &nbsp;·&nbsp; {fuel} &nbsp;·&nbsp; {trans} &nbsp;·&nbsp; {seller}</div>

      <div class="result-price-label">Estimated Resale Value</div>
      <div class="result-price">₹ {prediction:.2f}</div>
      <div class="result-price-sub">LAKHS</div>

      <hr class="result-divider">

      <div class="result-details">
        <div class="result-detail-item">
          <div class="result-detail-label">Present Price</div>
          <div class="result-detail-val">₹ {present_price:.1f}L</div>
        </div>
        <div class="result-detail-item">
          <div class="result-detail-label">KM Driven</div>
          <div class="result-detail-val">{km:,} km</div>
        </div>
        <div class="result-detail-item">
          <div class="result-detail-label">Car Age</div>
          <div class="result-detail-val">{age_years} yrs</div>
        </div>
        <div class="result-detail-item">
          <div class="result-detail-label">Depreciation</div>
          <div class="result-detail-val">{depreciation}%</div>
        </div>
        <div class="result-detail-item">
          <div class="result-detail-label">Prev Owners</div>
          <div class="result-detail-val">{owner}</div>
        </div>
      </div>

      <div class="badge-row">
        <span class="badge">{fuel}</span>
        <span class="badge">{trans}</span>
        <span class="badge">{seller}</span>
        <span class="badge">RF · 150 Trees</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.success("✅ Prediction completed — Random Forest Regressor")
