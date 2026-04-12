import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========================
# CONFIG PAGE
# ========================
st.set_page_config(
    page_title="EKlasifikasi Arsip",
    page_icon="📁",
    layout="centered"
)

# ========================
# CUSTOM CSS (BIAR KEREN & MOBILE FRIENDLY)
# ========================
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #2c3e50;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #7f8c8d;
        margin-bottom: 20px;
    }
    .box {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ========================
# HEADER
# ========================
st.markdown('<div class="title">📁 EKlasifikasi Arsip</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sistem Klasifikasi Arsip Pintar (AI + Pencarian Cepat)</div>', unsafe_allow_html=True)

# ========================
# API GEMINI (SECRET)
# ========================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
try:
    model = genai.GenerativeModel("gemini-1.5-flash")
except:
    model = genai.GenerativeModel("gemini-1.0-pro")

# ========================
# LOAD DATA
# ========================
@st.cache_data
def load_data():
    return pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")

data = load_data()

# ========================
# TF-IDF SETUP
# ========================
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['uraian'])

# ========================
# INPUT USER
# ========================
st.markdown('<div class="box">', unsafe_allow_html=True)

uraian_user = st.text_input("🔍 Masukkan uraian arsip", placeholder="Contoh: Surat permohonan cuti pegawai")

# ========================
# AUTO SARAN (REALTIME)
# ========================
if uraian_user:
    user_vec = vectorizer.transform([uraian_user])
    sim = cosine_similarity(user_vec, tfidf_matrix)

    top_auto = sim[0].argsort()[-3:][::-1]

    st.info("💡 Saran cepat:")

    for i in top_auto:
        st.write(f"**{data.iloc[i]['kode']}** - {data.iloc[i]['uraian']}")

st.markdown('</div>', unsafe_allow_html=True)

# ========================
# BUTTON AI
# ========================
if st.button("🚀 Cari Klasifikasi Pintar", use_container_width=True):

    if uraian_user.strip() != "":

        with st.spinner("🤖 AI sedang menganalisis..."):

            try:
                # STEP 1: TF-IDF ambil kandidat
                user_vector = vectorizer.transform([uraian_user])
                similarity = cosine_similarity(user_vector, tfidf_matrix)

                top_indices = similarity[0].argsort()[-5:][::-1]

                kandidat = ""
                for i in top_indices:
                    kandidat += f"{data.iloc[i]['kode']} - {data.iloc[i]['uraian']}\n"

                # STEP 2: GEMINI
                prompt = f"""
                Kamu adalah ahli klasifikasi arsip pemerintahan.

                Kandidat klasifikasi:
                {kandidat}

                Uraian arsip:
                "{uraian_user}"

                Pilih 3 klasifikasi paling tepat.

                Format:
                1. Kode - Uraian
                2. Kode - Uraian
                3. Kode - Uraian
                """

                response = model.generate_content(prompt)

                st.success("🎯 Hasil Klasifikasi AI")
                st.markdown(f'<div class="box">{response.text}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Terjadi error: {e}")

    else:
        st.warning("⚠️ Masukkan uraian arsip terlebih dahulu")

# ========================
# FOOTER
# ========================
st.markdown("""
---
<center style="color: gray; font-size: 12px;">
Aplikasi EKlasifikasi Arsip © 2026  
Dibuat dengan AI (Gemini) + Streamlit
</center>
""", unsafe_allow_html=True)
