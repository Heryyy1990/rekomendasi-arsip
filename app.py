import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========================
# CONFIG
# ========================
st.set_page_config(
    page_title="EKlasifikasi Arsip",
    page_icon="📁",
    layout="centered"
)

# ========================
# STYLE
# ========================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
.title {
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}
.stTextInput input {
    border-radius: 10px;
    padding: 10px;
}
.stButton>button {
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ========================
# HEADER
# ========================
st.markdown('<div class="title">📁 EKlasifikasi Arsip</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Klasifikasi Arsip (Hybrid: Cepat + Pintar)</div>', unsafe_allow_html=True)

# ========================
# API GEMINI
# ========================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ========================
# AUTO DETECT MODEL
# ========================
model = None
try:
    available_models = [
        m.name for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
    ]

    for m in available_models:
        if "gemini" in m:
            model = genai.GenerativeModel(m)
            break

except Exception as e:
    st.error(f"Gagal load model: {e}")

# fallback
if model is None:
    model = genai.GenerativeModel("models/gemini-1.0-pro")

# ========================
# LOAD DATA
# ========================
@st.cache_data
def load_data():
    return pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")

data = load_data()

# ========================
# TF-IDF
# ========================
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['uraian'])

# ========================
# INPUT USER
# ========================
uraian_user = st.text_input("🔍 Masukkan uraian atau kode klasifikasi:")

# ========================
# DETEKSI KODE LANGSUNG
# ========================
if uraian_user:
    kode_match = data[data['kode'].str.contains(uraian_user, case=False, na=False)]

    if not kode_match.empty:
        st.success("📌 Kode ditemukan")
        for _, row in kode_match.iterrows():
            st.write(f"**{row['kode']}** - {row['uraian']}")

# ========================
# SARAN CEPAT (TF-IDF)
# ========================
if uraian_user:
    user_vec = vectorizer.transform([uraian_user])
    sim = cosine_similarity(user_vec, tfidf_matrix)

    top_auto = sim[0].argsort()[-3:][::-1]

    st.info("💡 Saran cepat:")
    for i in top_auto:
        st.write(f"**{data.iloc[i]['kode']}** - {data.iloc[i]['uraian']}")

# ========================
# BUTTON AI
# ========================
if st.button("🚀 Cari Klasifikasi Pintar", use_container_width=True):

    if uraian_user.strip() != "":
        with st.spinner("🤖 AI sedang menganalisis..."):

            try:
                # STEP 1: TF-IDF kandidat
                user_vector = vectorizer.transform([uraian_user])
                similarity = cosine_similarity(user_vector, tfidf_matrix)

                top_indices = similarity[0].argsort()[-5:][::-1]

                kandidat = ""
                for i in top_indices:
                    kandidat += f"{data.iloc[i]['kode']} - {data.iloc[i]['uraian']}\n"

                # STEP 2: GEMINI (LOGIKA ARSIP)
                prompt = f"""
Kamu adalah arsiparis profesional di instansi pemerintah.

Ikuti langkah berikut:

1. Analisis isi:
Pahami inti masalah dari uraian arsip.

2. Pilih kode terdekat:
Dari kandidat, pilih yang paling relevan.

3. Tentukan sub masalah:
Persempit ke rincian masalah.

4. Tentukan sub-sub masalah:
Pilih klasifikasi paling spesifik.

Kandidat klasifikasi:
{kandidat}

Uraian arsip:
"{uraian_user}"

Tampilkan 3 hasil terbaik.

Format:
1. Kode - Uraian (alasan singkat)
2. Kode - Uraian (alasan singkat)
3. Kode - Uraian (alasan singkat)
"""

                response = model.generate_content(prompt)

                st.success("🎯 Hasil Klasifikasi AI")
                st.write(response.text)

            except Exception as e:
                st.error(f"Error AI: {e}")

    else:
        st.warning("Masukkan uraian arsip dulu")

# ========================
# FOOTER
# ========================
st.markdown("---")
st.caption("EKlasifikasi Arsip by Heryanto S.Pd © 2026 | Powered by Gemini AI")
