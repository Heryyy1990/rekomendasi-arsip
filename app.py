import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========================
# CONFIG
# ========================
st.set_page_config(page_title="EKlasifikasi Arsip", page_icon="📁", layout="centered")

# ========================
# STYLE (BERSIH & MOBILE)
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
st.markdown('<div class="subtitle">AI Klasifikasi Arsip (Cepat & Pintar)</div>', unsafe_allow_html=True)

# ========================
# GEMINI API
# ========================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# MODEL FIX
try:
    model = genai.GenerativeModel("models/gemini-1.5-flash")
except:
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
# INPUT
# ========================
uraian_user = st.text_input("🔍 Masukkan uraian arsip:")

# ========================
# SARAN OTOMATIS
# ========================
if uraian_user:
    user_vec = vectorizer.transform([uraian_user])
    sim = cosine_similarity(user_vec, tfidf_matrix)

    top_auto = sim[0].argsort()[-3:][::-1]

    st.info("💡 Saran cepat:")
    for i in top_auto:
        st.write(f"**{data.iloc[i]['kode']}** - {data.iloc[i]['uraian']}")

# ========================
# BUTTON
# ========================
if st.button("🚀 Cari Klasifikasi Pintar", use_container_width=True):

    if uraian_user.strip() != "":
        with st.spinner("AI sedang bekerja..."):

            try:
                # TF-IDF kandidat
                user_vector = vectorizer.transform([uraian_user])
                similarity = cosine_similarity(user_vector, tfidf_matrix)

                top_indices = similarity[0].argsort()[-5:][::-1]

                kandidat = ""
                for i in top_indices:
                    kandidat += f"{data.iloc[i]['kode']} - {data.iloc[i]['uraian']}\n"

                # GEMINI
                prompt = f"""
                Pilih 3 klasifikasi arsip paling tepat.

                Kandidat:
                {kandidat}

                Uraian:
                "{uraian_user}"

                Format:
                1. kode - uraian
                2. kode - uraian
                3. kode - uraian
                """

                response = model.generate_content(prompt)

                st.success("🎯 Hasil AI:")
                st.write(response.text)

            except Exception as e:
                st.error(f"Error: {e}")

    else:
        st.warning("Masukkan uraian dulu")

# ========================
# FOOTER
# ========================
st.markdown("---")
st.caption("EKlasifikasi Arsip by Heryanto © 2026")
