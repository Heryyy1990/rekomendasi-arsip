import streamlit as st
import pandas as pd
import re
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document

# ========================
# CONFIG
# ========================
st.set_page_config(page_title="EKlasifikasi Arsip PRO", page_icon="📁")
st.title("📁 EKlasifikasi Arsip PRO")
st.caption("Hybrid: Hierarki + TF-IDF + AI")

# ========================
# API GEMINI (AUTO SAFE)
# ========================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except:
    st.error("API KEY belum diatur")
    st.stop()

# AUTO MODEL
model = None
try:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            model = genai.GenerativeModel(m.name)
            break
except:
    pass

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
# PREPROCESS
# ========================
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['uraian_clean'] = data['uraian'].astype(str).apply(preprocess)

# ========================
# LEVEL
# ========================
data['level'] = data['kode'].apply(lambda x: x.count('.'))

# ========================
# WORD READER
# ========================
def read_docx(file):
    doc = Document(file)
    return " ".join([p.text for p in doc.paragraphs if len(p.text) > 20][:5])

# ========================
# INPUT
# ========================
uploaded = st.file_uploader("📂 Upload Word", type=["docx"])

if uploaded:
    uraian_user = read_docx(uploaded)
    st.success("Dokumen dibaca")
else:
    uraian_user = st.text_input("🔍 Masukkan uraian:")

# ========================
# FUNCTION TFIDF
# ========================
def tfidf_search(df, text, top_n=3):
    if len(df) == 0:
        return []

    vec = TfidfVectorizer(ngram_range=(1,2))
    tfidf = vec.fit_transform(df['uraian_clean'])
    user_vec = vec.transform([preprocess(text)])

    sim = cosine_similarity(user_vec, tfidf)[0]
    idx = sim.argsort()[-top_n:][::-1]

    return df.iloc[idx]

# ========================
# PROSES
# ========================
if uraian_user:

    # LEVEL 1
    lv1 = data[data['level'] == 0]
    hasil_lv1 = tfidf_search(lv1, uraian_user, 1)

    if hasil_lv1.empty:
        st.warning("Tidak ditemukan")
        st.stop()

    kode_lv1 = hasil_lv1.iloc[0]['kode']

    # LEVEL 2
    lv2 = data[(data['level'] == 1) & (data['kode'].str.startswith(kode_lv1))]
    hasil_lv2 = tfidf_search(lv2, uraian_user, 1)

    # LEVEL 3
    kandidat = []

    if not hasil_lv2.empty:
        kode_lv2 = hasil_lv2.iloc[0]['kode']
        lv3 = data[(data['level'] == 2) & (data['kode'].str.startswith(kode_lv2))]
        kandidat = tfidf_search(lv3, uraian_user, 3)

    # fallback jika tidak ada level 3
    if len(kandidat) == 0:
        kandidat = tfidf_search(lv2, uraian_user, 3)

    # ========================
    # TAMPILKAN KANDIDAT
    # ========================
    st.info("💡 Kandidat hasil sistem:")
    kandidat_text = ""
    for _, row in kandidat.iterrows():
        st.write(f"{row['kode']} - {row['uraian']}")
        kandidat_text += f"{row['kode']} - {row['uraian']}\n"

    # ========================
    # AI FINAL SELECTOR
    # ========================
    if st.button("🚀 Tentukan Klasifikasi Terbaik (AI)"):

        try:
            prompt = f"""
Kamu adalah arsiparis ahli.

Pilih 1 kode klasifikasi paling tepat.

Kandidat:
{kandidat_text}

Uraian:
{uraian_user}

Gunakan logika:
- pahami isi dokumen
- pilih paling mewakili

Jawaban:
Kode - Uraian
Alasan singkat
"""

            res = model.generate_content(prompt)
            st.success("🎯 Hasil Akhir AI")
            st.write(res.text)

        except Exception as e:
            st.warning("AI gagal / quota habis → pakai hasil terbaik sistem")

            terbaik = kandidat.iloc[0]
            st.success("🎯 Hasil Sistem")
            st.write(f"{terbaik['kode']} - {terbaik['uraian']}")
