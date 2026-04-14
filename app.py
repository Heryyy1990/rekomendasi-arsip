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
st.set_page_config(page_title="EKlasifikasi Arsip", page_icon="📁")

st.title("📁 EKlasifikasi Arsip")
st.caption("TF-IDF + AI (Stabil & Akurat)")

# ========================
# API GEMINI
# ========================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except:
    st.warning("API Key belum diatur, mode AI tidak aktif")

# auto detect model
model = None
try:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            model = genai.GenerativeModel(m.name)
            break
except:
    model = None

# ========================
# LOAD DATA
# ========================
@st.cache_data
def load_data():
    return pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")

try:
    data = load_data()
except:
    st.error("File klasifikasi_arsip.csv tidak ditemukan")
    st.stop()

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
# READ WORD
# ========================
def read_docx(file):
    try:
        doc = Document(file)
        teks = [p.text.strip() for p in doc.paragraphs if len(p.text.strip()) > 20]
        return " ".join(teks[:5])
    except:
        return ""

# ========================
# INPUT
# ========================
uploaded = st.file_uploader("📂 Upload dokumen Word (.docx)", type=["docx"])

if uploaded:
    uraian_user = read_docx(uploaded)
    if uraian_user:
        st.success("Dokumen berhasil dibaca")
    else:
        st.warning("Dokumen kosong / tidak terbaca")
else:
    uraian_user = st.text_input("🔍 Masukkan uraian arsip:")

# ========================
# TF-IDF GLOBAL
# ========================
def tfidf_global(df, text, top_n=5):
    vec = TfidfVectorizer(
        ngram_range=(1,2),
        sublinear_tf=True
    )
    
    tfidf = vec.fit_transform(df['uraian_clean'])
    user_vec = vec.transform([preprocess(text)])
    
    sim = cosine_similarity(user_vec, tfidf)[0]
    idx = sim.argsort()[-top_n:][::-1]
    
    hasil = df.iloc[idx].copy()
    hasil['skor'] = sim[idx]
    
    return hasil

# ========================
# PROSES
# ========================
if uraian_user:

    kandidat = tfidf_global(data, uraian_user, 5)

    # urutkan: skor + panjang kode (lebih spesifik)
    kandidat['panjang_kode'] = kandidat['kode'].astype(str).apply(len)
    kandidat = kandidat.sort_values(by=['skor','panjang_kode'], ascending=False)

    # ========================
    # TAMPILKAN KANDIDAT
    # ========================
    st.info("💡 Kandidat terbaik:")
    
    kandidat_text = ""
    for _, row in kandidat.iterrows():
        st.write(f"{row['kode']} - {row['uraian']} ({row['skor']*100:.2f}%)")
        kandidat_text += f"{row['kode']} - {row['uraian']}\n"

    # ========================
    # HASIL OTOMATIS (NON AI)
    # ========================
    terbaik = kandidat.iloc[0]
    
    st.success("🎯 Rekomendasi Sistem")
    st.write(f"**{terbaik['kode']} - {terbaik['uraian']}**")

    # ========================
    # AI (OPSIONAL)
    # ========================
    if model:
        if st.button("🚀 Perjelas dengan AI"):

            try:
                prompt = f"""
Pilih 1 klasifikasi paling tepat.

Kandidat:
{kandidat_text}

Uraian:
{uraian_user}

Jawaban:
Kode - Uraian
Alasan singkat
"""

                res = model.generate_content(prompt)

                if hasattr(res, "text"):
                    st.success("🎯 Hasil AI")
                    st.write(res.text)
                else:
                    raise Exception("Tidak ada respon AI")

            except Exception as e:
                st.warning("AI gagal / quota habis → gunakan hasil sistem")

# ========================
# FOOTER
# ========================
st.markdown("---")
st.caption("EKlasifikasi Arsip by Heryanto S.Pd © 2026 | Hybrid System")
