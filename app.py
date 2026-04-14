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
st.caption("Versi Stabil (Tanpa Ubah CSV) + AI")

# ========================
# GEMINI (TETAP ADA)
# ========================
model = None
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            model = genai.GenerativeModel(m.name)
            break
except:
    model = None

# ========================
# LOAD DATA (JANGAN DIUBAH)
# ========================
@st.cache_data
def load_data():
    return pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")

try:
    data = load_data()
except:
    st.error("Gagal membaca CSV. Pastikan kolom: kode, uraian")
    st.stop()

# validasi kolom
if "kode" not in data.columns or "uraian" not in data.columns:
    st.error("CSV harus punya kolom: kode dan uraian")
    st.stop()

# ========================
# PREPROCESS (SIMPLE & AMAN)
# ========================
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

data["uraian_clean"] = data["uraian"].astype(str).apply(preprocess)

# ========================
# DOCX READER (LEBIH AKURAT)
# ========================
def read_docx_best(file):
    try:
        doc = Document(file)
        text = " ".join([p.text for p in doc.paragraphs if len(p.text) > 20])

        sentences = text.split(".")
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

        # ambil 3 kalimat tengah (biasanya inti)
        if len(sentences) > 5:
            mid = len(sentences) // 2
            selected = sentences[mid-1:mid+2]
        else:
            selected = sentences[:3]

        return ". ".join(selected)

    except:
        return ""

# ========================
# INPUT
# ========================
uploaded = st.file_uploader("📂 Upload Word (.docx)", type=["docx"])

if uploaded:
    uraian_user = read_docx_best(uploaded)
    st.success("Dokumen dibaca")
else:
    uraian_user = st.text_input("🔍 Masukkan uraian arsip:")

# ========================
# TF-IDF GLOBAL (SIMPLE)
# ========================
def cari_kandidat(df, text, top_n=5):
    vec = TfidfVectorizer(ngram_range=(1,2))
    tfidf = vec.fit_transform(df["uraian_clean"])
    user_vec = vec.transform([preprocess(text)])

    sim = cosine_similarity(user_vec, tfidf)[0]
    idx = sim.argsort()[-top_n:][::-1]

    hasil = df.iloc[idx].copy()
    hasil["skor"] = sim[idx]

    return hasil

# ========================
# PROSES
# ========================
if uraian_user:

    kandidat = cari_kandidat(data, uraian_user, 5)

    st.info("💡 Kandidat terbaik:")
    kandidat_text = ""

    for _, row in kandidat.iterrows():
        st.write(f"{row['kode']} - {row['uraian']} ({row['skor']*100:.2f}%)")
        kandidat_text += f"{row['kode']} - {row['uraian']}\n"

    # hasil sistem
    terbaik = kandidat.iloc[0]
    st.success("🎯 Rekomendasi Sistem")
    st.write(f"**{terbaik['kode']} - {terbaik['uraian']}**")

    # ========================
    # AI FINAL (TETAP ADA 🔥)
    # ========================
    if model:
        if st.button("🚀 Validasi dengan AI"):
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
                st.success("🎯 Hasil AI")
                st.write(res.text)
            except:
                st.warning("AI gagal / quota habis")

# ========================
# FOOTER
# ========================
st.markdown("---")
st.caption("EKlasifikasi Arsip © 2026 | Clean Version")
