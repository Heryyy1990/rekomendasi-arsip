import streamlit as st
import pandas as pd
import re
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========================
# CONFIG
# ========================
st.set_page_config(page_title="EKlasifikasi Arsip", page_icon="📁")
st.title("📁 EKlasifikasi Arsip")
st.caption("Versi Stabil + Input Manual + AI")

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
# LOAD DATA (TIDAK DIUBAH)
# ========================
@st.cache_data
def load_data():
    return pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")

try:
    data = load_data()
except:
    st.error("Gagal membaca CSV. Pastikan ada kolom 'kode' dan 'uraian'")
    st.stop()

# validasi
if "kode" not in data.columns or "uraian" not in data.columns:
    st.error("CSV harus memiliki kolom: kode dan uraian")
    st.stop()

# ========================
# PREPROCESS SEDERHANA
# ========================
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

data["uraian_clean"] = data["uraian"].astype(str).apply(preprocess)

# ========================
# INPUT USER
# ========================
uraian_user = st.text_input("🔍 Masukkan uraian atau kode klasifikasi:")

# ========================
# DETEKSI KODE LANGSUNG 🔥
# ========================
if uraian_user:
    kode_input = uraian_user.strip()

    # cocokkan kode persis
    match = data[data["kode"].astype(str) == kode_input]

    if not match.empty:
        row = match.iloc[0]
        st.success("🎯 Hasil Pencarian Kode")
        st.write(f"**{row['kode']} - {row['uraian']}**")
        st.stop()

# ========================
# TF-IDF PENCARIAN
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
    # AI (TETAP ADA 🔥)
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
st.caption("EKlasifikasi Arsip © 2026 | by Heryanto S.Pd")
