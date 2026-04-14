import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from docx import Document

# ========================
# CONFIG
# ========================
st.set_page_config(page_title="EKlasifikasi Arsip PRO", page_icon="📁")

st.title("📁 EKlasifikasi Arsip PRO")
st.caption("AI Klasifikasi Arsip (Ultra Stabil Version)")

# ========================
# API GEMINI (AMAN)
# ========================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except:
    st.error("API KEY Gemini belum diatur di secrets")
    st.stop()

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
    st.warning(f"Gagal load model otomatis: {e}")

# fallback
if model is None:
    try:
        model = genai.GenerativeModel("models/gemini-1.0-pro")
    except:
        st.error("Semua model tidak tersedia")
        st.stop()

# debug model
st.caption(f"Model aktif: {model._model_name}")

# ========================
# LOAD DATA
# ========================
@st.cache_data
def load_data():
    return pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")

try:
    data_awal = load_data()
except:
    st.error("File klasifikasi_arsip.csv tidak ditemukan")
    st.stop()

# ========================
# SESSION
# ========================
if "history" not in st.session_state:
    st.session_state.history = []

# ========================
# FUNCTION WORD
# ========================
def read_docx(file):
    try:
        doc = Document(file)
        paragraphs = [p.text.strip() for p in doc.paragraphs if len(p.text.strip()) > 30]
        return " ".join(paragraphs[:5])
    except:
        return ""

# ========================
# INPUT
# ========================
uploaded_file = st.file_uploader("📂 Upload dokumen Word (.docx)", type=["docx"])

uraian_user = ""

if uploaded_file:
    uraian_user = read_docx(uploaded_file)

    if uraian_user == "":
        st.warning("Isi dokumen kosong atau tidak terbaca")
    else:
        st.success("Dokumen berhasil diproses")

else:
    uraian_user = st.text_input("🔍 Masukkan uraian atau kode klasifikasi:")

# ========================
# FILTER
# ========================
data = data_awal.copy()

filter_kata = st.text_input("📂 Filter bidang (opsional)")

if filter_kata:
    data = data[data['uraian'].str.contains(filter_kata, case=False, na=False)]

data = data.reset_index(drop=True)

if len(data) == 0:
    st.warning("Data tidak ditemukan")
    st.stop()

# ========================
# TF-IDF
# ========================
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['uraian'])

# ========================
# DETEKSI KODE
# ========================
if uraian_user:
    try:
        kode_match = data[data['kode'].str.contains(uraian_user, case=False, na=False)]
        if not kode_match.empty:
            st.success("📌 Kode ditemukan")
            for _, row in kode_match.iterrows():
                st.write(f"{row['kode']} - {row['uraian']}")
    except:
        pass

# ========================
# SARAN CEPAT
# ========================
if uraian_user:
    try:
        user_vec = vectorizer.transform([uraian_user])
        sim = cosine_similarity(user_vec, tfidf_matrix)

        top_auto = sim[0].argsort()[-3:][::-1]

        st.info("💡 Saran cepat:")
        for i in top_auto:
            if i < len(data):
                skor = sim[0][i] * 100
                st.write(f"{data.iloc[i]['kode']} - {data.iloc[i]['uraian']} ({skor:.2f}%)")
    except:
        st.warning("Tidak bisa menghitung saran")

# ========================
# AI BUTTON
# ========================
if st.button("🚀 Analisis Klasifikasi Utama", use_container_width=True):

    if uraian_user.strip() != "":
        with st.spinner("AI menganalisis..."):

            try:
                user_vector = vectorizer.transform([uraian_user])
                similarity = cosine_similarity(user_vector, tfidf_matrix)

                top_indices = similarity[0].argsort()[-5:][::-1]

                kandidat = ""
                for i in top_indices:
                    if i < len(data):
                        kandidat += f"{data.iloc[i]['kode']} - {data.iloc[i]['uraian']}\n"

                prompt = f"""
Kamu adalah arsiparis profesional.

Tentukan 1 klasifikasi utama dokumen ini.

Kandidat:
{kandidat}

Isi:
{uraian_user}

Jawaban:
Kode - Uraian
Alasan singkat
"""

                response = model.generate_content(prompt)

                if hasattr(response, "text"):
                    hasil = response.text
                else:
                    hasil = "AI tidak memberikan respon."

                st.success("🎯 Hasil")
                st.write(hasil)

                st.session_state.history.append({
                    "input": uraian_user[:100],
                    "hasil": hasil
                })

            except Exception as e:
                st.error(f"Error AI: {e}")

    else:
        st.warning("Masukkan uraian terlebih dahulu")

# ========================
# STATISTIK
# ========================
st.markdown("## 📊 Statistik")

jumlah = len(st.session_state.history)
st.write(f"Total analisis: {jumlah}")

if jumlah > 0:
    kata = [h["input"] for h in st.session_state.history]
    populer = Counter(kata).most_common(3)

    for k, v in populer:
        st.write(f"- {k} ({v}x)")

# ========================
# RIWAYAT
# ========================
if st.session_state.history:
    st.markdown("## 🧠 Riwayat")
    for h in reversed(st.session_state.history[-5:]):
        st.write(f"🔎 {h['input']}...")
        st.write(h['hasil'])
        st.markdown("---")

# ========================
# FOOTER
# ========================
st.markdown("---")
st.caption("EKlasifikasi Arsip PRO by Heryanto S.Pd © 2026")
