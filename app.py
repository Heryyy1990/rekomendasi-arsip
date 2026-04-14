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
st.caption("Versi Akurat: Smart Document + TF-IDF + AI")

# ========================
# API GEMINI (OPSIONAL)
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
# STOPWORDS
# ========================
stopwords_id = [
    "dan","yang","di","ke","dari","untuk","pada","dengan",
    "adalah","itu","ini","dalam","oleh","atau","sebagai"
]

# ========================
# KEYWORD MAPPING (OPSIONAL)
# ========================
keyword_map = {
    "cuti": "cuti pegawai",
    "izin": "cuti pegawai",
    "gaji": "penggajian pegawai",
    "honor": "penggajian pegawai",
}

# ========================
# PREPROCESS
# ========================
def preprocess(text):
    text = text.lower()

    for k,v in keyword_map.items():
        text = text.replace(k, v)

    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords_id]

    return " ".join(words)

data['uraian_clean'] = data['uraian'].astype(str).apply(preprocess)

# ========================
# SMART DOCX READER
# ========================
def read_docx_smart(file):
    try:
        doc = Document(file)
        paragraphs = [p.text.strip() for p in doc.paragraphs if len(p.text.strip()) > 30]

        if len(paragraphs) == 0:
            return ""

        full_text = " ".join(paragraphs)

        sentences = full_text.split(".")
        sentences = [s.strip() for s in sentences if len(s.strip()) > 40]

        sentences = sorted(sentences, key=len, reverse=True)[:5]

        return ". ".join(sentences)

    except:
        return ""

# ========================
# INPUT
# ========================
uploaded = st.file_uploader("📂 Upload dokumen Word (.docx)", type=["docx"])

if uploaded:
    uraian_user = read_docx_smart(uploaded)
    if uraian_user:
        st.success("Dokumen berhasil diproses (inti diambil)")
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
        sublinear_tf=True,
        max_features=5000
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

    # 🔥 OPSIONAL: RINGKAS DENGAN AI
    if model and len(uraian_user) > 300:
        try:
            prompt = f"Ringkas inti dokumen ini dalam 1 kalimat:\n{uraian_user}"
            res = model.generate_content(prompt)
            if hasattr(res, "text"):
                uraian_user = res.text
                st.info("🧠 Ringkasan AI digunakan")
        except:
            pass

    kandidat = tfidf_global(data, uraian_user, 5)

    # urutkan (skor + panjang kode)
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
    # HASIL SISTEM
    # ========================
    terbaik = kandidat.iloc[0]

    st.success("🎯 Rekomendasi Sistem")
    st.write(f"**{terbaik['kode']} - {terbaik['uraian']}**")

    # ========================
    # AI FINAL (OPSIONAL)
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
                    raise Exception()

            except:
                st.warning("AI gagal / quota habis → gunakan hasil sistem")

# ========================
# FOOTER
# ========================
st.markdown("---")
st.caption("EKlasifikasi Arsip © 2026 | by Heryanto S.Pd")
