import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document

# ========================
# CONFIG
# ========================
st.set_page_config(page_title="EKlasifikasi Arsip PRO", page_icon="📁")

st.title("📁 EKlasifikasi Arsip PRO")
st.caption("Klasifikasi Arsip Berbasis Hierarki + Analisis Teks")

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
# LEVEL KODE
# ========================
def get_level(kode):
    return kode.count('.')

data['level'] = data['kode'].apply(get_level)

data_lv1 = data[data['level'] == 0]
data_lv2 = data[data['level'] == 1]
data_lv3 = data[data['level'] == 2]

# ========================
# FUNCTION WORD
# ========================
def read_docx(file):
    doc = Document(file)
    paragraphs = [p.text.strip() for p in doc.paragraphs if len(p.text.strip()) > 20]
    return " ".join(paragraphs[:5])

# ========================
# INPUT
# ========================
uploaded_file = st.file_uploader("📂 Upload dokumen Word (.docx)", type=["docx"])

if uploaded_file:
    uraian_user = read_docx(uploaded_file)
    st.success("Dokumen berhasil dibaca")
else:
    uraian_user = st.text_input("🔍 Masukkan uraian arsip:")

# ========================
# FUNCTION CARI TERBAIK
# ========================
def cari_terbaik(df, text):
    if len(df) == 0:
        return None

    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        sublinear_tf=True
    )

    tfidf = vectorizer.fit_transform(df['uraian_clean'])
    user_vec = vectorizer.transform([preprocess(text)])

    sim = cosine_similarity(user_vec, tfidf)
    idx = sim[0].argmax()

    return df.iloc[idx], sim[0][idx]

# ========================
# PROSES UTAMA
# ========================
if uraian_user:

    # LEVEL 1
    hasil_lv1, skor1 = cari_terbaik(data_lv1, uraian_user)

    # LEVEL 2
    data_lv2_filter = data_lv2[data_lv2['kode'].str.startswith(hasil_lv1['kode'])]
    hasil_lv2, skor2 = cari_terbaik(data_lv2_filter, uraian_user) if len(data_lv2_filter) else (None, 0)

    # LEVEL 3
    data_lv3_filter = data_lv3[data_lv3['kode'].str.startswith(hasil_lv2['kode'])] if hasil_lv2 is not None else []
    hasil_lv3, skor3 = cari_terbaik(data_lv3_filter, uraian_user) if len(data_lv3_filter) else (None, 0)

    # ========================
    # HASIL FINAL
    # ========================
    if hasil_lv3 is not None:
        final = hasil_lv3
        skor = skor3
    elif hasil_lv2 is not None:
        final = hasil_lv2
        skor = skor2
    else:
        final = hasil_lv1
        skor = skor1

    # ========================
    # OUTPUT
    # ========================
    st.success("🎯 Klasifikasi Utama")
    st.write(f"**{final['kode']} - {final['uraian']}**")
    st.write(f"Skor kemiripan: {skor*100:.2f}%")

# ========================
# SARAN CEPAT (OPSIONAL)
# ========================
if uraian_user:

    vectorizer_all = TfidfVectorizer(ngram_range=(1,2))
    tfidf_all = vectorizer_all.fit_transform(data['uraian_clean'])
    user_vec_all = vectorizer_all.transform([preprocess(uraian_user)])

    sim_all = cosine_similarity(user_vec_all, tfidf_all)
    top_idx = sim_all[0].argsort()[-3:][::-1]

    st.info("💡 Saran alternatif:")
    for i in top_idx:
        if i < len(data):
            st.write(f"{data.iloc[i]['kode']} - {data.iloc[i]['uraian']} ({sim_all[0][i]*100:.2f}%)")

# ========================
# FOOTER
# ========================
st.markdown("---")
st.caption("EKlasifikasi Arsip PRO - Hierarchical System © 2026")
