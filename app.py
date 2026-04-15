import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Klasifikasi Arsip (Vector)", layout="wide")

st.title("📁 Klasifikasi Arsip Berbasis Vector (Lebih Akurat)")

# =========================
# INPUT CSV
# =========================
csv_url = st.text_input("Masukkan link RAW CSV GitHub")

@st.cache_data
def load_data(url):
    try:
        return pd.read_csv(url)
    except:
        return pd.read_csv(url, encoding="latin1")

if csv_url:
    df = load_data(csv_url)
    st.success("CSV berhasil dimuat")
    st.dataframe(df)

    # =========================
    # PILIH KOLOM
    # =========================
    kolom_teks = st.selectbox("Kolom Teks", df.columns)
    kolom_label = st.selectbox("Kolom Kategori (Label)", df.columns)

    # =========================
    # INPUT TEKS BARU
    # =========================
    st.subheader("🔍 Uji Klasifikasi")

    input_teks = st.text_input("Masukkan teks arsip")

    if st.button("Klasifikasikan"):

        # vectorize
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df[kolom_teks].astype(str))

        input_vec = vectorizer.transform([input_teks])

        # hitung similarity
        similarity = cosine_similarity(input_vec, tfidf_matrix)

        # ambil yang paling mirip
        idx = similarity.argmax()
        skor = similarity[0][idx]

        kategori = df.iloc[idx][kolom_label]

        st.success("Hasil Klasifikasi:")
        st.write(f"Kategori: **{kategori}**")
        st.write(f"Tingkat kemiripan: {round(skor, 3)}")

        # tampilkan referensi terdekat
        st.subheader("📄 Referensi Paling Mirip")
        st.write(df.iloc[idx][kolom_teks])
