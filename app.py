import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Rekomendasi Klasifikasi Arsip", layout="centered")

st.title("Rekomendasi Klasifikasi Arsip AI (Gratis)")
st.write("Masukkan uraian arsip untuk mendapatkan kode klasifikasi")

# Load CSV
data = pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")

# TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['uraian'])

# Input
uraian_user = st.text_input("Masukkan uraian arsip")

if st.button("Cari Klasifikasi"):

    if uraian_user.strip() != "":

        user_vector = vectorizer.transform([uraian_user])

        similarity = cosine_similarity(user_vector, tfidf_matrix)

        # Ambil Top 3
        top_indices = similarity[0].argsort()[-3:][::-1]

        st.success("Top 3 Rekomendasi Klasifikasi")

        for i in top_indices:
            kode = data.iloc[i]['kode']
            uraian = data.iloc[i]['uraian']
            skor = similarity[0][i]

            st.write("-----")
            st.write("Kode:", kode)
            st.write("Uraian:", uraian)
            st.write("Skor:", round(skor, 2))

    else:
        st.warning("Masukkan uraian arsip dulu")
