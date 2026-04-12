import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========================
# CONFIG
# ========================
st.set_page_config(page_title="EKlasifikasi Arsip", layout="centered")
st.title("EKlasifikasi Arsip")
st.write("Sistem klasifikasi arsip pintar (AI + Pencarian Cepat)")

# API KEY
genai.configure(api_key="API_KEY_KAMU")
model = genai.GenerativeModel("gemini-1.5-flash")

# ========================
# LOAD DATA
# ========================
data = pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")

# TF-IDF setup
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['uraian'])

# ========================
# AUTO SARAN SAAT MENGETIK
# ========================
uraian_user = st.text_input("Masukkan uraian arsip")

if uraian_user:
    user_vec = vectorizer.transform([uraian_user])
    sim = cosine_similarity(user_vec, tfidf_matrix)

    top_auto = sim[0].argsort()[-3:][::-1]

    st.info("💡 Saran cepat (berdasarkan kata kunci):")
    for i in top_auto:
        st.write(f"{data.iloc[i]['kode']} - {data.iloc[i]['uraian']}")

# ========================
# TOMBOL PROSES AI
# ========================
if st.button("🔍 Cari Klasifikasi Pintar"):

    if uraian_user.strip() != "":

        # ========================
        # STEP 1: TF-IDF (AMBIL TOP 5)
        # ========================
        user_vector = vectorizer.transform([uraian_user])
        similarity = cosine_similarity(user_vector, tfidf_matrix)

        top_indices = similarity[0].argsort()[-5:][::-1]

        kandidat = ""
        for i in top_indices:
            kandidat += f"{data.iloc[i]['kode']} - {data.iloc[i]['uraian']}\n"

        # ========================
        # STEP 2: GEMINI PILIH TERBAIK
        # ========================
        prompt = f"""
        Kamu adalah ahli klasifikasi arsip pemerintahan.

        Berikut kandidat klasifikasi:
        {kandidat}

        Uraian arsip:
        "{uraian_user}"

        Tugas:
        Pilih 3 klasifikasi paling tepat.

        Jawab format:
        1. Kode - Uraian
        2. Kode - Uraian
        3. Kode - Uraian
        """

        response = model.generate_content(prompt)

        st.success("🎯 Hasil Klasifikasi Pintar (AI)")
        st.write(response.text)

    else:
        st.warning("Masukkan uraian arsip dulu")
