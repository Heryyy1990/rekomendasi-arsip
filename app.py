import streamlit as st
import pandas as pd
from openai import OpenAI

# API KEY
client = OpenAI(api_key="ISI_API_KEY_DISINI")

# Judul aplikasi
st.title("Rekomendasi Klasifikasi Arsip AI")

st.write("Masukkan uraian arsip untuk mendapatkan kode klasifikasi")

# Load data
data = pd.read_csv("klasifikasi_arsip.csv")

# Input user
uraian_user = st.text_input("Masukkan uraian arsip")

if st.button("Cari Klasifikasi"):

    if uraian_user != "":

        daftar_klasifikasi = data.to_string()

        prompt = f"""
        Kamu adalah arsiparis.

        Tugas kamu memilih kode klasifikasi arsip yang paling sesuai.

        Data klasifikasi arsip:

        {daftar_klasifikasi}

        Uraian arsip:

        {uraian_user}

        Pilih 1 kode yang paling sesuai.

        Tampilkan:

        Kode:
        Uraian:
        Alasan:
        """

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        hasil = response.choices[0].message.content

        st.success("Hasil Rekomendasi")
        st.write(hasil)

    else:
        st.warning("Masukkan uraian arsip dulu")
