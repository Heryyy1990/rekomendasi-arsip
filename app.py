import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import json
import os

st.set_page_config(
    page_title="Klasifikasi Arsip Pemerintah",
    page_icon="🗂️",
    layout="wide"
)

st.title("🗂️ Sistem Rekomendasi Klasifikasi Arsip")
st.caption("Model Semantik (HuggingFace) + Validasi Arsiparis Digital (Google Gemini)")

@st.cache_resource
def load_semua():
    try:
    df = pd.read_csv("data/klasifikasi.csv", encoding="utf-8")
except:
    df = pd.read_csv("data/klasifikasi.csv", encoding="latin-1")
    df.columns = df.columns.str.strip()
    
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(
        df["uraian arsip"].tolist(),
        show_progress_bar=False
    )
    return df, model, embeddings

with st.spinner("⏳ Memuat model AI..."):
    df, model, embeddings = load_semua()

st.success(f"✅ Siap — {len(df)} kode klasifikasi tersedia")
st.divider()

st.subheader("📝 Masukkan Uraian Arsip")
uraian_input = st.text_area("Uraian arsip:", height=130)

proses = st.button("🔍 Klasifikasikan")

if proses and uraian_input.strip():

    q_emb = model.encode([uraian_input])
    skor = cosine_similarity(q_emb, embeddings)[0]
    top3_idx = np.argsort(skor)[::-1][:3]

    kandidat = [
        {
            "kode": str(df.iloc[i]["kode"]),
            "uraian": df.iloc[i]["uraian arsip"],
            "skor": float(skor[i])
        }
        for i in top3_idx
    ]

    st.subheader("📋 Tahap 1 — 3 Kandidat")

    for k in kandidat:
        st.write(k["kode"], "-", k["uraian"])

    st.subheader("🤖 Tahap 2 — Validasi Gemini")

    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

    if not api_key:
        st.error("API Key tidak ditemukan")
    else:
        genai.configure(api_key=api_key)

        kandidat_text = "\n".join([
            f"{i+1}. {k['kode']} - {k['uraian']}"
            for i, k in enumerate(kandidat)
        ])

        prompt = f"""
Dokumen:
{uraian_input}

Kandidat:
{kandidat_text}

Pilih 1 kode terbaik.
Jawab JSON:
{{
"kode_terpilih": "...",
"uraian_terpilih": "...",
"alasan": "..."
}}
"""

        try:
            model_gemini = genai.GenerativeModel("gemini-1.5-flash")
            response = model_gemini.generate_content(prompt)

            raw = response.text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()

            hasil = json.loads(raw)

            st.success("Hasil:")
            st.write(hasil)

        except Exception as e:
            st.error(str(e))
