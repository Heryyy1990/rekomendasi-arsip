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

# ── Header ────────────────────────────────────────────────────
st.title("🗂️ Sistem Rekomendasi Klasifikasi Arsip")
st.caption("Model Semantik (HuggingFace) + Validasi Arsiparis Digital (Google Gemini)")

# ========================
# LOAD DATA & MODEL
# ========================
@st.cache_resource
def load_semua():
    # Baca CSV dengan aman (hindari error encoding & format)
    try:
        df = pd.read_csv("data/klasifikasi.csv", encoding="utf-8")
    except:
        df = pd.read_csv(
            "data/klasifikasi.csv",
            encoding="latin-1",
            on_bad_lines="skip"
        )

    df.columns = df.columns.str.strip()

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    embeddings = model.encode(
        df["uraian arsip"].astype(str).tolist(),
        show_progress_bar=False
    )

    return df, model, embeddings

with st.spinner("⏳ Memuat model AI... (pertama kali ±1-2 menit)"):
    df, model, embeddings = load_semua()

st.success(f"✅ Siap — {len(df)} kode klasifikasi tersedia")
st.divider()

# ========================
# INPUT USER
# ========================
st.subheader("📝 Masukkan Uraian Arsip")

uraian_input = st.text_area(
    label="Uraian dokumen/arsip:",
    placeholder="Contoh: Surat keputusan pengangkatan Pegawai Negeri Sipil...",
    height=130
)

proses = st.button("🔍 Klasifikasikan")

# ========================
# PROSES
# ========================
if proses and uraian_input.strip():

    # ── TAHAP 1: Semantic Search ─────────────────────
    with st.spinner("🔎 Menganalisis makna arsip..."):
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

    st.subheader("📋 Tahap 1 — 3 Kandidat Kode")

    medal = ["🥇", "🥈", "🥉"]
    cols = st.columns(3)

    for i, (col, k) in enumerate(zip(cols, kandidat)):
        with col:
            st.metric(
                label=f"{medal[i]} Kandidat {i+1}",
                value=k["kode"],
                delta=f"{k['skor']:.1%}"
            )
            st.info(k["uraian"])

    # ── TAHAP 2: Gemini ─────────────────────
    st.divider()
    st.subheader("🤖 Tahap 2 — Validasi Gemini")

    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

    if not api_key:
        st.error("⚠️ API Key Google AI tidak ditemukan.")
    else:
        genai.configure(api_key=api_key)

        kandidat_text = "\n".join([
            f"{i+1}. Kode: {k['kode']} | Uraian: {k['uraian']}"
            for i, k in enumerate(kandidat)
        ])

        prompt = f"""
Anda adalah arsiparis profesional pemerintahan Indonesia.

DOKUMEN:
"{uraian_input}"

KANDIDAT:
{kandidat_text}

Gunakan langkah:
1. Identifikasi isi utama
2. Tentukan fungsi kegiatan
3. Cocokkan klasifikasi
4. Pilih paling spesifik
5. Perhatikan konteks unit kerja

Jawab hanya JSON:
{{
"kode_terpilih": "...",
"uraian_terpilih": "...",
"alasan": "..."
}}
"""

        with st.spinner("🧠 Gemini sedang menganalisis..."):
            try:
                model_gemini = genai.GenerativeModel("gemini-1.5-flash")
                response = model_gemini.generate_content(prompt)

                raw = response.text.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()

                hasil = json.loads(raw)

                st.success("✅ Kode Klasifikasi Final")

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.metric("📌 Kode", hasil["kode_terpilih"])
                    st.write(f"**Uraian:** {hasil['uraian_terpilih']}")

                with col2:
                    st.info(f"💡 **Alasan:**\n\n{hasil['alasan']}")

            except Exception as e:
                st.error(f"Terjadi error: {str(e)}")

elif proses:
    st.warning("⚠️ Masukkan uraian arsip terlebih dahulu.")
