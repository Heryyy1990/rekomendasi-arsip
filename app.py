import streamlit as st
import pandas as pd
import google.generativeai as genai
from thefuzz import fuzz
import os

st.set_page_config(page_title="AI Archivist Expert", page_icon="🏛️", layout="centered")

st.markdown("""
<style>
    .main-title { text-align: center; font-size: 30px; font-weight: bold; color: #1E3A8A; }
    .stButton>button { width: 100%; border-radius: 12px; height: 50px; background-color: #0F766E; color: white; font-weight: bold; border:none; }
    .local-result { padding: 15px; border-radius: 10px; background-color: #F0F9FF; border-left: 8px solid #0369A1; margin-bottom: 10px; }
    .ai-result { padding: 20px; border-radius: 10px; background-color: #F0FDF4; border-left: 8px solid #15803D; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if not os.path.exists("klasifikasi_arsip.csv"):
        st.error("File 'klasifikasi_arsip.csv' tidak ditemukan!")
        st.stop()
    df = pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df

df_data = load_data()

# =================================================================
# MESIN PENCARI DENGAN LOGIKA SUBJEK (ANTI-SOSIALISASI)
# =================================================================
def pencarian_presisi(query, df):
    query_clean = str(query).lower().strip()
    results = []
    
    # DAFTAR JANGKAR SUBJEK (Kearsipan, Kepegawaian, Keuangan)
    anchors = {
        "kearsipan": ["040", "000.5"],
        "arsip": ["040", "000.5"],
        "pegawai": ["800"],
        "cuti": ["800", "850"],
        "keuangan": ["900"],
        "anggaran": ["900"]
    }

    for _, row in df.iterrows():
        uraian = str(row['uraian']).lower()
        kode = str(row['kode']).strip()
        
        # Skor dasar
        score = fuzz.token_set_ratio(query_clean, uraian)
        
        # LOGIKA 1: Bonus "Jangkar" (Sangat Kuat)
        for key, prefixes in anchors.items():
            if key in query_clean:
                for prefix in prefixes:
                    if kode.startswith(prefix):
                        score += 200 # Loncatan skor agar rumpun ini jadi nomor 1
        
        # LOGIKA 2: Penalti Kata Umum (Agar "sosialisasi" tidak mengganggu)
        noise_words = ["sosialisasi", "penyuluhan", "pembinaan", "desiminasi"]
        if any(word in uraian for word in noise_words) and not any(word in query_clean for word in noise_words):
            score -= 50
            
        # LOGIKA 3: Exact Match Bonus
        if query_clean in uraian:
            score += 100

        if score > 50:
            results.append({
                "kode": kode,
                "uraian": str(row['uraian']).strip(),
                "score": score
            })
    
    return sorted(results, key=lambda x: x['score'], reverse=True)[:3]

# =================================================================
# UI
# =================================================================
st.markdown('<div class="main-title">🏛️ AI Archivist Expert</div>', unsafe_allow_html=True)
st.write("---")

uraian_input = st.text_input("✍️ Masukkan Perihal/Deskripsi Arsip:", placeholder="Contoh: kearsipan")

if uraian_input:
    if st.button("🚀 ANALISIS KLASIFIKASI"):
        kandidat = pencarian_presisi(uraian_input, df_data)
        
        st.markdown("### 📋 3 Rekomendasi Kode Terbaik")
        if not kandidat:
            st.warning("Tidak ditemukan hasil yang relevan.")
        else:
            for i, c in enumerate(kandidat):
                st.markdown(f"""
                <div class="local-result">
                    <b>{i+1}. Kode: {c['kode']}</b><br>
                    Uraian: {c['uraian']}
                </div>
                """, unsafe_allow_html=True)

            # Bagian AI (Opsional/Fallback)
            if "GEMINI_API_KEY" in st.secrets:
                try:
                    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    with st.spinner("🤖 Menyusun analisis pakar..."):
                        context_ai = "\n".join([f"- {c['kode']} | {c['uraian']}" for c in kandidat])
                        prompt = f"Berikan alasan singkat kenapa kode klasifikasi ini tepat untuk perihal '{uraian_input}':\n{context_ai}"
                        response = model.generate_content(prompt)
                        st.markdown(f'<div class="ai-result"><b>🧠 Analisis Pakar AI:</b><br>{response.text}</div>', unsafe_allow_html=True)
                except:
                    pass
