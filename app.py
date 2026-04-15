import streamlit as st
import pandas as pd
import google.generativeai as genai
from thefuzz import fuzz
import os

# =================================================================
# 1. KONFIGURASI HALAMAN
# =================================================================
st.set_page_config(page_title="AI Archivist Expert", page_icon="🏛️", layout="centered")

st.markdown("""
<style>
    .main-title { text-align: center; font-size: 32px; font-weight: bold; color: #1E3A8A; margin-bottom: 10px; }
    .sub-title { text-align: center; color: #64748B; margin-bottom: 30px; font-style: italic; }
    .stButton>button { width: 100%; border-radius: 12px; height: 55px; background-color: #0F766E; color: white; font-weight: bold; border:none; font-size: 18px; }
    .result-card { padding: 20px; border-radius: 15px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-top: 5px solid #0F766E; margin-top: 15px; }
    .recommendation-title { color: #0F766E; font-weight: bold; font-size: 18px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# =================================================================
# 2. SISTEM DATA & AI CORE
# =================================================================
@st.cache_data
def load_archive_data():
    if not os.path.exists("klasifikasi_arsip.csv"):
        st.error("❌ File 'klasifikasi_arsip.csv' tidak ditemukan!")
        st.stop()
    df = pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df['uraian_ai'] = df['uraian_ai'].fillna("")
    return df

def get_ai_model():
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("❌ API Key tidak ditemukan di Secrets Streamlit!")
        st.stop()
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel("gemini-1.5-flash")

df_data = load_archive_data()
model = get_ai_model()

# =================================================================
# 3. MESIN PENCARI (KANDIDAT MENTAH)
# =================================================================
def collect_candidates(query, df, limit=15):
    query_clean = str(query).lower().strip()
    results = []
    for _, row in df.iterrows():
        # Mencari di uraian asli dan konteks silsilah
        text_to_check = f"{row['uraian']} {row['uraian_ai']}".lower()
        score = fuzz.token_set_ratio(query_clean, text_to_check)
        
        if score > 30:
            results.append({
                "kode": str(row['kode']).strip(),
                "uraian": str(row['uraian']).strip(),
                "konteks": str(row['uraian_ai']).strip(),
                "score": score
            })
    return sorted(results, key=lambda x: x['score'], reverse=True)[:limit]

# =================================================================
# 4. ANTARMUKA PENGGUNA (UI)
# =================================================================
st.markdown('<div class="main-title">🏛️ AI Archivist Expert</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">"Analisis Kontekstual 3 Rekomendasi Terbaik"</div>', unsafe_allow_html=True)

uraian_input = st.text_area("✍️ Masukkan Perihal/Deskripsi Arsip:", 
                            placeholder="Contoh: Surat permohonan cuti tahunan pegawai...",
                            height=100)

if uraian_input:
    if st.button("🚀 ANALISIS KLASIFIKASI"):
        
        kandidat_mentah = collect_candidates(uraian_input, df_data)
        
        if not kandidat_mentah:
            st.warning("⚠️ Tidak ditemukan referensi yang relevan.")
        else:
            with st.spinner("🧠 AI sedang menimbang 3 rekomendasi terbaik..."):
                context_for_ai = "\n".join([f"- Kode: {c['kode']} | Deskripsi: {c['uraian']} | Kelompok: {c['konteks']}" for c in kandidat_mentah])
                
                # Prompt khusus untuk membatasi 3 hasil dengan alasan mendalam
                prompt = f"""
                Anda adalah Pakar Arsiparis Senior. 
                Pahami substansi dari uraian ini: "{uraian_input}"

                Dari daftar referensi berikut:
                {context_for_ai}

                TUGAS ANDA:
                1. Pilih HANYA 3 KODE yang paling relevan secara substansi.
                2. Berikan alasan cerdas untuk setiap pilihan tersebut yang menjelaskan mengapa kode itu cocok dengan uraian user.
                3. Jika perihal tentang "Cuti", pastikan kode yang berhubungan dengan Manajemen Pegawai/Kesejahteraan (800) menjadi prioritas utama.

                WAJIB GUNAKAN FORMAT INI UNTUK SETIAP REKOMENDASI:
                ---
                ### [NOMOR]. KODE: [KODE] - [URAIAN]
                **Alasan Pemilihan:** [Jelaskan secara logis berdasarkan fungsi dan substansi arsip]
                """
                
                try:
                    response = model.generate_content(prompt)
                    
                    # Menampilkan hasil
                    st.markdown("### 📋 Hasil Analisis AI")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"Terjadi kendala: {e}")

st.markdown("---")
st.caption("E-Archivist West Muna | Fokus pada Ketepatan Substansi")
