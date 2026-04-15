import streamlit as st
import pandas as pd
import google.generativeai as genai
from thefuzz import process, fuzz
import re

# ========================
# 1. KONFIGURASI HALAMAN
# ========================
st.set_page_config(page_title="AI Klasifikasi Arsip", page_icon="📁", layout="centered")

st.markdown("""
<style>
    .title { text-align: center; font-size: 28px; font-weight: bold; color: #1E3A8A; }
    .stButton>button { width: 100%; border-radius: 8px; height: 45px; background-color: #2563EB; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ========================
# 2. LOAD MASTER DATA
# ========================
@st.cache_data
def load_data():
    df = pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df

data = load_data()

# ========================
# 3. PENCARIAN LOKAL (FUZZY LOGIC)
# ========================
def get_local_candidates(query, df, limit=3):
    query = str(query).lower().strip()
    choices = df['uraian_ai'].tolist()
    
    raw_results = process.extract(query, choices, scorer=fuzz.token_set_ratio, limit=limit)
    
    candidates = []
    for res in raw_results:
        match_text = res[0]
        score = res[1]
        match_idx = df[df['uraian_ai'] == match_text].index[0]
        row = df.iloc[match_idx]
        
        candidates.append({
            "Kode": str(row['kode']),
            "Uraian": row['uraian'],
            "Skor": f"{score}%"
        })
    return candidates

# ========================
# 4. ANTARMUKA (UI)
# ========================
st.markdown('<div class="title">📁 AI Klasifikasi Arsip</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Sistem Rekomendasi Kearsipan Cerdas</p>", unsafe_allow_html=True)

# Cek API Key di Secrets
if "GEMINI_API_KEY" not in st.secrets:
    st.error("⚠️ GEMINI_API_KEY belum ditambahkan di Streamlit Secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

uraian_user = st.text_input("🔍 Masukkan Perihal Surat / Uraian Arsip:")

if uraian_user:
    if st.button("🚀 Proses Klasifikasi"):
        
        # --- TAHAP 1: FILTER LOKAL ---
        kandidat_lokal = get_local_candidates(uraian_user, data, limit=3)
        
        st.markdown("### 🔎 3 Kandidat Terdekat dari Master Data:")
        st.table(pd.DataFrame(kandidat_lokal))
        
        # --- TAHAP 2: VALIDASI AI ---
        with st.spinner("🤖 AI sedang memvalidasi kode yang paling sesuai..."):
            kandidat_str = "\n".join([f"- {c['Kode']} : {c['Uraian']}" for c in kandidat_lokal])
            
            # PROMPT BERSIH, TEGAS, TANPA ASUMSI NGAWUR
            prompt = f"""
            Anda adalah Arsiparis Ahli Pemerintah. 
            Tugas Anda adalah memilih 1 kode klasifikasi yang paling akurat berdasarkan fungsi kearsipannya.
            
            Uraian Arsip dari User: "{uraian_user}"
            
            Daftar Pilihan Kode:
            {kandidat_str}
            
            ATURAN MUTLAK:
            1. Anda WAJIB memilih salah satu kode dari Daftar Pilihan Kode di atas.
            2. DILARANG membuat atau merekomendasikan kode di luar daftar tersebut.
            
            Pilih satu yang paling relevan dengan konteks uraian user.
            
            FORMAT JAWABAN WAJIB:
            **KODE TERPILIH:** [Tulis Kode] - [Tulis Uraian]
            **ALASAN:** [Satu kalimat penjelasan logis kearsipan]
            """
            
            try:
                response = model.generate_content(prompt)
                st.success("✅ Validasi AI Selesai")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Gagal memanggil AI. Pastikan kuota/koneksi API aman. Detail: {e}")

st.markdown("---")
st.caption("Dikembangkan untuk efisiensi tata kelola kearsipan pemerintah daerah.")
