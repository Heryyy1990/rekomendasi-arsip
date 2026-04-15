import streamlit as st
import pandas as pd
import google.generativeai as genai
from thefuzz import fuzz
import os

# =================================================================
# 1. TAMPILAN ANTARMUKA
# =================================================================
st.set_page_config(page_title="AI Archivist Expert", page_icon="🏛️", layout="centered")

st.markdown("""
<style>
    .main-title { text-align: center; font-size: 30px; font-weight: bold; color: #1E3A8A; }
    .stButton>button { width: 100%; border-radius: 12px; height: 55px; background-color: #0F766E; color: white; font-weight: bold; border:none; font-size: 18px; }
    .result-card { padding: 25px; border-radius: 15px; background-color: #FFFFFF; border: 1px solid #E2E8F0; border-top: 8px solid #0F766E; margin-top: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# =================================================================
# 2. DATA & KONEKSI AI (FALLBACK SYSTEM)
# =================================================================
@st.cache_data
def load_data():
    if not os.path.exists("klasifikasi_arsip.csv"):
        st.error("File 'klasifikasi_arsip.csv' tidak ditemukan!")
        st.stop()
    return pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")

def get_brain():
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("API Key tidak ditemukan!")
        st.stop()
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # Mencari model yang tersedia secara dinamis untuk menghindari eror 404
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        target = next((m for m in models if "flash" in m), models[0])
        return genai.GenerativeModel(target)
    except:
        return genai.GenerativeModel("gemini-pro")

df_data = load_data()
brain = get_brain()

# =================================================================
# 3. LOGIKA BERPIKIR ARSIPARIS (PROMPT ENGINEERING)
# =================================================================
def analisis_inti_arsip(query, df):
    query_clean = str(query).lower().strip()
    
    # LANGKAH 1: Ambil jaring luas (Top 25) sebagai bahan bacaan AI
    candidates = []
    for _, row in df.iterrows():
        text = f"{row['kode']} {row['uraian']} {row['uraian_ai']}".lower()
        score = fuzz.token_set_ratio(query_clean, text)
        if score > 20: # Jaring sangat luas agar subjek inti tidak terlewat
            candidates.append(f"KODE: {row['kode']} | DESKRIPSI: {row['uraian']} | KELOMPOK: {row['uraian_ai']}")
    
    # Batasi bahan bacaan agar AI tidak bingung
    context = "\n".join(candidates[:25])
    
    # LANGKAH 2: Perintah Berpikir (Reasoning)
    prompt = f"""
    Anda adalah Pakar Arsiparis dengan kemampuan analisis substansi. 
    Abaikan kata-kata dekoratif (seperti: sosialisasi, pembinaan, rapat, laporan) jika itu bukan inti masalahnya.
    Fokuslah pada **SUBJEK UTAMA** atau **AKSI NYATA** dari uraian ini.

    URAIAN ARSIP: "{query}"

    DAFTAR REFERENSI KODE (Gunakan ini sebagai bahan pertimbangan):
    {context}

    TUGAS ANDA:
    1. Bedah uraian user. Tentukan apa 'Garis Besar' atau 'Inti' urusannya.
    2. Jika user menyebut 'Kearsipan', carilah kode yang secara fungsional menangani manajemen arsip (biasanya 040 atau 000.5).
    3. Jika user menyebut 'Cuti', carilah kode manajemen kesejahteraan pegawai (800).
    4. Pilih HANYA 1 KODE yang paling tepat sebagai keputusan final.

    FORMAT JAWABAN:
    ---
    ### 🎯 HASIL KEPUTUSAN: [KODE] - [URAIAN KODE]
    **ANALISIS SUBSTANSI (Garis Besar):** [Jelaskan apa inti dari uraian user menurut pemahaman Anda]
    **ALASAN KLASIFIKASI:** [Jelaskan mengapa kode ini dipilih berdasarkan fungsi organisasi, bukan sekadar kecocokan kata]
    """
    
    return brain.generate_content(prompt)

# =================================================================
# 4. HALAMAN UTAMA
# =================================================================
st.markdown('<div class="main-title">🏛️ AI Archivist Expert System</div>', unsafe_allow_html=True)
st.write("---")

uraian_input = st.text_area("✍️ Masukkan Uraian Arsip:", placeholder="Contoh: Sosialisasi tata naskah dinas dan kearsipan desa", height=120)

if uraian_input:
    if st.button("🚀 ANALISIS INTI & TENTUKAN KODE"):
        with st.spinner("Sedang memahami substansi arsip..."):
            try:
                response = analisis_inti_arsip(uraian_input, df_data)
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(response.text)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Gagal menganalisis: {e}")

st.markdown("---")
st.caption("E-Archivist West Muna | Decision Support System")
