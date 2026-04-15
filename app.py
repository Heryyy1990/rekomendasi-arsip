import streamlit as st
import pandas as pd
import google.generativeai as genai
from thefuzz import process, fuzz

# =================================================================
# 1. KONFIGURASI HALAMAN & STYLE
# =================================================================
st.set_page_config(
    page_title="AI Archivist Pro - Muna Barat", 
    page_icon="📁", 
    layout="centered"
)

st.markdown("""
<style>
    .main-title { text-align: center; font-size: 30px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { text-align: center; color: #64748B; margin-bottom: 30px; }
    .stButton>button { 
        width: 100%; 
        border-radius: 10px; 
        height: 50px; 
        background-color: #0F766E; 
        color: white; 
        font-weight: bold; 
        border: none;
    }
    .result-card { 
        padding: 20px; 
        border-radius: 12px; 
        background-color: #F0FDF4; 
        border-left: 6px solid #16A34A; 
        margin-top: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# =================================================================
# 2. PENGATURAN DATA & AI
# =================================================================
@st.cache_data
def load_data():
    # Memuat master data yang sudah berisi silsilah hierarki di 'uraian_ai'
    df = pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df

def get_gemini_model():
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("⚠️ GEMINI_API_KEY tidak ditemukan di Secrets!")
        st.stop()
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel("gemini-1.5-flash")

# Inisialisasi Data dan Model
data = load_data()
model = get_gemini_model()

# =================================================================
# 3. LOGIKA PENCARIAN (LOCAL THINKING)
# =================================================================
def get_hierarchical_candidates(query, df, limit=5):
    query = str(query).lower().strip()
    # Mencari di kolom 'uraian_ai' karena mengandung silsilah lengkap (Induk ➔ Ranting)
    choices = df['uraian_ai'].dropna().tolist()
    
    # Menggunakan token_set_ratio agar lebih fleksibel terhadap urutan kata
    raw_results = process.extract(query, choices, scorer=fuzz.token_set_ratio, limit=limit)
    
    candidates = []
    for match_text, score in raw_results:
        # Menemukan baris asli berdasarkan teks uraian_ai
        match_idx = df[df['uraian_ai'] == match_text].index[0]
        row = df.iloc[match_idx]
        candidates.append({
            "Kode": str(row['kode']).strip(),
            "Uraian": str(row['uraian']).strip(),
            "Konteks Silsilah": str(row['uraian_ai']).strip(),
            "Skor": f"{score}%"
        })
    return candidates

# =================================================================
# 4. ANTARMUKA PENGGUNA (UI)
# =================================================================
st.markdown('<div class="main-title">📁 AI Klasifikasi Arsip</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sistem Rekomendasi Berbasis Analisis Hierarki Kearsipan</div>', unsafe_allow_html=True)

uraian_user = st.text_input("🔍 Masukkan Perihal Surat / Uraian Arsip:", placeholder="Contoh: Sosialisasi tata kelola kearsipan")

if uraian_user:
    if st.button("🚀 JALANKAN ANALISIS KLASIFIKASI"):
        
        # --- TAHAP 1: PENCARIAN LOKAL (TRANSPARAN) ---
        st.markdown("### 🔎 Tahap 1: Pencarian Silsilah Lokal")
        st.info("Sistem menemukan kandidat berikut berdasarkan kemiripan silsilah di database:")
        
        candidates = get_hierarchical_candidates(uraian_user, data)
        # Menampilkan tabel hanya kolom Kode, Uraian, dan Skor
        st.table(pd.DataFrame(candidates)[["Kode", "Uraian", "Skor"]])
        
        # --- TAHAP 2: VALIDASI AI (HIERARKI) ---
        st.markdown("### 🤖 Tahap 2: Analisis Pakar (AI)")
        with st.spinner("Menganalisis hubungan Fungsi Utama dan Sub Bagian..."):
            
            # Mempersiapkan data kandidat untuk dikirim ke AI
            kandidat_str = "\n".join([f"- {c['Kode']} | {c['Uraian']} (Silsilah: {c['Konteks Silsilah']})" for c in candidates])
            
            prompt = f"""
            Anda adalah Arsiparis Profesional. Klasifikasikan perihal surat menggunakan metode Analisis Hierarki.
            
            PERIHAL SURAT: "{uraian_user}"
            
            DAFTAR KANDIDAT KODE & SILSILAH:
            {kandidat_str}
            
            INSTRUKSI VALIDASI:
            1. Tentukan 'BAGIAN INTI' (Fungsi Utama) dari urusan tersebut.
            2. Tentukan 'SUB BAGIAN' (Kegiatan spesifik).
            3. Pilih 3 rekomendasi terbaik dari kandidat yang diberikan, urutkan dari yang paling akurat.
            
            FORMAT HASIL (WAJIB):
            1. [KODE] - [URAIAN]
               - Inti: [Sebutkan urusan utamanya]
               - Alasan: [Penjelasan singkat]
            2. [KODE] - [URAIAN]
            3. [KODE] - [URAIAN]
            """
            
            try:
                response = model.generate_content(prompt)
                
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.success("🎯 Rekomendasi Akhir Berhasil Ditemukan")
                st.markdown(response.text)
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Terjadi kendala pada koneksi AI: {e}")

# =================================================================
# 5. FOOTER
# =================================================================
st.markdown("---")
st.caption("EKlasifikasi Pro | Dikembangkan untuk Efisiensi Tata Kelola Kearsipan Pemerintah")
