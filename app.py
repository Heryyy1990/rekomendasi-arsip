import streamlit as st
import pandas as pd
import google.generativeai as genai
from thefuzz import process, fuzz
import re

# ========================
# 1. STYLE & LAYOUT
# ========================
st.set_page_config(page_title="EKlasifikasi Pro v4", page_icon="📁", layout="centered")

st.markdown("""
<style>
    .title { text-align: center; font-size: 30px; font-weight: bold; color: #1E3A8A; }
    .stButton>button { width: 100%; border-radius: 10px; height: 50px; background-color: #047857; color: white; font-weight: bold; }
    .status-box { padding: 20px; border-radius: 10px; background-color: #F0FDF4; border: 1px solid #BBF7D0; color: #166534; }
</style>
""", unsafe_allow_html=True)

# ========================
# 2. INISIALISASI AI (ROBUST)
# ========================
def get_model():
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("⚠️ Masukkan API Key di Secrets Streamlit!")
        st.stop()
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # Menggunakan nama model yang paling stabil
    return genai.GenerativeModel("gemini-1.5-flash")

model = get_model()

# ========================
# 3. LOAD DATA (MASTER 3000 KODE)
# ========================
@st.cache_data
def load_data():
    # Pastikan file ini ada di GitHub Anda
    df = pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df

data = load_data()

# ========================
# 4. LOGIKA ANALISIS FUNGSI (ISO 15489)
# ========================
def get_functional_candidates(query, df):
    query = str(query).lower().strip()
    
    # Langkah 1: Pencarian berbasis skor gabungan (Fuzzy + Context)
    # Kami mencari di 'uraian_ai' karena mengandung kata kunci fungsional
    choices = df['uraian_ai'].tolist()
    
    # Ambil 10 kandidat teratas untuk dianalisis lebih dalam oleh AI
    raw_results = process.extract(query, choices, scorer=fuzz.token_set_ratio, limit=10)
    
    candidates = []
    for res in raw_results:
        match_text = res[0]
        score = res[1]
        match_idx = df[df['uraian_ai'] == match_text].index[0]
        row = df.iloc[match_idx]
        candidates.append({
            "kode": str(row['kode']),
            "uraian": row['uraian'],
            "context": row['uraian_ai']
        })
    return candidates

# ========================
# 5. ANTARMUKA (UI)
# ========================
st.markdown('<div class="title">📁 EKlasifikasi Pro v4</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Standar Internasional ISO 15489 - Analisis Fungsi Kearsipan</p>", unsafe_allow_html=True)

uraian_user = st.text_input("🔍 Masukkan Perihal atau Isi Ringkas Arsip:", 
                            placeholder="Contoh: Surat permohonan bantuan dana hibah pembangunan masjid")

if uraian_user:
    if st.button("🚀 Analisis Klasifikasi Sekarang"):
        with st.spinner("🤖 Melakukan Analisis Fungsi (Functional Analysis)..."):
            
            # AMBIL KANDIDAT BERDASARKAN KONTEKS
            candidates = get_functional_candidates(uraian_user, data)
            kandidat_str = "\n".join([f"- {c['kode']} | {c['uraian']} (Konteks: {c['context']})" for c in candidates])
            
            # PROMPT DENGAN LOGIKA ARSIPARIS DUNIA
            prompt = f"""
            Tugas Anda adalah melakukan klasifikasi arsip berdasarkan standar ISO 15489 (Functional Analysis).
            
            URAIAN ARSIP USER: "{uraian_user}"
            
            DAFTAR KODE TERSEDIA (KANDIDAT):
            {kandidat_str}
            
            LANGKAH BERPIKIR (LOGIKA):
            1. Tentukan FUNGSI utama (apakah ini urusan Keuangan, Kepegawaian, Pembangunan, atau Hukum?).
            2. Tentukan KEGIATAN spesifik di bawah fungsi tersebut.
            3. Pilih KODE yang paling menggambarkan transaksi akhir.
            
            HASIL YANG DIMINTA:
            Tampilkan 3 rekomendasi terbaik yang paling akurat.
            Setiap hasil harus mengikuti format:
            **[KODE] - [URAIAN]**
            *Analisis Fungsi:* [Penjelasan singkat kenapa kode ini dipilih berdasarkan fungsi organisasi]
            *Akurasi:* [X]%
            
            Berikan hasil yang sangat formal dan presisi.
            """
            
            try:
                response = model.generate_content(prompt)
                
                st.markdown('<div class="status-box">✅ Hasil Analisis Fungsi Berhasil</div>', unsafe_allow_html=True)
                st.divider()
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"Gagal memanggil AI: {e}. Coba lagi dalam beberapa saat.")

# ========================
# 6. FOOTER
# ========================
st.markdown("---")
st.caption("EKlasifikasi Pro v4 | Berdasarkan Standar Pengelolaan Arsip Internasional")
