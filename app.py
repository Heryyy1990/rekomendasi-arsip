import streamlit as st
import pandas as pd
import google.generativeai as genai
from thefuzz import process, fuzz
import re

# ========================
# 1. KONFIGURASI & STYLE
# ========================
st.set_page_config(page_title="EKlasifikasi Pro", page_icon="📁", layout="centered")

st.markdown("""
<style>
    .title { text-align: center; font-size: 32px; font-weight: bold; color: #1E3A8A; }
    .subtitle { text-align: center; color: #64748B; margin-bottom: 30px; }
    .stButton>button { width: 100%; border-radius: 10px; height: 50px; background-color: #2563EB; color: white; font-weight: bold; }
    .result-box { padding: 15px; border-radius: 10px; background-color: #F8FAFC; border-left: 5px solid #2563EB; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ========================
# 2. SISTEM OTOMATIS
# ========================
# Pastikan GEMINI_API_KEY sudah diset di Streamlit Secrets
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    st.error("⚠️ GEMINI_API_KEY tidak ditemukan di Secrets!")
    st.stop()

@st.cache_data
def load_data():
    # Load file klasifikasi_arsip.csv yang ada di folder yang sama
    df = pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df

data = load_data()

# ========================
# 3. LOGIKA SEARCH (PEMBERSIH VALUEERROR)
# ========================
def get_best_candidates(query, df, limit=5):
    query = str(query).lower().strip()
    # Kita cari di kolom 'uraian_ai' karena mengandung kata kunci pendukung
    choices = df['uraian_ai'].tolist()
    
    # Perbaikan Unpacking ValueError:
    # Kita ambil extract tanpa memaksakan 3 variabel langsung di loop
    raw_results = process.extract(query, choices, scorer=fuzz.token_set_ratio, limit=limit)
    
    candidates = []
    for res in raw_results:
        # res bisa berisi (string, score) atau (string, score, index)
        val_text = res[0]
        score = res[1]
        
        # Cari index aslinya di dataframe
        match_idx = df[df['uraian_ai'] == val_text].index[0]
        row = df.iloc[match_idx]
        
        candidates.append({
            "kode": str(row['kode']),
            "uraian": row['uraian'],
            "score": score
        })
    return candidates

# ========================
# 4. ANTARMUKA (UI)
# ========================
st.markdown('<div class="title">📁 EKlasifikasi Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Akurasi Tinggi untuk Arsiparis Profesional</div>', unsafe_allow_html=True)

uraian_user = st.text_input("🔍 Masukkan uraian, perihal surat, atau kata kunci:", placeholder="Contoh: Surat permohonan kenaikan pangkat")

if uraian_user:
    # Cek apakah input adalah KODE langsung
    kode_check = data[data['kode'].str.contains(f"^{re.escape(uraian_user)}$", case=False, na=False, regex=True)]
    
    if not kode_check.empty:
        st.success("📌 Kode Ditemukan Secara Presisi:")
        for _, r in kode_check.iterrows():
            st.info(f"**{r['kode']}** - {r['uraian']}")

    # Jalankan proses AI Rekomendasi
    if st.button("🚀 Cari Rekomendasi Akurat"):
        with st.spinner("🤖 Menganalisis Struktur Klasifikasi..."):
            
            # Langkah 1: Ambil kandidat lokal
            candidates = get_best_candidates(uraian_user, data)
            
            # Langkah 2: Konsultasi ke Gemini untuk Akurasi 100%
            kandidat_str = "\n".join([f"- {c['kode']} {c['uraian']}" for c in candidates])
            
            prompt = f"""
            Anda adalah pakar Arsiparis RI. Tugas Anda adalah memberikan 3 rekomendasi kode klasifikasi paling akurat.
            
            Uraian dari User: "{uraian_user}"
            
            Daftar Kandidat dari Database:
            {kandidat_str}
            
            INSTRUKSI KHUSUS:
            1. Analisis apakah urusan ini masuk ke Pemerintahan (000), Kepegawaian (800), Keuangan (900), atau lainnya.
            2. Urutkan 3 yang paling logis.
            3. Berikan persentase keyakinan (confidence score) untuk masing-masing.
            4. Tampilkan HANYA Kode, Uraian, dan Persentase dalam format list.
            
            Format Hasil:
            1. [KODE] - [URAIAN] - [SCORE]%
            2. [KODE] - [URAIAN] - [SCORE]%
            3. [KODE] - [URAIAN] - [SCORE]%
            """
            
            try:
                response = model.generate_content(prompt)
                st.markdown("### 🎯 Rekomendasi Hasil Terbaik:")
                st.write(response.text)
                
                # Footer untuk setiap hasil
                st.divider()
                st.caption("Gunakan rekomendasi dengan skor tertinggi untuk akurasi maksimal.")
                
            except Exception as e:
                st.error(f"Terjadi kesalahan pada AI: {e}")

# ========================
# 5. FOOTER
# ========================
st.markdown("---")
st.caption("EKlasifikasi Pro by Heryanto S.Pd | Sistem Berbasis Pengetahuan Kearsipan")
