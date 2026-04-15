import streamlit as st
import pandas as pd
import google.generativeai as genai
from thefuzz import fuzz

# =================================================================
# 1. KONFIGURASI HALAMAN
# =================================================================
st.set_page_config(page_title="AI Archivist Pro", page_icon="📁", layout="centered")

st.markdown("""
<style>
    .main-title { text-align: center; font-size: 30px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { text-align: center; color: #64748B; margin-bottom: 30px; }
    .stButton>button { width: 100%; border-radius: 10px; height: 50px; background-color: #0F766E; color: white; font-weight: bold; }
    .result-card { padding: 20px; border-radius: 12px; background-color: #F0FDF4; border-left: 6px solid #16A34A; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

# =================================================================
# 2. DATA & AI
# =================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df

def get_gemini_model():
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("⚠️ GEMINI_API_KEY tidak ditemukan di Secrets!")
        st.stop()
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel("gemini-1.5-flash")

data = load_data()
model = get_gemini_model()

# =================================================================
# 3. MESIN PENCARI CUSTOM (ANTI BIAS 1 KATA)
# =================================================================
def get_smart_candidates(query, df, limit=8):
    query_clean = str(query).lower().strip()
    query_words = set(query_clean.split())
    
    results = []
    for idx, row in df.iterrows():
        uraian_asli = str(row['uraian']).lower()
        uraian_ai = str(row['uraian_ai']).lower()
        
        # 1. Hitung jumlah kata yang cocok di Silsilah (uraian_ai)
        ai_words = set(uraian_ai.replace('.', ' ').replace(',', ' ').split())
        overlap_ai = len(query_words.intersection(ai_words))
        
        # 2. Hitung jumlah kata yang cocok langsung di Uraian asli
        asli_words = set(uraian_asli.replace('.', ' ').replace(',', ' ').split())
        overlap_asli = len(query_words.intersection(asli_words))
        
        # 3. Rasio kemiripan murni
        sort_score = fuzz.token_sort_ratio(query_clean, uraian_ai)
        
        # RUMUS SUPER: (Kecocokan Silsilah x 100) + (Kecocokan Asli x 50) + Kemiripan
        total_score = (overlap_ai * 100) + (overlap_asli * 50) + sort_score
        
        if total_score > 30: # Hanya ambil yang relevan
            results.append({
                "Kode": str(row['kode']).strip(),
                "Uraian": str(row['uraian']).strip(),
                "Konteks Silsilah": str(row['uraian_ai']).strip(),
                "Score": total_score
            })
            
    # Urutkan berdasarkan skor rumus dari yang paling tinggi
    results = sorted(results, key=lambda x: x['Score'], reverse=True)[:limit]
    
    # Rapikan untuk tampilan tabel
    for r in results:
        r['Relevansi'] = "Tinggi" if r['Score'] > 200 else "Sedang"
        
    return results

# =================================================================
# 4. ANTARMUKA PENGGUNA (UI)
# =================================================================
st.markdown('<div class="main-title">📁 AI Klasifikasi Arsip</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Algoritma Pencarian Cerdas (Bebas Bias Kata Tunggal)</div>', unsafe_allow_html=True)

uraian_user = st.text_input("🔍 Masukkan Perihal Surat / Uraian Arsip:", placeholder="Contoh: Sosialisasi kearsipan desa")

if uraian_user:
    if st.button("🚀 JALANKAN ANALISIS HIERARKI"):
        
        # --- TAHAP 1: PENCARIAN LOKAL KUSTOM ---
        candidates = get_smart_candidates(uraian_user, data, limit=8)
        
        st.markdown("### 🔎 Tahap 1: Tabel Kandidat (Lokal)")
        st.info("Kini tabel mengambil kandidat dari fungsi kata yang merata, tidak terfokus pada 1 kata saja.")
        # Tampilkan tabel yang lebih rapi (tanpa menampilkan kolom silsilah yang kepanjangan)
        st.table(pd.DataFrame(candidates)[["Kode", "Uraian", "Relevansi"]])
        
        # --- TAHAP 2: VALIDASI AI ---
        st.markdown("### 🤖 Tahap 2: Analisis Pakar (AI)")
        with st.spinner("Menganalisis Fungsi Utama..."):
            
            kandidat_str = "\n".join([f"- {c['Kode']} | {c['Uraian']} (Silsilah: {c['Konteks Silsilah']})" for c in candidates])
            
            prompt = f"""
            Anda adalah Arsiparis Profesional.
            
            URAIAN SURAT: "{uraian_user}"
            
            DAFTAR KANDIDAT KODE:
            {kandidat_str}
            
            INSTRUKSI VALIDASI:
            1. Tentukan 'BAGIAN INTI' (Fungsi Utama) dari urusan tersebut (contoh: Kearsipan, Keuangan, Kepegawaian).
            2. Tentukan 'SUB BAGIAN' (Kegiatan spesifik).
            3. Pilih 1 KODE TERBAIK dari kandidat yang secara hierarki paling cocok. Jika ada sosialisasi mengenai kearsipan, pilih kode kearsipan, BUKAN kode kesra/umum.
            
            FORMAT HASIL (WAJIB):
            **BAGIAN INTI:** [Sebutkan urusan utamanya]
            **SUB BAGIAN:** [Sebutkan kegiatannya]
            **KODE TERPILIH:** [Kode] - [Uraian]
            **ALASAN:** [Penjelasan singkat]
            """
            
            try:
                response = model.generate_content(prompt)
                
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.write(response.text)
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Koneksi AI bermasalah: {e}")

st.markdown("---")
st.caption("EKlasifikasi Pro | Pemecahan Masalah Metode Fuzzy Logic")
