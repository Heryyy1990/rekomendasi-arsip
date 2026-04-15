import streamlit as st
import pandas as pd
import google.generativeai as genai
from thefuzz import process, fuzz

# ========================
# 1. KONFIGURASI UI
# ========================
st.set_page_config(page_title="EKlasifikasi Hierarki", page_icon="📁", layout="centered")

st.markdown("""
<style>
    .title { text-align: center; font-size: 28px; font-weight: bold; color: #1E3A8A; }
    .stButton>button { width: 100%; border-radius: 8px; height: 45px; background-color: #0F766E; color: white; font-weight: bold; }
    .result-box { background-color: #F0FDF4; padding: 15px; border-radius: 8px; border-left: 5px solid #16A34A; }
</style>
""", unsafe_allow_html=True)

# ========================
# 2. LOAD DATA
# ========================
@st.cache_data
def load_data():
    df = pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df

data = load_data()

# ========================
# 3. PENCARIAN BERSIH (MENYEDIAKAN KANDIDAT)
# ========================
def get_candidates(query, df, limit=7):
    # Membersihkan kata umum agar tidak mengganggu pencarian inti
    stop_words = ["surat", "arsip", "dokumen", "administrasi", "perihal", "tentang"]
    clean_query = str(query).lower().strip()
    for word in stop_words:
        clean_query = clean_query.replace(word, "")
    clean_query = clean_query.strip()
    
    if not clean_query:
        clean_query = str(query).lower().strip()

    choices = df['uraian'].dropna().tolist()
    raw_results = process.extract(clean_query, choices, scorer=fuzz.token_set_ratio, limit=limit)
    
    candidates = []
    for match_text, score in raw_results:
        match_idx = df[df['uraian'] == match_text].index[0]
        row = df.iloc[match_idx]
        candidates.append({
            "Kode": str(row['kode']).strip(),
            "Uraian": str(row['uraian']).strip()
        })
    return candidates

# ========================
# 4. APLIKASI UTAMA
# ========================
st.markdown('<div class="title">📁 AI Klasifikasi Arsip</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Metode Pemikiran Hierarki (Inti ➔ Sub Bagian ➔ Kode)</p>", unsafe_allow_html=True)

uraian_user = st.text_input("🔍 Masukkan Perihal Arsip:", placeholder="Contoh: Sosialisasi tata kelola kearsipan desa")

if uraian_user:
    if st.button("🚀 Analisis Hierarki Kearsipan"):
        
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("⚠️ API Key belum dimasukkan di Streamlit Secrets!")
            st.stop()
            
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # 1. Tarik 7 Kandidat dari Database
        kandidat = get_candidates(uraian_user, data, limit=7)
        kandidat_str = "\n".join([f"- {c['Kode']} | {c['Uraian']}" for c in kandidat])
        
        st.markdown("### 🔎 Tabel Referensi Sistem Lokal")
        st.table(pd.DataFrame(kandidat))
        
        # 2. Paksa AI Berpikir Top-Down (Inti -> Sub -> Kode)
        st.markdown("### 🤖 Analisis Pohon Klasifikasi (AI)")
        with st.spinner("Mengidentifikasi Bagian Inti dan Sub Bagian..."):
            
            prompt = f"""
            Anda adalah Pakar Arsiparis. Tugas Anda adalah menemukan kode yang tepat dengan metode Analisis Hierarki (Top-Down).
            
            URAIAN SURAT: "{uraian_user}"
            
            KANDIDAT KODE YANG TERSEDIA:
            {kandidat_str}
            
            INSTRUKSI WAJIB (Ikuti urutan berpikir ini):
            1. Tentukan "Bagian Inti" (Fungsi Utama) dari surat tersebut. Apakah ini urusan Kepegawaian, Keuangan, Kearsipan/Perpustakaan, Umum, dll?
            2. Tentukan "Sub Bagian" (Kegiatan). Apa rincian spesifik dari inti tersebut?
            3. Berdasarkan Inti dan Sub Bagian di atas, pilih 1 (SATU) KODE FINAL dari daftar kandidat yang paling akurat.
            
            FORMAT JAWABAN (Harus persis seperti ini):
            **1. BAGIAN INTI:** [Sebutkan urusan utamanya]
            **2. SUB BAGIAN:** [Sebutkan kegiatannya]
            **3. KODE KLASIFIKASI:** [Kode Terpilih] - [Uraian Terpilih]
            """
            
            try:
                response = model.generate_content(prompt)
                
                # Menampilkan hasil dengan kotak hijau yang rapi
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.write(response.text)
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Gagal memanggil AI. Pastikan kuota/koneksi aman. Detail: {e}")

st.markdown("---")
st.caption("Logika Kearsipan Sesuai Tata Naskah Dinas")
