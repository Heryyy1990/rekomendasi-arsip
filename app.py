import streamlit as st
import pandas as pd
import google.generativeai as genai
from thefuzz import process, fuzz
import re

# ========================
# 1. KONFIGURASI HALAMAN
# ========================
st.set_page_config(page_title="EKlasifikasi Pro", page_icon="📁", layout="centered")

st.markdown("""
<style>
    .title { text-align: center; font-size: 30px; font-weight: bold; color: #1E3A8A; }
    .stButton>button { width: 100%; border-radius: 10px; height: 50px; background-color: #2563EB; color: white; font-weight: bold; }
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
# 3. PENCARIAN LOKAL (3 KANDIDAT)
# ========================
def get_local_candidates(query, df, limit=3):
    query = str(query).lower().strip()
    choices = df['uraian_ai'].tolist()
    
    # Ambil 3 terbaik
    raw_results = process.extract(query, choices, scorer=fuzz.token_set_ratio, limit=limit)
    
    candidates = []
    for res in raw_results:
        match_text = res[0]
        score = res[1]
        
        # Cari data aslinya
        match_idx = df[df['uraian_ai'] == match_text].index[0]
        row = df.iloc[match_idx]
        
        candidates.append({
            "Kode": str(row['kode']),
            "Uraian": row['uraian'],
            "Skor Kecocokan Lokal": f"{score}%"
        })
    return candidates

# ========================
# 4. FUNGSI AI DENGAN ANTI-EROR (FALLBACK)
# ========================
def generate_ai_response(prompt):
    if "GEMINI_API_KEY" not in st.secrets:
        return "⚠️ EROR: API Key tidak ditemukan di Streamlit Secrets."
    
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # Percobaan 1: Menggunakan Gemini 1.5 Flash
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e1:
        # Percobaan 2: Jika 404, pindah ke Gemini 1.0 Pro (Pasti Aman)
        try:
            model_fallback = genai.GenerativeModel("gemini-pro")
            response_fallback = model_fallback.generate_content(prompt)
            return f"*(Menggunakan model cadangan karena 1.5-flash sibuk/eror)*\n\n{response_fallback.text}"
        except Exception as e2:
            return f"⚠️ EROR FATAL API: Gagal memanggil semua model AI. Detail: {e2}"

# ========================
# 5. ANTARMUKA (UI)
# ========================
st.markdown('<div class="title">📁 EKlasifikasi Pro v5</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Transparansi Pencarian Lokal & Validasi AI</p>", unsafe_allow_html=True)

uraian_user = st.text_input("🔍 Masukkan Perihal atau Isi Ringkas Arsip:", placeholder="Contoh: Pembelian alat tulis kantor")

if uraian_user:
    if st.button("🚀 Proses Klasifikasi"):
        
        # --- TAHAP 1: TAMPILKAN HASIL LOKAL ---
        st.markdown("### 🔎 Tahap 1: Pencarian Lokal (Tanpa AI)")
        st.info("Berikut adalah 3 kode yang paling mirip dengan ketikan Anda berdasarkan Master Data:")
        
        kandidat_lokal = get_local_candidates(uraian_user, data, limit=3)
        # Menampilkan sebagai tabel agar rapi
        st.table(pd.DataFrame(kandidat_lokal))
        
        # --- TAHAP 2: VALIDASI AI ---
        st.markdown("### 🤖 Tahap 2: Validasi Analisis Fungsi oleh AI")
        with st.spinner("AI sedang memikirkan mana yang paling logis secara aturan kearsipan..."):
            
            kandidat_str = "\n".join([f"- Kode: {c['Kode']} | Uraian: {c['Uraian']}" for c in kandidat_lokal])
            
            prompt = f"""
            Tugas Anda adalah memvalidasi klasifikasi arsip berdasarkan standar kearsipan pemerintah.
            
            Uraian Surat: "{uraian_user}"
            
            3 Kandidat dari Sistem Lokal:
            {kandidat_str}
            
            INSTRUKSI:
            1. Analisis apakah surat tersebut secara fungsi lebih cocok ke urusan Umum, Keuangan, Kepegawaian, dsb.
            2. Pilih 1 KODE PALING TEPAT dari 3 kandidat di atas. Jika ketiganya kurang pas, pilih yang paling mendekati.
            3. Berikan alasan teknis kearsipan mengapa Anda memilih kode tersebut.
            
            FORMAT JAWABAN:
            **KODE TERPILIH:** [KODE] - [URAIAN]
            **ALASAN:** [Jelaskan singkat logika analisis fungsinya]
            """
            
            hasil_ai = generate_ai_response(prompt)
            
            st.success("Analisis Selesai!")
            st.write(hasil_ai)

# ========================
# 6. FOOTER
# ========================
st.markdown("---")
st.caption("Sistem Klasifikasi Hibrida (Fuzzy Matching + Generative AI)")
