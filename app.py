import streamlit as st
import pandas as pd
import google.generativeai as genai
from thefuzz import process, fuzz

# ========================
# 1. KONFIGURASI
# ========================
st.set_page_config(page_title="EKlasifikasi Pro", page_icon="📁", layout="centered")

st.markdown("""
<style>
    .title { text-align: center; font-size: 28px; font-weight: bold; color: #1E3A8A; }
    .stButton>button { width: 100%; border-radius: 8px; height: 45px; background-color: #047857; color: white; font-weight: bold; }
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
# 3. PENCARIAN CERDAS (BEBAS POLUSI)
# ========================
def get_clean_candidates(query, df, limit=5):
    # Buang kata umum agar fokus ke substansi (misal: "sosialisasi")
    stop_words = ["surat", "arsip", "dokumen", "administrasi", "tentang", "perihal"]
    
    clean_query = str(query).lower().strip()
    for word in stop_words:
        clean_query = clean_query.replace(word, "")
    clean_query = clean_query.strip()
    
    if clean_query == "":
        clean_query = str(query).lower().strip()

    # Cari di kolom 'uraian' ASLI (Bukan uraian_ai)
    choices = df['uraian'].dropna().tolist()
    
    raw_results = process.extract(clean_query, choices, scorer=fuzz.token_set_ratio, limit=limit)
    
    candidates = []
    for match_text, score in raw_results:
        match_idx = df[df['uraian'] == match_text].index[0]
        row = df.iloc[match_idx]
        candidates.append({
            "Kode": str(row['kode']).strip(),
            "Uraian": str(row['uraian']).strip(),
            "Kecocokan": f"{score}%"
        })
    return candidates

# ========================
# 4. ANTARMUKA (UI)
# ========================
st.markdown('<div class="title">📁 AI Klasifikasi Arsip</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Transparansi Pencarian Lokal & Analisis AI</p>", unsafe_allow_html=True)

uraian_user = st.text_input("🔍 Masukkan Perihal Arsip:", placeholder="Contoh: Sosialisasi kearsipan desa")

if uraian_user:
    if st.button("🚀 Proses Klasifikasi"):
        
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("⚠️ API Key belum dimasukkan di Streamlit Secrets!")
            st.stop()
            
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # --- TAHAP 1: TAMPILKAN HASIL LOKAL ---
        st.markdown("### 🔎 Tahap 1: Kandidat Sistem Lokal")
        st.info("Berikut 5 kode dengan kata kunci paling mirip sebelum divalidasi oleh AI:")
        
        # Ambil 5 kandidat
        kandidat_lokal = get_clean_candidates(uraian_user, data, limit=5)
        st.table(pd.DataFrame(kandidat_lokal)) # Menampilkan tabel secara terbuka!
        
        # --- TAHAP 2: VALIDASI AI ---
        st.markdown("### 🤖 Tahap 2: Keputusan Arsiparis (AI)")
        with st.spinner("AI sedang memilih kode yang paling tepat secara hierarki tata naskah..."):
            
            kandidat_str = "\n".join([f"- Kode: {c['Kode']} | Uraian: {c['Uraian']}" for c in kandidat_lokal])
            
            prompt = f"""
            Anda adalah Arsiparis Profesional.
            
            Uraian Arsip: "{uraian_user}"
            
            KANDIDAT KODE (HANYA BOLEH PILIH DARI SINI):
            {kandidat_str}
            
            TUGAS MUTLAK:
            1. Pilih HANYA SATU kode dari daftar kandidat di atas yang paling tepat sesuai fungsi kearsipannya.
            2. DILARANG merekomendasikan kode di luar 5 kandidat tersebut.
            
            FORMAT JAWABAN:
            **KODE TERPILIH:** [Kode] - [Uraian]
            **ALASAN:** [Jelaskan 1 kalimat kenapa kode ini paling tepat]
            """
            
            try:
                response = model.generate_content(prompt)
                st.success("✅ Analisis Selesai")
                st.write(response.text)
            except Exception as e:
                st.error(f"Gagal memanggil AI. Detail: {e}")

st.markdown("---")
st.caption("Dikembangkan untuk Aktualisasi Arsiparis Muna Barat")
