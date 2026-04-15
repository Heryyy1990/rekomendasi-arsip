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
# 3. PENCARIAN CERDAS (ANTI-POLUSI DATA)
# ========================
def get_clean_candidates(query, df, limit=10):
    # Hilangkan kata umum agar pencarian fokus ke inti masalah
    stop_words = ["surat", "arsip", "dokumen", "administrasi", "tentang", "perihal"]
    
    clean_query = str(query).lower().strip()
    for word in stop_words:
        clean_query = clean_query.replace(word, "")
    clean_query = clean_query.strip()
    
    # Jika setelah dibersihkan kosong (misal user cuma ketik "arsip"), kembalikan ke awal
    if clean_query == "":
        clean_query = str(query).lower().strip()

    # KITA GUNAKAN KOLOM 'uraian' ASLI, BUKAN 'uraian_ai'
    choices = df['uraian'].dropna().tolist()
    
    # Cari 10 kandidat terbaik agar AI punya pilihan lebih luas
    raw_results = process.extract(clean_query, choices, scorer=fuzz.token_set_ratio, limit=limit)
    
    candidates = []
    for match_text, score in raw_results:
        # Cari baris aslinya
        match_idx = df[df['uraian'] == match_text].index[0]
        row = df.iloc[match_idx]
        candidates.append({
            "Kode": str(row['kode']).strip(),
            "Uraian": str(row['uraian']).strip()
        })
    return candidates

# ========================
# 4. ANTARMUKA (UI)
# ========================
st.markdown('<div class="title">📁 AI Klasifikasi Arsip</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Sistem Cerdas Kearsipan (Bebas Polusi Data)</p>", unsafe_allow_html=True)

uraian_user = st.text_input("🔍 Masukkan Perihal Arsip:", placeholder="Contoh: Sosialisasi kearsipan desa")

if uraian_user:
    if st.button("🚀 Cari Kode Klasifikasi"):
        
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("⚠️ API Key belum dimasukkan di Streamlit Secrets!")
            st.stop()
            
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        with st.spinner("🔍 Memindai database dan menganalisis fungsi arsip..."):
            
            # 1. Dapatkan 10 kandidat teratas dari data lokal
            kandidat_lokal = get_clean_candidates(uraian_user, data, limit=10)
            kandidat_str = "\n".join([f"- Kode: {c['Kode']} | Uraian: {c['Uraian']}" for c in kandidat_lokal])
            
            # 2. Minta AI memilih 1 yang paling akurat dari 10 kandidat tersebut
            prompt = f"""
            Anda adalah Arsiparis Profesional. Klasifikasikan arsip berikut.
            
            Uraian Arsip: "{uraian_user}"
            
            DAFTAR KANDIDAT KODE:
            {kandidat_str}
            
            TUGAS:
            1. Pilih HANYA 1 (satu) kode dari DAFTAR KANDIDAT di atas yang paling sesuai secara hierarki tata naskah dinas.
            2. Jika tentang "sosialisasi/penyuluhan kearsipan", cari kode yang berkaitan dengan Pembinaan Kearsipan, Perpustakaan, atau Pendidikan.
            
            FORMAT JAWABAN:
            **KODE TERPILIH:** [Tulis Kode] - [Tulis Uraian]
            **ALASAN:** [Jelaskan singkat 1 kalimat]
            """
            
            try:
                response = model.generate_content(prompt)
                st.success("✅ Klasifikasi Berhasil Ditemukan")
                st.write(response.text)
                
                with st.expander("Lihat Daftar Kandidat Sistem Lokal"):
                    st.table(pd.DataFrame(kandidat_lokal))
                    
            except Exception as e:
                st.error(f"Gagal memanggil AI. Detail: {e}")

st.markdown("---")
st.caption("Dikembangkan untuk Aktualisasi Arsiparis Muna Barat")
