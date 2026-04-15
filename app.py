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
    .result-card { padding: 15px; border-radius: 10px; background-color: #F1F5F9; border-left: 5px solid #2563EB; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# ========================
# 2. INISIALISASI AI (ANTI 404)
# ========================
def get_gemini_model():
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("⚠️ Secrets 'GEMINI_API_KEY' belum diatur!")
        return None
    
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    try:
        # Mencari model yang didukung secara dinamis
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Prioritas: 1.5 Flash -> 1.5 Pro -> 1.0 Pro
        target_models = ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-1.0-pro"]
        
        for tm in target_models:
            if tm in models:
                return genai.GenerativeModel(tm)
        
        # Jika tidak ada yang cocok, ambil yang pertama tersedia
        return genai.GenerativeModel(models[0])
    except Exception as e:
        # Fallback paling aman
        return genai.GenerativeModel("gemini-1.5-flash")

model = get_gemini_model()

# ========================
# 3. DATA & LOGIKA SEARCH
# ========================
@st.cache_data
def load_data():
    df = pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df

data = load_data()

def hybrid_thinking(query, df):
    query = str(query).lower().strip()
    # Gunakan 'uraian_ai' untuk pencarian lokal agar lebih kaya konteks
    choices = df['uraian_ai'].tolist()
    
    # Ambil 5 kandidat untuk dipertimbangkan AI
    raw_results = process.extract(query, choices, scorer=fuzz.token_set_ratio, limit=5)
    
    candidates = []
    for res in raw_results:
        val_text = res[0]
        score = res[1]
        # Cari index baris di dataframe
        match_idx = df[df['uraian_ai'] == val_text].index[0]
        row = df.iloc[match_idx]
        candidates.append({"kode": str(row['kode']), "uraian": row['uraian']})
    
    return candidates

# ========================
# 4. ANTARMUKA (UI)
# ========================
st.markdown('<div class="title">📁 EKlasifikasi Pro v3</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Sistem Digitalisasi Arsip Berbasis Kecerdasan Buatan</p>", unsafe_allow_html=True)

uraian_user = st.text_input("🔍 Masukkan Perihal atau Uraian Arsip:", placeholder="Contoh: Surat permohonan cuti tahunan")

if uraian_user:
    # 1. Cek Cepat (Kecocokan Kode Langsung)
    exact_match = data[data['kode'].str.fullmatch(re.escape(uraian_user), case=False, na=False)]
    if not exact_match.empty:
        st.success("📌 Kode Ditemukan:")
        for _, r in exact_match.iterrows():
            st.write(f"**{r['kode']}** - {r['uraian']}")

    # 2. Proses Rekomendasi Pintar
    if st.button("🚀 Cari Rekomendasi Paling Akurat"):
        with st.spinner("🤖 AI sedang menganalisis database kearsipan..."):
            
            # Ambil kandidat lokal
            candidates = hybrid_thinking(uraian_user, data)
            kandidat_str = "\n".join([f"- {c['kode']} {c['uraian']}" for c in candidates])
            
            prompt = f"""
            Tugas: Pilih 3 kode klasifikasi paling tepat dari daftar kandidat untuk uraian surat berikut.
            
            Uraian Surat: "{uraian_user}"
            
            Daftar Kandidat (Ditemukan di Database):
            {kandidat_str}
            
            Instruksi:
            1. Analisis hubungan antara uraian surat dengan fungsi organisasi.
            2. Urutkan berdasarkan yang paling spesifik.
            3. Jawab dengan format: [KODE] - [URAIAN] - [Keyakinan %]
            
            Tampilkan 3 baris saja.
            """
            
            try:
                response = model.generate_content(prompt)
                st.markdown("### 🎯 Rekomendasi Terbaik:")
                st.write(response.text)
            except Exception as e:
                st.error(f"Eror saat memanggil AI: {e}. Pastikan API Key Anda aktif.")

# ========================
# 5. FOOTER
# ========================
st.markdown("---")
st.caption("EKlasifikasi Pro | Dikembangkan untuk Archivist Muna Barat")
