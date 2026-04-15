import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# =================================================================
# 1. KONFIGURASI HALAMAN
# =================================================================
st.set_page_config(page_title="AI Archivist Pro", page_icon="📁", layout="centered")

st.markdown("""
<style>
    .main-title { text-align: center; font-size: 30px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { text-align: center; color: #64748B; margin-bottom: 30px; }
    .stButton>button { width: 100%; border-radius: 10px; height: 50px; background-color: #0F766E; color: white; font-weight: bold; border:none;}
    .result-card { padding: 20px; border-radius: 12px; background-color: #F0FDF4; border-left: 6px solid #16A34A; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

# =================================================================
# 2. PEMBUATAN VEKTOR LOKAL (TANPA API GOOGLE)
# =================================================================
@st.cache_resource
def inisialisasi_sistem_lokal():
    try:
        # Membaca langsung dari CSV, bukan PKL
        df = pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        df['uraian_ai'] = df['uraian_ai'].fillna("arsip")
        
        # Membuat vektor makna langsung di dalam aplikasi (Local Vector)
        vectorizer = TfidfVectorizer()
        matriks_vektor = vectorizer.fit_transform(df['uraian_ai'])
        
        return df, vectorizer, matriks_vektor
    except Exception as e:
        st.error(f"Gagal membaca file klasifikasi_arsip.csv: {e}")
        st.stop()

# Load Data
df_data, mesin_vektor, database_vektor = inisialisasi_sistem_lokal()

# Konfigurasi AI untuk Tahap 2 (Hanya jika API Key valid untuk teks)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    try:
        # Fallback paling aman untuk menghindari 404 di Text Generation
        model_ai = genai.GenerativeModel("gemini-pro") 
    except:
        model_ai = None
else:
    model_ai = None

# =================================================================
# 3. MESIN PENCARI MAKNA (LOCAL RAG)
# =================================================================
def cari_kandidat(query, df, vectorizer, matriks_db, limit=6):
    # Mengubah teks pengguna menjadi vektor lokal
    query_vec = vectorizer.transform([str(query).lower().strip()])
    
    # Menghitung kemiripan matematika
    skor = cosine_similarity(query_vec, matriks_db).flatten()
    top_indices = skor.argsort()[-limit:][::-1]
    
    kandidat = []
    for idx in top_indices:
        if skor[idx] > 0.05: # Ambil yang relevan
            row = df.iloc[idx]
            kandidat.append({
                "Kode": str(row['kode']).strip(),
                "Uraian": str(row['uraian']).strip(),
                "Konteks": str(row['uraian_ai']).strip(),
                "Akurasi": f"{round(skor[idx] * 100, 1)}%"
            })
    return kandidat

# =================================================================
# 4. ANTARMUKA (UI)
# =================================================================
st.markdown('<div class="main-title">📁 AI Klasifikasi Arsip</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Mesin Pencari Makna Lokal Berbasis TF-IDF Vector</div>', unsafe_allow_html=True)

uraian_user = st.text_input("🔍 Masukkan Perihal Surat / Uraian Arsip:", placeholder="Contoh: Sosialisasi kearsipan desa")

if uraian_user:
    if st.button("🚀 EKSEKUSI KLASIFIKASI"):
        
        # --- TAHAP 1: PENCARIAN VEKTOR LOKAL ---
        kandidat_makna = cari_kandidat(uraian_user, df_data, mesin_vektor, database_vektor, limit=6)
        
        st.markdown("### 🔎 Tahap 1: Hasil Pencarian Vektor Lokal")
        if not kandidat_makna:
            st.warning("Tidak ditemukan kecocokan di database.")
        else:
            st.table(pd.DataFrame(kandidat_makna)[["Kode", "Uraian", "Akurasi"]])
            
            # --- TAHAP 2: VALIDASI AI ---
            if model_ai:
                st.markdown("### 🤖 Tahap 2: Keputusan Akhir AI")
                with st.spinner("AI sedang mengunci fungsi utama..."):
                    kandidat_str = "\n".join([f"- {c['Kode']} | {c['Uraian']} (Konteks: {c['Konteks']})" for c in kandidat_makna])
                    
                    prompt = f"""
                    Anda adalah Ahli Tata Naskah Dinas.
                    
                    URAIAN SURAT: "{uraian_user}"
                    KANDIDAT KODE:
                    {kandidat_str}
                    
                    STANDAR MUTLAK:
                    - "Sosialisasi" tentang "Kearsipan" -> INTI = KEARSIPAN (040/000.5), BUKAN UMUM.
                    - "SK/Keputusan" tentang "Pegawai/Mutasi" -> INTI = KEPEGAWAIAN (800).
                    
                    Pilih HANYA 1 KODE TERBAIK.
                    
                    FORMAT HASIL:
                    **BAGIAN INTI:** [Nama Urusan]
                    **KODE TERPILIH:** [Kode] - [Uraian]
                    **ALASAN:** [Jelaskan 1 kalimat]
                    """
                    try:
                        response = model_ai.generate_content(prompt)
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.write(response.text)
                        st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error("Gagal menghubungi AI Google untuk analisis akhir. Tabel di atas adalah hasil pencarian final.")
            else:
                st.info("Sistem AI generasi teks tidak aktif. Silakan pilih kode yang paling sesuai dari tabel di atas.")

st.markdown("---")
st.caption("EKlasifikasi Pro | Sistem Analisis Subjek Tanpa API External")
