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
    .stButton>button { width: 100%; border-radius: 10px; height: 50px; background-color: #0F766E; color: white; font-weight: bold; }
    .result-card { padding: 20px; border-radius: 12px; background-color: #F0FDF4; border-left: 6px solid #16A34A; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

# =================================================================
# 2. DATA, MODEL AI, & MESIN PEMBOBOTAN KATA (TF-IDF)
# =================================================================
@st.cache_data
def load_data_and_vectorizer():
    # Load Master Data
    df = pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    
    # Pastikan data aman
    df['uraian_ai'] = df['uraian_ai'].fillna("")
    
    # Inisialisasi Mesin Pembobot Kata (TF-IDF)
    vectorizer = TfidfVectorizer()
    # Melatih mesin untuk mengenali bobot setiap kata di Silsilah Hierarki
    tfidf_matrix = vectorizer.fit_transform(df['uraian_ai'])
    
    return df, vectorizer, tfidf_matrix

def get_gemini_model():
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("⚠️ GEMINI_API_KEY tidak ditemukan di Secrets!")
        st.stop()
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        target_model = next((m for m in available_models if "1.5-flash" in m), None)
        if not target_model:
            target_model = next((m for m in available_models if "gemini-pro" in m or "1.0-pro" in m), None)
        return genai.GenerativeModel(target_model or available_models[0])
    except:
        return genai.GenerativeModel("gemini-pro")

# Memuat data ke memori
data, vectorizer, tfidf_matrix = load_data_and_vectorizer()
model = get_gemini_model()

# =================================================================
# 3. MESIN PENCARI SEMANTIK (MEMAHAMI INTI KATA)
# =================================================================
def get_semantic_candidates(query, df, vectorizer, tfidf_matrix, limit=8):
    # Mengubah kalimat ketikan user menjadi vektor angka berbobot
    query_vec = vectorizer.transform([str(query).lower().strip()])
    
    # Menghitung kemiripan (Cosine Similarity) antara ketikan user dengan seluruh database
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Ambil index dengan skor tertinggi
    top_indices = similarity_scores.argsort()[-limit:][::-1]
    
    candidates = []
    for idx in top_indices:
        score = similarity_scores[idx]
        if score > 0.05: # Ambil yang punya relevansi saja
            row = df.iloc[idx]
            candidates.append({
                "Kode": str(row['kode']).strip(),
                "Uraian": str(row['uraian']).strip(),
                "Konteks": str(row['uraian_ai']).strip(),
                "Kecocokan": f"{round(score * 100, 1)}%"
            })
            
    return candidates

# =================================================================
# 4. ANTARMUKA PENGGUNA (UI)
# =================================================================
st.markdown('<div class="main-title">📁 AI Klasifikasi Arsip</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Algoritma Semantik (Memahami Subjek Utama)</div>', unsafe_allow_html=True)

uraian_user = st.text_input("🔍 Masukkan Perihal Surat / Uraian Arsip:", placeholder="Contoh: Sosialisasi kearsipan desa")

if uraian_user:
    if st.button("🚀 JALANKAN ANALISIS KLASIFIKASI"):
        
        # --- TAHAP 1: PENCARIAN SEMANTIK ---
        candidates = get_semantic_candidates(uraian_user, data, vectorizer, tfidf_matrix, limit=8)
        
        st.markdown("### 🔎 Tahap 1: Tabel Kandidat Prioritas Subjek")
        if not candidates:
            st.warning("Tidak ditemukan kandidat yang cocok di database.")
        else:
            # Tampilkan ke user tanpa kolom Konteks yang panjang
            st.table(pd.DataFrame(candidates)[["Kode", "Uraian", "Kecocokan"]])
            
            # --- TAHAP 2: VALIDASI AI ---
            st.markdown("### 🤖 Tahap 2: Keputusan Akhir AI")
            with st.spinner("AI sedang mengunci fungsi utama kearsipan..."):
                
                kandidat_str = "\n".join([f"- {c['Kode']} | {c['Uraian']} (Silsilah: {c['Konteks']})" for c in candidates])
                
                prompt = f"""
                Anda adalah Ahli Tata Naskah Dinas dan Kearsipan.
                
                URAIAN SURAT: "{uraian_user}"
                
                KANDIDAT KODE (Urut berdasarkan algoritma pembobot):
                {kandidat_str}
                
                INSTRUKSI VALIDASI:
                1. Analisis 'BAGIAN INTI' (Subjek Utama) dari uraian tersebut (contoh: apakah tentang Kearsipan, Kepegawaian, Hukum, dll?).
                2. Pilih HANYA 1 KODE dari kandidat yang secara hierarki mewakili inti dan kegiatan surat.
                
                FORMAT HASIL (WAJIB):
                **BAGIAN INTI:** [Sebutkan urusan utamanya]
                **KODE TERPILIH:** [Kode] - [Uraian]
                **ALASAN:** [Jelaskan 1 kalimat]
                """
                
                try:
                    response = model.generate_content(prompt)
                    
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.write(response.text)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Koneksi AI bermasalah: {e}")

st.markdown("---")
st.caption("EKlasifikasi Pro | Sistem Analisis Subjek Sesuai Tata Naskah")
