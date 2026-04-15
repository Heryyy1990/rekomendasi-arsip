import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

st.set_page_config(page_title="AI Archivist Pro", page_icon="🏛️", layout="centered")

st.markdown("""
<style>
    .main-title { text-align: center; font-size: 32px; font-weight: 900; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { text-align: center; color: #475569; margin-bottom: 30px; font-size: 16px;}
    .stButton>button { width: 100%; border-radius: 8px; height: 50px; background-color: #0F766E; color: white; font-weight: bold; border:none;}
    .result-card { padding: 25px; border-radius: 12px; background-color: #F0FDF4; border-left: 8px solid #16A34A; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

# =================================================================
# 1. PERSIAPAN SISTEM & AI
# =================================================================
if not os.path.exists("klasifikasi_arsip_vektor_hf.pkl"):
    st.error("🚨 File 'klasifikasi_arsip_vektor_hf.pkl' tidak ditemukan di GitHub!")
    st.stop()

@st.cache_resource
def load_system():
    # 1. Load Data Vektor
    df = pd.read_pickle("klasifikasi_arsip_vektor_hf.pkl")
    db_embeddings = np.array(df['vektor'].tolist())
    
    # 2. Load Model Vektor Open Source (Tanpa API)
    model_vektor = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return df, db_embeddings, model_vektor

df_vektor, matriks_vektor, mesin_pencari = load_system()

# 3. Setup Gemini (Hanya untuk merangkum teks, tidak pakai vektor google)
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    try:
        model_pemikir = genai.GenerativeModel("gemini-1.5-flash")
    except:
        model_pemikir = genai.GenerativeModel("gemini-pro")
else:
    model_pemikir = None

# =================================================================
# 2. FUNGSI PENCARI MAKNA (HUGGING FACE)
# =================================================================
def cari_makna(query, df, matriks_db, mesin, limit=6):
    # Ubah teks user jadi vektor
    query_emb = mesin.encode([str(query).strip()])
    
    # Hitung jarak makna
    skor = cosine_similarity(query_emb, matriks_db)[0]
    top_indices = skor.argsort()[-limit:][::-1]
    
    kandidat = []
    for idx in top_indices:
        if skor[idx] > 0.3: # Batas relevansi
            row = df.iloc[idx]
            kandidat.append({
                "Kode": str(row['kode']).strip(),
                "Uraian": str(row['uraian']).strip(),
                "Konteks": str(row['uraian_ai']).strip(),
                "Akurasi": f"{round(skor[idx] * 100, 1)}%"
            })
    return kandidat

# =================================================================
# 3. ANTARMUKA (UI)
# =================================================================
st.markdown('<div class="main-title">🏛️ AI Archivist Enterprise</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sistem Vector RAG (HuggingFace + Google AI)</div>', unsafe_allow_html=True)

uraian_user = st.text_input("🔍 Ketik Uraian/Perihal Arsip:", placeholder="Contoh: Pencairan dana desa")

if uraian_user:
    if st.button("🚀 EKSEKUSI KLASIFIKASI"):
        
        with st.spinner("🤖 Mengukur jarak semantik (Hugging Face)..."):
            kandidat_makna = cari_makna(uraian_user, df_vektor, matriks_vektor, mesin_pencari, limit=6)
            
        st.markdown("### 🔎 Bukti Transparansi Mesin (Tahap 1)")
        if not kandidat_makna:
            st.warning("Uraian terlalu melenceng dari standar kearsipan pemerintah.")
        else:
            st.table(pd.DataFrame(kandidat_makna)[["Kode", "Uraian", "Akurasi"]])
            
            st.markdown("### 🧠 Putusan Final Arsiparis AI (Tahap 2)")
            if model_pemikir:
                with st.spinner("Gemini menerapkan panduan tata naskah..."):
                    kandidat_str = "\n".join([f"- {c['Kode']} | {c['Konteks']}" for c in kandidat_makna])
                    prompt = f"""
                    Anda Auditor Kearsipan. Kunci 1 KODE KLASIFIKASI yang paling akurat.
                    URAIAN PENGGUNA: "{uraian_user}"
                    KANDIDAT DARI DATABASE:
                    {kandidat_str}
                    STANDAR OPERASIONAL:
                    - "Sosialisasi" tentang "Kearsipan" -> INTI = KEARSIPAN (040/000.5).
                    - "SK/Keputusan" tentang "Nasib Pegawai/Mutasi" -> INTI = KEPEGAWAIAN (800).
                    
                    PILIH 1 KODE SAJA.
                    FORMAT PUTUSAN:
                    **BAGIAN INTI:** [Nama Urusan]
                    **KODE FINAL:** [Kode] - [Uraian]
                    **ANALISIS ARSIPARIS:** [Alasan]
                    """
                    try:
                        hasil_final = model_pemikir.generate_content(prompt)
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.write(hasil_final.text)
                        st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Gagal mengambil putusan akhir: {e}")
            else:
                st.info("API Gemini tidak tersedia. Silakan gunakan tabel di atas.")

st.markdown("---")
st.caption("Hybrid Engine: HuggingFace Sentence-Transformers & Google Generative AI")
