import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process, fuzz # Pastikan instal: pip install thefuzz

# ========================
# CONFIG & SECRETS
# ========================
st.set_page_config(page_title="AI Archivist Pro", page_icon="📁", layout="centered")

# Mengambil API Key dari Secrets Streamlit
API_KEY = st.secrets.get("GEMINI_API_KEY")

# ========================
# LOAD MASTER DATA
# ========================
@st.cache_data
def load_data():
    # Memastikan file dibaca dengan benar sesuai master data Anda
    df = pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df

data = load_data()

# ========================
# LOGIC: HYBRID SEARCH
# ========================
def hybrid_search(query, df, limit=3):
    """
    Menggabungkan TF-IDF dan Fuzzy Matching untuk akurasi maksimal.
    """
    # 1. Fuzzy Matching pada 'uraian_ai' karena lebih kaya kata kunci 
    choices = df['uraian_ai'].tolist()
    fuzzy_results = process.extract(query, choices, scorer=fuzz.token_set_ratio, limit=10)
    
    candidates = []
    for uraian_ai_val, score, index in fuzzy_results:
        row = df.iloc[index]
        candidates.append({
            "kode": str(row['kode']),
            "uraian": row['uraian'],
            "score": score
        })
    
    # Sort berdasarkan skor tertinggi dan ambil 3
    return sorted(candidates, key=lambda x: x['score'], reverse=True)[:limit]

# ========================
# UI HEADER
# ========================
st.markdown("<h1 style='text-align: center;'>📁 AI Klasifikasi Arsip</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Sistem Otomasi Kearsipan Muna Barat</p>", unsafe_allow_html=True)

if not API_KEY:
    st.error("⚠️ API Key tidak ditemukan di Streamlit Secrets!")
else:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    uraian_user = st.text_input("🔍 Masukkan Perihal Surat / Uraian Arsip:", placeholder="Contoh: Daftar gaji pegawai bulan April")

    if uraian_user:
        # Ambil 3 kandidat terbaik secara lokal
        top_candidates = hybrid_search(uraian_user, data)
        
        with st.spinner("🤖 AI sedang memvalidasi akurasi..."):
            # Susun kandidat untuk Gemini
            context_list = "\n".join([f"{c['kode']} - {c['uraian']}" for c in top_candidates])
            
            # PROMPT PAKAR ARSIPARIS
            prompt = f"""
            Anda adalah pakar Arsiparis Pemerintah Indonesia.
            Tugas: Validasi dan urutkan 3 kode klasifikasi berikut berdasarkan tingkat relevansi yang paling presisi dengan uraian user.

            Uraian User: "{uraian_user}"

            Daftar Kandidat (Kode - Nama Kategori):
            {context_list}

            ATURAN:
            1. Tampilkan hanya 3 hasil terbaik.
            2. Sertakan persentase keyakinan Anda (0-100%).
            3. Gunakan format: [Kode] - [Uraian] - [Skor%]
            4. Berikan 1 kalimat alasan teknis kearsipan di bawah setiap hasil.
            """

            try:
                response = model.generate_content(prompt)
                
                st.success("🎯 Rekomendasi Klasifikasi Terbaik:")
                st.markdown(response.text)
                
                # Opsi tambahan: Tampilkan tabel perbandingan skor lokal jika perlu
                with st.expander("Lihat Skor Kecocokan Teknis (Lokal)"):
                    st.table(pd.DataFrame(top_candidates))

            except Exception as e:
                st.error(f"Gagal memproses AI: {e}")

# ========================
# FOOTER
# ========================
st.markdown("---")
st.caption("Proyek Aktualisasi Digitalisasi Arsip © 2026 | Kecerdasan Buatan Terintegrasi")
