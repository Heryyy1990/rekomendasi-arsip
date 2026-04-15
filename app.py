import streamlit as st
import pandas as pd
import google.generativeai as genai
from thefuzz import fuzz
import os

# =================================================================
# 1. KONFIGURASI HALAMAN
# =================================================================
st.set_page_config(page_title="AI Archivist Expert", page_icon="🏛️", layout="centered")

st.markdown("""
<style>
    .main-title { text-align: center; font-size: 32px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { text-align: center; color: #64748B; margin-bottom: 30px; font-style: italic; }
    .stButton>button { width: 100%; border-radius: 12px; height: 55px; background-color: #0F766E; color: white; font-weight: bold; border:none; }
    .result-card { padding: 20px; border-radius: 15px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-top: 5px solid #0F766E; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

# =================================================================
# 2. SISTEM DATA & AUTO-MODEL AI
# =================================================================
@st.cache_data
def load_archive_data():
    if not os.path.exists("klasifikasi_arsip.csv"):
        st.error("❌ File 'klasifikasi_arsip.csv' tidak ditemukan!")
        st.stop()
    df = pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df['uraian_ai'] = df['uraian_ai'].fillna("")
    return df

def get_safe_ai_model():
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("❌ API Key tidak ditemukan di Streamlit Secrets!")
        st.stop()
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # SISTEM AUTO-FALLBACK: Mencari model apa pun yang aktif agar tidak eror 404
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Prioritas 1.5 Flash, jika gagal pakai Pro, jika gagal pakai yang tersedia
        target = next((m for m in available_models if "1.5-flash" in m), 
                 next((m for m in available_models if "gemini-pro" in m), 
                 available_models[0]))
        return genai.GenerativeModel(target)
    except Exception:
        # Pilihan terakhir jika gagal list model
        return genai.GenerativeModel("gemini-pro")

df_data = load_archive_data()
model = get_safe_ai_model()

# =================================================================
# 3. MESIN PENCARI KANDIDAT
# =================================================================
def collect_candidates(query, df, limit=20):
    query_clean = str(query).lower().strip()
    results = []
    for _, row in df.iterrows():
        text_to_check = f"{row['uraian']} {row['uraian_ai']}".lower()
        score = fuzz.token_set_ratio(query_clean, text_to_check)
        if score > 25: # Jaring lebih luas untuk AI
            results.append({
                "kode": str(row['kode']).strip(),
                "uraian": str(row['uraian']).strip(),
                "konteks": str(row['uraian_ai']).strip(),
                "score": score
            })
    return sorted(results, key=lambda x: x['score'], reverse=True)[:limit]

# =================================================================
# 4. ANTARMUKA PENGGUNA (UI)
# =================================================================
st.markdown('<div class="main-title">🏛️ AI Archivist Expert</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">"Analisis Kontekstual 3 Rekomendasi Terbaik"</div>', unsafe_allow_html=True)

uraian_input = st.text_area("✍️ Masukkan Perihal/Deskripsi Arsip:", placeholder="Contoh: cuti")

if uraian_input:
    if st.button("🚀 ANALISIS KLASIFIKASI"):
        
        kandidat_mentah = collect_candidates(uraian_input, df_data)
        
        if not kandidat_mentah:
            st.warning("⚠️ Tidak ditemukan referensi yang relevan.")
        else:
            with st.spinner("🧠 AI sedang menelaah 3 rekomendasi terbaik..."):
                context_for_ai = "\n".join([f"- Kode: {c['kode']} | Deskripsi: {c['uraian']} | Kelompok: {c['konteks']}" for i, c in enumerate(kandidat_mentah)])
                
                prompt = f"""
                Analisis perihal ini: "{uraian_input}"
                
                Gunakan daftar referensi ini:
                {context_for_ai}

                TUGAS:
                1. Pilih HANYA 3 KODE yang paling tepat secara substansi (bukan sekadar mirip kata).
                2. Untuk tiap pilihan, jelaskan alasan kenapa kode itu paling logis untuk arsip tersebut.
                3. Jika tentang "Cuti", prioritaskan kode di rumpun Kepegawaian (800).

                FORMAT OUTPUT:
                ---
                ### [NOMOR]. KODE: [KODE] - [URAIAN]
                **Alasan Pemilihan:** [Penjelasan logis]
                """
                
                try:
                    response = model.generate_content(prompt)
                    st.markdown("### 📋 Hasil Analisis Kontekstual")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"⚠️ Terjadi gangguan pada server AI Google. Silakan coba klik tombol lagi dalam beberapa saat.")

st.markdown("---")
st.caption("E-Archivist West Muna | Sistem Stabil & Akurat")
