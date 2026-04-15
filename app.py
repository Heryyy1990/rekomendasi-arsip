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
    .main-title { text-align: center; font-size: 30px; font-weight: bold; color: #1E3A8A; }
    .stButton>button { width: 100%; border-radius: 12px; height: 50px; background-color: #0F766E; color: white; font-weight: bold; border:none; }
    .local-result { padding: 15px; border-radius: 10px; background-color: #F0F9FF; border-left: 5px solid #0369A1; margin-bottom: 10px; }
    .ai-result { padding: 20px; border-radius: 10px; background-color: #F0FDF4; border-left: 5px solid #15803D; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# =================================================================
# 2. LOAD DATA
# =================================================================
@st.cache_data
def load_data():
    if not os.path.exists("klasifikasi_arsip.csv"):
        st.error("File 'klasifikasi_arsip.csv' tidak ditemukan!")
        st.stop()
    df = pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df['uraian_ai'] = df['uraian_ai'].fillna("")
    return df

df_data = load_data()

# =================================================================
# 3. MESIN PENCARI LOKAL (ANTI-GAGAL)
# =================================================================
def pencarian_lokal(query, df):
    query_clean = str(query).lower().strip()
    results = []
    
    for _, row in df.iterrows():
        uraian = str(row['uraian']).lower()
        konteks = str(row['uraian_ai']).lower()
        kode = str(row['kode']).strip()
        
        # Skor kemiripan teks
        score = fuzz.token_set_ratio(query_clean, f"{uraian} {konteks}")
        
        # LOGIKA JANGKAR: Jika cari "cuti", prioritaskan kode 800 (Kepegawaian)
        if "cuti" in query_clean and kode.startswith("8"):
            score += 50
            
        if score > 40:
            results.append({
                "kode": kode,
                "uraian": str(row['uraian']).strip(),
                "konteks": str(row['konteks'] if 'konteks' in row else row['uraian_ai']).strip(),
                "score": score
            })
    
    # Ambil 3 terbaik
    return sorted(results, key=lambda x: x['score'], reverse=True)[:3]

# =================================================================
# 4. ANTARMUKA UTAMA
# =================================================================
st.markdown('<div class="main-title">🏛️ AI Archivist Expert</div>', unsafe_allow_html=True)
st.write("---")

uraian_input = st.text_input("✍️ Masukkan Perihal/Deskripsi Arsip:", placeholder="Contoh: cuti")

if uraian_input:
    if st.button("🚀 ANALISIS KLASIFIKASI"):
        
        # TAHAP 1: SELALU TAMPILKAN HASIL LOKAL (AGAR TIDAK KOSONG)
        kandidat = pencarian_lokal(uraian_input, df_data)
        
        st.markdown("### 📋 3 Rekomendasi Kode (Hasil Database)")
        if not kandidat:
            st.warning("Tidak ditemukan hasil di database lokal.")
        else:
            for i, c in enumerate(kandidat):
                with st.container():
                    st.markdown(f"""
                    <div class="local-result">
                        <b>{i+1}. Kode: {c['kode']}</b><br>
                        Uraian: {c['uraian']}
                    </div>
                    """, unsafe_allow_html=True)

            # TAHAP 2: COBA ANALISIS AI (HANYA JIKA SERVER GEMINI AKTIF)
            st.markdown("---")
            if "GEMINI_API_KEY" in st.secrets:
                try:
                    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                    # Cari model yang tersedia secara otomatis
                    models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    target_model = next((m for m in models if "flash" in m), models[0])
                    model = genai.GenerativeModel(target_model)
                    
                    with st.spinner("🤖 AI sedang menyusun alasan pemilihan..."):
                        context_ai = "\n".join([f"- {c['kode']} | {c['uraian']}" for c in kandidat])
                        prompt = f"Analisis kenapa 3 kode ini cocok untuk perihal '{uraian_input}':\n{context_ai}\n\nBerikan penjelasan singkat untuk masing-masing."
                        
                        response = model.generate_content(prompt)
                        st.markdown('<div class="ai-result">', unsafe_allow_html=True)
                        st.markdown("### 🧠 Analisis Pakar AI")
                        st.write(response.text)
                        st.markdown('</div>', unsafe_allow_html=True)
                except:
                    st.info("ℹ️ Analisis tambahan AI tidak tersedia saat ini karena gangguan server Google, namun hasil database di atas tetap valid.")
            else:
                st.info("ℹ️ Masukkan API Key untuk mengaktifkan Analisis Pakar AI.")
