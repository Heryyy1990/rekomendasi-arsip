import streamlit as st
import pandas as pd
import google.generativeai as genai
from thefuzz import process, fuzz

st.set_page_config(page_title="AI Archivist Pro - Muna Barat", layout="wide")

# --- AUTO-LOAD CONFIGURATION ---
# Ambil API Key dari Secrets Streamlit
API_KEY = st.secrets.get("GEMINI_API_KEY")

# Masukkan link RAW CSV Master Data Anda di sini
GITHUB_RAW_URL = "URL_MASTER_DATA_RAW_ANDA_DI_SINI"

st.title("📁 AI Klasifikasi Arsip Profesional")
st.caption("Sistem Rekomendasi 3 Kode Klasifikasi dengan Skor Kecocokan")

@st.cache_data
def load_master_from_github(url):
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Gagal memuat Master Data: {e}")
        return None

def get_top_3(target_text, master_df):
    text_to_search = str(target_text).lower().strip()
    # Menggunakan uraian_ai untuk akurasi kata kunci
    choices = master_df['uraian_ai'].tolist()
    
    # Ambil 3 terbaik secara lokal (Local Thinking)
    results = process.extract(text_to_search, choices, scorer=fuzz.token_set_ratio, limit=3)
    
    recs = []
    for uraian_val, score, index in results:
        row = master_df.iloc[index]
        # Format sesuai permintaan: Kode - Uraian (Score%)
        recs.append(f"{row['kode']} - {row['uraian']} ({score}%)")
    return recs

# --- ALUR UTAMA ---
if not API_KEY:
    st.error("⚠️ API Key tidak ditemukan! Harap masukkan GEMINI_API_KEY di menu Secrets Streamlit.")
else:
    # Memuat master data secara otomatis
    master_df = load_master_from_github(GITHUB_RAW_URL)
    
    if master_df is not None:
        st.success(f"✅ Sistem Aktif. {len(master_df)} Kode Klasifikasi telah sinkron dari GitHub.")
        
        # Area upload file yang ingin diklasifikasi
        input_file = st.file_uploader("Upload Data yang ingin diklasifikasi (CSV)", type=["csv"])
        
        if input_file:
            df_target = pd.read_csv(input_file)
            kolom_perihal = st.selectbox("Pilih kolom perihal/deskripsi arsip", df_target.columns)
            
            if st.button("🔍 Mulai Klasifikasi"):
                genai.configure(api_key=API_KEY)
                
                r1, r2, r3 = [], [], []
                progress = st.progress(0)
                
                for i, row in df_target.iterrows():
                    top_3 = get_top_3(row[kolom_perihal], master_df)
                    r1.append(top_3[0])
                    r2.append(top_3[1])
                    r3.append(top_3[2])
                    progress.progress((i + 1) / len(df_target))

                # Menyusun DataFrame hasil akhir
                df_target['Rekomendasi 1'] = r1
                df_target['Rekomendasi 2'] = r2
                df_target['Rekomendasi 3'] = r3
                
                # Filter hanya kolom yang dibutuhkan
                display_cols = [kolom_perihal, 'Rekomendasi 1', 'Rekomendasi 2', 'Rekomendasi 3']
                
                st.subheader("📊 Hasil Analisis Rekomendasi")
                st.dataframe(df_target[display_cols])
                
                csv = df_target[display_cols].to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Hasil", csv, "hasil_arsip.csv", "text/csv")
