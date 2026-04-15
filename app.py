import streamlit as st
import pandas as pd
import google.generativeai as genai
from thefuzz import process, fuzz
import time

# Konfigurasi Halaman
st.set_page_config(page_title="AI Arsiparis Pro", layout="wide")

st.title("📁 Sistem Klasifikasi Arsip Pintar")
st.caption("Pencarian 3 Rekomendasi Terbaik dengan Skor Kecocokan")

# --- SIDEBAR: Pengaturan ---
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    api_key = st.text_input("Masukkan Gemini API Key", type="password")
    
    st.divider()
    st.subheader("📊 Master Data")
    # Mengacu pada file klasifikasi_arsip.csv yang Anda miliki 
    master_file = st.file_uploader("Upload Master Data Klasifikasi", type=["csv"])
    
    st.divider()
    st.info("Sesuai permintaan: Sistem akan menampilkan 3 rekomendasi dengan persentase kecocokan.")

# --- FUNGSI LOCAL THINKING (3 REKOMENDASI) ---
def get_top_3_recommendations(target_text, master_df):
    """
    Mencari 3 kecocokan terbaik secara lokal menggunakan Fuzzy Logic 
    pada kolom 'uraian_ai' untuk akurasi kata kunci.
    """
    # Membersihkan teks input
    text_to_search = str(target_text).lower().strip()
    
    # Menggunakan uraian_ai sebagai basis pencarian karena lebih lengkap 
    choices = master_df['uraian_ai'].tolist()
    
    # Mencari 3 besar (limit=3)
    results = process.extract(text_to_search, choices, scorer=fuzz.token_set_ratio, limit=3)
    
    recommendations = []
    for uraian_val, score, index in results:
        row = master_df.iloc[index]
        recommendations.append({
            "kode": str(row['kode']).strip(),
            "uraian": row['uraian'],
            "persentase": f"{score}%"
        })
    return recommendations

# --- LOGIK PROSES ---
if master_file and api_key:
    master_df = pd.read_csv(master_file)
    master_df.columns = master_df.columns.str.strip() # Bersihkan nama kolom
    
    st.success(f"✅ Master data siap: {len(master_df)} baris kode termuat.")
    
    st.divider()
    input_file = st.file_uploader("Upload CSV Data yang ingin diklasifikasikan", type=["csv"])
    
    if input_file:
        df_target = pd.read_csv(input_file)
        kolom_perihal = st.selectbox("Pilih kolom Perihal/Uraian", df_target.columns)
        
        if st.button("🔍 Mulai Klasifikasi"):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Wadah hasil
            rec1_list, rec2_list, rec3_list = [], [], []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, row in df_target.iterrows():
                teks_surat = str(row[kolom_perihal])
                
                # 1. LOCAL THINKING (Ambil 3 Rekomendasi)
                top_3 = get_top_3_recommendations(teks_surat, master_df)
                
                # 2. AI VALIDATION (Opsional untuk memastikan urutan paling logis)
                # AI hanya diminta memvalidasi urutan dari 3 kandidat tersebut
                context_candidates = "\n".join([f"{c['kode']} - {c['uraian']}" for c in top_3])
                
                prompt = f"""
                Teks: "{teks_surat}"
                Kandidat:
                {context_candidates}
                
                Tugas: Pastikan Kode dan Uraian di atas sudah yang paling relevan. 
                Jawab dengan format yang sama tanpa penjelasan tambahan.
                """
                
                # Menambahkan hasil ke list (Kode - Uraian - Persentase)
                # Format: "Kode - Uraian (Score%)"
                rec1_list.append(f"{top_3[0]['kode']} - {top_3[0]['uraian']} ({top_3[0]['persentase']})")
                rec2_list.append(f"{top_3[1]['kode']} - {top_3[1]['uraian']} ({top_3[1]['persentase']})")
                rec3_list.append(f"{top_3[2]['kode']} - {top_3[2]['uraian']} ({top_3[2]['persentase']})")
                
                # Update UI
                progress = (i + 1) / len(df_target)
                progress_bar.progress(progress)
                status_text.text(f"Memproses {i+1} dari {len(df_target)} data...")

            # Menambahkan ke DataFrame hasil akhir
            df_target['Rekomendasi 1'] = rec1_list
            df_target['Rekomendasi 2'] = rec2_list
            df_target['Rekomendasi 3'] = rec3_list
            
            status_text.success("✅ Selesai! Menampilkan hasil akhir.")
            
            # Hanya menampilkan kolom input asli dan 3 rekomendasi hasil akhirnya
            display_cols = [kolom_perihal, 'Rekomendasi 1', 'Rekomendasi 2', 'Rekomendasi 3']
            st.dataframe(df_target[display_cols])
            
            # Tombol Download
            csv = df_target[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Hasil CSV", csv, "hasil_klasifikasi_3_opsi.csv", "text/csv")

else:
    st.info("Silakan lengkapi API Key dan Master Data di sidebar untuk mulai bekerja sebagai Arsiparis.")
