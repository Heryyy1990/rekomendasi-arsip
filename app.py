import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="SIKAP - Klasifikasi Arsip Pintar", page_icon="🗂️", layout="wide")

# --- STYLE CSS ---
st.markdown("""
    <style>
    .sikap-title { font-size: 3.5rem; font-weight: 900; text-align: center; margin-bottom: -10px; background: linear-gradient(45deg, #00BFA5, #0288D1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .sikap-subtitle { font-size: 1.2rem; text-align: center; margin-bottom: 30px; color: #555; }
    </style>
""", unsafe_allow_html=True)

# --- SISTEM PENYIMPANAN SEMENTARA (SESSION STATE) ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'riwayat' not in st.session_state:
    st.session_state['riwayat'] = []
if 'feedback_data' not in st.session_state:
    st.session_state['feedback_data'] = []

def halaman_login():
    st.markdown("<div class='sikap-title'>SIKAP</div>", unsafe_allow_html=True)
    st.markdown("<div class='sikap-subtitle'>Sistem Informasi Klasifikasi Arsip Pintar</div>", unsafe_allow_html=True)
    with st.container():
        left, mid, right = st.columns([1, 2, 1])
        with mid:
            with st.form("login"):
                user = st.text_input("Username")
                pwd = st.text_input("Password", type="password")
                if st.form_submit_button("Masuk"):
                    if user == "admin" and pwd == "mubarkaya": 
                        st.session_state['logged_in'] = True
                        st.rerun()
                    else:
                        st.error("Username atau Password salah")

# --- INISIALISASI AI (SECRETS) ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    st.error("🚨 API Key tidak ditemukan di Streamlit Secrets! Silakan atur di pengaturan aplikasi.")
    st.stop()

# --- LOAD DATA & EMBEDDINGS (SUPER KILAT) ---
@st.cache_resource
def muat_sistem():
    df = pd.read_csv('klasifikasi_enriched.csv', sep=None, engine='python', dtype=str)
    if len(df.columns) >= 2:
        df.columns = ['kode', 'uraian'] + list(df.columns[2:])
    
    if os.path.exists('embeddings.npy'):
        embeddings_matrix = np.load('embeddings.npy')
        if len(embeddings_matrix) == len(df):
            df['embedding'] = list(embeddings_matrix)
        else:
            st.error(f"🚨 Ketidakcocokan data! CSV: {len(df)}, NPY: {len(embeddings_matrix)}")
            st.stop()
    else:
        st.error("🚨 File 'embeddings.npy' tidak ditemukan di GitHub!")
        st.stop()
        
    return df

def cari_arsip(query, df):
    response = client.models.embed_content(model="gemini-embedding-001", contents=query)
    vektor_user = response.embeddings[0].values
    vektor_db = np.array(df['embedding'].tolist())
    skor = cosine_similarity([vektor_user], vektor_db)[0]
    df['skor'] = skor
    return df.sort_values(by='skor', ascending=False).head(3)

# --- HALAMAN UTAMA (DENGAN TABS) ---
def halaman_utama():
    df = muat_sistem()
    
    st.markdown("<div class='sikap-title'>SIKAP</div>", unsafe_allow_html=True)
    st.markdown("<div class='sikap-subtitle'>Dinas Perpustakaan dan Kearsipan Muna Barat</div>", unsafe_allow_html=True)

    # MEMBUAT 4 TABS
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 Pencarian AI", "📖 Klasifikasi Manual", "🕒 Riwayat", "💬 Feedback"])

    # TAB 1: PENCARIAN PINTAR (AI)
    with tab1:
        st.write("Cari kode arsip dengan mengetikkan perihal atau isi ringkas surat.")
        query = st.text_input("📝 Masukkan Perihal Surat:", placeholder="Contoh: Undangan rapat evaluasi anggaran...")

        if st.button("Cari dengan AI", type="primary"):
            if query:
                with st.spinner("SIKAP sedang menganalisis..."):
                    hasil = cari_arsip(query, df)
                    
                    # Simpan ke riwayat
                    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    kode_teratas = hasil.iloc[0]['kode']
                    st.session_state['riwayat'].insert(0, {"waktu": waktu, "pencarian": query, "hasil_teratas": kode_teratas})
                    
                    st.success("Berikut adalah rekomendasi kode klasifikasi:")
                    for i, row in hasil.iterrows():
                        with st.expander(f"🏅 Rekomendasi {i+1}: {row['kode']} (Skor Kemiripan: {row['skor']:.2%})", expanded=(i==hasil.index[0])):
                            st.write(f"**Uraian:** {row['uraian']}")
                            if 'penjelasan' in df.columns:
                                st.info(f"**Penjelasan:** {row.get('penjelasan', '-')}")
                            if 'contoh_surat' in df.columns:
                                st.warning(f"**Contoh Surat:** {row.get('contoh_surat', '-')}")
            else:
                st.warning("Silakan masukkan perihal surat terlebih dahulu.")

    # TAB 2: KLASIFIKASI MANUAL (TABEL DATA)
    with tab2:
        st.write("Jelajahi seluruh daftar kode klasifikasi arsip secara manual.")
        cari_manual = st.text_input("🔍 Filter tabel (ketik kata kunci atau kode):")
        
        # Tampilkan data tanpa kolom embedding agar tabel tidak berat/error
        df_tampil = df.drop(columns=['embedding', 'skor'], errors='ignore')
        
        if cari_manual:
            # Filter berdasarkan kode atau uraian
            mask = df_tampil.astype(str).apply(lambda x: x.str.contains(cari_manual, case=False)).any(axis=1)
            st.dataframe(df_tampil[mask], use_container_width=True, hide_index=True)
        else:
            st.dataframe(df_tampil, use_container_width=True, hide_index=True)

    # TAB 3: RIWAYAT PENCARIAN
    with tab3:
        st.write("Riwayat pencarian kode arsip pada sesi ini:")
        if not st.session_state['riwayat']:
            st.info("Belum ada riwayat pencarian.")
        else:
            for item in st.session_state['riwayat']:
                st.markdown(f"**[{item['waktu']}]** Dicari: *\"{item['pencarian']}\"* ➡️ Rekomendasi: **{item['hasil_teratas']}**")
            
            if st.button("Hapus Riwayat"):
                st.session_state['riwayat'] = []
                st.rerun()

    # TAB 4: FEEDBACK USER
    with tab4:
        st.write("Bantu kami meningkatkan aplikasi SIKAP dengan memberikan masukan Anda.")
        with st.form("form_feedback"):
            nama_user = st.text_input("Nama (Opsional):")
            rating = st.slider("Seberapa terbantu Anda dengan SIKAP?", 1, 5, 5)
            pesan_feedback = st.text_area("Masukan / Saran perbaikan:")
            submit_feedback = st.form_submit_button("Kirim Masukan")
            
            if submit_feedback:
                if pesan_feedback:
                    waktu_fb = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state['feedback_data'].append({
                        "waktu": waktu_fb,
                        "nama": nama_user if nama_user else "Anonim",
                        "rating": rating,
                        "pesan": pesan_feedback
                    })
                    st.success("Terima kasih atas masukan Anda! Feedback berhasil disimpan.")
                else:
                    st.warning("Mohon isi kolom masukan terlebih dahulu.")
        
        # Tampilkan feedback yang sudah masuk (bisa disembunyikan jika hanya untuk admin)
        with st.expander("Lihat Feedback Masuk"):
            if st.session_state['feedback_data']:
                for fb in st.session_state['feedback_data']:
                    st.markdown(f"⭐ **{fb['rating']}/5** | 👤 {fb['nama']} | 🕒 {fb['waktu']}")
                    st.write(f"💬 *\"{fb['pesan']}\"*")
                    st.divider()
            else:
                st.write("Belum ada masukan.")

# --- EKSEKUSI ---
if not st.session_state['logged_in']:
    halaman_login()
else:
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🚪 Keluar"):
            st.session_state['logged_in'] = False
            st.rerun()
    halaman_utama()
