import streamlit as st
import pandas as pd
import re
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from thefuzz import process, fuzz
from groq import Groq


# --- INISIALISASI SESSION STATE LOGIN & HISTORY ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'role' not in st.session_state:
    st.session_state['role'] = None
if 'nama' not in st.session_state:
    st.session_state['nama'] = ""
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# --- FUNGSI VALIDASI LOGIN (BACA DARI PENGGUNA.CSV) ---
def validasi_login(user, pwd):
    try:
        df_user = pd.read_csv('pengguna.csv', sep=',') # Pastikan Anda sudah membuat file pengguna.csv
        user_data = df_user[(df_user['username'] == user) & (df_user['password'] == pwd)]
        if not user_data.empty:
            return True, user_data.iloc[0]['role'], user_data.iloc[0]['nama_lengkap']
    except Exception as e:
        st.error(f"File pengguna.csv tidak ditemukan atau format salah: {e}")
    return False, None, None

# --- FUNGSI RIWAYAT PERMANEN (CSV) ---
def simpan_riwayat_csv(nama_user, pencarian):
    file_riwayat = 'riwayat_pencarian.csv'
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_baru = pd.DataFrame({'waktu': [waktu], 'nama': [nama_user], 'pencarian': [pencarian]})
    
    if not os.path.isfile(file_riwayat):
        df_baru.to_csv(file_riwayat, index=False)
    else:
        df_baru.to_csv(file_riwayat, mode='a', header=False, index=False)

def baca_riwayat_csv(nama_user):
    file_riwayat = 'riwayat_pencarian.csv'
    if os.path.isfile(file_riwayat):
        try:
            df_riwayat = pd.read_csv(file_riwayat)
            riwayat_user = df_riwayat[df_riwayat['nama'] == nama_user]['pencarian'].tolist()
            return list(dict.fromkeys(riwayat_user)) # Hapus duplikat
        except:
            return []
    return []

# --- HALAMAN LOGIN ---
def halaman_login():
    st.markdown("<div class='sikap-title'>SIKAP</div>", unsafe_allow_html=True)
    st.markdown("<div class='sikap-subtitle'>Silakan Masuk untuk Melanjutkan</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("form_login"):
            user_input = st.text_input("Username")
            pwd_input = st.text_input("Password", type="password")
            submit = st.form_submit_button("Masuk", use_container_width=True)
            
            if submit:
                is_valid, role, nama = validasi_login(user_input, pwd_input)
                if is_valid:
                    st.session_state['logged_in'] = True
                    st.session_state['role'] = role
                    st.session_state['nama'] = nama
                    st.rerun()
                else:
                    st.error("Username atau Password salah!")

# 1. Menarik API Key dengan aman (Bisa jalan di lokal maupun di Streamlit Cloud)
try:
    # Membaca dari Streamlit Secrets jika di Cloud
    api_key = st.secrets["GROQ_API_KEY"]
except:
    # Masukkan API Key manual HANYA untuk tes di laptop lokal (Hapus sebelum di-push ke GitHub!)
    api_key = "MASUKKAN_API_KEY_GROQ_DI_SINI_UNTUK_TES_LOKAL" 

client = Groq(api_key=api_key)


# --- 1. MEMUAT DATABASE (VERSI BERSIH UNTUK FULL AI) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('klasifikasi_arsip_emas.csv', sep=',', on_bad_lines='skip', dtype=str)
    except:
        df = pd.read_csv('klasifikasi_arsip_emas.csv', sep=';', on_bad_lines='skip', dtype=str)
    
    if len(df.columns) == 1:
        col_name = df.columns[0]
        df[['kode', 'uraian']] = df[col_name].str.split(r'[,;]', n=1, expand=True)
        df = df.drop(columns=[col_name])
        
    df['uraian'] = df['uraian'].astype(str).str.replace(r';$', '', regex=True).str.strip().fillna("")
    df['kode'] = df['kode'].astype(str).str.strip().fillna("000")

    # Membangun Jalur Hierarki untuk dibaca Groq
    kode_dict = dict(zip(df['kode'], df['uraian']))
    def bangun_hierarki(kode):
        jalur = []
        curr = str(kode).strip()
        while curr:
            if curr in kode_dict:
                jalur.insert(0, kode_dict[curr]) 
            if '.' in curr:
                curr = curr.rsplit('.', 1)[0]
            else:
                if len(curr) == 3 and curr.endswith('00'): break 
                elif len(curr) > 3: curr = curr[:-1]
                elif len(curr) == 3:
                    if curr.endswith('0'): curr = curr[0] + '00'
                    else: curr = curr[0:2] + '0'
                else: break
        return " > ".join(jalur)
    
    df['uraian_lengkap'] = df['kode'].apply(bangun_hierarki)
    return df

# =====================================================================
# MESIN UTAMA SIKAP (FULL-AI DUAL AGENT ARCHITECTURE)
# Bebas Sastrawi, Bebas TF-IDF. 100% Nalar Murni Groq.
# =====================================================================
def smart_classify(user_input, df, top_n=3):
    import streamlit as st
    import re
    
    st.info("🧠 SIKAP sedang menyaring database agar muat dalam memori AI...")
    
    # ==========================================
    # TAHAP 1: RESEPSIONIS (Mencari Sub-Rumpun Sangat Spesifik)
    # ==========================================
    prompt_rumpun = f"""
    Tugas: Tentukan 3 KODE SUB-RUMPUN (2 digit awal) yang paling relevan untuk surat ini.
    Perihal Surat: "{user_input}"
    
    Contoh Sub-Rumpun:
    00.1 (Organisasi), 00.2 (Perlengkapan/Aset), 50.1 (Pertanian), 50.17 (Pertanahan), 80.1 (Pengadaan Pegawai), 90.1 (Anggaran).
    
    ATURAN: Balas HANYA dengan 3 kode sub-rumpun (2-3 digit depan), pisahkan dengan koma.
    Contoh balasan: 000.2, 500.1, 800.1
    """
    
    try:
        # Gunakan model 8B yang cepat dan murah untuk screening awal
        chat_1 = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_rumpun}],
            model="llama-3.1-8b-instant", 
            temperature=0.0,
        )
        balasan_rumpun = chat_1.choices[0].message.content.strip()
        
        # Ambil kode-kode angka yang disebutkan AI
        sub_kandidat = re.findall(r'\d{3}(?:\.\d+)?', balasan_rumpun)
        
        if not sub_kandidat:
            # Fallback jika AI bingung
            df_filter = df.head(50)
        else:
            # Filter database berdasarkan kode yang disebutkan (hanya ambil 50 baris pertama agar tidak kena limit)
            df_filter = df[df['kode'].str.startswith(tuple(sub_kandidat))].head(50)
            
            # Jika hasil filter terlalu sedikit, tambahkan rumpun besarnya
            if len(df_filter) < 10:
                rumpun_besar = [c[0] for c in sub_kandidat]
                df_filter = df[df['kode'].str.startswith(tuple(rumpun_besar))].head(50)

        # Ubah dataframe yang sudah disaring menjadi teks untuk Agent 2
        katalog_teks = ""
        for _, row in df_filter.iterrows():
            katalog_teks += f"[{row['kode']}] {row['uraian_lengkap']}\n"
            
        # ==========================================
        # TAHAP 2: DETEKTIF PROFESOR (70B) - Kapasitas Terbatas
        # ==========================================
        prompt_detektif = f"""
        Anda adalah Arsiparis Ahli. Cocokkan surat dengan KATALOG di bawah.
        
        SURAT: "{user_input}"
        
        ATURAN: Balas HANYA dengan format: 1|KODE_1, 2|KODE_2, 3|KODE_3.
        
        KATALOG:
        {katalog_teks}
        """
        
        # Sekarang model 70B hanya menerima 50 baris, pasti muat di limit 12.000 token!
        chat_2 = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_detektif}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
        )
        
        balasan_akhir = chat_2.choices[0].message.content.strip()
        
        hasil_akhir = []
        for baris in balasan_akhir.split('\n'):
            parts = baris.split('|')
            if len(parts) >= 2:
                kode_ai = re.sub(r'[^\d\.]', '', parts[1].strip())
                idx_match = df.index[df['kode'] == kode_ai].tolist()
                if idx_match:
                    skor = 0.99 - (len(hasil_akhir) * 0.14) 
                    hasil_akhir.append((idx_match[0], skor))
            if len(hasil_akhir) == top_n:
                break
                
        return hasil_akhir
        
    except Exception as e:
        # Jika masih error limit, kita gunakan pencarian teks sederhana sebagai cadangan terakhir
        st.warning("⚠️ Server Groq sangat sibuk. Menggunakan pencarian teks cadangan...")
        # Simple string matching sebagai cadangan darurat
        df['score'] = df['uraian_lengkap'].apply(lambda x: 1.0 if user_input.lower() in x.lower() else 0.0)
        top_fallback = df.sort_values('score', ascending=False).head(top_n)
        return [(idx, 0.5) for idx in top_fallback.index]
    
# --- 4. ANTARMUKA UTAMA ---
def halaman_utama():
    st.markdown("<div class='sikap-title'>SIKAP</div>", unsafe_allow_html=True)
    st.markdown("<div class='sikap-subtitle'>Sistem Informasi Klasifikasi Arsip Pintar</div>", unsafe_allow_html=True)

    try:
        df = load_data()
        
        with st.sidebar:
            # --- TAMBAHAN: INFO USER & TOMBOL LOGOUT ---
            st.success(f"Masuk sebagai:\n**{st.session_state['nama']}**")
            if st.button("Keluar", use_container_width=True):
                st.session_state['logged_in'] = False
                st.session_state['role'] = None
                st.session_state['nama'] = ""
                st.rerun()
            st.divider()
            # -------------------------------------------

            st.header("🕒 Riwayat Pencarian")
            
            # Tarik data dari CSV saat pertama kali masuk
            if not st.session_state.search_history:
                st.session_state.search_history = baca_riwayat_csv(st.session_state['nama'])

            if st.session_state.search_history:
                for riwayat in reversed(st.session_state.search_history[-10:]):
                    st.caption(f"• {riwayat}")
                if st.button("Sembunyikan Riwayat", use_container_width=True):
                    st.session_state.search_history = []
                    st.rerun()
            else:
                st.info("Belum ada pencarian.")

        # --- TAMBAHAN: LOGIKA TAB ADMIN ---
        if st.session_state.get('role') == 'admin':
            tab_ai, tab_katalog, tab_admin = st.tabs(["🤖 Pencarian AI (Cerdas)", "📂 Jelajah Kode Klasifikasi (Manual)", "⚙️ Panel Admin"])
        else:
            tab_ai, tab_katalog = st.tabs(["🤖 Pencarian AI (Cerdas)", "📂 Jelajah Kode Klasifikasi (Manual)"])

        # ================= TAB 1: PENCARIAN AI =================
        with tab_ai:
            st.write("Masukkan perihal atau deskripsi surat, biarkan kecerdasan buatan mencari kode klasifikasi yang paling tepat untuk Anda.")
            user_input = st.text_input("📝 Perihal Surat / Dokumen:", placeholder="Contoh: permohonan cuti tahunan pegawai atau undangan rapat tapd...", key="input_ai")

            if user_input:
                if user_input not in st.session_state.search_history:
                    st.session_state.search_history.append(user_input)
                    # Simpan ke CSV secara permanen
                    simpan_riwayat_csv(st.session_state['nama'], user_input)

                with st.spinner('Menganalisis bahasa dan mencari kecocokan...'):
                    results = smart_classify(user_input, df)
                    
                    if results:
                        st.success("Analisis selesai! Berikut adalah rekomendasi kode untuk dokumen Anda:")
                        pilihan_feedback = [] 
                        
                        for i, (idx, score) in enumerate(results):
                            res = df.iloc[idx]
                            pilihan_feedback.append(f"{res['kode']} - {res['uraian'].title()}")
                            
                            with st.expander(f"🏅 Rekomendasi #{i+1}: Kode {res['kode']} (Keyakinan: {score:.1%})", expanded=(i==0)):
                                st.markdown(f"**Uraian Akhir:** {res['uraian'].title()}")
                                st.markdown("**Struktur Hierarki:**")
                                hierarki = get_hierarchy(res['kode'], df)
                                for h in hierarki:
                                    st.markdown(h, unsafe_allow_html=True)
                        
                        st.divider()
                        
                        st.markdown("#### 💡 Bantu SIKAP Menjadi Lebih Pintar")
                        st.write("Dari rekomendasi di atas, mana kode yang paling tepat menurut Anda?")
                        
                        with st.form("feedback_form"):
                            jawaban_benar = st.radio("Pilih kode yang benar:", pilihan_feedback)
                            submit_feedback = st.form_submit_button("Kirim Masukan")
                            
                            if submit_feedback:
                                waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                data_feedback = f"{waktu_sekarang} | Input: {user_input} | Terpilih: {jawaban_benar}\n"
                                with open("feedback_ai_log.txt", "a", encoding="utf-8") as f:
                                    f.write(data_feedback)
                                st.success(f"Terima kasih! Pilihan Anda telah disimpan untuk evaluasi AI ke depannya.")
                    else:
                        st.warning("Tidak ditemukan klasifikasi yang cocok. Coba gunakan kata kunci lain.")

        # ================= TAB 2: JELAJAH KODE =================
        with tab_katalog:
            st.write("Jelajahi Pohon Hierarki Klasifikasi Arsip (Klik pada Folder Utama untuk membuka isinya):")
            
            daftar_primer = [f"{i}00" for i in range(10)]
            
            for p in daftar_primer:
                cek_df = df[df['kode'] == p]
                uraian_primer = cek_df.iloc[0]['uraian'].title() if not cek_df.empty else "Detail Klasifikasi"
                
                with st.expander(f"📁 RUMPUN {p} - {uraian_primer}"):
                    
                    hasil_filter = df[df['kode'].str.startswith(p)]
                    
                    if not hasil_filter.empty:
                        hasil_filter = hasil_filter[hasil_filter['kode'].str.match(r'^\d')]
                        
                        nodes = {}
                        for _, row in hasil_filter.iterrows():
                            k = str(row['kode']).strip()
                            u = str(row['uraian']).title()
                            nodes[k] = {'uraian': u, 'children': []}
                            
                        for k in nodes:
                            if k == p: continue 
                            curr = k
                            parent = None
                            while True:
                                if '.' in curr:
                                    curr = curr.rsplit('.', 1)[0]
                                else:
                                    if len(curr) == 3 and curr.endswith('00'):
                                        parent = curr
                                        break
                                    elif len(curr) > 3:
                                        curr = curr[:-1]
                                    elif len(curr) == 3:
                                        if curr.endswith('0'): curr = curr[0] + '00'
                                        else: curr = curr[0:2] + '0'
                                    else:
                                        break
                                if curr in nodes:
                                    parent = curr
                                    break
                            
                            if not parent or parent not in nodes:
                                parent = p
                            
                            if parent in nodes and parent != k:
                                nodes[parent]['children'].append(k)
                                
                        def render_tree(k):
                            node = nodes[k]
                            u = node['uraian']
                            children = node['children']
                            
                            children.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
                            
                            titik_count = str(k).count('.')
                            actual_level = titik_count if titik_count < 4 else 3
                            
                            warna_level = ["#B71C1C", "#1565C0", "#2E7D32", "#E65100"]
                            warna_bg = warna_level[actual_level]
                            
                            levels_name = ["Primer", "Sekunder", "Tersier", "Kuartier"]
                            label = levels_name[actual_level]
                            
                            html = ""
                            if children:
                                html += f'<details style="margin-bottom: 8px;"><summary style="cursor: pointer; list-style: none; outline: none;"><span style="background-color: {warna_bg}; color: #ffffff; padding: 6px 12px; border-radius: 6px; font-weight: normal; font-size: 0.95em; display: inline-block; box-shadow: 0px 2px 4px rgba(0,0,0,0.2);"><strong>📁 {k}</strong> &nbsp;|&nbsp; {u} <i style="opacity: 0.8;">({label})</i></span></summary><div style="margin-left: 20px; padding-left: 10px; border-left: 2px dashed #ccc; padding-top: 8px;">'
                                for c in children:
                                    html += render_tree(c)
                                html += '</div></details>'
                            else:
                                html += f'<div style="margin-bottom: 8px;"><span style="background-color: {warna_bg}; color: #ffffff; padding: 6px 12px; border-radius: 6px; font-weight: normal; font-size: 0.95em; display: inline-block; box-shadow: 0px 2px 4px rgba(0,0,0,0.2);"><strong>📁 {k}</strong> &nbsp;|&nbsp; {u} <i style="opacity: 0.8;">({label})</i></span></div>'
                            return html
                            
                        if p in nodes:
                            full_html = ""
                            sorted_children = sorted(nodes[p]['children'], key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
                            for child_kode in sorted_children:
                                full_html += render_tree(child_kode)
                            st.markdown(full_html, unsafe_allow_html=True)
                        
                    else:
                        st.caption("Tidak ada data klasifikasi di dalam rumpun ini.")

        # ================= TAB 3: PANEL ADMIN (BARU) =================
        if st.session_state.get('role') == 'admin':
            with tab_admin:
                st.header("⚙️ Panel Kontrol Administrator")
                st.write("Selamat datang di halaman khusus Admin.")
                
                st.subheader("Daftar Pengguna")
                try:
                    df_user = pd.read_csv('pengguna.csv')
                    st.dataframe(df_user, use_container_width=True)
                except:
                    st.warning("File pengguna.csv belum dibuat atau tidak ditemukan.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")

# --- 5. PENGATUR HALAMAN (ROUTER) ---
if not st.session_state.get('logged_in', False):
    # Memanggil fungsi halaman_login() yang harusnya sudah Anda letakkan di bagian atas file
    halaman_login()
else:
    halaman_utama()
