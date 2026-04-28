import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import json
import time
from datetime import datetime
from thefuzz import process, fuzz
from groq import Groq
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

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
        df_user = pd.read_csv('pengguna.csv', sep=',') 
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

# --- 1. KONFIGURASI API (GROQ & GEMINI) ---
try:
    # Membaca dari Streamlit Secrets jika di Cloud
    groq_key = st.secrets["GROQ_API_KEY"]
    gemini_key = st.secrets["GEMINI_API_KEY"]
except:
    # Masukkan manual HANYA untuk lokal (Hapus sebelum di-push ke GitHub!)
    groq_key = "MASUKKAN_API_KEY_GROQ_DI_SINI" 
    gemini_key = "MASUKKAN_API_KEY_GEMINI_DI_SINI"

client_groq = Groq(api_key=groq_key)
genai.configure(api_key=gemini_key)

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="SIKAP - Klasifikasi Arsip Pintar", page_icon="🗂️", layout="wide")

# --- UI & CSS CUSTOM ---
st.markdown("""
    <style>
    .sikap-title {
        font-size: 4.5rem; 
        font-weight: 900; 
        text-align: center;
        margin-bottom: -15px; 
        letter-spacing: 3px;
        background: linear-gradient(45deg, #00BFA5, #0288D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sikap-subtitle {
        font-size: 1.3rem; 
        text-align: center; 
        font-weight: 600;
        margin-bottom: 40px;
        letter-spacing: 1px;
        background: linear-gradient(45deg, #FF512F, #DD2476);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .streamlit-expanderHeader {
        font-weight: bold;
        color: #0288D1;
    }
    details > summary {
        list-style: none !important;
        outline: none !important;
    }
    details > summary::-webkit-details-marker {
        display: none !important; 
    }
    </style>
""", unsafe_allow_html=True)


# --- 2. MEMUAT DATABASE DAN EMBEDDING PINTAR ---
@st.cache_data
def load_data():
    try:
        # Coba baca dengan koma
        df = pd.read_csv('klasifikasi_enriched.csv', sep=',', on_bad_lines='skip', dtype=str)
        # Jika semua data menumpuk di 1 kolom, berarti Excel menggunakan titik-koma (;)
        if len(df.columns) == 1:
            df = pd.read_csv('klasifikasi_enriched.csv', sep=';', on_bad_lines='skip', dtype=str)
    except Exception as e:
        st.error(f"Gagal memuat CSV. Error: {e}")
        return pd.DataFrame()
    
    # --- MENGEMBALIKAN FITUR "SABUK PENGAMAN" DARI KODE LAMA ANDA ---
    # Jika ada masalah pada judul kolom Excel, kita paksa kolom 1 jadi 'kode' dan kolom 2 jadi 'uraian'
    if len(df.columns) >= 2:
        kolom_baru = list(df.columns)
        kolom_baru[0] = 'kode'
        kolom_baru[1] = 'uraian'
        df.columns = kolom_baru
        
    df['uraian'] = df['uraian'].astype(str).str.replace(r';$', '', regex=True).str.strip().fillna("")
    df['kode'] = df['kode'].astype(str).str.strip().fillna("000")
    df = df.fillna("")
    
    # --- LOGIKA BARU UNTUK AI GEMINI ---
    def gabung_teks(row):
        teks = f"Uraian: {row.get('uraian', '')}. "
        # Gunakan get() yang lebih aman jika kolom penjelas tidak sengaja terhapus di Excel
        if 'penjelasan' in df.columns and row.get('penjelasan'): teks += f"Penjelasan: {row['penjelasan']}. "
        if 'konteks' in df.columns and row.get('konteks'): teks += f"Konteks: {row['konteks']}. "
        if 'sinonim' in df.columns and row.get('sinonim'): teks += f"Sinonim: {row['sinonim']}. "
        if 'contoh_surat' in df.columns and row.get('contoh_surat'): teks += f"Contoh Surat: {row['contoh_surat']}."
        return teks
    
    df['teks_embedding'] = df.apply(gabung_teks, axis=1)

    # Cek apakah memori AI sudah ada
    if os.path.exists('embeddings.npy'):
        embeddings_matrix = np.load('embeddings.npy')
        # Sabuk pengaman ke-2: Jika Anda menambah baris di Excel, AI otomatis sadar dan reset ingatan
        if len(embeddings_matrix) != len(df):
            os.remove('embeddings.npy')
            st.warning("Perubahan jumlah baris CSV terdeteksi! Menyesuaikan ulang memori... Silakan Refresh (F5).")
            st.stop()
        df['embedding'] = list(embeddings_matrix)
    else:
        st.warning("⚠️ Membangun database kecerdasan untuk pertama kali. Mohon tunggu beberapa detik...")
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(df), batch_size):
            batch_texts = df['teks_embedding'].iloc[i:i+batch_size].tolist()
            try:
                response = genai.embed_content(
                    model="models/gemini-embedding-2",
                    content=batch_texts
                )
                all_embeddings.extend(response['embedding'])
                time.sleep(1) # Jeda aman anti-blokir
            except Exception as e:
                st.error(f"Gagal melakukan embedding: {e}")
                
        if all_embeddings:
            embeddings_matrix = np.array(all_embeddings)
            np.save('embeddings.npy', embeddings_matrix) 
            df['embedding'] = list(embeddings_matrix)
            st.success("Database berhasil diperkaya! Silakan mulai pencarian surat.")
            
    return df

# --- FUNGSI PEMBUAT BADGE UNTUK TAB 1 ---
def get_badge_html(kode, uraian, level):
    levels_name = ["Primer", "Sekunder", "Tersier", "Kuartier", "Kuintier"]
    label = levels_name[level] if level < len(levels_name) else f"Level {level+1}"
    warna_level = ["#B71C1C", "#1565C0", "#2E7D32", "#E65100", "#4A148C"]
    warna_bg = warna_level[level] if level < len(warna_level) else "#424242"
    indent = level * 30 
    return f"<div style='margin-left: {indent}px; margin-bottom: 8px;'><span style='background-color: {warna_bg}; color: #ffffff; padding: 6px 12px; border-radius: 6px; font-weight: normal; font-size: 0.95em; display: inline-block; box-shadow: 0px 2px 4px rgba(0,0,0,0.2);'><strong>📁 {kode}</strong> &nbsp;|&nbsp; {uraian} <i style='opacity: 0.8;'>({label})</i></span></div>"

def get_hierarchy(kode_target, df):
    parts = str(kode_target).split('.')
    hierarchy_list = []
    current_code = ""
    for i, part in enumerate(parts):
        current_code = (current_code + "." + part) if current_code else part
        match = df[df['kode'] == current_code]
        uraian = match.iloc[0]['uraian'].title() if not match.empty else "Detail Klasifikasi"
        html_string = get_badge_html(current_code, uraian, i)
        hierarchy_list.append(html_string)
    return hierarchy_list

# --- 3. OTAK BARU SIKAP: HYBRID GROQ + GEMINI ---
def panggil_groq_json(prompt_text):
    chat_completion = client_groq.chat.completions.create(
        messages=[{"role": "user", "content": prompt_text}],
        model="llama-3.1-8b-instant", 
        response_format={"type": "json_object"},
        temperature=0.1
    )
    return json.loads(chat_completion.choices[0].message.content)

def cari_top_k(query_embed, df_target, k=3):
    embeddings_list = np.vstack(df_target['embedding'].values)
    query_embed_arr = np.array(query_embed).reshape(1, -1)
    similarities = cosine_similarity(query_embed_arr, embeddings_list)[0]
    
    top_indices = np.argsort(similarities)[-k:][::-1]
    kandidat = []
    for idx in top_indices:
        row = df_target.iloc[idx]
        kandidat.append({"kode": row['kode'], "uraian": row['uraian'], "idx_asli": df_target.index[idx], "score": float(similarities[idx])})
    return kandidat

import time

# --- 1. MESIN PENALARAN BARU (Pengganti Groq) ---
def panggil_gemini_json(prompt_text):
    model = genai.GenerativeModel(
        model_name="models/gemini-2.5-flash",
        generation_config={
            "response_mime_type": "application/json", # Garansi tidak ada teks "sampah"
            "temperature": 0.1
        }
    )
    response = model.generate_content(prompt_text)
    return json.loads(response.text)

# --- 2. FUNGSI KLASIFIKASI (Murni Gemini) ---
def smart_classify(isi_surat, df):
    try:
        # TAHAP 1: EKSTRAKSI (Insting Arsiparis Tetap Dipertahankan)
        st.write("Mengekstrak inti surat...")
        prompt_ekstrak = f"""Anda adalah Arsiparis Ahli di instansi pemerintah. Analisis surat berikut.
        ATURAN MUTLAK EKSTRAKSI:
        1. ABAIKAN KATA PENGANTAR: Buang kata "Penyampaian", "Pengantar", "Permohonan", "Undangan".
        2. ABAIKAN LOKASI/OBJEK FISIK: Jangan jadikan "Gedung", "Perpustakaan", atau "Puskesmas" sebagai inti jika suratnya tentang perizinan atau tanah.
        3. ABAIKAN SUMBER DANA: Abaikan kata "DAK", "APBD", "BOS" kecuali surat tersebut MURNI membahas pencairan uang/SP2D.
        4. HIERARKI DOKUMEN: Jika ada kata "Sertifikat Tanah" atau "Aset Lahan", maka inti surat ADALAH PERTANAHAN, BUKAN BAST/Hukum hibahnya.
        
        Surat: {isi_surat}
        Output JSON format: {{"inti_surat": "...", "kata_kunci": ["...", "..."]}}"""
        
        ekstrak = panggil_gemini_json(prompt_ekstrak)
        inti = ekstrak.get('inti_surat', isi_surat)
        st.info(f"**Inti Surat yang Dianalisis:** {inti}")
        
        time.sleep(1.5) # Jeda aman anti-limit
        
        # TAHAP 2: PENCARIAN VEKTOR (Tetap menggunakan gemini-embedding-2)
        st.write("Mengubah makna menjadi vektor...")
        res_embed = genai.embed_content(model="models/gemini-embedding-2", content=inti)
        vektor_query = res_embed['embedding']
        
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        vektor_db = np.array(df['embedding'].tolist())
        df['skor'] = cosine_similarity([vektor_query], vektor_db)[0]
        
        time.sleep(1.5) # Jeda aman anti-limit

        # TAHAP 3: BEDAH PRIMER
        st.write("🔍 Membedah tingkat Primer...")
        df_primer = df[df['kode'].str.endswith('00')].nlargest(3, 'skor')
        pilihan_primer = df_primer[['kode', 'uraian']].to_dict('records')
        
        prompt_primer = f"""Pilih 1 kode Primer yang paling akurat untuk: "{inti}". 
        Pilihan: {pilihan_primer}. Output JSON: {{"kode": "..."}}"""
        
        hasil_primer = panggil_gemini_json(prompt_primer)
        kode_primer = hasil_primer.get('kode', pilihan_primer[0]['kode'])
        uraian_primer = df[df['kode'] == kode_primer]['uraian'].values[0]
        
        st.success(f"📁 **{kode_primer}** | {uraian_primer} (Primer)")
        
        time.sleep(1.5) # Jeda aman anti-limit

        # TAHAP 4: BEDAH SEKUNDER
        st.write("🔍 Membedah tingkat Sekunder...")
        prefix_sekunder = kode_primer[:1] 
        df_sekunder = df[(df['kode'].str.startswith(prefix_sekunder)) & (~df['kode'].str.endswith('00')) & (df['kode'].str.len() <= 6)]
        
        if df_sekunder.empty:
            return kode_primer, uraian_primer
            
        df_sekunder = df_sekunder.nlargest(3, 'skor')
        pilihan_sekunder = df_sekunder[['kode', 'uraian']].to_dict('records')
        
        prompt_sekunder = f"""Pilih 1 kode Sekunder turunan dari {kode_primer} yang paling akurat untuk: "{inti}". 
        Pilihan: {pilihan_sekunder}. Output JSON: {{"kode": "..."}}"""
        
        hasil_sekunder = panggil_gemini_json(prompt_sekunder)
        kode_sekunder = hasil_sekunder.get('kode', pilihan_sekunder[0]['kode'])
        uraian_sekunder = df[df['kode'] == kode_sekunder]['uraian'].values[0]
        
        st.success(f"📂 **{kode_sekunder}** | {uraian_sekunder} (Sekunder)")
        
        return kode_sekunder, uraian_sekunder

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
        return None

# --- 4. ANTARMUKA UTAMA ---
def halaman_utama():
    st.markdown("<div class='sikap-title'>SIKAP</div>", unsafe_allow_html=True)
    st.markdown("<div class='sikap-subtitle'>Sistem Informasi Klasifikasi Arsip Pintar</div>", unsafe_allow_html=True)

    try:
        df = load_data()
        if df.empty: return
        
        with st.sidebar:
            st.success(f"Masuk sebagai:\n**{st.session_state['nama']}**")
            if st.button("Keluar", use_container_width=True):
                st.session_state['logged_in'] = False
                st.session_state['role'] = None
                st.session_state['nama'] = ""
                st.rerun()
            st.divider()

            st.header("🕒 Riwayat Pencarian")
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

        if st.session_state.get('role') == 'admin':
            tab_ai, tab_katalog, tab_admin = st.tabs(["🤖 Pencarian AI (Hybrid)", "📂 Jelajah Kode Klasifikasi", "⚙️ Panel Admin"])
        else:
            tab_ai, tab_katalog = st.tabs(["🤖 Pencarian AI (Hybrid)", "📂 Jelajah Kode Klasifikasi"])

        # ================= TAB 1: PENCARIAN AI =================
        with tab_ai:
            st.write("Ketikkan inti surat Anda. SIKAP akan mengekstrak informasi dan berpikir secara hierarkis untuk menemukan kode kearsipan yang paling akurat.")
            user_input = st.text_input("📝 Perihal Surat / Dokumen:", placeholder="Contoh: permohonan penerbitan sertifikat tanah untuk perpustakaan...", key="input_ai")

            if user_input:
                if user_input not in st.session_state.search_history:
                    st.session_state.search_history.append(user_input)
                    simpan_riwayat_csv(st.session_state['nama'], user_input)

                results = smart_classify(user_input, df)
                    
                if results:
                    st.success("Analisis selesai! AI telah mengambil keputusan tunggal berdasarkan hierarki:")
                    pilihan_feedback = [] 
                    
                    for i, (idx, score) in enumerate(results):
                        res = df.iloc[idx]
                        pilihan_feedback.append(f"{res['kode']} - {res['uraian'].title()}")
                        
                        with st.expander(f"🏅 Rekomendasi Utama: Kode {res['kode']}", expanded=True):
                            st.markdown(f"**Uraian Akhir:** {res['uraian'].title()}")
                            st.markdown("**Struktur Hierarki:**")
                            hierarki = get_hierarchy(res['kode'], df)
                            for h in hierarki:
                                st.markdown(h, unsafe_allow_html=True)
                else:
                    st.warning("Tidak ditemukan klasifikasi yang cocok. Coba perjelas bahasa Anda.")

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

        # ================= TAB 3: PANEL ADMIN =================
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
    halaman_login()
else:
    halaman_utama()
