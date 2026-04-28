import streamlit as st
import pandas as pd
import re
import os
import time
import json
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process, fuzz
import google.generativeai as genai

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
            return list(dict.fromkeys(riwayat_user))
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

# --- 1. MENARIK API KEY GEMINI DENGAN AMAN ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = "MASUKKAN_API_KEY_GEMINI_DI_SINI_UNTUK_TES_LOKAL" 
genai.configure(api_key=api_key)

# --- FUNGSI PEMANGGIL GEMINI JSON (STANDAR BARU) ---
def panggil_gemini_json(prompt_text):
    model = genai.GenerativeModel(
        model_name="models/gemini-2.5-flash",
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.0 # Mengunci agar tidak berhalusinasi
        }
    )
    response = model.generate_content(prompt_text)
    return json.loads(response.text)

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

# --- INISIALISASI NLP (Sastrawi) ---
@st.cache_resource
def init_nlp():
    stemmer = StemmerFactory().create_stemmer()
    remover = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, remover

stemmer, remover = init_nlp()

# --- KAMUS JARGON & SINGKATAN BIROKRASI (ASLI 100%) ---
kamus_birokrasi = {
    "apbd": "anggaran pendapatan dan belanja daerah",
    "apbn": "anggaran pendapatan dan belanja negara",
    "tapd": "tim anggaran pemerintah daerah",
    "dpa": "dokumen pelaksanaan anggaran",
    "rka": "rencana kerja anggaran",
    "skpd": "satuan kerja perangkat daerah",
    "ppkd": "pejabat pengelola keuangan daerah",
    "ppa": "prioritas plafon anggaran",
    "spp": "surat permintaan pembayaran",
    "spm": "surat perintah membayar",
    "sp2d": "surat perintah pencairan dana",
    "up": "uang persediaan",
    "gu": "ganti uang",
    "tu": "tambah uang",
    "ls": "langsung",
    "bud": "bendahara umum daerah",
    "bku": "buku kas umum",
    "sakd": "sistem akuntansi keuangan daerah",
    "phln": "pinjaman hibah luar negeri",
    "bumd": "badan usaha milik daerah",
    "blud": "badan layanan umum daerah",
    "dau": "dana alokasi umum",
    "dak": "dana alokasi khusus",
    "dbh": "dana bagi hasil",
    "asn": "aparatur sipil negara",
    "pns": "pegawai negeri sipil",
    "cpns": "calon pegawai negeri sipil",
    "pppk": "pegawai pemerintah dengan perjanjian kerja",
    "p3k": "pegawai pemerintah dengan perjanjian kerja",
    "nip": "nomor induk pegawai",
    "bkn": "badan kepegawaian negara",
    "skp": "sasaran kinerja pegawai", 
    "duk": "daftar urut kepangkatan",
    "karpeg": "kartu pegawai",
    "kpe": "kartu pegawai elektronik",
    "karis": "kartu istri",
    "karsu": "kartu suami",
    "lp2p": "laporan pajak penghasilan pribadi",
    "kp4": "keterangan penerimaan pembayaran penghasilan pegawai",
    "baperjakat": "badan pertimbangan jabatan dan pangkat",
    "bpjs": "badan penyelenggara jaminan sosial",
    "diklat": "pendidikan dan pelatihan",
    "bimtek": "bimbingan teknis",
    "lhp": "laporan hasil pemeriksaan",
    "lha": "laporan hasil audit",
    "lhpo": "laporan hasil pemeriksaan operasional",
    "lhe": "laporan hasil evaluasi",
    "lhai": "laporan hasil audit investigasi",
    "tpk": "tindak pidana korupsi",
    "gcg": "good corporate governance",
    "perda": "peraturan daerah",
    "perbup": "peraturan bupati",
    "perwali": "peraturan wali kota",
    "mou": "memorandum of understanding nota kesepakatan",
    "sop": "standar operasional prosedur",
    "haki": "hak atas kekayaan intelektual",
    "dprd": "dewan perwakilan rakyat daerah",
    "musrenbang": "musyawarah perencanaan pembangunan",
    "lkpj": "laporan keterangan pertanggungjawaban",
    "lppd": "laporan penyelenggaraan pemerintahan daerah",
    "bmd": "barang milik daerah",
    "kak": "kerangka acuan kerja",
    "sppd": "surat perintah perjalanan dinas",
    "spt": "surat perintah tugas",
    "nodin": "nota dinas",
    "bap": "berita acara pemeriksaan",
    "bast": "berita acara serah terima",
    "kpu": "komisi pemilihan umum",
    "kpud": "komisi pemilihan umum daerah",
    "dp4": "daftar penduduk potensial pemilih",
    "dps": "daftar pemilih sementara",
    "dpt": "daftar pemilih tetap",
    "panwasda": "panitia pengawas daerah",
    "ppk": "panitia pemilihan kecamatan",
    "pps": "panitia pemungutan suara",
    "kpps": "kelompok penyelenggara pemungutan suara",
    "ormas": "organisasi kemasyarakatan",
    "lsm": "lembaga swadaya masyarakat",
    "parpol": "partai politik",
    "anri": "arsip nasional republik indonesia",
    "jra": "jadwal retensi arsip",
    "sikn": "sistem informasi kearsipan nasional",
    "jikn": "jaringan informasi kearsipan nasional",
    "spam": "sistem penyediaan air minum",
    "psat": "pangan segar asal tumbuhan",
    "bumdes": "badan usaha milik desa",
    "bos": "bantuan operasional sekolah",
    "paud": "pendidikan anak usia dini",
    "rtrw": "rencana tata ruang wilayah",
    "rdtr": "rencana detail tata ruang",
    "rtbl": "rencana tata bangunan dan lingkungan",
    "amdal": "analisis mengenai dampak lingkungan",
    "ukl": "upaya pengelolaan lingkungan",
    "upl": "upaya pemantauan lingkungan",
    "b3": "bahan berbahaya dan beracun",
    "sar": "search and rescue pencarian dan pertolongan"
}

# --- FUNGSI PENERJEMAH SINGKATAN ---
def terjemahkan_singkatan(text):
    kata_kata = str(text).lower().split()
    kata_terjemahan = [kamus_birokrasi.get(kata, kata) for kata in kata_kata]
    return " ".join(kata_terjemahan)

# --- FUNGSI PEMBERSIH UTAMA ---
def preprocess_text(text):
    text = str(text).lower()
    text = terjemahkan_singkatan(text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = remover.remove(text)
    text = stemmer.stem(text)
    return text

# --- 2. MEMUAT DATABASE (DENGAN SUNTIKAN KONTEKS HIERARKI & ANTI-LIMIT EMBEDDING) ---
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
        
    if len(df.columns) >= 2:
        kolom_baru = list(df.columns)
        kolom_baru[0] = 'kode'
        kolom_baru[1] = 'uraian'
        df.columns = kolom_baru
    
    df['uraian'] = df['uraian'].astype(str).str.replace(r';$', '', regex=True).str.strip().fillna("")
    df['kode'] = df['kode'].astype(str).str.strip().fillna("000")

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
    df['clean_uraian'] = df['uraian_lengkap'].apply(preprocess_text)
    
    # PERSIAPAN EMBEDDING UNTUK OTAK BARU SIKAP
    def gabung_teks(row):
        teks = f"Uraian: {row.get('uraian', '')}. Konteks: {row.get('uraian_lengkap', '')}. "
        # Jika CSV Emas Anda punya kolom lain, akan otomatis ditangkap
        if 'penjelasan' in df.columns: teks += f"Penjelasan: {row.get('penjelasan', '')}. "
        if 'contoh_surat' in df.columns: teks += f"Contoh Surat: {row.get('contoh_surat', '')}."
        return teks
        
    df['teks_embedding'] = df.apply(gabung_teks, axis=1)

    # SISTEM ANTI-LIMIT GOOGLE 429
    if os.path.exists('embeddings.npy'):
        embeddings_matrix = np.load('embeddings.npy')
        if len(embeddings_matrix) != len(df):
            os.remove('embeddings.npy')
            st.warning("Perubahan CSV terdeteksi! Membangun ulang memori AI... Silakan Refresh (F5).")
            st.stop()
        df['embedding'] = list(embeddings_matrix)
    else:
        st.warning("⚠️ Menerjemahkan kamus kearsipan ke otak AI. Proses berjalan SATU KALI saja. Mohon biarkan halaman terbuka...")
        batch_size = 5 
        all_embeddings = []
        
        progress_text = "Membangun otak pencarian..."
        my_bar = st.progress(0, text=progress_text)
        
        for i in range(0, len(df), batch_size):
            batch_texts = df['teks_embedding'].iloc[i:i+batch_size].tolist()
            sukses = False
            percobaan = 0
            while not sukses and percobaan < 5:
                try:
                    response = genai.embed_content(model="models/gemini-embedding-2", content=batch_texts)
                    all_embeddings.extend(response['embedding'])
                    sukses = True
                    time.sleep(2) 
                except Exception as e:
                    if "429" in str(e) or "Quota" in str(e):
                        percobaan += 1
                        time.sleep(10) 
                    else:
                        st.error(f"Gagal melakukan embedding: {e}")
                        st.stop()
            persentase = min(100, int(((i + batch_size) / len(df)) * 100))
            my_bar.progress(persentase / 100.0, text=f"{progress_text} ({persentase}%)")
                
        if all_embeddings:
            embeddings_matrix = np.array(all_embeddings)
            np.save('embeddings.npy', embeddings_matrix) 
            df['embedding'] = list(embeddings_matrix)
            st.success("Database berhasil diperkaya! Aplikasi akan dimuat ulang...")
            time.sleep(2)
            st.rerun()
            
    return df

# --- FUNGSI PEMBUAT BADGE UNTUK TAB 1 (ASLI 100%) ---
def get_badge_html(kode, uraian, level):
    levels_name = ["Primer", "Sekunder", "Tersier", "Kuartier", "Kuintier"]
    label = levels_name[level] if level < len(levels_name) else f"Level {level+1}"
    warna_level = ["#B71C1C", "#1565C0", "#2E7D32", "#E65100", "#4A148C"]
    warna_bg = warna_level[level] if level < len(warna_level) else "#424242"
    indent = level * 30 
    return f"<div style='margin-left: {indent}px; margin-bottom: 8px;'>" \
           f"<span style='background-color: {warna_bg}; color: #ffffff; padding: 6px 12px; border-radius: 6px; font-weight: normal; font-size: 0.95em; display: inline-block; box-shadow: 0px 2px 4px rgba(0,0,0,0.2);'>" \
           f"<strong>📁 {kode}</strong> &nbsp;|&nbsp; {uraian} <i style='opacity: 0.8;'>({label})</i>" \
           f"</span></div>"

# --- FITUR HIERARKI TAB 1 (ASLI 100%) ---
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

# --- 3. LOGIKA AI HYBRID (GEMINI DOT-BASED HIERARCHY) ---
def smart_classify(user_input, df, top_n=3):
    try:
        # FASE 1: EKSTRAKSI MENGGUNAKAN PROMPT "THE FINAL BOSS" MILIK ANDA
        prompt_ekstrak = f"""Anda adalah Sistem AI Ahli Kearsipan Pemerintahan Daerah. Tugas Anda menganalisis perihal surat dan mengekstrak "Inti Substansi" (maksimal 2-3 frasa).
        GUNAKAN LOGIKA BERPIKIR BERIKUT SECARA BERURUTAN:
        1. HAPUS KATA PENGANTAR: Buang kata basa-basi (contoh: penyampaian, permohonan, undangan, laporan, tindak lanjut, usulan, hal, mengenai, draf, rancangan, penerbitan, fasilitasi, perihal, rekomendasi, sosialisasi).
        2. HAPUS ENTITAS & LOKASI: Buang nama instansi, nama tempat, nama orang, jabatan, dan tahun/tanggal.
        3. CARI SUBSTANSI UTAMA: Temukan urusan aslinya (fasilitatif maupun substantif teknis daerah).
        4. RESOLUSI JEBAKAN "ARSIP": Jangan jadikan "arsip" sebagai inti jika hanya lokasi. Gunakan jika murni teknis (jadwal retensi).
        5. RESOLUSI JEBAKAN ASET/BANGUNAN: Jika urusan tanah/bangunan, ambil status hukumnya (Sertifikat Tanah). Jangan jadikan nama proyek sebagai inti.
        
        [KASUS KEUANGAN, ANGGARAN & ASET]
        Input: "Penyampaian dokumen rencana kerja anggaran (RKA) dan dokumen pelaksanaan anggaran (DPA) tahun anggaran 2026" | Output: rencana kerja anggaran, dpa
        
        [KASUS KEPEGAWAIAN, PENGAWASAN & HUKUM]
        Input: "Usulan penetapan angka kredit (PAK) jabatan fungsional arsiparis tingkat ahli" | Output: penetapan angka kredit, jabatan fungsional
        
        SEKARANG, KERJAKAN DENGAN POLA LOGIKA YANG SAMA:
        Input: "{user_input}"
        
        Output JSON format WAJIB: {{"inti_surat": "..."}}"""
        
        hasil_ekstrak = panggil_gemini_json(prompt_ekstrak)
        inti_dari_llm = hasil_ekstrak.get('inti_surat', user_input)
        st.info(f"🧠 SIKAP (Gemini Flash) menangkap inti surat Anda sebagai: **{inti_dari_llm}**")
        time.sleep(1.5)

        # FASE 2: PENCARIAN VEKTOR
        res_embed = genai.embed_content(model="models/gemini-embedding-2", content=inti_dari_llm)
        vektor_query = res_embed['embedding']
        vektor_db = np.array(df['embedding'].tolist())
        df['skor'] = cosine_similarity([vektor_query], vektor_db)[0]
        df_sorted = df.sort_values(by='skor', ascending=False)
        time.sleep(1.5)

        # FASE 3: BEDAH HIERARKI BERJENJANG (Primer -> Sekunder -> Tersier -> Kuartier)
        df_primer = df_sorted[~df_sorted['kode'].astype(str).str.contains(r'\.', na=False)].head(3)
        if df_primer.empty: df_primer = df_sorted.head(3)
            
        pilihan_primer = df_primer[['kode', 'uraian_lengkap']].to_dict('records')
        prompt_primer = f"Pilih 1 kode Primer yang paling akurat untuk: '{inti_dari_llm}'. Pilihan: {pilihan_primer}. Output JSON: {{\"kode\": \"...\"}}"
        
        kode_terakhir = str(panggil_gemini_json(prompt_primer).get('kode', pilihan_primer[0]['kode']))
        df_kandidat_terakhir = df_primer # Menyimpan posisi kandidat terakhir untuk UI
        
        tingkatan = ["Sekunder", "Tersier", "Kuartier", "Kuintier"]
        
        for tingkat in tingkatan:
            time.sleep(1.5)
            prefix_anak = str(kode_terakhir) + "."
            df_anak = df_sorted[(df_sorted['kode'].astype(str).str.startswith(prefix_anak)) & (df_sorted['kode'] != kode_terakhir)]
            
            if df_anak.empty: break
                
            df_kandidat_terakhir = df_anak.head(3) # Simpan top 3 kandidat di level ini
            pilihan_anak = df_kandidat_terakhir[['kode', 'uraian_lengkap']].to_dict('records')
            prompt_anak = f"Pilih 1 kode {tingkat} turunan dari {kode_terakhir} yang paling akurat untuk: '{inti_dari_llm}'. Pilihan: {pilihan_anak}. Output JSON: {{\"kode\": \"...\"}}"
            
            kode_baru = str(panggil_gemini_json(prompt_anak).get('kode', pilihan_anak[0]['kode']))
            if kode_baru not in df['kode'].values: kode_baru = str(pilihan_anak[0]['kode'])
            kode_terakhir = kode_baru

        # KEMBALIKAN DALAM FORMAT LIST TUPLE AGAR FORM FEEDBACK UI ANDA TETAP BEKERJA
        hasil_akhir = []
        # Memasukkan kode juara 1 yang terpilih
        idx_juara = df[df['kode'] == kode_terakhir].index[0]
        hasil_akhir.append((idx_juara, 0.99))
        
        # Memasukkan kode juara 2 dan 3 dari level pencarian terdalam (sebagai alternatif di form UI)
        for i, row in df_kandidat_terakhir.iterrows():
            if i != idx_juara and len(hasil_akhir) < 3:
                hasil_akhir.append((i, row['skor'] * 0.90)) # Skor diturunkan sedikit

        return hasil_akhir

    except Exception as e:
        st.error(f"Terjadi kesalahan AI: {e}")
        # Fallback ke pencarian TF-IDF Klasik milik Anda jika API mati
        clean_input = preprocess_text(user_input)
        vectorizer = TfidfVectorizer(ngram_range=(1, 3)) 
        all_docs = df['clean_uraian'].tolist() + [clean_input]
        tfidf_matrix = vectorizer.fit_transform(all_docs)
        cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        skor_awal = [{'idx': idx, 'skor': score} for idx, score in enumerate(cosine_sim)]
        return [(item['idx'], item['skor']) for item in sorted(skor_awal, key=lambda x: x['skor'], reverse=True)[:top_n]]

# --- 4. ANTARMUKA UTAMA (TIDAK ADA YANG DIUBAH) ---
def halaman_utama():
    st.markdown("<div class='sikap-title'>SIKAP</div>", unsafe_allow_html=True)
    st.markdown("<div class='sikap-subtitle'>Sistem Informasi Klasifikasi Arsip Pintar</div>", unsafe_allow_html=True)

    try:
        df = load_data()
        
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
                                    elif len(curr) > 3: curr = curr[:-1]
                                    elif len(curr) == 3:
                                        if curr.endswith('0'): curr = curr[0] + '00'
                                        else: curr = curr[0:2] + '0'
                                    else: break
                                if curr in nodes:
                                    parent = curr
                                    break
                            
                            if not parent or parent not in nodes: parent = p
                            if parent in nodes and parent != k: nodes[parent]['children'].append(k)
                                
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
                                for c in children: html += render_tree(c)
                                html += '</div></details>'
                            else:
                                html += f'<div style="margin-bottom: 8px;"><span style="background-color: {warna_bg}; color: #ffffff; padding: 6px 12px; border-radius: 6px; font-weight: normal; font-size: 0.95em; display: inline-block; box-shadow: 0px 2px 4px rgba(0,0,0,0.2);"><strong>📁 {k}</strong> &nbsp;|&nbsp; {u} <i style="opacity: 0.8;">({label})</i></span></div>'
                            return html
                            
                        if p in nodes:
                            full_html = ""
                            sorted_children = sorted(nodes[p]['children'], key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
                            for child_kode in sorted_children: full_html += render_tree(child_kode)
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
