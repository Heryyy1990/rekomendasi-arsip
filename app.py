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

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="SIKAP - Klasifikasi Arsip Pintar", page_icon="🗂️", layout="wide")



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
    st.markdown("""
<div class="sikap-wrapper">
    <div class="sikap-title">SIKAP</div>
    <div class="sikap-subtitle">Sistem Informasi Klasifikasi Arsip Pintar</div>
</div>
""", unsafe_allow_html=True)

    with st.form("form_login"):
        st.markdown("""
        <div class="login-header-container">
            <div class="login-title">Selamat Datang</div>
        </div>
        <div class="login-subtitle">
        <b>Masuk untuk mengakses sistem klasifikasi arsip<br>
        secara cepat, akurat, dan pintar.</b>
        </div>
        """, unsafe_allow_html=True)

        user_input = st.text_input(
            "Username",
            placeholder="Masukkan username Anda"
        )

        pwd_input = st.text_input(
            "Password",
            type="password",
            placeholder="Masukkan password Anda"
        )

        submit = st.form_submit_button(
            "Masuk",
            use_container_width=True
        )
        
        # FOOTER BARU: Icon Futuristik + Crafted by Heryanto, S.Pd.
        st.markdown("""
        <div class="login-footer">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#009DFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: sub; margin-right: 6px;">
                <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                <polyline points="2 17 12 22 22 17"></polyline>
                <polyline points="2 12 12 17 22 12"></polyline>
            </svg>
            <span>Crafted by <b>Heryanto, S.Pd.</b></span>
        </div>
        """, unsafe_allow_html=True)

        if submit:
            is_valid, role, nama = validasi_login(user_input, pwd_input)

            if is_valid:
                st.session_state['logged_in'] = True
                st.session_state['role'] = role
                st.session_state['nama'] = nama
                st.rerun()
            else:
                st.error("Username atau Password salah!")

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

# 2. Fungsi "Otak Ekstraktor"
def ekstrak_inti_surat(teks_user):
    # TERA PROMPT: Transplantasi Otak Logika Klasifikasi Arsip (The Final Boss Version)
    prompt = f"""
    Anda adalah Sistem AI Ahli Kearsipan Pemerintahan Daerah. Tugas Anda menganalisis perihal surat dan mengekstrak "Inti Substansi" (maksimal 2-3 frasa) untuk mesin pencari klasifikasi.
    
    GUNAKAN LOGIKA BERPIKIR BERIKUT SECARA BERURUTAN:
    1. HAPUS KATA PENGANTAR: Buang kata basa-basi (contoh: penyampaian, permohonan, undangan, laporan, tindak lanjut, usulan, hal, mengenai, draf, rancangan, penerbitan, fasilitasi, perihal, rekomendasi, sosialisasi).
    2. HAPUS ENTITAS & LOKASI: Buang nama instansi (Dinas, Badan, Kementerian, KPU, Bawaslu, RSUD), nama tempat (Provinsi, Kabupaten, Desa), nama orang, jabatan (Bupati, Kadis, Kades), dan tahun/tanggal.
    3. CARI SUBSTANSI UTAMA: Temukan urusan aslinya (fasilitatif maupun substantif teknis daerah).
    4. RESOLUSI JEBAKAN "ARSIP": 
       - JANGAN jadikan "arsip" sebagai inti jika itu hanya lokasi/tujuan (misal: "Bimtek kearsipan" -> intinya "Bimbingan Teknis").
       - GUNAKAN "arsip" JIKA teknis murni (misal: "jadwal retensi arsip", "pemusnahan arsip").
    5. RESOLUSI JEBAKAN ASET/BANGUNAN:
       - Jika urusannya adalah tanah/lahan/bangunan, ambil status hukumnya (Sertifikat Tanah, Pengadaan Lahan, Hibah Tanah).
       - JANGAN jadikan NAMA BANGUNAN/PROYEK (seperti Perpustakaan, Puskesmas, Sekolah, Jembatan) sebagai inti substansi.

    BERIKUT ADALAH BANK DATA CONTOH POLA PIKIR YANG WAJIB ANDA TIRU 100%:
    
    [KASUS KEUANGAN, ANGGARAN & ASET]
    Input: "Penyampaian dokumen rencana kerja anggaran (RKA) dan dokumen pelaksanaan anggaran (DPA) tahun anggaran 2026"
    Output: rencana kerja anggaran, dpa
    Input: "Permohonan penerbitan surat perintah pencairan dana (SP2D) dan SPPR untuk kegiatan sosialisasi"
    Output: pencairan dana, sp2d, sppr
    Input: "Usulan persetujuan pinjaman hibah luar negeri (PHLN) dan dana tugas pembantuan"
    Output: pinjaman hibah luar negeri, phln, tugas pembantuan
    
    [KASUS PENGAWASAN, KEPEGAWAIAN & HUKUM]
    Input: "Tindak lanjut temuan laporan hasil pemeriksaan (LHP) dan Laporan Auditor Independen (LAI) BPK RI"
    Output: tindak lanjut temuan, laporan hasil pemeriksaan, laporan auditor independen
    Input: "Laporan hasil audit investigasi (LHAI) yang mengandung unsur tindak pidana korupsi (TPK)"
    Output: laporan hasil audit investigasi, tindak pidana korupsi
    Input: "Usulan penetapan angka kredit (PAK) jabatan fungsional arsiparis tingkat ahli"
    Output: penetapan angka kredit, jabatan fungsional
    
    [KASUS INFRASTRUKTUR, PEKERJAAN UMUM & TATA RUANG]
    Input: "Laporan progres pemeliharaan jalan bebas hambatan dan pengelolaan irigasi rawa"
    Output: pemeliharaan jalan bebas hambatan, pengelolaan irigasi rawa
    Input: "Pengajuan Rencana Detail Tata Ruang (RDTR) dan Rencana Tata Bangunan dan Lingkungan (RTBL)"
    Output: rencana detail tata ruang, rencana tata bangunan dan lingkungan
    Input: "Persetujuan penataan bangunan dan pengelolaan gedung rumah negara"
    Output: penataan bangunan, pengelolaan rumah negara

    [KASUS KEPENDUDUKAN, KESEHATAN & KESRA]
    Input: "Laporan pelaksanaan Sistem Informasi Administrasi Kependudukan (SIAK) dan pencatatan sipil"
    Output: sistem informasi administrasi kependudukan, pencatatan sipil
    Input: "Pelaksanaan program Jaminan Kesehatan Nasional (JKN) dan National Health Account (NHA)"
    Output: jaminan kesehatan nasional, national health account
    Input: "Data Forum Komunikasi Umat Beragama (FKUB) dan penyelesaian kasus aliran keagamaan"
    Output: forum komunikasi umat beragama, kasus aliran keagamaan
    
    [KASUS TEKNOLOGI INFORMASI, KOMUNIKASI & PERSANDIAN]
    Input: "Permohonan layanan sertifikasi elektronik dan evaluasi tata kelola e-government tingkat kabupaten"
    Output: sertifikasi elektronik, e government
    Input: "Pemantauan layanan jaringan telekomunikasi dan pengawasan keamanan informasi"
    Output: jaringan telekomunikasi, keamanan informasi

    [KASUS PEMILU, KESBANGPOL & KETERTIBAN]
    Input: "Penyampaian daftar pemilih sementara (DPS) dan daftar penduduk potensial pemilih (DP4) Pilkada"
    Output: daftar pemilih sementara, daftar penduduk potensial pemilih
    Input: "Penyusunan Rencana Anggaran Satuan Kerja (RASK) dan pembiayaan kegiatan operasional (PPKO) pemilu"
    Output: rencana anggaran satuan kerja, pembiayaan kegiatan operasional pemilu
    
    [KASUS PENANAMAN MODAL, LINGKUNGAN HIDUP, BENCANA & PERTANIAN]
    Input: "Fasilitasi penyelesaian masalah pencabutan pembatalan perizinan penanaman modal asing"
    Output: pencabutan pembatalan perizinan penanaman modal
    Input: "Pembahasan dokumen analisis mengenai dampak lingkungan (AMDAL) dan UKL-UPL pabrik kelapa sawit"
    Output: analisis mengenai dampak lingkungan, amdal, ukl upl
    Input: "Laporan operasi pencarian dan pertolongan (SAR) korban banjir bandang"
    Output: operasi pencarian pertolongan, sar, korban banjir
    Input: "Pengendalian Organisme Pengganggu Tumbuhan (OPT) dan Pengendalian Hama Terpadu (PHT)"
    Output: organisme pengganggu tumbuhan, pengendalian hama terpadu
    
    [KASUS PERTAMBANGAN, ENERGI & PERHUBUNGAN]
    Input: "Penerbitan Sertifikat Laik Operasi (SLO) dan Izin Usaha Pertambangan (IUP) Batubara"
    Output: sertifikat laik operasi, izin usaha pertambangan batubara
    Input: "Sertifikasi uji tipe kendaraan bermotor dan pengesahan kualifikasi petugas terminal"
    Output: sertifikasi uji tipe kendaraan bermotor, kualifikasi petugas terminal
    
    [KASUS PEMERINTAHAN UMUM]
    Input: "Penyampaian laporan hasil perjalanan dinas ke Arsip Nasional"
    Output: perjalanan dinas
    Input: "Persetujuan draf jadwal retensi arsip dan pemusnahan arsip inaktif"
    Output: jadwal retensi arsip, pemusnahan arsip inaktif
    
    SEKARANG, KERJAKAN DENGAN POLA LOGIKA YANG SAMA:
    Input: "{teks_user}"
    Output:
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant", # Model terbaru, pengganti llama3-8b
            temperature=0.0, # 0.0 membuat AI tidak berhalusinasi/kreatif, murni mengekstrak
        )
       # Mengambil balasan cerewet dari Groq (Biarkan dia berpikir agar pintar)
        inti_teks_mentah = chat_completion.choices[0].message.content.strip()
        
        # PISAU BEDAH PYTHON: Kita ambil baris paling bawah saja dari curhatan Groq
        # Karena kesimpulan jawaban selalu ada di baris paling bawah.
        daftar_baris = [baris for baris in inti_teks_mentah.split('\n') if baris.strip() != '']
        inti_teks_bersih = daftar_baris[-1].replace('**', '').strip()
        
        # Membersihkan tanda kutip
        inti_teks_bersih = inti_teks_bersih.replace('"', '').replace("'", "")
        return inti_teks_bersih
    except Exception as e:
        st.error(f"🚨 ERROR GROQ (Tahap Ekstraksi): {e}")
        return teks_user
        
# --- UI & CSS CUSTOM ---
st.markdown("""
<style>
/* ============================= */
/* IMPORT FONT POPPINS & GLOBAL */
/* ============================= */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&display=swap');

* {
    font-family: 'Poppins', sans-serif !important;
    box-sizing: border-box !important;
}

:root {
    --bg-app: radial-gradient(circle at top right, #E0F2FE 0%, transparent 40%), linear-gradient(135deg, #F1F5F9 0%, #E2E8F0 100%);
    --card-bg: rgba(255, 255, 255, 0.95);
    --card-border: rgba(0, 157, 255, 0.15);
    --card-shadow: 0 20px 40px rgba(15, 23, 42, 0.06), 0 0 0 1px rgba(255, 255, 255, 0.8);
    --text-title: #0F172A;
    --text-subtitle: #475569;
    --input-bg: #F8FAFC; 
    --input-border: rgba(0, 157, 255, 0.4); 
    --input-focus-bg: #FFFFFF;
    --icon-bg: rgba(0, 157, 255, 0.08);
    --icon-border: rgba(0, 157, 255, 0.5);
}

/* ============================= */
/* VARIABEL TEMA GELAP (OTOMATIS) */
/* ============================= */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-app: radial-gradient(circle at top, rgba(0,191,255,0.08), transparent 40%), linear-gradient(135deg, #020617 0%, #060f26 50%, #020617 100%);
        --card-bg: rgba(10, 20, 40, 0.5);
        --card-border: rgba(0, 194, 255, 0.25);
        --card-shadow: 0 15px 50px rgba(0,0,0,0.6);
        --text-title: #FFFFFF;
        --text-subtitle: #94A3B8;
        --input-bg: rgba(255, 255, 255, 0.05);
        --input-border: rgba(120, 180, 255, 0.4); 
        --input-focus-bg: rgba(0, 157, 255, 0.05);
        --icon-bg: rgba(0, 157, 255, 0.1);
        --icon-border: #009DFF;
    }
}

.stApp {
    background: var(--bg-app) !important;
    min-height: 100vh;
}

/* ============================= */
/* TITLE & WRAPPER SIKAP */
/* ============================= */
.sikap-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin-top: 6vh;
    margin-bottom: 2rem;
    padding: 0 15px;
}

.sikap-title {
    font-size: clamp(3.5rem, 8vw, 6.5rem) !important;
    font-weight: 900;
    line-height: 1.1;
    letter-spacing: 6px !important; 
    background: linear-gradient(90deg, #21E6C1 0%, #009DFF 50%, #1E88FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    filter: drop-shadow(0 0 15px rgba(0,194,255,0.2));
}

.sikap-subtitle {
    font-size: clamp(0.85rem, 3vw, 1.15rem) !important;
    font-weight: 700 !important;
    color: var(--text-subtitle);
    text-align: center;
    margin-top: 5px;
    padding-bottom: 20px; 
    position: relative;
}

.sikap-subtitle::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0; 
    width: 100%; 
    height: 1.5px;
    background: linear-gradient(90deg, transparent, #009DFF, transparent); 
}

.sikap-subtitle::before {
    content: "";
    position: absolute;
    bottom: -1.5px; 
    left: 50%;
    transform: translateX(-50%);
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #00C2FF;
    box-shadow: 0 0 10px 2px rgba(0, 194, 255, 0.8);
    z-index: 1;
}

/* ============================= */
/* LOGIN CARD */
/* ============================= */
div[data-testid="stForm"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 24px !important;
    padding: 40px 30px !important;
    backdrop-filter: blur(16px) !important;
    box-shadow: var(--card-shadow) !important;
    width: 100% !important;
    max-width: 460px !important; 
    margin: 0 auto !important;
}

.login-header-container {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 10px; /* Jarak dirapatkan sedikit karena ikon hilang */
}

.login-title {
    font-size: clamp(1.6rem, 5vw, 2.1rem) !important;
    font-weight: 800 !important;
    color: var(--text-title);
    letter-spacing: -0.5px;
    text-align: center !important;
}
.login-subtitle {
    text-align: center;
    font-size: 0.95rem;
    color: var(--text-subtitle);
    margin-bottom: 30px;
    line-height: 1.6;
}

/* ============================= */
/* LABEL USERNAME & PASSWORD */
/* ============================= */
.stTextInput label {
    color: var(--text-title) !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    margin-bottom: 8px !important;
}

/* ============================= */
/* INPUT BOX */
/* ============================= */
div[data-baseweb="input"], 
div[data-baseweb="base-input"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

div[data-baseweb="input"] {
    background: var(--input-bg) !important;
    border: 2px solid var(--input-border) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    overflow: hidden !important; 
    min-height: 56px !important; 
}

div[data-baseweb="input"]:focus-within {
    border: 2px solid #009DFF !important;
    background: var(--input-focus-bg) !important;
    box-shadow: 0 0 15px rgba(0, 157, 255, 0.2) !important;
}

input[type="text"], input[type="password"] {
    height: 56px !important; 
    padding: 15px 45px 15px 55px !important; 
    line-height: 1.2 !important; 
    color: var(--text-title) !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    background-color: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background-repeat: no-repeat !important;
    background-position: 18px center !important; 
    background-size: 20px !important;
}

input[aria-label="Username"] {
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23009DFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>') !important;
}

input[aria-label="Password"] {
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23009DFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>') !important;
}

input[type="text"]::placeholder, input[type="password"]::placeholder {
    color: #94A3B8 !important;
    font-weight: 500 !important;
}

div[data-testid="stTextInputPassword"] button {
    color: #009DFF !important;
    background: transparent !important;
    padding-right: 15px !important;
}

div[data-testid="InputInstructions"], .st-emotion-cache-12oz5g7, small {
    display: none !important;
}

/* ============================= */
/* BUTTON MASUK (BERSIH & CENTER) */
/* ============================= */
.stFormSubmitButton > button {
    display: block !important; 
    width: 100% !important;
    height: 58px !important; 
    border: none !important;
    border-radius: 12px !important;
    margin-top: 25px !important;
    
    font-family: 'Poppins', sans-serif !important;
    font-size: 1.2rem !important; 
    font-weight: 700 !important; 
    letter-spacing: 4px !important; 
    text-transform: uppercase !important; 
    text-align: center !important; 
    
    color: #FFFFFF !important; 
    background: linear-gradient(90deg, #009DFF 0%, #0A6CFF 100%) !important;
    transition: all .25s ease !important;
    box-shadow: 0 8px 20px rgba(0, 140, 255, 0.3) !important; 
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important; 
}

.stFormSubmitButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 25px rgba(0,140,255,0.45) !important;
}

.login-footer {
    text-align: center;
    color: var(--text-subtitle);
    font-size: 0.95rem;
    margin-top: 25px;
    font-weight: 500;
}

/* ============================= */
/* FIX RESPONSIVE HP */
/* ============================= */
@media screen and (max-width: 480px) { 
    div[data-testid="stForm"] {
        padding: 30px 12px !important; 
    }
    
    .login-title {
        font-size: 1.5rem !important;
    }
    
    .login-subtitle {
        font-size: 0.72rem !important; 
        margin-bottom: 25px !important;
        line-height: 1.5 !important;
        letter-spacing: -0.2px !important; 
    }
    
    input[type="text"], input[type="password"] {
        font-size: 0.9rem !important;
        padding-left: 45px !important;
    }
}
</style>
""", unsafe_allow_html=True)

# --- INISIALISASI SESSION STATE ---
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

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
    "rapbd": "rencana anggaran pendapatan dan belanja daerah",
    "apbdp": "anggaran pendapatan dan belanja daerah perubahan",
    "tapd": "tim anggaran pemerintah daerah",
    "dpa": "dokumen pelaksanaan anggaran",
    "rdpa": "rancangan dokumen pelaksanaan anggaran",
    "rka": "rencana kerja anggaran",
    "rkaskpd": "rencana kerja anggaran satuan kerja perangkat daerah",
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
    "bln": "bantuan luar negeri",
    "wa": "withdrawal authorization",
    "nol": "no objection letter",
    "bumd": "badan usaha milik daerah",
    "blud": "badan layanan umum daerah",
    "dau": "dana alokasi umum",
    "dak": "dana alokasi khusus",
    "dbh": "dana bagi hasil",
    "pnbp": "penerimaan negara bukan pajak",
    "sppr": "surat permintaan pembayaran rutinitas",
    "spdr": "surat penyediaan dana rutin",
    "rask": "rencana anggaran satuan kerja",
    "drask": "dokumen rancangan anggaran satuan kerja",
    "ppko": "penyediaan pembiayaan kegiatan operasional",
    "pph": "pajak penghasilan",
    "ppn": "pajak pertambahan nilai",
    "lhp": "laporan hasil pemeriksaan",
    "lha": "laporan hasil audit",
    "lhpo": "laporan hasil pemeriksaan operasional",
    "lhe": "laporan hasil evaluasi",
    "lhai": "laporan hasil audit investigasi",
    "la": "laporan akuntan",
    "lai": "laporan auditor independen",
    "tl": "tindak lanjut",
    "tpk": "tindak pidana korupsi",
    "gcg": "good corporate governance",
    "asn": "aparatur sipil negara",
    "pns": "pegawai negeri sipil",
    "cpns": "calon pegawai negeri sipil",
    "pppk": "pegawai pemerintah dengan perjanjian kerja",
    "p3k": "pegawai pemerintah dengan perjanjian kerja",
    "nip": "nomor induk pegawai",
    "sdm": "sumber daya manusia",
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
    "diklat": "pendidikan dan pelatihan",
    "bimtek": "bimbingan teknis",
    "perda": "peraturan daerah",
    "perbup": "peraturan bupati",
    "perwali": "peraturan wali kota",
    "mou": "memorandum of understanding nota kesepakatan",
    "sop": "standar operasional prosedur",
    "haki": "hak atas kekayaan intelektual",
    "dprd": "dewan perwakilan rakyat daerah",
    "dpr": "dewan perwakilan rakyat",
    "musrenbang": "musyawarah perencanaan pembangunan",
    "lkpj": "laporan keterangan pertanggungjawaban",
    "lppd": "laporan penyelenggaraan pemerintahan daerah",
    "amj": "akhir masa jabatan",
    "bmd": "barang milik daerah",
    "kak": "kerangka acuan kerja",
    "sppd": "surat perintah perjalanan dinas",
    "spt": "surat perintah tugas",
    "nodin": "nota dinas",
    "bap": "berita acara pemeriksaan",
    "bast": "berita acara serah terima",
    "nkri": "negara kesatuan republik indonesia",
    "satpol pp": "satuan polisi pamong praja",
    "lnl": "lembaga nirlaba lainnya",
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
    "kckr": "karya cetak dan karya rekam",
    "ti": "teknologi informasi",
    "bpjs": "badan penyelenggara jaminan sosial",
    "bos": "bantuan operasional sekolah",
    "paud": "pendidikan anak usia dini",
    "psg": "pendidikan sistem ganda",
    "pkl": "praktek kerja lapang",
    "fkub": "forum komunikasi umat beragama",
    "hiv": "human immunodeficiency virus",
    "aids": "acquired immunodeficiency syndrome",
    "napza": "narkotika psikotropika dan zat adiktif",
    "bkkbn": "badan kependudukan dan keluarga berencana nasional",
    "ape": "anugerah parahita ekapraya",
    "siak": "sistem informasi administrasi kependudukan",
    "nha": "national health account",
    "jkn": "jaminan kesehatan nasional",
    "kuk": "konsorsium upaya kesehatan",
    "spam": "sistem penyediaan air minum",
    "psat": "pangan segar asal tumbuhan",
    "bumdes": "badan usaha milik desa",
    "rtrw": "rencana tata ruang wilayah",
    "rdtr": "rencana detail tata ruang",
    "rtbl": "rencana tata bangunan dan lingkungan",
    "amdal": "analisis mengenai dampak lingkungan",
    "ukl": "upaya pengelolaan lingkungan",
    "upl": "upaya pemantauan lingkungan",
    "rkl": "rencana pengelolaan lingkungan",
    "rpl": "rencana pemantauan lingkungan",
    "b3": "bahan berbahaya dan beracun",
    "sar": "search and rescue pencarian dan pertolongan",
    "pvtt": "perlindungan varietas tanaman",
    "uttp": "ukur takar timbang dan perlengkapannya",
    "opt": "organisme pengganggu tumbuhan",
    "pht": "pengendalian hama terpadu",
    "ukm": "usaha kecil menengah",
    "ukmk": "usaha kecil menengah dan koperasi",
    "llp": "lembaga layanan pemasaran",
    "lpb": "lembaga pengembangan bisnis",
    "pma": "penanam modal asing",
    "sim": "sistem informasi manajemen",
    "tkp": "tempat khusus parkir",
    "tkdn": "tingkat komponen dalam negeri",
    "rkib": "rencana kebutuhan impor barang",
    "rib": "rencana impor barang",
    "pod": "plan of development",
    "kks": "kontrak kerja sama",
    "sni": "standar nasional indonesia",
    "rsni": "rancangan standar nasional indonesia",
    "skkni": "standar kompetensi kerja nasional indonesia",
    "rskkni": "rancangan standar kompetensi kerja nasional indonesia",
    "npt": "nomor pelumas terdaftar",
    "wps": "welding procedure specification",
    "pqr": "procedure qualification record",
    "ebt": "energi baru terbarukan",
    "ebtke": "energi baru terbarukan dan konservasi energi",
    "skt": "surat keterangan terdaftar",
    "skpi": "sertifikasi kelayakan penggunaan instalasi",
    "iup": "izin usaha pertambangan",
    "ipb": "izin panas bumi",
    "ipl": "izin pemanfaatan langsung",
    "pltp": "pembangkit listrik tenaga panas bumi",
    "pln": "perusahaan listrik negara",
    "ipj": "izin pemanfaatan jaringan",
    "bbn": "bahan bakar nabati",
    "hip": "harga indeks pasar",
    "iga": "investment grade audit",
    "re": "rasio elektrifikasi",
    "rd": "rasio desa berlistrik",
    "io": "izin operasi",
    "iupl": "izin usaha penyediaan tenaga listrik",
    "slo": "sertifikat laik operasi",
    "lsk": "lembaga sertifikasi kompetensi",
    "lit": "lembaga inspeksi teknis",
    "wk": "wilayah kerja",
    "kk": "kontrak karya",
    "obvitnas": "obyek vital nasional",
    "cnc": "clear and clean",
    "pkp2b": "perjanjian karya pengusahaan batubara",
    "k3": "keselamatan dan kesehatan kerja",
    "pltsa": "pembangkit listrik tenaga sampah",
    "ppns": "penyidik pegawai negeri sipil",
    "tdem": "time domain electromagnetic",
    "cdm": "clean development mechanism",
    "ppm": "program pengembangan dan pemberdayaan masyarakat"
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

# --- 1. MEMUAT DATABASE (DENGAN SUNTIKAN KONTEKS HIERARKI) ---
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

    # --- LOGIKA BARU: MEMBANGUN JALUR HIERARKI BREADCRUMBS ---
    kode_dict = dict(zip(df['kode'], df['uraian']))
    
    def bangun_hierarki(kode):
        jalur = []
        curr = str(kode).strip()
        
        while curr:
            if curr in kode_dict:
                # Masukkan di awal agar urutannya: Rumpun > Induk > Anak
                jalur.insert(0, kode_dict[curr]) 
                
            # Lacak Bapak/Induknya menggunakan logika standar arsip
            if '.' in curr:
                curr = curr.rsplit('.', 1)[0]
            else:
                if len(curr) == 3 and curr.endswith('00'):
                    break 
                elif len(curr) > 3:
                    curr = curr[:-1]
                elif len(curr) == 3:
                    if curr.endswith('0'): curr = curr[0] + '00'
                    else: curr = curr[0:2] + '0'
                else:
                    break
                    
        # Gabungkan menjadi satu kalimat utuh
        return " > ".join(jalur)
    
    # Kolom baru ini yang akan menjadi "Mata" bagi AI
    df['uraian_lengkap'] = df['kode'].apply(bangun_hierarki)
    
    # TF-IDF dan Sastrawi sekarang membersihkan dan menghafal jalur hierarki secara penuh
    df['clean_uraian'] = df['uraian_lengkap'].apply(preprocess_text)
    
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

# --- 2. FITUR HIERARKI TAB 1 (ASLI 100%) ---
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

# --- 3. LOGIKA AI HYBRID (RERANKING) ---
def smart_classify(user_input, df, top_n=3):
    # 1. Biarkan LLM mengekstrak "inti" dari uraian panjang user
    inti_dari_llm = ekstrak_inti_surat(user_input)
    st.info(f"🧠 SIKAP menangkap inti surat Anda sebagai: **{inti_dari_llm}**")
    
    # 2. Lakukan pembersihan teks (Sastrawi) pada hasil ekstraksi
    clean_input = preprocess_text(inti_dari_llm)
    
   # 3. TF-IDF & Fuzzy Matching (Tugasnya mengambil 10 Nominasi Terbaik)
    vectorizer = TfidfVectorizer(ngram_range=(1, 3)) 
    all_docs = df['clean_uraian'].tolist() + [clean_input]
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    skor_awal = []
    for idx, score in enumerate(cosine_sim):
        # GANTI partial_ratio MENJADI token_set_ratio
        fuzzy_score = fuzz.token_set_ratio(clean_input, df.iloc[idx]['clean_uraian']) / 100
        
        # --- TAMBAHAN: DEPTH BONUS (BOBOT KEDALAMAN) ---
        kode_item = str(df.iloc[idx]['kode'])
        jumlah_titik = kode_item.count('.')
        depth_bonus = jumlah_titik * 0.05 
        
        # Ubah porsi bobotnya: Berikan kekuatan lebih besar pada TF-IDF (Score)
        combined_score = (score * 0.70) + (fuzzy_score * 0.30) + depth_bonus 
        
        skor_awal.append({'idx': idx, 'skor': combined_score})
        
    # Ambil 10 besar nominasi untuk dinilai ulang oleh AI
    top_10_kandidat = sorted(skor_awal, key=lambda x: x['skor'], reverse=True)[:10]
    
 # 4. FASE JURI AI (Llama-3 memilih 3 terbaik dari 10 nominasi matematis)
    daftar_kandidat = ""
    for i, item in enumerate(top_10_kandidat):
        baris = df.iloc[item['idx']]
        # PERUBAHAN: AI SEKARANG MELIHAT JALUR LENGKAP (Konteks), BUKAN CUMA UJUNGNYA
        daftar_kandidat += f"[{i+1}] Kode: {baris['kode']} | Konteks Hierarki: {baris['uraian_lengkap'].title()}\n"
        
    prompt_juri = f"""
    Pilih 3 nomor urut opsi yang paling tepat untuk urusan: "{inti_dari_llm}"
    
    Daftar Opsi (Baca dengan teliti jalur konteks hierarkinya):
    {daftar_kandidat}
    
    ATURAN MUTLAK:
    1. Kamu HANYA BOLEH membalas dengan 3 angka urutan (antara 1 sampai 10) yang dipisah koma.
    2. JIKA ADA BEBERAPA KODE DARI RUMPUN YANG SAMA, KAMU WAJIB MEMILIH KODE TURUNAN YANG PALING DALAM/SPESIFIK. Haram hukumnya memilih kode induk jika ada kode anaknya yang lebih detail dan relevan.
    3. JANGAN tulis kodenya. JANGAN ada teks apapun selain 3 angka.
    Contoh balasan yang benar: 1, 5, 8
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_juri}],
            model="llama-3.3-70b-versatile", # Model raksasa yang sangat teliti dalam menjuri 
            temperature=0.0, 
        )
        balasan_juri = chat_completion.choices[0].message.content.strip()
        
        # LOGIKA ANTI-JEBOL: Ambil semua angka, tapi HANYA simpan angka 1-10 yang unik
        angka_mentah = re.findall(r'\d+', balasan_juri)
        angka_pilihan = []
        for angka in angka_mentah:
            angka_int = int(angka)
            # Pastikan itu nomor urut nominasi (1-10), BUKAN kode klasifikasi seperti 800 atau 000
            if 1 <= angka_int <= 10:
                if angka_int not in angka_pilihan:
                    angka_pilihan.append(angka_int)
            if len(angka_pilihan) == 3: # Berhenti jika sudah dapat 3 juara
                break
                
        hasil_akhir = []
        for nomor in angka_pilihan:
            idx_kandidat = nomor - 1 
            if 0 <= idx_kandidat < len(top_10_kandidat):
                # Bobot keyakinan simulasi yang menurun (99%, 85%, 70%)
                skor_simulasi = 0.99 - (len(hasil_akhir) * 0.14)
                hasil_akhir.append((top_10_kandidat[idx_kandidat]['idx'], skor_simulasi))
                
        if hasil_akhir:
            return hasil_akhir
            
    except Exception as e:
        st.error(f"🚨 ERROR GROQ (Tahap Juri AI): {e}")
        
    # Fallback
    return [(item['idx'], item['skor']) for item in top_10_kandidat[:top_n]]
    
# --- 4. ANTARMUKA UTAMA (STYLE DASHBOARD ENTERPRISE) ---
def halaman_utama():
    # INISIALISASI ROUTING HALAMAN
    if 'page' not in st.session_state:
        st.session_state.page = 'Beranda'

    def ganti_halaman(nama_halaman):
        st.session_state.page = nama_halaman

    # CSS GLOBAL HALAMAN UTAMA
    st.markdown("""
    <style>
    /* Reset & Force Light Theme */
    .stApp { background-color: #F8FAFC !important; }

    /* Font Poppins untuk Teks */
    .hero-title, .hero-subtitle, .search-box-title, .search-box-desc,
    .section-title, p, h1, h2, h3, h4, span, div {
        font-family: 'Poppins', sans-serif;
    }

    /* Penyelamat Ikon Streamlit & Material Symbols */
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');
    .material-symbols-rounded, [data-testid="stHeader"] *, [data-testid="stSidebarCollapseButton"] *, .st-emotion-cache-1wivap2, .st-emotion-cache-1104e76 {
        font-family: 'Material Symbols Rounded', sans-serif !important;
    }
    header[data-testid="stHeader"] { background: transparent !important; }
    .block-container { padding-top: 2rem !important; max-width: 1100px !important; }

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    .sidebar-title-container { display: flex; align-items: center; gap: 10px; padding: 10px 0 20px 0; }
    .sidebar-logo {
        background: #009DFF; color: white; width: 35px; height: 35px; border-radius: 8px;
        display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.2rem;
    }
    .sidebar-title { color: #009DFF; font-weight: 900; letter-spacing: 1px; margin: 0; font-size: 1.8rem; line-height: 1; }

    /* Tombol Sidebar Normal */
    section[data-testid="stSidebar"] .stButton button {
        height: 45px !important; border: none !important; background-color: transparent !important;
        font-size: 0.95rem !important; color: #475569 !important; justify-content: flex-start !important;
        box-shadow: none !important; transform: none !important; font-family: 'Poppins', sans-serif !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover { background-color: #F8FAFC !important; color: #0F172A !important; }
    
    /* PAKSA WARNA BIRU UNTUK TOMBOL AKTIF (Melawan tema merah bawaan) */
    button[kind="primary"] {
        background-color: #009DFF !important; border-color: #009DFF !important; color: white !important; font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] .stButton button[kind="primary"] {
        background-color: #E0F2FE !important; color: #009DFF !important;
    }

    /* UMUM */
    .section-title { font-weight: 700; color: #0F172A; font-size: 1.15rem; margin-bottom: 15px; margin-top: 25px; font-family: 'Poppins', sans-serif !important;}
    div[data-testid="stDataFrame"] { background: #FFFFFF; border-radius: 12px; padding: 10px; border: 1px solid #E2E8F0; }
    div[data-testid="InputInstructions"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    try:
        df = load_data()
        nama_user = st.session_state.get('nama', 'Administrator')
        role_user = st.session_state.get('role', 'admin')

        # ================= SIDEBAR NAVIGASI =================
        with st.sidebar:
            st.markdown("""
            <div class="sidebar-title-container">
                <div class="sidebar-logo">S</div>
                <h1 class="sidebar-title">SIKAP</h1>
            </div>
            <p style="color:#64748B; font-size:0.75rem; margin-top:-15px; margin-bottom:30px;">Sistem Informasi Klasifikasi<br>Arsip Pintar</p>
            """, unsafe_allow_html=True)
            
            st.caption("MENU UTAMA")
            st.button("🏠 Beranda", use_container_width=True, type="primary" if st.session_state.page == 'Beranda' else "secondary", on_click=ganti_halaman, args=('Beranda',))
            st.button("🤖 Pencarian AI (Cerdas)", use_container_width=True, type="primary" if st.session_state.page == 'Pencarian AI' else "secondary", on_click=ganti_halaman, args=('Pencarian AI',))
            st.button("📁 Jelajah Kode Klasifikasi", use_container_width=True, type="primary" if st.session_state.page == 'Jelajah Kode' else "secondary", on_click=ganti_halaman, args=('Jelajah Kode',))
            st.button("🕒 Riwayat Pencarian", use_container_width=True, type="primary" if st.session_state.page == 'Riwayat' else "secondary", on_click=ganti_halaman, args=('Riwayat',))
            
            if role_user == 'admin':
                st.button("⚙️ Panel Admin", use_container_width=True, type="primary" if st.session_state.page == 'Admin' else "secondary", on_click=ganti_halaman, args=('Admin',))
            
            st.divider()
            
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; padding:10px; background:#F8FAFC; border-radius:10px; border:1px solid #E2E8F0;">
                <div style="width:35px; height:35px; background:#E0F2FE; color:#009DFF; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:bold;">
                    {nama_user[0:2].upper()}
                </div>
                <div>
                    <div style="font-size:0.85rem; font-weight:700; color:#0F172A;">{nama_user}</div>
                    <div style="font-size:0.7rem; color:#64748B;">{role_user.title()}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            if st.button("Keluar", use_container_width=True):
                st.session_state['logged_in'] = False
                st.session_state['role'] = None
                st.session_state['nama'] = ""
                st.rerun()

        # ================= MAIN CONTENT (ROUTING) =================
            
        if st.session_state.page == 'Beranda':
            
            # CSS KHUSUS BERANDA (Tarikan Ekstrim)
            st.markdown("""
            <style>
            /* 1. Desain Banner Biru Utama */
            .hero-banner {
                background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%); 
                border-radius: 16px; 
                padding: 30px 20px 40px 20px; 
                display: flex; flex-direction: column; align-items: center; text-align: center; 
                border: 1px solid #BAE6FD; margin-bottom: 0px; 
            }
            .hero-content { width: 100%; max-width: 700px; margin: 0 auto; }
            .hero-title { font-size: 2rem; font-weight: 800; color: #0F172A; margin-bottom: 5px; font-family: 'Poppins', sans-serif !important;}
            .hero-title span { color: #009DFF; }
            .hero-subtitle { font-size: 0.95rem; color: #475569; margin-bottom: 25px; font-family: 'Poppins', sans-serif !important;}
            
            /* 2. Desain Kotak Tempat Search */
            .search-card-bg { 
                background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%); 
                border-radius: 12px; 
                /* PERHATIAN: Padding bawah dilebarkan jadi 85px agar input punya tempat bersandar */
                padding: 20px 25px 85px 25px; 
                box-shadow: 0 10px 25px rgba(37, 99, 235, 0.3); border: 1px solid #1D4ED8;
                width: 100%; max-width: 600px; margin: 0 auto;
            }
            .search-title { font-size: 1.1rem; font-weight: 700; color: #FFFFFF; font-family: 'Poppins', sans-serif !important;}
            
            /* ========================================================= */
            /* 3. TAKTIK SNIPER: TARIKAN EKSTRIM (-120px)                */
            /* ========================================================= */
            div[data-testid="stHorizontalBlock"]:has(input) {
                /* INI DIA KUNCINYA: Tarik paksa ke atas dua kali lipat lebih kuat */
                margin-top: -120px !important; 
                
                /* Posisi Kanan (Sudah pas, kita biarkan) */
                transform: translateX(10px) !important; 
                
                width: 100% !important; max-width: 580px !important; 
                margin-left: auto !important; margin-right: auto !important;
                position: relative; z-index: 99;
            }
            
            /* Desain Input Streamlit */
            div[data-testid="stHorizontalBlock"]:has(input) div[data-baseweb="input"] {
                height: 48px !important; border-radius: 8px !important; background: #FFFFFF !important;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
            }
            div[data-testid="stHorizontalBlock"]:has(input) div[data-baseweb="input"]:focus-within {
                border-color: #009DFF !important; box-shadow: 0 0 0 2px rgba(0, 157, 255, 0.3) !important;
            }
            div[data-testid="stHorizontalBlock"]:has(input) input { font-size: 0.95rem !important; padding-left: 15px !important; font-family: 'Poppins', sans-serif !important;}
            
            /* 4. DESAIN ICON KACA PEMBESAR MODERN + ANIMASI SELURUH TOMBOL */
            
            /* Nama animasi disamakan jadi floatButton, dan pakai translateY untuk kotak */
            @keyframes floatButton {
                0% { transform: translateY(0px); }
                50% { transform: translateY(-6px); } /* Kotak naik ke atas */
                100% { transform: translateY(0px); }
            }

            div[data-testid="stHorizontalBlock"]:has(input) button {
                height: 54px !important; border-radius: 8px !important; width: 100% !important; 
                background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%) !important; 
                border: none !important; font-size: 0 !important; color: transparent !important; position: relative;
                transition: all 0.3s ease; /* Transisi agar mulus */
            }

            /* Memanggil nama animasi yang benar: floatButton */
            div[data-testid="stHorizontalBlock"]:has(input) button:hover {
                animation: floatButton 1.5s ease-in-out infinite !important;
                box-shadow: 0 8px 18px rgba(0, 114, 255, 0.5) !important; /* Bayangan membesar saat tombol naik */
            }
            
            /* SVG Icon ditanam MATI di tengah tombol (tidak ada animasi di sini) */
            div[data-testid="stHorizontalBlock"]:has(input) button::before {
                content: ""; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                width: 20px; height: 20px;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='24' height='24' stroke='white' stroke-width='2.5' fill='none' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'%3E%3C/circle%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'%3E%3C/line%3E%3C/svg%3E");
                background-size: cover; 
            }
            </style>
            """, unsafe_allow_html=True)

            # HTML BANNER UTAMA 
            st.markdown(f"""
<div class="hero-banner">
    <div class="hero-content">
        <div class="hero-title">Selamat datang, <span>{nama_user.split()[0]}</span></div>
        <div class="hero-subtitle">Kelola dan temukan kode klasifikasi arsip dengan mudah, cepat, dan akurat.</div>
        <div class="search-card-bg">
            <div class="search-title">Cari kode klasifikasi arsip</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

            # FORM PENCARIAN 
            col_in, col_btn = st.columns([5, 1])
            with col_in:
                user_input = st.text_input("Pencarian AI", placeholder="Ketik perihal surat di sini...", label_visibility="collapsed", key="input_beranda")
            with col_btn:
                btn_cari = st.button("🔍", key="btn_cari_beranda")

            if user_input or btn_cari:
                st.session_state.temp_search = user_input 
                ganti_halaman('Pencarian AI')
                st.rerun()
                
            # =========================================================
            # 5. AKSES CEPAT (KARTU MENU)
            # =========================================================
            
            # CSS Khusus untuk Kartu Akses Cepat
            st.markdown("""
            <style>
            .section-title { font-weight: 700; color: #0F172A; font-size: 1.15rem; margin-bottom: 15px; margin-top: 10px; font-family: 'Poppins', sans-serif !important;}
            .card-container { position: relative; height: 160px; margin-bottom: 10px; border-radius: 16px;}
            
            /* Trik Tombol Gaib */
            .card-container div[data-testid="stButton"] { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 10; }
            .card-container div[data-testid="stButton"] button { width: 100%; height: 100%; opacity: 0; cursor: pointer; border: none; background: transparent; }

            /* Desain Visual Kartu */
            .saas-card {
                position: absolute; top: 0; left: 0; width: 100%; height: 100%;
                background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px;
                padding: 24px 20px; display: flex; align-items: flex-start; gap: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.02); transition: all 0.3s ease;
                z-index: 1; /* Di bawah tombol gaib */
            }
            .card-container:hover .saas-card {
                border-color: #009DFF; box-shadow: 0 10px 25px rgba(0, 157, 255, 0.1); transform: translateY(-5px);
            }

            /* Ikon Lingkaran Kiri */
            .saas-icon-box {
                width: 50px; height: 50px; min-width: 50px; border-radius: 50%;
                display: flex; align-items: center; justify-content: center; font-size: 1.6rem;
            }
            .bg-blue { background-color: #F0F9FF; color: #009DFF; }
            .bg-orange { background-color: #FFF7ED; color: #F97316; }
            .bg-purple { background-color: #FAF5FF; color: #A855F7; }

            /* Area Teks Kanan */
            .saas-card-content { flex-grow: 1; display: flex; flex-direction: column; height: 100%; justify-content: flex-start;}
            .saas-card-title { font-weight: 700; color: #0F172A; font-size: 1rem; margin-bottom: 5px; font-family: 'Poppins', sans-serif !important;}
            .saas-card-desc { font-size: 0.8rem; color: #64748B; line-height: 1.4; font-family: 'Poppins', sans-serif !important;}
            
            /* Panah Kanan Bawah */
            .saas-card-arrow { align-self: flex-end; margin-top: auto; color: #009DFF; font-size: 1.2rem; display: flex; }
            </style>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-title">⚡ Akses Cepat</div>', unsafe_allow_html=True)
            
            # Membuat 3 Kolom untuk Menu
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="card-container">
                    <div class="saas-card">
                        <div class="saas-icon-box bg-blue"><span class="material-symbols-rounded">smart_toy</span></div>
                        <div class="saas-card-content">
                            <div class="saas-card-title">Pencarian AI (Cerdas)</div>
                            <div class="saas-card-desc">Temukan kode klasifikasi dengan bantuan AI.</div>
                            <div class="saas-card-arrow"><span class="material-symbols-rounded">arrow_forward</span></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                # Tombol Gaib Streamlit
                if st.button("btn_ai", key="btn_akses_ai"):
                    ganti_halaman('Pencarian AI')
                    st.rerun()

            with col2:
                st.markdown("""
                <div class="card-container">
                    <div class="saas-card">
                        <div class="saas-icon-box bg-orange"><span class="material-symbols-rounded">folder</span></div>
                        <div class="saas-card-content">
                            <div class="saas-card-title">Jelajah Kode Klasifikasi</div>
                            <div class="saas-card-desc">Telusuri dan jelajahi struktur hierarki arsip.</div>
                            <div class="saas-card-arrow"><span class="material-symbols-rounded">arrow_forward</span></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                # Tombol Gaib Streamlit
                if st.button("btn_jelajah", key="btn_akses_jelajah"):
                    ganti_halaman('Jelajah Kode')
                    st.rerun()

            with col3:
                st.markdown("""
                <div class="card-container">
                    <div class="saas-card">
                        <div class="saas-icon-box bg-purple"><span class="material-symbols-rounded">schedule</span></div>
                        <div class="saas-card-content">
                            <div class="saas-card-title">Riwayat Pencarian</div>
                            <div class="saas-card-desc">Lihat riwayat pencarian yang telah dilakukan.</div>
                            <div class="saas-card-arrow"><span class="material-symbols-rounded">arrow_forward</span></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                # Tombol Gaib Streamlit
                if st.button("btn_riwayat", key="btn_akses_riwayat"):
                    ganti_halaman('Riwayat')
                    st.rerun()

            # TABEL RIWAYAT
            st.markdown("""
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="section-title">Riwayat Pencarian Terakhir</div>
                <div style="color:#009DFF; font-size:0.85rem; font-weight:600; cursor:pointer;">Lihat semua &gt;</div>
            </div>
            """, unsafe_allow_html=True)
            
            if not st.session_state.search_history:
                st.session_state.search_history = baca_riwayat_csv(st.session_state['nama'])
                
            if st.session_state.search_history:
                history_data = {"WAKTU": [], "KATA KUNCI": [], "METODE": []}
                for riwayat in reversed(st.session_state.search_history[-3:]):
                    history_data["WAKTU"].append("Baru Saja")
                    history_data["KATA KUNCI"].append(riwayat)
                    history_data["METODE"].append("AI (Cerdas)")
                st.dataframe(history_data, use_container_width=True, hide_index=True)
            else:
                st.info("Belum ada riwayat pencarian.")

        # --- HALAMAN 2: PENCARIAN AI ---
        elif st.session_state.page == 'Pencarian AI':
            st.markdown('<div class="section-title">🤖 Pencarian AI (Cerdas)</div>', unsafe_allow_html=True)
            st.write("Sistem cerdas akan menganalisis bahasa natural Anda untuk menemukan kode klasifikasi.")
            
            default_val = st.session_state.pop('temp_search', '') 
            user_input = st.text_input("Ketik perihal surat:", value=default_val, placeholder="Contoh: penyusunan rencana kerja anggaran...", key="input_halaman_ai")

            if user_input:
                if user_input not in st.session_state.search_history:
                    st.session_state.search_history.append(user_input)
                    simpan_riwayat_csv(st.session_state['nama'], user_input)

                with st.spinner('AI sedang membedah dokumen Anda...'):
                    results = smart_classify(user_input, df)
                    if results:
                        st.success("✨ Analisis selesai! Berikut rekomendasi untuk dokumen Anda:")
                        for i, (idx, score) in enumerate(results):
                            res = df.iloc[idx]
                            with st.expander(f"🏅 Kode {res['kode']} (Keyakinan: {score:.1%})", expanded=(i==0)):
                                st.markdown(f"**Uraian:** {res['uraian'].title()}")
                                hierarki = get_hierarchy(res['kode'], df)
                                for h in hierarki: st.markdown(h, unsafe_allow_html=True)
                    else:
                        st.warning("Tidak ditemukan klasifikasi yang cocok.")

        # --- HALAMAN 3: JELAJAH KODE ---
        elif st.session_state.page == 'Jelajah Kode':
            st.markdown('<div class="section-title">📁 Jelajah Kode Klasifikasi</div>', unsafe_allow_html=True)
            st.write("Telusuri seluruh struktur hierarki klasifikasi arsip secara manual.")
            
            import re
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

        # --- HALAMAN 4: RIWAYAT ---
        elif st.session_state.page == 'Riwayat':
            st.markdown('<div class="section-title">🕒 Riwayat Pencarian Lengkap</div>', unsafe_allow_html=True)
            if not st.session_state.search_history:
                st.session_state.search_history = baca_riwayat_csv(st.session_state['nama'])
                
            if st.session_state.search_history:
                for riwayat in reversed(st.session_state.search_history):
                    st.markdown(f"🔹 {riwayat}")
                if st.button("Hapus Riwayat"):
                    st.session_state.search_history = []
                    st.rerun()
            else:
                st.info("Belum ada riwayat.")

        # --- HALAMAN 5: ADMIN PANEL ---
        elif st.session_state.page == 'Admin':
            st.markdown('<div class="section-title">⚙️ Panel Administrator</div>', unsafe_allow_html=True)
            st.warning("Area terbatas. Mengelola database pengguna dan log sistem.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")

# --- 5. PENGATUR HALAMAN (ROUTER) ---
if not st.session_state.get('logged_in', False):
    halaman_login()
else:
    halaman_utama()
