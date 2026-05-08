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

<div class="sikap-title">
SIKAP
</div>

<div class="sikap-subtitle">
Sistem Informasi Klasifikasi Arsip Pintar
</div>

</div>
""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([0.6, 2.8, 0.6])
    with col2:
        with st.form("form_login"):
            user_input = st.text_input("Username")
            pwd_input = st.text_input(
                "Password", 
                type="password", 
                label_visibility="visible"
            )
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
/* WRAPPER JUDUL */
/* ============================= */

.sikap-wrapper{
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    margin-top:20px;
    margin-bottom:40px;
}

/* ============================= */
/* TITLE */
/* ============================= */

.sikap-title {

    font-size: 6rem;

    font-weight: 900;

    line-height: 1;

    letter-spacing: 2px;

    background: linear-gradient(45deg, #00BFA5, #0288D1);

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;

    margin:0;
    padding:0;

    text-align:center;
}

.sikap-subtitle {

    font-size: 0.95rem;

    font-weight: 600;

    text-align:center;

    letter-spacing: 0.8px;

    margin-top: 2px;

    margin-left: 2px;

    background: linear-gradient(
        45deg,
        #FF512F,
        #DD2476
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    white-space: normal;
}

/* ============================= */
/* CARD LOGIN */
/* ============================= */

div[data-testid="stForm"] {

    background: rgba(255,255,255,0.04);

    padding: 35px;

    border-radius: 22px;

    border: 1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(8px);

    box-shadow:
        0 8px 32px rgba(0,0,0,0.15);

    margin-top: 10px;
}

/* ============================= */
/* INPUT */
/* ============================= */

.stTextInput input {

    border-radius: 16px !important;

    border: 1px solid rgba(255,255,255,0.12) !important;

    padding: 14px 16px !important;

    font-size: 1rem !important;

    background: rgba(255,255,255,0.04) !important;

    color: white !important;

    outline: none !important;

    box-shadow: none !important;

    transition: all .25s ease;
}

/* ============================= */
/* INPUT FOCUS */
/* ============================= */

.stTextInput input:focus {

    border: 1px solid #00BCD4 !important;

    outline: none !important;

    box-shadow:
        0 0 0 1px rgba(0,188,212,.25),
        0 0 12px rgba(0,188,212,.18) !important;
}

/* ============================= */
/* TOMBOL MASUK */
/* ============================= */

.stFormSubmitButton > button {

    width: 100%;

    border-radius: 14px !important;

    padding: 14px 20px !important;

    font-size: 1rem !important;

    font-weight: 700 !important;

    border: none !important;

    color: white !important;

    background:
        linear-gradient(
            135deg,
            #0288D1,
            #03A9F4
        ) !important;

    box-shadow:
        0 6px 18px rgba(2,136,209,.35),
        inset 0 1px 1px rgba(255,255,255,.25);

    transition: all .25s ease;
}

/* ============================= */
/* HOVER BUTTON */
/* ============================= */

.stFormSubmitButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 10px 24px rgba(2,136,209,.45),
        inset 0 1px 1px rgba(255,255,255,.3);

    background:
        linear-gradient(
            135deg,
            #039BE5,
            #29B6F6
        ) !important;
}

/* ============================= */
/* HERO BOX */
/* ============================= */

.hero-box{

    width:100%;

    padding:22px 28px;

    border-radius:24px;

    background:
        linear-gradient(
            135deg,
            rgba(0,191,165,0.12),
            rgba(2,136,209,0.12)
        );

    border:1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(10px);

    margin-bottom:25px;

    box-shadow:
        0 8px 24px rgba(0,0,0,0.12);

}

.hero-text{

    text-align:center;

    font-size:1.05rem;

    line-height:1.8;

    font-weight:500;

    color:inherit;

    letter-spacing:0.3px;

}

/* ============================= */
/* TAB MODERN */
/* ============================= */

.stTabs [data-baseweb="tab-list"]{

    gap:10px;

    margin-bottom:10px;
}

.stTabs [data-baseweb="tab"]{

    height:52px;

    padding:0 22px;

    border-radius:14px;

    background:rgba(255,255,255,0.04);

    transition:all .2s ease;

    font-weight:600;
}

.stTabs [aria-selected="true"]{

    background:
        linear-gradient(
            135deg,
            #0288D1,
            #00BFA5
        ) !important;

    color:white !important;

    box-shadow:
        0 6px 16px rgba(2,136,209,.35);
}

/* HILANGKAN GARIS MERAH/BIRU STREAMLIT */

div[data-baseweb="input"] {

    border: none !important;

    box-shadow: none !important;

    background: transparent !important;
}

div[data-baseweb="base-input"] {

    border: none !important;

    box-shadow: none !important;

    background: transparent !important;
}

[data-testid="stTextInput"] {

    border: none !important;

    box-shadow: none !important;
}

/* SAMAKAN LEBAR PASSWORD DENGAN USERNAME */

[data-testid="stTextInput"] {

    width: 100% !important;
}

[data-testid="stTextInput"] > div {

    width: 100% !important;
}

[data-testid="stTextInput"] input {

    width: 100% !important;
    min-width: 100% !important;
    box-sizing: border-box !important;
}

/* KHUSUS PASSWORD */

[data-testid="stTextInput"] div:has(input[type="password"]) input {

    padding-right: 48px !important;
}

/* FIX PASSWORD ICON MOBILE */

[data-testid="stTextInput"] div[data-baseweb="base-input"] {

    display: flex !important;
    align-items: center !important;
    overflow: hidden !important;

    border-radius: 16px !important;
}

/* ICON MATA */

[data-testid="stTextInput"] button {

    margin-right: 12px !important;

    background: transparent !important;

    border: none !important;

    box-shadow: none !important;
}

/* INPUT PASSWORD */

[data-testid="stTextInput"] input[type="password"] {

    padding-right: 12px !important;
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
    
# --- 4. ANTARMUKA UTAMA ---
def halaman_utama():
    st.markdown("""
<div class="sikap-wrapper">

<div class="sikap-title">
SIKAP
</div>

<div class="sikap-subtitle">
Sistem Informasi Klasifikasi Arsip Pintar
</div>

</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="hero-box">
    <div class="hero-text">
        🔍 Sistem klasifikasi arsip cerdas berbasis AI untuk membantu identifikasi kode arsip secara cepat, akurat, dan modern.
    </div>
</div>
""", unsafe_allow_html=True)

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
