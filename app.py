import streamlit as st
import pandas as pd
import re
import os
import plotly.express as px
import json
import io
import pytz
import threading
import time
import logging
# Membungkam peringatan (warning) gaib dari Google API saat pertama kali load
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from thefuzz import process, fuzz
from groq import Groq
from google import genai
from google.genai import types


# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="SIKAP - Klasifikasi Arsip Pintar", page_icon="🗂️", layout="wide", initial_sidebar_state="auto")

# ====================================================
# MESIN SINKRONISASI GOOGLE DRIVE (VERSI TAHAN BANTING)
# ====================================================
@st.cache_resource
def init_drive():
    key_dict = json.loads(st.secrets["gcp_service_account"])
    creds = service_account.Credentials.from_service_account_info(
        key_dict, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build('drive', 'v3', credentials=creds)

def get_drive_file_id(drive_service, file_name, folder_id):
    try:
        query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])
        if items:
            return items[0]['id']
        return None
    except Exception:
        # Jika koneksi ke Drive putus, diamkan saja
        return None

def sync_from_drive(file_name):
    try:
        drive_service = init_drive()
        folder_id = st.secrets["drive_folder_id"]
        file_id = get_drive_file_id(drive_service, file_name, folder_id)
        if file_id:
            request = drive_service.files().get_media(fileId=file_id)
            
            # JURUS ANTI-CORRUPT: Download ke memori (RAM) dulu
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                
            # Jika sukses 100% tanpa putus, baru timpa file aslinya!
            with open(file_name, 'wb') as f:
                f.write(buffer.getvalue())
    except Exception:
        # Jika gagal sedot, file CSV lokal yang asli tetap utuh & aman!
        pass

# --- ROBOT ASISTEN GAIB (BACKGROUND THREAD) ---
def retry_sync_gaib(file_name):
    time.sleep(5) # Robot diam-diam menunggu 5 detik di belakang layar
    try:
        drive_service = init_drive()
        folder_id = st.secrets["drive_folder_id"]
        file_id = get_drive_file_id(drive_service, file_name, folder_id)
        media = MediaFileUpload(file_name, mimetype='text/csv', resumable=True)
        if file_id:
            drive_service.files().update(fileId=file_id, media_body=media).execute()
        else:
            file_metadata = {'name': file_name, 'parents': [folder_id]}
            drive_service.files().create(body=file_metadata, media_body=media).execute()
    except Exception:
        pass # Kalau 5 detik kemudian MASIH putus, robot menyerah dalam diam.

# --- FUNGSI UPLOAD UTAMA ---
def sync_to_drive(file_name):
    if not os.path.exists(file_name):
        return
    try:
        drive_service = init_drive()
        folder_id = st.secrets["drive_folder_id"]
        file_id = get_drive_file_id(drive_service, file_name, folder_id)
        media = MediaFileUpload(file_name, mimetype='text/csv', resumable=True)
        if file_id:
            drive_service.files().update(fileId=file_id, media_body=media).execute()
        else:
            file_metadata = {'name': file_name, 'parents': [folder_id]}
            drive_service.files().create(body=file_metadata, media_body=media).execute()
    except Exception:
        # JIKA GAGAL: Lepaskan Robot Asisten Gaib untuk mencoba lagi 5 detik kemudian!
        threading.Thread(target=retry_sync_gaib, args=(file_name,), daemon=True).start()

# --- SEDOT DATA SAAT APLIKASI PERTAMA DIBUKA ---
def initial_sync():
    # Gunakan session_state agar hanya jalan 1x (Mencegah error berulang)
    if 'drive_synced' not in st.session_state:
        try:
            sync_from_drive('pengguna.csv')
            sync_from_drive('riwayat_pencarian.csv')
            sync_from_drive('feedback_ai.csv')
        except Exception:
            pass # Tangkap sisa error tak terduga
        finally:
            # Berhasil atau gagal, WAJIB stempel selesai!
            st.session_state['drive_synced'] = True

initial_sync()
# ====================================================
# --- INISIALISASI SESSION STATE LOGIN & HISTORY ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'role' not in st.session_state:
    st.session_state['role'] = None
if 'nama' not in st.session_state:
    st.session_state['nama'] = ""
if 'username' not in st.session_state:
    st.session_state['username'] = ""
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

# --- FUNGSI RIWAYAT PERMANEN (SOFT DELETE) ---
def simpan_riwayat_csv(nama_user, pencarian):
    file_riwayat = 'riwayat_pencarian.csv'
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # KUNCI: Kita tambahkan kolom baru bernama 'status_tampil'
    df_baru = pd.DataFrame({'waktu': [waktu], 'nama': [nama_user], 'pencarian': [pencarian], 'status_tampil': ['aktif']})
    
    if os.path.isfile(file_riwayat) and os.path.getsize(file_riwayat) > 0:
        try:
            df_lama = pd.read_csv(file_riwayat)
            df_final = pd.concat([df_lama, df_baru], ignore_index=True)
        except:
            df_final = df_baru
    else:
        df_final = df_baru
        
    df_final.to_csv(file_riwayat, index=False)
    sync_to_drive(file_riwayat)

def baca_riwayat_csv(nama_user):
    file_riwayat = 'riwayat_pencarian.csv'
    if os.path.isfile(file_riwayat):
        try:
            df_riwayat = pd.read_csv(file_riwayat)
            if 'nama' in df_riwayat.columns and 'pencarian' in df_riwayat.columns:
                # KUNCI: Filter data, JANGAN baca yang statusnya 'dihapus'
                if 'status_tampil' in df_riwayat.columns:
                    df_riwayat = df_riwayat[df_riwayat['status_tampil'] != 'dihapus']
                    
                riwayat_user = df_riwayat[df_riwayat['nama'] == nama_user]['pencarian'].tolist()
                return list(dict.fromkeys(riwayat_user)) # Hapus duplikat
        except:
            return []
    return []


# --- FUNGSI FEEDBACK PEMBELAJARAN AI (CSV) ---
def simpan_feedback_csv(nama_user, input_user, kode_terpilih):
    file_feedback = 'feedback_ai.csv'
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_baru = pd.DataFrame({'waktu': [waktu], 'nama': [nama_user], 'perihal': [input_user], 'kode_terpilih': [kode_terpilih]})
    
    # PERBAIKAN: Baca, Gabungkan, lalu Simpan (Anti Hilang Header)
    if os.path.isfile(file_feedback) and os.path.getsize(file_feedback) > 0:
        try:
            df_lama = pd.read_csv(file_feedback)
            df_final = pd.concat([df_lama, df_baru], ignore_index=True)
        except:
            df_final = df_baru
    else:
        df_final = df_baru
        
    df_final.to_csv(file_feedback, index=False)
    sync_to_drive(file_feedback)

# --- HALAMAN LOGIN ---
def halaman_login():
    # JURUS MEMBELAH LAYAR JADI 2 KOLOM
    # vertical_alignment="center" memastikan Kiri (Teks) dan Kanan (Form) sejajar sempurna di tengah!
    col_kiri, col_kanan = st.columns(2, gap="large", vertical_alignment="center")

    # KOLOM KIRI: Tempat Judul
    with col_kiri:
        st.markdown("""
        <div class="sikap-wrapper">
            <div class="sikap-title">SIKAP</div>
            <div class="sikap-subtitle">Sistem Informasi Klasifikasi Arsip Pintar</div>
        </div>
        """, unsafe_allow_html=True)

    # KOLOM KANAN: Tempat Form Login
    with col_kanan:
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

            # Cukup satu kali eksekusi submit (kode asli Bapak ada 2 kali tercopy)
            if submit:
                is_valid, role, nama = validasi_login(user_input, pwd_input)

                if is_valid:
                    st.session_state['logged_in'] = True
                    st.session_state['role'] = role
                    st.session_state['nama'] = nama
                    st.session_state['username'] = user_input
                    st.session_state.search_history = baca_riwayat_csv(nama)
                    st.rerun()
                else:
                    st.error("Username atau Password salah!")

# Groq (fallback)
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    api_key = "MASUKKAN_API_KEY_GROQ_DI_SINI_UNTUK_TES_LOKAL"
 
client = Groq(api_key=api_key)
 
# Gemini (primer)
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    client_gemini = genai.Client(api_key=gemini_api_key)
    GEMINI_TERSEDIA = True
except Exception:
    GEMINI_TERSEDIA = False

st.sidebar.caption(f"Gemini: {'✅' if GEMINI_TERSEDIA else '❌'}")


KATA_PENGANTAR = {
    "penyampaian", "permohonan", "undangan", "laporan", "tindak",
    "lanjut", "usulan", "hal", "mengenai", "draf", "rancangan",
    "penerbitan", "fasilitasi", "perihal", "rekomendasi", "sosialisasi",
    "pelaksanaan", "penyelenggaraan", "pembahasan", "pengajuan",
    "penyusunan", "pembuatan", "pemberitahuan", "pelaporan",
    "penetapan", "pengiriman", "pengesahan", "permintaan", "data",
    "hasil", "tindaklanjut", "konfirmasi", "koordinasi", "pemantauan",
}
 
# Nilai valid atribut jenjang per rumpun — hasil validasi Gemini
REFERENSI_JENJANG = {
    "umum": [
        "kepala daerah", "dprd", "pegawai umum",
        "nasional", "provinsi", "kabupaten", "kecamatan", "desa"
    ],
    "pemerintahan": [
        "gubernur", "bupati", "sekda provinsi", "sekda kabupaten"
    ],
    "politik": [
        "dpr pusat", "dprd provinsi", "dprd kabupaten", "dprd kota"
    ],
    "keamanan": ["harian", "bulanan", "tahunan"],
    "kesejahteraan": [
        "paud", "sd", "smp", "sma", "smk", "perguruan tinggi",
        "daerah", "nasional", "internasional",
        "rumah sakit", "puskesmas"
    ],
    "perekonomian": [
        "ikm", "industri kecil menengah", "industri besar",
        "dalam negeri", "luar negeri", "ekspor"
    ],
    "pekerjaan umum": [
        "nasional", "provinsi", "kabupaten", "kota", "desa"
    ],
    "pengawasan": [
        "kementerian", "lembaga pusat", "pemerintah provinsi",
        "pemerintah kabupaten", "pemerintah kota", "kelurahan", "desa"
    ],
    "kepegawaian": [
        "cpns", "pns", "pppk", "honorer",
        "golongan i", "golongan ii", "golongan iii", "golongan iv",
        "eselon i", "eselon ii", "eselon iii", "eselon iv",
        "jabatan fungsional"
    ],
    "keuangan": [
        "apbn", "apbd provinsi", "apbd kabupaten", "apbd kota", "apbdes"
    ],
}
 
 
def _fallback_ekstraksi_manual(teks):
    """
    Lapis ke-3: ekstraksi Python murni, tidak pakai LLM, tidak bisa gagal.
    Mengembalikan (inti_string, atribut_dict) dengan nilai default.
    """
    teks_bersih = re.sub(r'[^a-zA-Z0-9\s(),/-]', ' ', teks.lower()).strip()
    kata = teks_bersih.split()
    while kata and kata[0] in KATA_PENGANTAR:
        kata.pop(0)
    inti = " ".join(kata[:10]) if kata else teks.lower()[:80]
    atribut_default = {
        "konteks": "fasilitatif",
        "domain": "umum",
        "objek": inti,
        "jenjang": "",
        "kegiatan": "pelaksanaan",
        "produk": "laporan",
        "inti": inti,
        "_model": "python-manual",
    }
    return inti, atribut_default
 
 
def _validasi_json_atribut(data: dict) -> bool:
    """
    Validasi output JSON dari LLM.
    Cek kelengkapan kunci dan nilai 'inti' yang bermakna.
    """
    kunci_wajib = {"konteks", "domain", "objek", "jenjang",
                   "kegiatan", "produk", "inti"}
    if not kunci_wajib.issubset(data.keys()):
        return False
    inti = str(data.get("inti", "")).strip()
    if len(inti) < 3 or len(inti) > 200:
        return False
    penanda_gagal = [
        "maaf", "saya tidak", "i cannot", "i don't", "i'm sorry",
        "as an ai", "note:", "catatan:", "semoga membantu",
    ]
    if any(p in inti.lower() for p in penanda_gagal):
        return False
        
    # PERBAIKAN CLAUDE: Validasi Konsistensi
    konteks = str(data.get("konteks", "")).lower().strip()
    domain = str(data.get("domain", "")).lower().strip()
    if konteks == "substantif" and domain == "umum":
        return False # Tolak dan paksa turun ke fallback
        
    return True
 
 
def _parse_json_atribut(raw: str) -> dict | None:
    """
    Parse JSON dari output LLM.
    Toleran terhadap pembungkus markdown (```json ... ```).
    """
    try:
        bersih = re.sub(r'```(?:json)?', '', raw).strip().rstrip('`').strip()
        awal = bersih.find('{')
        akhir = bersih.rfind('}')
        if awal == -1 or akhir == -1:
            return None
        return json.loads(bersih[awal:akhir + 1])
    except Exception:
        return None
 
 
# =========================================================
# OTAK 1: PROMPT UNTUK QWEN-32B (KAPASITAS ANALISIS TINGGI)
# =========================================================
def _bangun_prompt_qwen(teks_user: str) -> str:
    referensi_str = "\n".join(
        f"  - {rumpun}: {', '.join(nilai)}"
        for rumpun, nilai in REFERENSI_JENJANG.items()
    )
    return f"""Anda adalah Arsiparis Senior ahli kearsipan pemerintahan daerah Indonesia.
Tugas: Analisis perihal surat berikut dan ekstrak 6 atribut terstruktur.
 
LANGKAH BERPIKIR (jalankan berurutan, wajib):
1. KONTEKS: Apakah surat ini tentang kerumahtanggaan/administrasi internal kantor (fasilitatif) ATAU pelayanan teknis ke masyarakat/program daerah (substantif)?
2. DOMAIN: Bidang fungsi pemerintahan utama (umum/pemerintahan/kepegawaian/keuangan/kesejahteraan/perekonomian/pekerjaan umum/pengawasan/keamanan/politik)?
3. OBJEK: Apa yang menjadi target/sasaran utama? (bukan aktor, bukan lokasi)
4. JENJANG: Apakah ada dimensi hierarki yang disebutkan atau tersirat? Gunakan referensi:
{referensi_str}
   Jika tidak ada jenjang yang jelas, isi string kosong "".
5. KEGIATAN: Jenis tindakan (perencanaan/pelaksanaan/pelaporan/pengawasan/evaluasi/koordinasi/pengadaan/penetapan)?
6. PRODUK: Output/tujuan surat (sk/laporan/persetujuan/undangan/instruksi/perjanjian/sppd/rekomendasi)?
7. INTI: Gabungkan objek + jenjang (jika ada) + kegiatan menjadi frasa pencarian singkat maksimal 15 kata. Harus cukup spesifik untuk membedakan kode arsip yang mirip.
 
PRINSIP KRITIS:
- Jika objek spesifik teridentifikasi → ABAIKAN rumpun umum meski kata kuncinya cocok.
  Contoh: "perizinan pertanian" bukan 100.3.12 (hukum umum), tapi masuk 500.6 (perekonomian pertanian).
- Fasilitatif = Urusan administrasi, kerumahtanggaan, kepegawaian, dan KEUANGAN (APBD/APBN/Aset) internal pemerintah.
- Substantif = Urusan teknis, layanan masyarakat, batas wilayah pemerintahan, pembangunan fisik (jalan/bangunan), dan program sektoral dinas.
- PENTING: Urusan "batas wilayah", "pemekaran", atau "otonomi" selalu berstatus SUBSTANTIF dengan domain PEMERINTAHAN, bukan umum.
- Kata "penelitian" yang disertai objek spesifik (batuan, kelautan, dll) = substantif, bukan umum.
- AWAS JEBAKAN "DALAM RANGKA": Jika ada pola "[Aksi] dalam rangka / untuk / guna [Tujuan]", OBJEK UTAMANYA adalah [Tujuan]. Abaikan aksinya.
- BUANG KONSIDERAN HUKUM: Hapus mutlak semua frasa dasar hukum seperti "sebagaimana amanat", "Undang-Undang", "Peraturan Menteri", "Nomor...", dan "Tahun...". Fokus HANYA pada substansi kegiatannya (misal: "Survei Kepuasan Masyarakat", "Pelayanan Publik").
 
CONTOH 1:
Input: "Perjalanan dinas Bupati ke Jakarta konsultasi APBD"
Output: {{"konteks":"fasilitatif","domain":"umum","objek":"perjalanan dinas","jenjang":"kepala daerah","kegiatan":"pelaksanaan","produk":"sppd","inti":"perjalanan dinas kepala daerah dalam negeri"}}
 
CONTOH 2:
Input: "Laporan pertanggungjawaban penggunaan dana BOS SMPN 1 Tikep"
Output: {{"konteks":"substantif","domain":"kesejahteraan","objek":"bantuan operasional sekolah","jenjang":"smp","kegiatan":"pelaporan","produk":"laporan","inti":"bantuan operasional sekolah bos smp pelaporan"}}
 
CONTOH 3:
Input: "Izin penelitian sampel batuan tambang dari Universitas Halu Oleo"
Output: {{"konteks":"substantif","domain":"perekonomian","objek":"penelitian kegeologian batuan","jenjang":"","kegiatan":"pelaksanaan","produk":"rekomendasi","inti":"izin penelitian batuan tambang kegeologian"}}
 
CONTOH 4:
Input: "Kenaikan pangkat PNS golongan III ke golongan IV"
Output: {{"konteks":"fasilitatif","domain":"kepegawaian","objek":"kenaikan pangkat","jenjang":"golongan iv","kegiatan":"penetapan","produk":"sk","inti":"kenaikan pangkat pns golongan iv"}}
 
CONTOH 5:
Input: "Rekonsiliasi BMD dan tindak lanjut temuan BPK"
Output: {{"konteks":"fasilitatif","domain":"keuangan","objek":"barang milik daerah","jenjang":"","kegiatan":"pengawasan","produk":"laporan","inti":"rekonsiliasi barang milik daerah tindak lanjut temuan bpk"}}

CONTOH 6:
Input: "Penetapan batas wilayah administrasi antar kecamatan"
Output: {{"konteks":"substantif","domain":"pemerintahan","objek":"batas wilayah administrasi","jenjang":"kecamatan","kegiatan":"penetapan","produk":"sk","inti":"penetapan batas wilayah administrasi kecamatan"}}

CONTOH 7:
Input: "Perubahan struktur organisasi dan tata kerja perangkat daerah"
Output: {{"konteks":"substantif","domain":"pemerintahan","objek":"organisasi perangkat daerah","jenjang":"kabupaten","kegiatan":"penetapan","produk":"sk","inti":"struktur organisasi tata kerja perangkat daerah"}}

CONTOH 8:
Input: "Pembayaran tagihan biaya pengadaan komputer dan kursi kantor"
Output: {{"konteks":"fasilitatif","domain":"umum","objek":"komputer dan kursi kantor","jenjang":"","kegiatan":"pengadaan","produk":"surat","inti":"pengadaan komputer kursi kantor"}}
 
SEKARANG KERJAKAN:
Input: "{teks_user}"
Keluarkan hasil murni dalam format JSON. Jangan tulis tag <think>, jangan beri penjelasan, jangan tambahkan markdown ```json. HANYA format JSON valid yang diawali dengan {{ dan diakhiri dengan }}.
"""


# =========================================================
# OTAK 2: PROMPT UNTUK LLAMA-8B (MODE EKSTRAKTOR MURNI)
# =========================================================
def _bangun_prompt_llama(teks_user: str) -> str:
    referensi_str = "\n".join(
        f"  - {rumpun}: {', '.join(nilai)}"
        for rumpun, nilai in REFERENSI_JENJANG.items()
    )
    
    return f"""Anda adalah asisten ekstraksi teks. Tugas Anda HANYA membersihkan teks dan menemukan subjek utamanya.

ATURAN WAJIB (WAJIB DIPATUHI):
1. Abaikan kata-kata transaksi berikut (jangan jadikan objek):
   biaya, pembayaran, pencairan, pengadaan, pembelian, pemeliharaan, honor, dana, anggaran, termin, permintaan data, undangan.
2. Temukan BENDA FISIK atau SUBJEK UTAMA dari kalimat tersebut.
3. AWAS JEBAKAN "DALAM RANGKA": Jika ada kata "dalam rangka", "untuk", atau "guna", ambil subjek SETELAH kata tersebut, abaikan kata di depannya.
4. BUANG KONSIDERAN HUKUM: Hapus mutlak semua kata yang merujuk pada aturan hukum (Undang-Undang, Peraturan Menteri, Nomor, Tahun, sebagaimana amanat). Jangan pernah jadikan ini sebagai objek/inti!
5. Untuk field "domain", SELALU isi dengan "tidak_diketahui".

URUTAN CARA BERPIKIR (Terapkan urutan ini di dalam JSON):
1. Cari tahu apa Benda Fisik / Subjek utamanya. Tulis di field "objek".
2. Masukkan isi field objek tersebut ke dalam field "inti".

REFERENSI JENJANG:
{referensi_str}

ATURAN OUTPUT JSON:
Keluarkan HANYA JSON yang valid dengan urutan key PERSIS seperti di bawah ini, tanpa teks pengantar, markdown, atau penjelasan apa pun:

{{
  "konteks": "<isi dengan fasilitatif atau substantif>",
  "objek": "<TULIS BENDA FISIK/SUBJEKNYA DI SINI. Jangan masukkan kata transaksi uang/biaya>",
  "inti": "<WAJIB tulis ulang isi dari field 'objek' di sini, lalu tambahkan kata kegiatannya (maksimal 8 kata)>",
  "domain": "tidak_diketahui",
  "kegiatan": "<aksi/proses utama seperti pengadaan, pelaporan, penetapan>",
  "jenjang": "<cocokkan dengan Referensi Jenjang di atas, atau kosongkan jika tidak ada>",
  "produk": "<jenis dokumen output seperti laporan, surat, kuitansi, sk>"
}}

Kalimat yang harus diekstrak:
"{teks_user}"
"""


def _panggil_qwen3(prompt: str, max_retries: int = 3) -> str | None:
    for percobaan in range(max_retries):
        try:
            chat = client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[
                    {"role": "system", "content": "Anda adalah asisten arsiparis. Berikan analisis dan keluarkan hasil akhirnya dalam bentuk block markdown JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                top_p=0.95,
                max_completion_tokens=2048,
            )
            return chat.choices[0].message.content.strip()

        except Exception as e:
            import streamlit as st
            st.error(f"Peringatan Sistem: Gagal memanggil Qwen (Percobaan {percobaan+1}). Detail: {e}")
            
            pesan = str(e).lower()
            if any(k in pesan for k in ["429", "503", "rate", "quota", "overloaded"]):
                if percobaan < max_retries - 1:
                    jeda = (2 ** percobaan) * 2
                    time.sleep(jeda)
                    continue
            break
    return None


def _panggil_llama_ekstraksi(prompt_6_atribut: str) -> str | None:
    """
    Fallback Llama Cadangan PINTAR.
    """
    try:
        chat = client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[
                {"role": "system", "content": "Anda adalah asisten arsiparis. Keluarkan hasil murni dalam bentuk JSON."},
                {"role": "user", "content": prompt_6_atribut}
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return chat.choices[0].message.content.strip()
    except Exception as e:
        import streamlit as st
        st.warning(f"Fallback Llama juga gagal: {e}")
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def ekstrak_inti_surat(teks_user: str) -> tuple[str, dict]:
    """
    Mengekstrak 6 atribut terstruktur dari perihal surat.
    Hierarki fallback:
      1. Qwen3 32B   → Prompt orisinal (reasoning tinggi, analisa substantif)
      2. Llama 8B    → Prompt hukum universal (mencegah bias semantik)
      3. Python murni → tidak bisa gagal
    """
    # SIAPKAN DUA OTAK BERBEDA
    prompt_qwen = _bangun_prompt_qwen(teks_user)
    prompt_llama = _bangun_prompt_llama(teks_user)
 
    # === LAPIS 1: Qwen3 32B ===
    raw_qwen = _panggil_qwen3(prompt_qwen) # Qwen membaca prompt orisinal
    if raw_qwen:
        data = _parse_json_atribut(raw_qwen)
        if data and _validasi_json_atribut(data):
            data["_model"] = "qwen3-32b"
            inti = str(data["inti"]).strip().lower()
            return inti, data
 
    # === LAPIS 2: Llama Cadangan Pintar ===
    raw_llama = _panggil_llama_ekstraksi(prompt_llama) # Llama membaca hukum universal
    if raw_llama:
        data_llama = _parse_json_atribut(raw_llama)
        if data_llama and _validasi_json_atribut(data_llama):
            data_llama["_model"] = "llama-8b-fallback"
            inti = str(data_llama["inti"]).strip().lower()
            return inti, data_llama
 
    # === LAPIS 3: Python manual ===
    return _fallback_ekstraksi_manual(teks_user)



        
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
    /* Background Utama & Kartu */
    --bg-app: #F8FAFC; 
    --card-bg: #FFFFFF;
    --card-border: #E2E8F0;
    --card-shadow: 0 4px 6px rgba(0,0,0,0.02);
    
    /* Teks */
    --text-title: #0F172A;
    --text-subtitle: #475569;
    --text-muted: #64748B;
    
    /* Input & Expander */
    --input-bg: #F8FAFC; 
    --input-border: rgba(0, 157, 255, 0.4); 
    --expander-bg: #F8FAFC;
    
    /* Komponen Spesifik */
    --hero-bg: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
    --hero-border: #BAE6FD;
    --sidebar-bg: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
    --sidebar-hover: rgba(255, 255, 255, 0.15);
}

/* ============================= */
/* VARIABEL TEMA GELAP (OTOMATIS) */
/* ============================= */
@media (prefers-color-scheme: dark) {
    :root {
        /* Background Utama & Kartu */
        --bg-app: #0F172A; /* Biru sangat gelap */
        --card-bg: #1E293B;
        --card-border: #334155;
        --card-shadow: 0 10px 25px rgba(0,0,0,0.5);
        
        /* Teks */
        --text-title: #F8FAFC;
        --text-subtitle: #94A3B8;
        --text-muted: #CBD5E1;
        
        /* Input & Expander */
        --input-bg: #0F172A; 
        --input-border: rgba(120, 180, 255, 0.4); 
        --expander-bg: #1E293B;
        
        /* Komponen Spesifik */
        --hero-bg: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        --hero-border: #334155;
        --sidebar-bg: linear-gradient(135deg, #020617 0%, #0F172A 100%);
        --sidebar-hover: rgba(255, 255, 255, 0.05);
    }
}

/* Terapkan ke Background Utama Streamlit + Jurus Tirai Fade-In */
.stApp {
    background: var(--bg-app) !important;
    animation: smoothLoad 0.8s ease-in-out forwards !important; /* <--- Animasi masuk yang elegan */
}

/* Kunci Animasi Memudar (Fade In) */
@keyframes smoothLoad {
    0% { opacity: 0; }
    100% { opacity: 1; }
}

/* JURUS SAPU BERSIH: Sembunyikan kotak merah/kuning bawaan Streamlit di detik pertama */
[data-testid="stException"], 
[data-testid="stAlert"] {
    animation: hideError 1s forwards !important;
}

@keyframes hideError {
    0% { display: none !important; opacity: 0; }
    99% { display: none !important; opacity: 0; }
    100% { display: block; opacity: 1; }
}

/* ============================= */
/* TITLE & WRAPPER SIKAP */
/* ============================= */
.sikap-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center !important; /* KUNCI: Memusatkan elemen di dalam kolom */
    justify-content: center;
    margin-top: 0vh;
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
    text-align: center !important; /* KUNCI: Teks rata tengah */
    filter: drop-shadow(0 0 15px rgba(0,194,255,0.2));
    width: 100%;
}

.sikap-subtitle {
    font-size: clamp(0.85rem, 3vw, 1.15rem) !important;
    font-weight: 700 !important;
    color: var(--text-subtitle);
    text-align: center !important; /* KUNCI: Teks rata tengah */
    margin-top: 5px;
    padding-bottom: 20px; 
    position: relative;
    width: 100%;
}

/* Garis Biru Futuristik - Kembali memudar dari tengah ke pinggir */
.sikap-subtitle::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 50% !important; /* KUNCI: Balik ke tengah */
    transform: translateX(-50%) !important;
    width: 100%; 
    height: 1.5px;
    background: linear-gradient(90deg, transparent, #009DFF, transparent) !important; 
}

/* Titik Cahaya - Kembali persis di tengah garis */
.sikap-subtitle::before {
    content: "";
    position: absolute;
    bottom: -1.5px; 
    left: 50% !important; /* KUNCI: Balik ke tengah */
    transform: translateX(-50%) !important;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #00C2FF;
    box-shadow: 0 0 10px 2px rgba(0, 194, 255, 0.8);
    z-index: 1;
}

/* ========================================================= */
/* JURUS PELINDUNG HP: KEMBALIKAN SEMUANYA KE TENGAH!        */
/* ========================================================= */
@media screen and (max-width: 768px) {
    .sikap-wrapper { 
        align-items: center !important; /* Kembalikan ke tengah */
        margin-top: 6vh !important; 
        padding: 0 15px !important;
    }
    .sikap-title, .sikap-subtitle { 
        text-align: center !important; /* Teks kembali rata tengah */
    }
    /* Garis memudar dari kiri-tengah-kanan */
    .sikap-subtitle::after { 
        background: linear-gradient(90deg, transparent, #009DFF, transparent) !important; 
    }
    /* Titik nyala kembali ke tengah bawah */
    .sikap-subtitle::before { 
        left: 50% !important; 
        transform: translateX(-50%) !important; 
    }
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
/* INPUT BOX (VERSI ANTI HILANG) */
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

/* Kotak Input Teksnya Sendiri */
input[type="text"], input[type="password"] {
    height: 56px !important; 
    padding: 15px 45px 15px 55px !important; 
    line-height: 1.2 !important; 
    color: var(--text-title) !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    background-color: transparent !important; /* Wajib transparan agar tembus pandang */
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

/* ======================================================= */
/* JURUS PAMUNGKAS: ANTI HILANG & ANTI AUTOFILL CHROME     */
/* ======================================================= */

/* 1. Gambar ditanam di Tembok Belakang (Wadah Luar), bukan di Kotak Inputnya */
div[data-baseweb="input"]:has(input[placeholder="Masukkan username Anda"]) {
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23009DFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>') !important;
    background-repeat: no-repeat !important;
    background-position: 18px center !important; 
    background-size: 20px !important;
}

div[data-baseweb="input"]:has(input[placeholder="Masukkan password Anda"]) {
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23009DFF" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>') !important;
    background-repeat: no-repeat !important;
    background-position: 18px center !important; 
    background-size: 20px !important;
}

/* 2. JURUS SULAP: Tahan warna putih Autofill Chrome selama 5000 detik! */
input:-webkit-autofill,
input:-webkit-autofill:hover, 
input:-webkit-autofill:focus, 
input:-webkit-autofill:active {
    -webkit-text-fill-color: var(--text-title) !important;
    transition: background-color 5000s ease-in-out 0s !important;
}

/* ======================================================= */

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
    "sd": "sekolah dasar",
    "smp": "sekolah menengah pertama",
    "sma": "sekolah menengah atas",
    "smk": "sekolah menengah kejuruan",
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
    "ppm": "program pengembangan dan pemberdayaan masyarakat",
    "sirup": "sistem informasi rencana umum pengadaan",
    "rup": "rencana umum pengadaan",
}

# --- FUNGSI PENERJEMAH SINGKATAN ---
def terjemahkan_singkatan(text):
    kata_kata = str(text).lower().split()
    kata_terjemahan = [kamus_birokrasi.get(kata, kata) for kata in kata_kata]
    return " ".join(kata_terjemahan)

# --- FUNGSI PEMBERSIH UTAMA ---
def preprocess_text(text):
    text = str(text).lower()
    # KUNCI: Bersihkan tanda baca DULU agar kata singkatan berdiri telanjang!
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Baru masukkan ke mesin terjemahan
    text = terjemahkan_singkatan(text)
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
    
    # --- FUNGSI PEMBUAT BADGE UNTUK TAB 1 (PENCARIAN AI - PC & HP AMAN) ---
def get_badge_html(kode, uraian, level):
    levels_name = ["Primer", "Sekunder", "Tersier", "Kuartier", "Kuintier"]
    label = levels_name[level] if level < len(levels_name) else f"Level {level+1}"
    
    warna_level = ["#EF4444", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]
    warna_bg = warna_level[level] if level < len(warna_level) else "#424242"
    
    # Margin untuk Laptop (Aman)
    indent = f"clamp(0px, {level * 5}vw, {level * 30}px)" 
    
    # =======================================================================
    # JURUS GAIB: CSS INI HANYA AKTIF DI HP (LAPTOP TIDAK DISENTUH SAMA SEKALI)
    # =======================================================================
    css_hp = """
    <style>
    @media screen and (max-width: 768px) {
        .wadah-kapsul {
            /* Di HP, margin kiri kita pangkas agar tidak mendorong kotak sampai habis layar */
            margin-left: calc(var(--level) * 15px) !important;
        }
        .kapsul-pintar {
            /* Di HP, bentuk kapsul berubah jadi kotak biasa agar teks lega dan tidak memanjang ke bawah */
            border-radius: 12px !important; 
            padding: 10px 15px !important;
            display: flex !important;
            flex-direction: column !important; /* Trik: Kode di atas, teks turun ke bawah */
            width: 100% !important; 
        }
        .kapsul-pintar > div:first-child {
            margin-bottom: 5px !important; /* Jarak antara baris kode dan teks uraian */
        }
    }
    </style>
    """
    
    # KODE UTAMA (Sesuai kode aslimu yang sudah bagus di laptop, ditambah class pemanggil)
    return f"{css_hp}" \
           f"<div class='wadah-kapsul' style='--level: {level}; margin-left: {indent}; margin-bottom: 8px; display: flex;'>" \
           f"<div class='kapsul-pintar' style='background-color: {warna_bg}; color: #ffffff; padding: 6px 18px; border-radius: 50px; font-weight: normal; font-size: 0.9rem !important; display: inline-flex; align-items: flex-start; width: fit-content; max-width: 100%; box-shadow: 0px 3px 6px rgba(0,0,0,0.15);'>" \
           f"<div style='display: flex; align-items: flex-start; flex-shrink: 0; margin-right: 8px;'>" \
           f"<span class='material-symbols-rounded' style='font-size: 1.15rem; margin-right: 4px; margin-top: 2px;'>folder</span>" \
           f"<strong style='margin-top: 2px;'>{kode}</strong>" \
           f"<span style='margin: 2px 4px 0 4px; opacity: 0.6;'>|</span>" \
           f"</div>" \
           f"<div style='text-align: left; line-height: 1.4; margin-top: 2px; word-wrap: break-word; overflow-wrap: break-word;'>" \
           f"{uraian} <i style='opacity: 0.8; margin-left: 4px; display: inline-block; font-size: 0.8rem;'>({label})</i>" \
           f"</div>" \
           f"</div>" \
           f"</div>"

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
@st.cache_data(show_spinner=False, ttl=3600)
def smart_classify(user_input, df, top_n=3):
 
    # 1. Ekstraksi 6 atribut
    inti_dari_llm, atribut_6 = ekstrak_inti_surat(user_input)
 
    # Debug — hapus setelah selesai testing
    st.caption(
        f"🧠 Inti: **{inti_dari_llm}** &nbsp;|&nbsp; "
        f"Model: `{atribut_6.get('_model', '?')}`"
    )
    with st.expander("🔍 Detail 6 Atribut", expanded=False):
        st.json({k: v for k, v in atribut_6.items() if not k.startswith('_')})
 
    # 2. Preprocessing teks
    input_bersih = preprocess_text(inti_dari_llm)
 
    # 2b. Filter rumpun berdasarkan domain
    PETA_DOMAIN_RUMPUN = {
        "umum":           "000",
        "pemerintahan":   "100",
        "politik":        "200",
        "keamanan":       "300",
        "kesejahteraan":  "400",
        "perekonomian":   "500",
        "pekerjaan umum": "600",
        "pengawasan":     "700",
        "kepegawaian":    "800",
        "keuangan":       "900",
    }
 
    domain_terdeteksi = str(atribut_6.get("domain", "")).lower().strip()
    model_dipakai     = str(atribut_6.get("_model", "")).lower()
    rumpun_target     = PETA_DOMAIN_RUMPUN.get(domain_terdeteksi)
 
    # =======================================================
    # PENDEKATAN K.I.S.S (KEEP IT SIMPLE, STUPID)
    # Hanya percaya filter domain jika modelnya Qwen-32B.
    # Jika Llama-8B atau Python, filter MATI. Biarkan TF-IDF mencari bebas!
    # =======================================================
    FILTER_AKTIF = (
        rumpun_target is not None
        and "qwen" in model_dipakai 
    )
 
    if FILTER_AKTIF:
        df_subset = df[df['kode'].str.startswith(rumpun_target)].copy()
        # Jika hasil filter terlalu sedikit, batalkan filter (Safety Net)
        if len(df_subset) < 20:
            df_subset    = df.copy()
            FILTER_AKTIF = False
    else:
        df_subset = df.copy()

    # Simpan indeks asli df sebelum reset
    df_subset = df_subset.reset_index(drop=False)
 
    # Debug filter
    if FILTER_AKTIF:
        st.caption(
            f"🎯 Filter: rumpun **{rumpun_target}** "
            f"({len(df_subset)} dari {len(df)} baris)"
        )
    else:
        st.caption(f"🔍 Filter nonaktif — {len(df)} baris")
 
    # 3. TF-IDF & Fuzzy + LANGKAH 5: Depth Bonus Kondisional
    vectorizer    = TfidfVectorizer(ngram_range=(1, 3))
    semua_dokumen = df_subset['clean_uraian'].tolist() + [input_bersih]
    matriks_tfidf = vectorizer.fit_transform(semua_dokumen)
 
    kemiripan_kosinus = cosine_similarity(
        matriks_tfidf[-1], matriks_tfidf[:-1]
    )[0]
 
    skor_awal = []
    for indeks, nilai_skor in enumerate(kemiripan_kosinus):
        skor_samar = fuzz.token_set_ratio(
            input_bersih, df_subset.iloc[indeks]['clean_uraian']
        ) / 100
 
        # Hitung skor dasar dulu
        skor_dasar = (nilai_skor * 0.70) + (skor_samar * 0.30)
 
        # LANGKAH 5: Bonus kedalaman HANYA jika skor dasar sudah relevan
        kode_item    = str(df_subset.iloc[indeks]['kode'])
        jumlah_titik = kode_item.count('.')
        if skor_dasar >= 0.15:
            bonus_kedalaman = jumlah_titik * 0.01
        else:
            bonus_kedalaman = 0
 
        skor_gabungan = skor_dasar + bonus_kedalaman
 
        idx_asli = df_subset.iloc[indeks]['index']
        skor_awal.append({'idx': idx_asli, 'skor': skor_gabungan})
 
    dua_puluh_kandidat_teratas = sorted(
        skor_awal, key=lambda x: x['skor'], reverse=True
    )[:20]
 
    # LANGKAH 6: Smart Routing — bypass jury jika kandidat #1 sudah dominan
    skor_pertama = dua_puluh_kandidat_teratas[0]['skor']
    skor_kedua   = dua_puluh_kandidat_teratas[1]['skor'] if len(dua_puluh_kandidat_teratas) > 1 else 0
    selisih_skor = skor_pertama - skor_kedua
 
    if skor_pertama >= 0.80 and selisih_skor >= 0.15:
        # Fast track: Llama tidak dipanggil, langsung kembalikan top-3
        print(f"[SIKAP] Smart Routing aktif — skor={skor_pertama:.3f}, selisih={selisih_skor:.3f}")
        st.caption("⚡ Fast track")
        hasil_fast = []
        for item in dua_puluh_kandidat_teratas[:top_n]:
            skor_sim = 0.99 - (len(hasil_fast) * 0.14)
            hasil_fast.append((item['idx'], skor_sim))
        return hasil_fast, inti_dari_llm
 
    # 4. Juri AI — hanya dipanggil jika smart routing tidak aktif
    kandidat_untuk_juri = dua_puluh_kandidat_teratas[:20] # KITA KEMBALIKAN KE 20 AGAR DIA BISA MELIHAT DANA BOS

    daftar_kandidat = ""
    for urutan, item in enumerate(kandidat_untuk_juri):
        baris_data = df.iloc[item['idx']]
        daftar_kandidat += (
            f"[OPSI {urutan+1}] Kode: {baris_data['kode']} | "
            f"Konteks Hierarki: {baris_data['uraian_lengkap'].title()}\n"
        )

    perintah_juri = f"""Anda adalah Arsiparis Senior.
Tugas: Pilih 3 nomor urut OPSI yang paling tepat untuk urusan berikut.

URUSAN SURAT: "{inti_dari_llm}"

DAFTAR KANDIDAT:
{daftar_kandidat}

LANGKAH BERPIKIR (Wajib diisi):
- Domain urusan: [isi]
- Kode Anak terdalam yang relevan: [isi]
- Alasan menolak kode induk: [isi]

ATURAN MUTLAK:
1. DILARANG KERAS memilih kode induk jika ada kode anaknya yang relevan (Kuartier/Tersier diutamakan).
2. Tuliskan NOMOR URUT OPSINYA saja (1-20), bukan kode klasifikasinya!
3. Tulis hasil akhir PERSIS seperti format ini:
HASIL AKHIR: OPSI X, OPSI Y, OPSI Z"""

    try:
        penyelesaian_obrolan = client.chat.completions.create(
            messages=[{"role": "user", "content": perintah_juri}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
        )
        balasan_juri = penyelesaian_obrolan.choices[0].message.content.strip()

        # ========================================================
        # CCTV UI: Intip jalan pikiran Llama langsung di layar web!
        with st.expander("🧠 Intip Cara Berpikir Juri AI (Chain-of-Thought)", expanded=False):
            st.info("Berikut adalah rekaman proses analisis Juri Llama dalam menyeleksi 20 kandidat:")
            st.text(balasan_juri)
        # ========================================================

        # Penangkap Angka Anti-Meleset
        angka_pilihan = []
        for baris in balasan_juri.split('\n'):
            if 'HASIL AKHIR' in baris.upper():
                # Wajib menangkap angka HANYA setelah kata OPSI
                angka_mentah = re.findall(r'OPSI\s*(\d+)', baris.upper())
                
                # Jika Llama ngeyel tidak pakai kata OPSI, tangkap angka biasa
                if not angka_mentah:
                    angka_mentah = re.findall(r'\d+', baris)
                    
                for angka in angka_mentah:
                    angka_bulat = int(angka)
                    if 1 <= angka_bulat <= len(kandidat_untuk_juri) and angka_bulat not in angka_pilihan:
                        angka_pilihan.append(angka_bulat)
                    if len(angka_pilihan) == 3:
                        break
                break

        # Fallback jika baris HASIL AKHIR gaib
        if not angka_pilihan:
            for angka in re.findall(r'\d+', balasan_juri):
                angka_bulat = int(angka)
                if 1 <= angka_bulat <= len(kandidat_untuk_juri) and angka_bulat not in angka_pilihan:
                    angka_pilihan.append(angka_bulat)
                if len(angka_pilihan) == 3:
                    break

        hasil_akhir = []
        for nomor in angka_pilihan:
            indeks_kandidat = nomor - 1
            if 0 <= indeks_kandidat < len(kandidat_untuk_juri):
                skor_simulasi = 0.99 - (len(hasil_akhir) * 0.14)
                hasil_akhir.append(
                    (kandidat_untuk_juri[indeks_kandidat]['idx'], skor_simulasi)
                )

        if hasil_akhir:
            return hasil_akhir, inti_dari_llm

    except Exception as e:
        st.error(f"🚨 KESALAHAN SISTEM (Tahap Juri Penilai): {e}")

    # Fallback murni jika Juri error total
    return [
        (item['idx'], item['skor'])
        for item in kandidat_untuk_juri[:top_n]
    ], inti_dari_llm


    
# --- 4. ANTARMUKA UTAMA (STYLE DASHBOARD ENTERPRISE) ---
def dapatkan_sapaan():
    # Mengatur zona waktu spesifik ke WITA
    tz_wita = pytz.timezone('Asia/Makassar')
    waktu_sekarang = datetime.now(tz_wita)
    jam = waktu_sekarang.hour

    if 4 <= jam < 11:
        return "Selamat Pagi!"
    elif 11 <= jam < 15:
        return "Selamat Siang!"
    elif 15 <= jam < 18:
        return "Selamat Sore!"
    else:
        return "Selamat Malam!"

# =========================================================
# JURUS ANTI-LEMOT: PEMBEKUAN HTML UNTUK JELAJAH KODE
# =========================================================
@st.cache_data(show_spinner=False)
def render_pohon_html(p, _df):
    hasil_filter = _df[_df['kode'].str.startswith(p)]
    if hasil_filter.empty:
        return ""
        
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
        warna_level = ["#EF4444", "#3B82F6", "#10B981", "#F59E0B"]
        warna_bg = warna_level[actual_level]
        levels_name = ["Primer", "Sekunder", "Tersier", "Kuartier"]
        label = levels_name[actual_level]
        
        badge_style = f"background-color: {warna_bg}; color: #ffffff; padding: 8px 16px; border-radius: 16px; font-weight: normal; font-size: 0.95rem !important; display: inline-flex; align-items: flex-start; max-width: 100%; box-shadow: 0px 3px 6px rgba(0,0,0,0.15);"
        icon_code_html = f"<div style='display: flex; align-items: flex-start; flex-shrink: 0; white-space: nowrap; margin-right: 8px;'><span class='material-symbols-rounded' style='font-size: 1.15rem; margin-right: 6px; margin-top: 2px;'>folder</span><strong style='margin-top: 2px;'>{k}</strong><span style='margin: 2px 4px 0 4px; opacity: 0.6;'>|</span></div>"
        text_html = f"<div style='flex-grow: 1; text-align: justify; line-height: 1.5; margin-top: 2px; word-break: break-word;'>{u} <i style='opacity: 0.8; margin-left: 6px; white-space: nowrap;'>({label})</i></div>"
        indikator_html = f"<div style='display: flex; align-items: flex-start; margin-left: 10px; margin-top: 2px; flex-shrink: 0;'><span class='material-symbols-rounded arrow-icon' style='font-size: 1.2rem; background: rgba(0,0,0,0.15); border-radius: 50%; padding: 2px;'>keyboard_arrow_right</span></div>" if children else ""
        
        html = ""
        if children:
            html += f'<details class="anak-cucu" style="margin-bottom: 8px;"><summary style="cursor: pointer; list-style: none; outline: none;"><div style="{badge_style}">{icon_code_html}{text_html}{indikator_html}</div></summary><div style="margin-left: 20px; padding-left: 10px; border-left: 2px dashed #CBD5E1; padding-top: 8px;">'
            for c in children:
                html += render_tree(c)
            html += '</div></details>'
        else:
            html += f'<div style="margin-bottom: 8px;"><div style="{badge_style}">{icon_code_html}{text_html}{indikator_html}</div></div>'
        return html

    full_html = ""
    if p in nodes:
        sorted_children = sorted(nodes[p]['children'], key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
        for child_kode in sorted_children:
            full_html += render_tree(child_kode)
    return full_html
    
def halaman_utama():
    # INISIALISASI ROUTING HALAMAN
    if 'page' not in st.session_state:
        st.session_state.page = 'Beranda'

    def ganti_halaman(nama_halaman):
        st.session_state.page = nama_halaman

    # CSS GLOBAL HALAMAN UTAMA
    st.markdown("""
    <style>
    /* Hapus paksaan warna putih, biarkan mengikuti root/sistem */
    .stApp { background-color: var(--bg-app) !important; }

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
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--card-border) !important;
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
        font-size: 0.95rem !important; color: var(--text-subtitle) !important; justify-content: flex-start !important;
        box-shadow: none !important; transform: none !important; font-family: 'Poppins', sans-serif !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover { background-color: var(--sidebar-hover) !important; color: var(--text-title) !important; }
    
    /* PAKSA WARNA BIRU UNTUK TOMBOL AKTIF */
    button[kind="primary"] {
        background-color: #009DFF !important; border-color: #009DFF !important; color: white !important; font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] .stButton button[kind="primary"] {
        background-color: rgba(0, 157, 255, 0.1) !important; color: #009DFF !important;
    }

    /* UMUM */
    .section-title { font-weight: 700; color: var(--text-title); font-size: 1.15rem; margin-bottom: 15px; margin-top: 25px; font-family: 'Poppins', sans-serif !important;}
    div[data-testid="stDataFrame"] { background: var(--card-bg); border-radius: 12px; padding: 10px; border: 1px solid var(--card-border); }
    div[data-testid="InputInstructions"] { display: none !important; }
    
    /* ========================================================= */
    /* PINDAHAN CSS KARTU AKSES CEPAT (ANTI TELANJANG / FOUC)    */
    /* ========================================================= */
    .card-container { position: relative; height: 160px; margin-bottom: 10px; border-radius: 16px;}
    .element-container:has(.card-container) + .element-container { margin-top: -170px !important; position: relative !important; z-index: 10 !important; }
    .element-container:has(.card-container) + .element-container button { height: 160px !important; width: 100% !important; opacity: 0 !important; cursor: pointer !important; background: transparent !important; border: none !important; }
    .element-container:has(+ .element-container:hover) .saas-card,
    div[data-testid="column"]:hover .saas-card { border-color: #009DFF !important; box-shadow: 0 10px 25px rgba(0, 157, 255, 0.2) !important; transform: translateY(-5px) !important; }
    .saas-card { position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: var(--sidebar-bg); border: 1px solid var(--card-border); border-radius: 16px; padding: 24px 20px; display: flex; align-items: flex-start; gap: 15px; box-shadow: var(--card-shadow); transition: all 0.3s ease !important; z-index: 1; }
    .saas-icon-box { width: 50px; height: 50px; min-width: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; }
    .bg-blue { background-color: rgba(0, 157, 255, 0.15); color: #009DFF; }
    .bg-orange { background-color: rgba(249, 115, 22, 0.15); color: #F97316; }
    .bg-purple { background-color: rgba(168, 85, 247, 0.15); color: #A855F7; }
    .saas-card-content { flex-grow: 1; display: flex; flex-direction: column; height: 100%; justify-content: flex-start;}
    .saas-card-title { font-weight: 700; color: #FFFFFF; font-size: 1rem; margin-bottom: 5px; font-family: 'Poppins', sans-serif !important;}
    .saas-card-desc { font-size: 0.8rem; color: #E2E8F0; line-height: 1.4; font-family: 'Poppins', sans-serif !important;}
    .saas-card-arrow { align-self: flex-end; margin-top: auto; color: #009DFF; font-size: 1.2rem; display: flex; }
    
    /* JURUS ANTI KAMUFLASE INPUT */
    div[data-testid="stHorizontalBlock"]:has(input) div[data-baseweb="input"] { background-color: #FFFFFF !important; }
    div[data-testid="stHorizontalBlock"]:has(input) div[data-baseweb="input"] input { color: #1E1E1E !important; -webkit-text-fill-color: #1E1E1E !important; font-weight: 500 !important; }
    div[data-testid="stHorizontalBlock"]:has(input) div[data-baseweb="input"] input::placeholder { color: #888888 !important; -webkit-text-fill-color: #888888 !important; }

    /* RESPONSIVE HP UNTUK KARTU (Sama persis dengan milik Bapak) */
    @media screen and (max-width: 768px) {
        .card-container { height: 120px; margin-bottom: 15px;}
        .saas-card { padding: 12px 15px; gap: 10px; }
        .saas-icon-box { width: 40px; height: 40px; min-width: 40px; font-size: 1.3rem; }
        .saas-card-title { font-size: 0.85rem; }
        .saas-card-desc { font-size: 0.7rem; }
        .saas-card-arrow { font-size: 1rem; }
        .element-container:has(.card-container) + .element-container { margin-top: -145px !important; }
        .element-container:has(.card-container) + .element-container button { height: 120px !important; }
    }
    </style>
    """, unsafe_allow_html=True)

    try:
        df = load_data()
        nama_user = st.session_state.get('nama', 'Administrator')
        role_user = st.session_state.get('role', 'admin')

        # ================= SIDEBAR NAVIGASI =================
        with st.sidebar:
            st.markdown("""
            <style>
            /* 1. Ubah Background Sidebar Jadi Biru Gelap Gradasi (Kebal Dark Mode) */
            [data-testid="stSidebar"], 
            [data-testid="stSidebar"] > div:first-child {
                background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
                background-color: transparent !important;
                border-right: none !important;
            }
            [data-testid="stSidebar"] * {
                color: white !important;
            }
            
            /* 2. Desain Wadah Menu HTML */
            .sidebar-menu {
                height: 48px; display: flex; align-items: center; gap: 15px; 
                padding: 0 15px; border-radius: 8px; margin-bottom: 5px;
                color: #FFFFFF !important; font-weight: 500; font-family: 'Poppins', sans-serif;
                transition: all 0.3s ease !important;
            }
            .sidebar-menu span.material-symbols-rounded { font-size: 1.4rem; }

            /* 3. Trik Tarik Tombol Gaib ke Atas */
            [data-testid="stSidebar"] .element-container:has(.sidebar-menu) + .element-container {
                margin-top: -53px !important; position: relative !important; z-index: 10 !important;
            }
            [data-testid="stSidebar"] .element-container:has(.sidebar-menu) + .element-container button {
                height: 48px !important; width: 100% !important; opacity: 0 !important;
                cursor: pointer !important; background: transparent !important; border: none !important;
            }

            /* 4. Efek Hover (Bergeser ke Kanan) */
            [data-testid="stSidebar"] .element-container:has(+ .element-container:hover) .sidebar-menu {
                background-color: rgba(255, 255, 255, 0.15) !important; transform: translateX(8px);
            }
            
            /* Divider dan Judul Menu */
            .sidebar-divider { height: 1px; background-color: rgba(255,255,255,0.2); margin: 20px 0; }
            .sidebar-title-menu { color: #93C5FD !important; font-size: 0.85rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 10px; padding-left: 5px; text-transform: uppercase; font-family: 'Poppins', sans-serif;}
            
            /* Tombol Keluar Default */
            [data-testid="stSidebar"] button[kind="secondary"] {
                color: #0F172A !important; background: #FFFFFF !important; border: none !important; font-weight: 700 !important; font-family: 'Poppins', sans-serif !important; border-radius: 8px !important; margin-top: 10px !important;
            }
            [data-testid="stSidebar"] button[kind="secondary"]:hover {
                background: #F1F5F9 !important; transform: translateY(-2px);
            }

            /* ========================================================= */
            /* 5. TARIKAN HEADER & LOGO "S" FUTURISTIK                   */
            /* ========================================================= */
            
            /* KEMBALIKAN TOMBOL HIDE SIDEBAR: Jangan di-display none! */
            [data-testid="stSidebarHeader"] { 
                background: transparent !important;
                padding-bottom: 0px !important; 
            }
            
            /* Pastikan icon panah hide sidebar warnanya putih agar terlihat */
            [data-testid="stSidebarHeader"] svg {
                color: #FFFFFF !important;
            }
            
            .sidebar-title-container { 
                /* TARIKAN AMAN: Ubah angka -10px ini jika kurang lurus dengan banner utama */
                margin-top: -10px !important; 
                display: flex; align-items: center; gap: 14px; 
                padding: 0 0 15px 0;
            }
            
            .sidebar-logo { 
                background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%) !important; 
                width: 44px !important; height: 44px !important; 
                border-radius: 12px !important; 
                display: flex; align-items: center; justify-content: center; 
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15), inset 0 -3px 0 rgba(226, 232, 240, 1) !important;
            }
            
            /* ========================================================= */
            /* 6. KUNCI MATI LEBAR SIDEBAR (JURUS SENSOR KURSOR)         */
            /* ========================================================= */
            
            /* Membasmi elemen tanpa nama yang memiliki kursor penarik batas */
            div[style*="cursor: col-resize"],
            div[style*="cursor: ew-resize"] {
                display: none !important;
                width: 0px !important;
                pointer-events: none !important;
                opacity: 0 !important;
            }
            
            .sidebar-title { color: #FFFFFF !important; font-weight: 900 !important; letter-spacing: 1px !important; font-size: 1.8rem !important; margin: 0 !important;}
            </style>
            """, unsafe_allow_html=True)

            # HTML Logo Modern (Menggunakan SVG Vector)
            st.markdown("""
            <div class="sidebar-title-container">
                <div class="sidebar-logo">
                    <svg viewBox="0 -2 24 24" width="100%" height="70%" style="filter: drop-shadow(0px 2px 3px rgba(0,157,255,0.4)); transform: scale(1.6);">
                        <defs>
                            <linearGradient id="gradS" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" style="stop-color:#00C6FF;stop-opacity:1" />
                                <stop offset="100%" style="stop-color:#0072FF;stop-opacity:1" />
                            </linearGradient>
                        </defs>
                        <!-- Gambar S Modern -->
                        <path d="M16,7 A4,4 0 0,0 12,3 A4,4 0 0,0 8,7 C8,9.5 16,10.5 16,13 A4,4 0 0,1 12,17 A4,4 0 0,1 8,13" fill="none" stroke="url(#gradS)" stroke-width="4" stroke-linecap="round"/>
                    </svg>
                </div>
                <h1 class="sidebar-title">SIKAP</h1>
            </div>
            <p style="color:#93C5FD !important; font-size:0.75rem; margin-top:-15px; margin-bottom:30px; font-weight:500;">Sistem Informasi Klasifikasi<br>Arsip Pintar</p>
            """, unsafe_allow_html=True)
            
            # --- KELOMPOK MENU UTAMA ---
            st.markdown('<div class="sidebar-title-menu">🧭 Menu Utama</div>', unsafe_allow_html=True)

            st.markdown('<div class="sidebar-menu"><span class="material-symbols-rounded">home</span><span>Beranda</span></div>', unsafe_allow_html=True)
            if st.button(" ", key="sb_beranda", use_container_width=True):
                ganti_halaman('Beranda')
                st.rerun()

            st.markdown('<div class="sidebar-menu"><span class="material-symbols-rounded">smart_toy</span><span>Pencarian AI</span></div>', unsafe_allow_html=True)
            if st.button(" ", key="sb_ai", use_container_width=True):
                ganti_halaman('Pencarian AI')
                st.rerun()

            st.markdown('<div class="sidebar-menu"><span class="material-symbols-rounded">folder</span><span>Jelajah Kode</span></div>', unsafe_allow_html=True)
            if st.button(" ", key="sb_jelajah", use_container_width=True):
                ganti_halaman('Jelajah Kode')
                st.rerun()

            st.markdown('<div class="sidebar-menu"><span class="material-symbols-rounded">schedule</span><span>Riwayat</span></div>', unsafe_allow_html=True)
            if st.button(" ", key="sb_riwayat", use_container_width=True):
                ganti_halaman('Riwayat')
                st.rerun()

            if role_user == 'admin':
                st.markdown('<div class="sidebar-menu"><span class="material-symbols-rounded">admin_panel_settings</span><span>Panel Admin</span></div>', unsafe_allow_html=True)
                if st.button(" ", key="sb_admin", use_container_width=True):
                    ganti_halaman('Admin')
                    st.rerun()

            # --- KELOMPOK LAINNYA ---
            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-title-menu">📌 Lainnya</div>', unsafe_allow_html=True)

            st.markdown('<div class="sidebar-menu"><span class="material-symbols-rounded">menu_book</span><span>Petunjuk Penggunaan</span></div>', unsafe_allow_html=True)
            if st.button(" ", key="sb_petunjuk", use_container_width=True):
                ganti_halaman('Petunjuk')
                st.rerun()

            st.markdown('<div class="sidebar-menu"><span class="material-symbols-rounded">info</span><span>Tentang SIKAP</span></div>', unsafe_allow_html=True)
            if st.button(" ", key="sb_tentang", use_container_width=True):
                ganti_halaman('Tentang')
                st.rerun()
                
            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
            
            # Tombol Profil Modern
            st.markdown('<div class="sidebar-menu"><span class="material-symbols-rounded">manage_accounts</span><span>Profil Saya</span></div>', unsafe_allow_html=True)
            if st.button(" ", key="sb_profil", width="stretch"):
                ganti_halaman('Profil Saya')
                st.rerun()

            # Tombol Keluar Modern
            st.markdown('<div class="sidebar-menu"><span class="material-symbols-rounded">logout</span><span>Keluar</span></div>', unsafe_allow_html=True)
            if st.button(" ", key="sb_keluar", width="stretch"):
                st.session_state['logged_in'] = False
                st.session_state['role'] = None
                st.session_state['nama'] = ""
                st.session_state.search_history = [] # Bersihkan riwayat saat keluar
                st.rerun()

        # ================= MAIN CONTENT (ROUTING) =================
            
        if st.session_state.page == 'Beranda':
            
            # CSS KHUSUS BERANDA (Tarikan Ekstrim)
            st.markdown("""
            <style>
            /* 1. Desain Banner Biru Utama */
            .hero-banner {
                background: var(--hero-bg); 
                border-radius: 24px; 
                padding: 50px 20px; 
                display: flex; flex-direction: column; align-items: center; text-align: center; 
                border: 1px solid var(--hero-border);
                margin-bottom: 0px; 
            }
            .hero-content { width: 100%; max-width: 850px; margin: 0 auto; }
            
            .hero-welcome-text { 
                font-size: 2rem; font-weight: 900; color: #009DFF; 
                text-transform: uppercase; letter-spacing: 3px; margin-bottom: 8px;
            }
            
            .hero-user-name { 
                font-size: clamp(1.5rem, 5vw, 1.5rem); font-weight: 700; 
                color: var(--text-title); 
                line-height: 1.1; margin-bottom: 15px; word-break: break-word; 
            }
            
            .hero-subtitle { 
                font-size: 1rem; color: var(--text-subtitle); margin-bottom: 30px;
                max-width: 600px; margin-left: auto; margin-right: auto;
            }
            
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
                padding-left: 25px !important;
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
            sapaan_dinamis = dapatkan_sapaan()
            
            # HTML BANNER UTAMA 
            st.markdown(f"""
<div class="hero-banner">
    <div class="hero-content">
        <div class="hero-welcome-text">{sapaan_dinamis}</div>
        <div class="hero-user-name">{nama_user}</div>
        <div class="hero-subtitle">Kelola dan temukan kode klasifikasi arsip dengan mudah, cepat, dan akurat.</div>
        <div class="search-card-bg">
            <div class="search-title">Cari kode klasifikasi arsip</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
            
            # =========================================================
            # CALLBACK: FUNGSI PINDAH HALAMAN TANPA NGE-FLASH
            # =========================================================
            def jalankan_pencarian():
                if st.session_state.input_beranda:
                    st.session_state.temp_search = st.session_state.input_beranda
                st.session_state.page = 'Pencarian AI'

            # FORM PENCARIAN 
            col_in, col_btn = st.columns([5, 1])
            with col_in:
                user_input = st.text_input("Pencarian AI", placeholder="Ketik perihal surat di sini...", label_visibility="collapsed", key="input_beranda", on_change=jalankan_pencarian)
            with col_btn:
                btn_cari = st.button("🔍", key="btn_cari_beranda", on_click=jalankan_pencarian)
                
            # =========================================================
            # 5. AKSES CEPAT (FINAL DENGAN ANIMASI SIBLING)
            # =========================================================
            
            st.markdown("""
            <style>
            .section-title { font-weight: 700; color: var(--text-title); font-size: 2rem; margin-bottom: 15px; margin-top: 10px; font-family: 'Poppins', sans-serif !important;}
            
            /* Wadah utama kartu */
            .card-container { position: relative; height: 160px; margin-bottom: 10px; border-radius: 16px;}
            
            /* ========================================================= */
            /* JURUS TARIKAN KE ATAS                                     */
            /* ========================================================= */
            
            /* 1. Tarik wadah tombol yang jatuh di bawah kartu NAIK KE ATAS (-170px) */
            .element-container:has(.card-container) + .element-container {
                margin-top: -170px !important; 
                position: relative !important;
                z-index: 10 !important;
            }
            
            /* 2. Jadikan tombolnya transparan dan tingginya sama dengan kartu */
            .element-container:has(.card-container) + .element-container button {
                height: 160px !important;
                width: 100% !important;
                opacity: 0 !important;
                cursor: pointer !important;
                background: transparent !important;
                border: none !important;
            }
            
            /* ========================================================= */
            /* 3. PERBAIKAN ANIMASI (Trik Sibling)                       */
            /* ========================================================= */
            /* "Jika tombol gaib sedang di-hover, angkat kartu yang ada sebelumnya!" */
            .element-container:has(+ .element-container:hover) .saas-card,
            div[data-testid="column"]:hover .saas-card {
                border-color: #009DFF !important; 
                box-shadow: 0 10px 25px rgba(0, 157, 255, 0.2) !important; 
                transform: translateY(-5px) !important;
            }

            /* ========================================================= */
            /* DESAIN VISUAL KARTU                                       */
            /* ========================================================= */
            .saas-card {
                position: absolute; top: 0; left: 0; width: 100%; height: 100%;
                background: var(--sidebar-bg); border: 1px solid var(--card-border); border-radius: 16px;
                padding: 24px 20px; display: flex; align-items: flex-start; gap: 15px;
                box-shadow: var(--card-shadow); 
                transition: all 0.3s ease !important; /* Paksa transisi agar mulus */
                z-index: 1;
            }

            .saas-icon-box {
                width: 50px; height: 50px; min-width: 50px; border-radius: 50%;
                display: flex; align-items: center; justify-content: center; font-size: 1.6rem;
            }
            .bg-blue { background-color: rgba(0, 157, 255, 0.15); color: #009DFF; }
            .bg-orange { background-color: rgba(249, 115, 22, 0.15); color: #F97316; }
            .bg-purple { background-color: rgba(168, 85, 247, 0.15); color: #A855F7; }

            .saas-card-content { flex-grow: 1; display: flex; flex-direction: column; height: 100%; justify-content: flex-start;}
            .saas-card-title { font-weight: 700; color: #FFFFFF; font-size: 1rem; margin-bottom: 5px; font-family: 'Poppins', sans-serif !important;}
            .saas-card-desc { font-size: 0.8rem; color: #E2E8F0; line-height: 1.4; font-family: 'Poppins', sans-serif !important;}
            .saas-card-arrow { align-self: flex-end; margin-top: auto; color: #009DFF; font-size: 1.2rem; display: flex; }

            /* ========================================================= */
            /* JURUS ANTI KAMUFLASE (WARNA TEKS INPUT TETAP TERLIHAT)    */
            /* ========================================================= */
            
            /* 1. Pastikan kotak inputnya tetap putih bersih */
            div[data-testid="stHorizontalBlock"]:has(input) div[data-baseweb="input"] {
                background-color: #FFFFFF !important; 
            }

            /* 2. Paksa teks yang Bapak ketik berwarna gelap (Hitam/Abu Tua) */
            div[data-testid="stHorizontalBlock"]:has(input) div[data-baseweb="input"] input {
                color: #1E1E1E !important; 
                -webkit-text-fill-color: #1E1E1E !important; 
                font-weight: 500 !important;
            }
            
            /* 3. Sesuaikan warna teks bayangan (Placeholder) agar elegan */
            div[data-testid="stHorizontalBlock"]:has(input) div[data-baseweb="input"] input::placeholder {
                color: #888888 !important;
                -webkit-text-fill-color: #888888 !important;
            }

            /* ========================================================= */
            /* JURUS RESPONSIVE HP (ANTI GESER KIRI / KANAN)             */
            /* ========================================================= */
            @media screen and (max-width: 768px) {
                /* 1. Sesuaikan Ukuran Teks Banner */
                .hero-banner { padding: 30px 15px; border-radius: 16px; }
                .hero-welcome-text { font-size: 1.5rem; letter-spacing: 1px; }
                .hero-user-name { font-size: 1.3rem; margin-bottom: 10px; }
                .hero-subtitle { font-size: 0.85rem; margin-bottom: 20px; }
                
                /* 2. Beri Ruang di Kotak Biru untuk Input */
                .search-card-bg { padding: 15px 15px 85px 15px; border-radius: 12px; }
                .search-title { font-size: 0.95rem; }
                
                /* ===================================================== */
                /* 3. RESET TOTAL POSISI KOTAK SEARCH                    */
                /* ===================================================== */
                div[data-testid="stHorizontalBlock"]:has(input) {
                    margin-top: -130px !important; 
                    transform: translateX(0) !important; /* MATIKAN PAKSA EFEK GESER DARI PC */
                    left: 0 !important;
                    right: 0 !important;
                    
                    width: 100% !important; /* Pakai seluruh lebar yang tersedia */
                    width: 90% !important;
                    margin-left: 0 !important; 
                    padding-left: 20px !important;
                    padding-right: 50px !important;
                    padding: 0 20px !important; /* Jepit dari kiri-kanan pakai padding agar ke tengah */
                    
                    display: flex !important;
                    flex-direction: row !important; 
                    flex-wrap: nowrap !important;
                    align-items: center !important;
                    gap: 3px !important;
                    box-sizing: border-box !important; /* Wajib agar padding tidak merusak lebar */
                }

                /* Targetkan Kolom 1 (Area Input Teks) */
                div[data-testid="stHorizontalBlock"]:has(input) > div[data-testid="column"]:nth-child(1) {
                    flex: 1 1 auto !important;
                    width: 100% !important;
                    min-width: 0 !important; 
                    margin: 0 !important;
                }

                /* Targetkan Kolom 2 (Area Tombol Cari) */
                div[data-testid="stHorizontalBlock"]:has(input) > div[data-testid="column"]:nth-child(2) {
                    flex: 0 0 55px !important;
                    width: 55px !important; 
                    min-width: 55px !important;
                    max-width: 55px !important;
                    flex-shrink: 0 !important;
                    margin: 0 !important;
                }

                /* Seragamkan Tinggi Tombol dan Kotak Input */
                div[data-testid="stHorizontalBlock"]:has(input) button {
                    height: 52px !important;
                    min-height: 52px !important; /* <--- ANTI GEPENG ATAS-BAWAH */
                    width: 55px !important;      /* <--- KUNCI LEBAR TOMBOL SAMA DENGAN KOLOM */
                    min-width: 55px !important;  /* <--- ANTI GEPENG KIRI-KANAN */
                    flex-shrink: 0 !important;   /* <--- JURUS MUTLAK ANTI MENGKERUT */
                    padding: 0 !important;
                    border-radius: 8px !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                }
                div[data-testid="stHorizontalBlock"]:has(input) div[data-baseweb="input"] {
                    height: 52px !important;
                }

                /* ===================================================== */
                /* 4. Rapikan Kartu Akses Cepat                          */
                /* ===================================================== */
                .card-container { height: 120px; margin-bottom: 15px;}
                .saas-card { padding: 12px 15px; gap: 10px; }
                .saas-icon-box { width: 40px; height: 40px; min-width: 40px; font-size: 1.3rem; }
                .saas-card-title { font-size: 0.85rem; }
                .saas-card-desc { font-size: 0.7rem; }
                .saas-card-arrow { font-size: 1rem; }
                
                .element-container:has(.card-container) + .element-container { margin-top: -145px !important; }
                .element-container:has(.card-container) + .element-container button { height: 120px !important; }
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-title" style="text-align: center; color: #009DFF; font-weight: 900; letter-spacing: 1px; text-transform: uppercase;"> Akses Cepat</div>', unsafe_allow_html=True)
            
            # Membuat 3 Kolom
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
                st.button(" ", key="btn_akses_ai", use_container_width=True, on_click=ganti_halaman, args=('Pencarian AI',))

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
                st.button("  ", key="btn_akses_jelajah", use_container_width=True, on_click=ganti_halaman, args=('Jelajah Kode',))

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
                st.button("   ", key="btn_akses_riwayat", use_container_width=True, on_click=ganti_halaman, args=('Riwayat',))
                    
        # --- HALAMAN 2: PENCARIAN AI ---
        elif st.session_state.page == 'Pencarian AI':
            st.markdown('<div class="section-title" style="display:flex; align-items:center; gap:8px;"><span class="material-symbols-rounded" style="color:#009DFF; font-size:1.8rem;">smart_toy</span> Pencarian AI (Cerdas)</div>', unsafe_allow_html=True)
            st.write("Sistem cerdas akan menganalisis bahasa natural Anda untuk menemukan kode klasifikasi.")
            
            # =========================================================
            # CSS KHUSUS UNTUK MEMPERCANTIK HASIL PENCARIAN & FEEDBACK
            # =========================================================
            st.markdown("""
            <style>
            /* 1. Kotak Expander Hasil Pencarian (DENGAN OUTLINE BIRU) */
            div[data-testid="stExpander"] {
                border: 2px solid rgba(0, 157, 255, 0.5) !important; /* <--- OUTLINE BIRU DIKEMBALIKAN */
                border-radius: 16px !important;
                box-shadow: 0 4px 15px rgba(0, 157, 255, 0.1) !important; /* <--- GLOW BIRU HALUS */
                background: #FFFFFF !important;
                margin-bottom: 15px !important;
                overflow: hidden !important;
            }
            
            /* 2. PERBAIKAN TOTAL: BUNUH TEKS HANTU "arrow" ("arr") + Garis Bawah Biru */
            div[data-testid="stExpander"] > details > summary {
                padding: 15px 20px !important;
                background: #F8FAFC !important;
                border-bottom: 1px solid rgba(0, 157, 255, 0.2) !important; /* <--- GARIS BAWAH BIRU SENADA */
                border-radius: 16px 16px 0 0 !important;
                font-size: 0px !important; /* JURUS ANTI HANTU */
                color: transparent !important;
            }
            
            /* Bangkitkan HANYA teks judul (p) kita agar tampil sempurna */
            div[data-testid="stExpander"] summary p {
                font-weight: 700 !important;
                color: #0F172A !important;
                font-size: 1.05rem !important;
                font-family: 'Poppins', sans-serif !important;
                margin: 0 !important;
            }

            /* Pastikan icon panah asli Streamlit tetap ada dan warnanya pas */
            div[data-testid="stExpander"] summary svg {
                color: var(--text-muted) !important; /* <--- UBAH DI SINI */
                width: 24px !important;
                height: 24px !important;
            }
            
            /* 3. Beri Ruang Lega di Dalam Expander Agar Tidak Tumpang Tindih */
            div[data-testid="stExpanderDetails"] {
                padding: 25px 20px !important;
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            
            /* 4. Desain Area Feedback (Tanpa Wrapper HTML agar tidak bentrok) */
            .feedback-title {
                font-size: 1rem;
                font-weight: 700;
                color: var(--text-title); /* <--- UBAH DI SINI */
                margin-top: 40px;
                margin-bottom: 20px;
                text-align: center;
                font-family: 'Poppins', sans-serif !important;
            }
            
            /* JURUS JITU: Target langsung tombol Streamlit yang ada di dalam kolom */
            div[data-testid="stHorizontalBlock"] button {
                border-radius: 16px !important; /* <--- SUDUT MEMBULAT 16px */
                font-weight: 700 !important;
                border: 2px solid rgba(0, 157, 255, 0.6) !important; /* <--- OUTLINE BIRU */
                background: linear-gradient(135deg, #F0F9FF 0%, #FFFFFF 100%) !important;
                color: #009DFF !important;
                height: 55px !important;
                box-shadow: 0 4px 15px rgba(0, 157, 255, 0.1) !important; /* Glow tipis */
                transition: all 0.3s ease !important;
            }
            div[data-testid="stHorizontalBlock"] button:hover {
                border-color: #009DFF !important;
                transform: translateY(-3px);
                box-shadow: 0 6px 18px rgba(0, 157, 255, 0.3) !important; /* Glow menguat saat kursor mendekat */
            }
            </style>
            """, unsafe_allow_html=True)

            default_val = st.session_state.pop('temp_search', '') 
            user_input = st.text_input("Ketik perihal surat:", value=default_val, placeholder="Contoh: penyusunan rencana kerja anggaran...", key="input_halaman_ai")

            if user_input:
                if user_input not in st.session_state.search_history:
                    st.session_state.search_history.append(user_input)
                    simpan_riwayat_csv(st.session_state['nama'], user_input)

                with st.spinner('AI sedang membedah dokumen Anda...'):
                    # TANGKAP HASIL INTI SURAT DARI FUNGSI
                    results, inti_dari_llm = smart_classify(user_input, df) 
                    
                    if results:
                        # KOTAK HASIL EKSTRAKSI DULU, BARU BANNER SUKSES (ADAPTIF LIGHT/DARK MODE)
                        st.markdown(f"""
                        <div style="background: var(--card-bg); border: 1px dashed var(--input-border); padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; display:flex; align-items:flex-start; gap:10px; box-shadow: var(--card-shadow);">
                            <span class="material-symbols-rounded" style="color:#009DFF; font-size: 1.4rem; margin-top: 2px;">psychology</span>
                            <div style="font-size: 0.9rem; color: var(--text-title); line-height: 1.5;">
                                <strong>Inti Substansi (Hasil Bedah AI):</strong> <br>
                                <span style="color: var(--text-subtitle); font-style: italic;">"{inti_dari_llm.title()}"</span>
                            </div>
                        </div>

                        <div style="background: #E0F2FE; border-left: 4px solid #009DFF; padding: 16px 20px; border-radius: 8px; margin-bottom: 25px; display:flex; align-items:center; gap:10px;">
                            <span class="material-symbols-rounded" style="color:#009DFF;">check_circle</span>
                            <div><span style="font-weight: 700; color: #0369A1;">Analisis Selesai!</span> <span style="color:#0F172A; font-size:0.95rem;">Berikut rekomendasi klasifikasi terbaik untuk dokumen Anda:</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        rekomendasi_kode = [] # List untuk menyimpan kode yang dihasilkan AI
                        
                        # LOOPING HASIL PENCARIAN
                        for i, (idx, score) in enumerate(results):
                            res = df.iloc[idx]
                            rekomendasi_kode.append(res['kode']) # Tangkap kodenya
                            
                            with st.expander(f"❯ KODE {res['kode']} (Keyakinan: {score:.1%})", expanded=(i==0)):
                                st.markdown(f"<div style='font-size:1.15rem; font-weight:700; color:#009DFF; margin-bottom:15px; line-height:1.4; word-wrap: break-word; overflow-wrap: break-word;'>{res['uraian'].title()}</div>", unsafe_allow_html=True)
                                st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#64748B; margin-bottom:12px; letter-spacing:1px; text-transform:uppercase;'>JALUR HIERARKI KODE:</div>", unsafe_allow_html=True)
                                hierarki = get_hierarchy(res['kode'], df)
                                for h in hierarki: 
                                    st.markdown(h, unsafe_allow_html=True)
                                    
                        # --- FITUR FEEDBACK (PILIH KODE TERBAIK) ---
                        st.markdown('<div class="feedback-title" style="display:flex; justify-content:center; align-items:center; gap:8px;"><span class="material-symbols-rounded" style="color:#009DFF; font-size:1.5rem;">fact_check</span> Bantu SIKAP Belajar! Mana kode yang paling tepat menurut Anda?</div>', unsafe_allow_html=True)
                        
                        # Buat kolom tombol secara dinamis sesuai jumlah rekomendasi AI
                        cols = st.columns(len(rekomendasi_kode))
                        for i, col in enumerate(cols):
                            with col:
                                if st.button(f"Pilih Kode {rekomendasi_kode[i]}", key=f"fb_pilih_{rekomendasi_kode[i]}_{user_input}", use_container_width=True):
                                    simpan_feedback_csv(st.session_state['nama'], user_input, rekomendasi_kode[i])
                                    st.success(f"✨ Terima kasih! Anda memvalidasi **Kode {rekomendasi_kode[i]}** sebagai jawaban yang paling tepat. Pilihan ini akan terekam di sistem kami.")
                        st.markdown('</div>', unsafe_allow_html=True)

                    else:
                        st.warning("Maaf, tidak ditemukan klasifikasi yang cocok dengan kata kunci tersebut.")

        # --- HALAMAN 3: JELAJAH KODE ---
        elif st.session_state.page == 'Jelajah Kode':
            # Judul dengan icon modern menyesuaikan sidebar
            st.markdown('<div class="section-title" style="display:flex; align-items:center; gap:8px;"><span class="material-symbols-rounded" style="color:#009DFF; font-size:1.8rem;">folder</span> Jelajah Kode Klasifikasi</div>', unsafe_allow_html=True)
            st.write("Telusuri seluruh struktur hierarki klasifikasi arsip secara manual.")
            
            # =========================================================
            # CSS ANTI-HANTU & DESAIN KOTAK EXPANDER SERAGAM
            # =========================================================
            st.markdown("""
            <style>
            div[data-testid="stExpander"] {
                border: 2px solid var(--input-border) !important;
                border-radius: 16px !important;
                box-shadow: 0 4px 15px rgba(0, 157, 255, 0.05) !important;
                background: var(--card-bg) !important;
                margin-bottom: 15px !important;
                overflow: hidden !important;
            }
            
            /* KUNCI PERBAIKAN: Tanda "> details >" ini wajib ada agar anak-cucunya tidak ikut kena tembak jurus 0px! */
            div[data-testid="stExpander"] > details > summary {
                padding: 15px 20px !important;
                background: var(--expander-bg) !important;
                border-bottom: 1px solid var(--card-border) !important;
                border-radius: 16px 16px 0 0 !important;
                font-size: 0px !important; /* JURUS ANTI HANTU (Aman, cuma buat Bapaknya) */
                color: transparent !important;
            }
            
            div[data-testid="stExpander"] > details > summary p {
                font-weight: 700 !important;
                color: var(--text-title) !important;
                font-size: 1.05rem !important;
                font-family: 'Poppins', sans-serif !important;
                margin: 0 !important;
            }
            
            div[data-testid="stExpander"] > details > summary svg {
                color: var(--text-muted) !important;
                width: 24px !important;
                height: 24px !important;
            }
            
            div[data-testid="stExpanderDetails"] {
                padding: 25px 20px !important;
            }
            
            /* Sembunyikan panah default HTML yg jelek di anak cucu */
            details.anak-cucu > summary::marker, 
            details.anak-cucu > summary::-webkit-details-marker {
                display: none !important;
            }
            
            /* ========================================================= */
            /* TAMBAHAN ANIMASI PANAH ELEGAN                             */
            /* ========================================================= */
            .arrow-icon {
                transition: transform 0.3s ease-in-out !important;
                display: inline-block !important; 
            }
            details[open] > summary .arrow-icon {
                transform: rotate(90deg) !important; 
            }

            </style>
            """, unsafe_allow_html=True)
            
            import re
            daftar_primer = [f"{i}00" for i in range(10)]
            
            for p in daftar_primer:
                cek_df = df[df['kode'] == p]
                uraian_primer = cek_df.iloc[0]['uraian'].title() if not cek_df.empty else "Detail Klasifikasi"
                
                with st.expander(f"❯ {p} - {uraian_primer}"):
                    # Panggil fungsi beku! Mesin tidak akan capek merakit ulang.
                    pohon_html = render_pohon_html(p, df)
                    
                    if pohon_html:
                        st.markdown(pohon_html, unsafe_allow_html=True)
                    else:
                        st.caption("Tidak ada data klasifikasi di dalam rumpun ini.")

        # --- HALAMAN 4: RIWAYAT ---
        elif st.session_state.page == 'Riwayat':
            st.markdown('<div class="section-title" style="display:flex; align-items:center; gap:8px;"><span class="material-symbols-rounded" style="color:#009DFF; font-size:1.8rem;">history</span> Riwayat Pencarian Lengkap</div>', unsafe_allow_html=True)
    
            if st.session_state.search_history:
                for riwayat in reversed(st.session_state.search_history):
                    st.markdown(f"🔹 {riwayat}")
                
                # --- TOMBOL HAPUS SEMU (SOFT DELETE) ---
                if st.button("Hapus Riwayat"):
                    # 1. Kosongkan tampilan di layar user
                    st.session_state.search_history = []
                    
                    # 2. Ubah status di CSV menjadi "dihapus" (bukan dibuang fisik)
                    file_riwayat = 'riwayat_pencarian.csv'
                    if os.path.exists(file_riwayat):
                        try:
                            df_riwayat = pd.read_csv(file_riwayat, dtype=str)
                            if 'nama' in df_riwayat.columns:
                                # Jika kolom status belum ada (untuk file CSV lama), kita buatkan
                                if 'status_tampil' not in df_riwayat.columns:
                                    df_riwayat['status_tampil'] = 'aktif'
                                    
                                # Ganti status menjadi 'dihapus' HANYA untuk user yang sedang login
                                mask = df_riwayat['nama'] == st.session_state['nama']
                                df_riwayat.loc[mask, 'status_tampil'] = 'dihapus'
                                
                                # Simpan perubahan
                                df_riwayat.to_csv(file_riwayat, index=False)
                                sync_to_drive(file_riwayat)
                        except Exception as e:
                            st.error(f"Gagal menyembunyikan riwayat: {e}")
                            
                    st.success("Riwayat berhasil dihapus dari tampilan Anda!")
                    time.sleep(1) # Jeda agar pesan terbaca sebelum refresh
                    st.rerun()
            else:
                st.info("Belum ada riwayat.")

        # --- HALAMAN 5: ADMIN PANEL ---
        elif st.session_state.page == 'Admin':
            # Judul Halaman Utama Admin
            st.markdown('<div class="section-title" style="display:flex; align-items:center; gap:8px;"><span class="material-symbols-rounded" style="color:#009DFF; font-size:1.8rem;">admin_panel_settings</span> Panel Administrator SIKAP</div>', unsafe_allow_html=True)
            
            # --- FUNGSI BANTUAN KHUSUS ADMIN ---
            def admin_load_csv(file_name, sep=','):
                # KUNCI: Pastikan file ada DAN isinya tidak kosong (0 bytes)
                if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
                    try:
                        return pd.read_csv(file_name, sep=sep, dtype=str)
                    except:
                        try:
                            return pd.read_csv(file_name, sep=';', dtype=str)
                        except:
                            return pd.DataFrame() # Tangkap segala jenis error aneh
                return pd.DataFrame()

            def admin_save_csv(df, file_name, sep=','):
                df.to_csv(file_name, index=False, sep=sep)
                sync_to_drive(file_name) # UPLOAD KE DRIVE
                st.toast(f"Data berhasil disinkronkan permanen ke Cloud!", icon="☁️")

            # --- KOMPONEN UI KUSTOM (MATERIAL DESIGN) ---
            def modern_alert(icon, text, color="#009DFF", bg="#E0F2FE"):
                st.markdown(f"""
                <div style="background: {bg}; border-left: 4px solid {color}; padding: 12px 16px; border-radius: 8px; margin-bottom: 20px; display:flex; align-items:flex-start; gap:12px;">
                    <span class="material-symbols-rounded" style="color:{color}; font-size:1.4rem; margin-top:2px;">{icon}</span>
                    <div style="color: #334155; font-size: 0.9rem; font-weight: 500; line-height: 1.5;">{text}</div>
                </div>
                """, unsafe_allow_html=True)

            def metric_card(icon, title, value, color="#009DFF", bg="var(--expander-bg)"):
                return f"""
                <div style="background: var(--card-bg); border: 1px solid var(--card-border); padding: 20px; border-radius: 16px; display:flex; align-items:center; gap:15px; box-shadow: var(--card-shadow);">
                    <div style="background: {bg}; width: 54px; height: 54px; border-radius: 14px; display:flex; align-items:center; justify-content:center;">
                        <span class="material-symbols-rounded" style="color:{color}; font-size:2rem;">{icon}</span>
                    </div>
                    <div>
                        <div style="color: var(--text-subtitle); font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom:4px;">{title}</div>
                        <div style="color: var(--text-title); font-size: 1.6rem; font-weight: 800; line-height: 1;">{value}</div>
                    </div>
                </div>
                """

            # --- TAB MENU ADMIN ---
            tab_dash, tab_user, tab_data, tab_log = st.tabs([
                "Dashboard Statistik", 
                "Manajemen Pengguna", 
                "Manajemen Klasifikasi", 
                "Log Monitoring"
            ])

            # ==========================================
            # TAB 1: DASHBOARD STATISTIK (DENGAN PLOTLY)
            # ==========================================
            with tab_dash:
                st.markdown('<div class="section-title" style="display:flex; align-items:center; gap:8px; margin-top:10px;"><span class="material-symbols-rounded" style="color:#009DFF; font-size:1.6rem;">analytics</span> Ringkasan Sistem</div>', unsafe_allow_html=True)
                
                df_users = admin_load_csv('pengguna.csv')
                df_dataset = admin_load_csv('klasifikasi_arsip_emas.csv')
                df_logs = admin_load_csv('riwayat_pencarian.csv')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(metric_card("group", "Pengguna", f"{len(df_users)}", "#3B82F6", "#DBEAFE"), unsafe_allow_html=True)
                with col2:
                    st.markdown(metric_card("folder_managed", "Klasifikasi", f"{len(df_dataset)}", "#10B981", "#D1FAE5"), unsafe_allow_html=True)
                with col3:
                    st.markdown(metric_card("manage_search", "Total Pencarian", f"{len(df_logs)}", "#F59E0B", "#FEF3C7"), unsafe_allow_html=True)
                
                st.write("<br>", unsafe_allow_html=True)
                
                if not df_logs.empty and 'nama' in df_logs.columns:
                    st.markdown('<div style="font-weight:700; color:#0F172A; margin-bottom:15px; font-family:Poppins;">Aktivitas Pencarian per Pengguna</div>', unsafe_allow_html=True)
                    
                    # Persiapan Data untuk Plotly
                    log_counts = df_logs['nama'].value_counts().reset_index()
                    log_counts.columns = ['Nama Pengguna', 'Total Pencarian']
                    
                    # Grafik Plotly Modern
                    fig = px.bar(
                        log_counts, 
                        x='Nama Pengguna', 
                        y='Total Pencarian',
                        text='Total Pencarian',
                        color_discrete_sequence=['#009DFF']
                    )
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0, r=0, t=20, b=0),
                        xaxis=dict(showgrid=False, title=None, tickfont=dict(family='Poppins')),
                        yaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=None, tickfont=dict(family='Poppins'))
                    )
                    
                    fig.update_traces(
                        textposition='outside', 
                        textfont=dict(family='Poppins', size=13, color='#0F172A'),
                        cliponaxis=False # <--- PENGGANTI YANG BENAR
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                else:
                    modern_alert("info", "Belum ada data aktivitas untuk divisualisasikan.", "#64748B", "#F1F5F9")

            # ==========================================
            # TAB 2: CRUD PENGGUNA
            # ==========================================
            with tab_user:
                st.markdown('<div class="section-title" style="display:flex; align-items:center; gap:8px; margin-top:10px;"><span class="material-symbols-rounded" style="color:#009DFF; font-size:1.6rem;">manage_accounts</span> Pengaturan Akun</div>', unsafe_allow_html=True)
                modern_alert("lightbulb", "Gunakan tabel di bawah untuk mengelola akun. Klik <b>+</b> untuk menambah baris, atau pilih baris dan tekan <b>Delete</b> untuk menghapus.", "#009DFF", "#E0F2FE")
                
                df_users = admin_load_csv('pengguna.csv')
                if df_users.empty:
                    df_users = pd.DataFrame(columns=["username", "password", "role", "nama_lengkap"])
                    
                edited_users = st.data_editor(df_users, num_rows="dynamic", use_container_width=True, key="edit_user")
                
                if st.button("💾 Perbarui Data Pengguna", type="primary", use_container_width=True):
                    admin_save_csv(edited_users, 'pengguna.csv')

            # ==========================================
            # TAB 3: CRUD DATASET KLASIFIKASI
            # ==========================================
            with tab_data:
                st.markdown('<div class="section-title" style="display:flex; align-items:center; gap:8px; margin-top:10px;"><span class="material-symbols-rounded" style="color:#009DFF; font-size:1.6rem;">dataset</span> Master Data Klasifikasi</div>', unsafe_allow_html=True)
                modern_alert("warning", "<b>Peringatan:</b> Perubahan pada tabel ini akan langsung memengaruhi logika klasifikasi AI dan menu Jelajah Kode.", "#F59E0B", "#FEF3C7")
                
                df_dataset = admin_load_csv('klasifikasi_arsip_emas.csv')
                if df_dataset.empty:
                    df_dataset = pd.DataFrame(columns=["kode", "uraian"])
                    
                edited_dataset = st.data_editor(df_dataset, num_rows="dynamic", use_container_width=True, key="edit_data", height=500)
                
                if st.button("💾 Perbarui Master Klasifikasi", type="primary", use_container_width=True):
                    admin_save_csv(edited_dataset, 'klasifikasi_arsip_emas.csv')
                    st.cache_data.clear() # Membersihkan cache agar data baru langsung dimuat di halaman utama
                    st.success("Data berhasil diperbarui dan cache sistem telah dibersihkan.")

            # ==========================================
            # TAB 4: LOG MONITORING
            # ==========================================
            with tab_log:
                st.markdown('<div class="section-title" style="display:flex; align-items:center; gap:8px; margin-top:10px;"><span class="material-symbols-rounded" style="color:#009DFF; font-size:1.6rem;">monitoring</span> Jejak Audit Pencarian</div>', unsafe_allow_html=True)
                
                df_logs = admin_load_csv('riwayat_pencarian.csv')
                if not df_logs.empty:
                    st.dataframe(df_logs.iloc[::-1], use_container_width=True)
                    
                    csv_log = df_logs.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Ekspor Seluruh Log (CSV)",
                        data=csv_log,
                        file_name=f'audit_sikap_{datetime.now().strftime("%Y%m%d")}.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                else:
                    modern_alert("hourglass_empty", "Belum ada riwayat pencarian terekam.", "#64748B", "#F1F5F9")

        # --- HALAMAN 6: PETUNJUK PENGGUNAAN ---
        elif st.session_state.page == 'Petunjuk':
            st.markdown('<div class="section-title" style="display:flex; align-items:center; gap:8px;"><span class="material-symbols-rounded" style="color:#009DFF; font-size:1.8rem;">menu_book</span> Panduan Penggunaan SIKAP</div>', unsafe_allow_html=True)
            
            # =========================================================
            # CSS ANTI-HANTU & RESPONSIVE IMAGE UNTUK HP VS LAPTOP
            # =========================================================
            st.markdown("""
            <style>
            /* --- KOTAK EXPANDER & ANTI HANTU "ARROW" --- */
            div[data-testid="stExpander"] {
                border: 2px solid var(--input-border) !important;
                border-radius: 16px !important;
                box-shadow: 0 4px 15px rgba(0, 157, 255, 0.05) !important;
                background: var(--card-bg) !important;
                margin-bottom: 15px !important;
                overflow: hidden !important;
            }
            
            div[data-testid="stExpander"] > details > summary {
                padding: 15px 20px !important;
                background: var(--expander-bg) !important;
                border-bottom: 1px solid var(--card-border) !important;
                border-radius: 16px 16px 0 0 !important;
                font-size: 0px !important; 
                color: transparent !important;
            }
            
            div[data-testid="stExpander"] > details > summary p {
                font-weight: 700 !important;
                color: var(--text-title) !important;
                font-size: 1.05rem !important;
                font-family: 'Poppins', sans-serif !important;
                margin: 0 !important;
            }
            
            div[data-testid="stExpander"] > details > summary svg {
                color: var(--text-muted) !important;
                width: 24px !important;
                height: 24px !important;
            }
            
            div[data-testid="stExpanderDetails"] {
                padding: 25px 20px !important;
            }

            /* --- JURUS PENYELAMAT GAMBAR (ANTI-COLLAPSE) --- */
            /* Kita atur centering pada wadah luar (div), bukan pada tag img langsung */
            div[data-testid="stImage"] {
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
                width: 100% !important;
                margin-top: 10px !important;
                margin-bottom: 15px !important;
            }
            
            div[data-testid="stImage"] img {
                width: 75% !important; /* Di laptop tampil elegan proporsional 75% */
                height: auto !important;
                border-radius: 14px !important;
                border: 1px solid var(--card-border) !important;
                box-shadow: 0 10px 30px rgba(0,0,0,0.05) !important;
            }
            
            /* --- PENGATURAN GAMBAR KHUSUS DI HP (MOBILE) --- */
            @media screen and (max-width: 768px) {
                div[data-testid="stImage"] img {
                    width: 100% !important; /* Di HP otomatis 100% full memenuhi layar */
                    border-radius: 8px !important;
                }
                div[data-testid="stExpanderDetails"] {
                    padding: 15px 10px !important;
                }
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Banner Pembuka
            st.markdown("""
            <div style="background: var(--card-bg); border-left: 4px solid #009DFF; padding: 16px 20px; border-radius: 8px; margin-bottom: 25px; box-shadow: var(--card-shadow); border-top: 1px solid var(--card-border); border-right: 1px solid var(--card-border); border-bottom: 1px solid var(--card-border);">
                <div style="font-weight: 700; color: var(--text-title); font-size: 1.1rem; margin-bottom: 5px;">Selamat Datang di Menu Panduan!</div>
                <div style="color: var(--text-subtitle); font-size: 0.95rem; line-height: 1.5;">
                    Sistem Informasi Klasifikasi Arsip Pintar (SIKAP) dirancang untuk membantu Anda menemukan kode klasifikasi arsip daerah dengan cepat dan presisi. Silakan buka menu di bawah ini untuk mempelajari langkah demi langkah penggunaan sistem.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # --- BAB 1: LOGIN ---
            with st.expander("❯ 1. Cara Masuk (Login)"):
                st.image("panduan_login.png", caption="Tampilan Halaman Login SIKAP", use_container_width=True)
                st.markdown("""
                **Cara Masuk ke Aplikasi (Login):**
                * **❶ Username:** Masukkan nama pengguna yang telah didaftarkan oleh Administrator pada kolom pertama.
                * **❷ Password:** Ketik kata sandi Anda pada kolom kedua. *(Tips: Klik ikon mata di sebelah kanan untuk melihat sandi yang Anda ketik agar tidak salah).*
                * **❸ Tombol Masuk:** Setelah memastikan *username* dan *password* benar, klik tombol biru **MASUK** untuk mulai menggunakan SIKAP.
                """)

            # --- BAB 2: BERANDA ---
            with st.expander("❯ 2. Mengenal Halaman Utama (Beranda)"):
                st.image("panduan_beranda.png", caption="Tampilan Navigasi Halaman Utama SIKAP", use_container_width=True)
                st.markdown("""
                **Mengenal Fitur di Halaman Beranda:**
                * **❶ Pencarian Cepat:** Anda dapat langsung mengetikkan perihal surat pada kolom yang tersedia di tengah layar untuk mencari kode klasifikasi tanpa harus berpindah menu.
                * **❷ Akses Cepat:** Terdapat tiga kartu visual yang berfungsi sebagai jalan pintas (*shortcut*) menuju fitur-fitur utama aplikasi, yaitu Pencarian AI, Jelajah Kode, dan Riwayat Pencarian.
                * **❸ Menu Navigasi:** Di sebelah kiri layar, terdapat *sidebar* biru yang berisi seluruh menu aplikasi untuk mempermudah Anda berpindah halaman.
                """)

            # --- BAB 3: PENCARIAN AI ---
            with st.expander("❯ 3. Cara Menggunakan Pencarian AI (Cerdas)"):
                st.markdown("""
                Ini adalah fitur unggulan SIKAP. Anda tidak perlu menghafal ratusan kode klasifikasi, cukup ceritakan perihal surat Anda secara natural menggunakan bahasa sehari-hari!
                
                **Langkah Awal (Memulai Pencarian):**
                * Klik menu **Pencarian AI** di sidebar kiri atau gunakan kotak pencarian di halaman Beranda.
                * **❶ Ketik Perihal Surat:** Pada kolom yang tersedia, ketikkan perihal surat atau dokumen Anda (Contoh: *"perjalanan dinas pegawai ke kendari"*). Kemudian tekan **Enter** pada *keyboard* Anda.
                """)
                st.image("panduan_ai_awal.png", caption="Proses Mengetik Perihal Surat", use_container_width=True)

                st.markdown("""
                ---
                **Langkah Lanjutan (Membaca Hasil & Validasi):**
                Setelah AI memproses teks Anda, sistem akan menampilkan layar hasil analisis seperti di bawah ini:
                """)
                st.image("panduan_ai_hasil.png", caption="Tampilan Hasil Bedah AI dan Rekomendasi Kode", use_container_width=True)

                st.markdown("""
                * **❷ Inti Substansi (Hasil Bedah AI):** Sistem pertama-tama akan membuang kata-basi dan menangkap inti urusannya saja (Misal: *"Perjalanan Dinas"*). Ini membuktikan AI memahami esensi surat Anda.
                * **❸ Rekomendasi Kode:** AI akan memberikan 3 rekomendasi kode terbaik beserta persentase keyakinannya. Anda dapat **mengklik kotak tersebut (ikon ❯)** untuk melihat rincian nama klasifikasi dan jalur hierarkinya secara lengkap.
                * **❹ Bantu SIKAP Belajar (Validasi):** Setelah membaca detail dari ketiga kode, klik salah satu tombol di paling bawah untuk memilih mana kode yang *paling tepat*. Pilihan Anda akan disimpan untuk membuat AI SIKAP semakin cerdas di masa depan!
                """)

            # --- BAB 4: JELAJAH KODE ---
            with st.expander("❯ 4. Cara Menjelajah Kode Secara Manual"):
                st.markdown("""
                Jika Anda ingin melihat struktur kode secara keseluruhan layaknya buku induk klasifikasi arsip, Anda bisa menggunakan fitur ini untuk menelusuri hierarki tradisional.
                
                **Langkah 1: Membuka Rumpun Utama**
                * **❶ Buka Halaman:** Klik menu **Jelajah Kode** di sidebar kiri. Anda akan melihat daftar Rumpun Utama (Primer) klasifikasi arsip dari 000 hingga 900.
                * **❷ Pilih Rumpun:** Klik pada salah satu kotak Rumpun yang ingin Anda lihat (misalnya: *> 000 - Umum*).
                """)
                st.image("panduan_jelajah_awal.png", caption="Memilih Rumpun Utama Klasifikasi", use_container_width=True)

                st.markdown("""
                ---
                **Langkah 2: Menelusuri Hierarki (Anak & Cucu Kode)**
                Setelah Anda mengklik salah satu Rumpun Utama, struktur di dalamnya akan terbuka kebawah.
                """)
                st.image("panduan_jelajah_buka.png", caption="Struktur Detail Hierarki Kode yang Terbuka", use_container_width=True)

                st.markdown("""
                * **❸ Buka Turunan Kode:** Perhatikan ikon panah kecil (❯) di ujung kanan setiap kotak. Klik kotak tersebut untuk melihat rincian kode turunan di bawahnya.
                * **❹ Pahami Tingkatan (Level):** Untuk memudahkan penglihatan, sistem SIKAP memberikan warna berbeda pada setiap tingkatan klasifikasi:
                    * **Abu-abu (Primer):** Tingkat Rumpun Utama (Misal: 000).
                    * **Biru (Sekunder):** Sub-rumpun urusan (Misal: 000.2).
                    * **Hijau (Tersier):** Rincian urusan (Misal: 000.2.1).
                    * **Kuning/Oranye (Kuartier):** Rincian urusan yang paling spesifik (Misal: 000.2.1.1).
                """)

            # --- BAB 5: RIWAYAT ---
            with st.expander("❯ 5. Melihat dan Menghapus Riwayat Pencarian"):
                st.image("panduan_riwayat.png", caption="Tampilan Halaman Riwayat Pencarian", use_container_width=True)
                st.markdown("""
                Sistem SIKAP secara otomatis mengingat perihal surat yang pernah Anda cari selama menggunakan aplikasi agar Anda tidak perlu mengetik ulang pencarian yang sama.
                
                **Cara Mengelola Riwayat:**
                * **❶ Buka Halaman:** Klik menu **Riwayat** di sidebar kiri.
                * **❷ Daftar Riwayat:** Di sini Anda dapat melihat daftar rekam jejak pencarian Anda, diurutkan dari yang paling baru.
                * **❸ Hapus Riwayat:** Jika ingin membersihkan daftar ini, cukup klik tombol **Hapus Riwayat**. Seluruh riwayat Anda akan langsung dibersihkan dari layar.
                """)

            # --- BAB 6: PETUNJUK PENGGUNAAN (HALAMAN INI) ---
            with st.expander("❯ 6. Membaca Petunjuk Penggunaan"):
                st.image("panduan_petunjuk.png", caption="Tampilan Halaman Petunjuk Penggunaan", use_container_width=True)
                st.markdown("""
                Halaman ini berisi kumpulan panduan lengkap mengenai tata cara penggunaan aplikasi SIKAP dari awal hingga akhir.
                
                **Cara Mengakses Panduan:**
                * **❶ Buka Halaman:** Klik menu **Petunjuk Penggunaan** pada bagian *Lainnya* di sidebar sebelah kiri.
                * **❷ Pilih Topik:** Klik pada salah satu baris panduan (seperti yang sedang Anda lakukan saat ini) untuk melihat langkah-langkah detail beserta gambarnya.
                """)

            # --- BAB 7: TENTANG SIKAP ---
            with st.expander("❯ 7. Informasi Tentang SIKAP"):
                st.image("panduan_tentang.png", caption="Tampilan Halaman Tentang SIKAP", use_container_width=True)
                st.markdown("""
                Halaman ini berisi profil singkat mengenai identitas, tujuan pembuatan, dan versi dari aplikasi SIKAP.
                
                **Cara Mengakses Informasi Aplikasi:**
                * **❶ Buka Halaman:** Klik menu **Tentang SIKAP** pada bagian *Lainnya* di sidebar sebelah kiri.
                * **❷ Lihat Detail Informasi:** Pada halaman ini, Anda dapat membaca deskripsi mengenai latar belakang inovasi SIKAP, beserta informasi **Versi Aplikasi** dan **Status Rilis** untuk memastikan sistem berjalan pada versi resmi.
                """)

            # --- BAB 8: PROFIL & LOGOUT ---
            with st.expander("❯ 8. Pengaturan Profil dan Keluar (Logout)"):
                st.image("panduan_profil.png", caption="Tampilan Halaman Pengaturan Profil", use_container_width=True)
                st.markdown("""
                Halaman ini digunakan untuk mengelola data akun pribadi Anda agar tetap aman.
                
                **Cara Mengubah Profil & Kata Sandi:**
                * **❶ Buka Halaman:** Klik menu **Profil Saya** di sidebar sebelah kiri.
                * **❷ Ubah Nama:** Pada kolom **Nama Lengkap**, Anda dapat memperbaiki ejaan nama atau menambahkan gelar terbaru Anda.
                * **❸ Ubah Kata Sandi (Opsional):** Jika ingin mengganti kata sandi lama, masukkan sandi baru pada kedua kolom yang disediakan. *Catatan: Biarkan kedua kolom ini KOSONG jika Anda hanya ingin mengubah nama tanpa mengganti kata sandi.*
                * **❹ Simpan:** Klik tombol biru **Simpan Perubahan Profil** untuk memperbarui data Anda seketika.
                
                ---
                **Cara Mengakhiri Sesi (Keluar):**
                * **❺ Keluar Aplikasi:** Untuk menjaga keamanan akun, selalu biasakan untuk menekan menu **Keluar** di ujung bawah sidebar sebelah kiri setelah Anda selesai menggunakan aplikasi SIKAP.
                """)

        # --- HALAMAN 7: TENTANG SIKAP ---
        elif st.session_state.page == 'Tentang':
            st.markdown('<div class="section-title">ℹ️ Tentang SIKAP</div>', unsafe_allow_html=True)
            st.write("SIKAP (Sistem Informasi Klasifikasi Arsip Pintar) adalah inovasi digital cerdas yang dirancang untuk membantu aparatur pemerintah daerah dalam mengklasifikasikan dan menemukan kode arsip secara presisi, cepat, dan modern.")
            
            st.markdown("""
            <div style="margin-top: 20px; padding: 15px; border-radius: 8px; background-color: var(--card-bg); color: var(--text-title); border-left: 4px solid #009DFF; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                <strong>Versi Aplikasi:</strong> Ver. 1.0<br>
                <strong>Status:</strong> Rilis Perdana
            </div>
            """, unsafe_allow_html=True)
            
        # --- HALAMAN 8: PROFIL SAYA ---
        elif st.session_state.page == 'Profil Saya':
            st.markdown('<div class="section-title" style="display:flex; align-items:center; gap:8px;"><span class="material-symbols-rounded" style="color:#009DFF; font-size:1.8rem;">manage_accounts</span> Pengaturan Profil</div>', unsafe_allow_html=True)
            
            # Form Pengaturan Akun
            st.markdown('<div style="display:flex; align-items:center; gap:8px; margin-top:20px; margin-bottom:10px;"><span class="material-symbols-rounded" style="color:#475569; font-size:1.4rem;">person_edit</span><span style="font-weight:700; color:#0F172A; font-size:1.1rem; font-family:\'Poppins\';">Ubah Data Pengguna</span></div>', unsafe_allow_html=True)
            
            with st.container():
                # Input Data
                nama_baru = st.text_input("Nama Lengkap", value=st.session_state['nama'])
                role_pilihan = st.selectbox("Role / Jabatan", ["admin", "user"], index=0 if st.session_state['role'] == "admin" else 1)
                
                st.markdown("<hr style='margin: 20px 0; opacity: 0.1;'>", unsafe_allow_html=True)
                st.caption("Biarkan kosong jika tidak ingin mengubah kata sandi")
                pass_baru = st.text_input("Kata Sandi Baru", type="password")
                konfirmasi_pass = st.text_input("Konfirmasi Kata Sandi Baru", type="password")
                
                if st.button("Simpan Perubahan Profil", type="primary", width="stretch"):
                    if pass_baru and pass_baru != konfirmasi_pass:
                        st.markdown('<div style="background:#FEE2E2; border-left:4px solid #EF4444; padding:12px; border-radius:8px; display:flex; align-items:center; gap:8px;"><span class="material-symbols-rounded" style="color:#EF4444;">error</span><span style="color:#7F1D1D; font-size:0.9rem;">Kata sandi tidak cocok.</span></div>', unsafe_allow_html=True)
                    else:
                        # Proses Update Database
                        df_users = pd.read_csv('pengguna.csv', dtype=str)
                        # Cari baris berdasarkan username yang unik
                        mask = df_users['username'] == st.session_state['username']
                        
                        if not df_users[mask].empty:
                            # 1. INGAT NAMA LAMA SEBELUM DITIMPA
                            nama_lama = st.session_state['nama'] 
                            
                            df_users.loc[mask, 'nama_lengkap'] = nama_baru
                            df_users.loc[mask, 'role'] = role_pilihan
                            if pass_baru:
                                df_users.loc[mask, 'password'] = pass_baru
                            
                            # Simpan Permanen ke pengguna.csv
                            df_users.to_csv('pengguna.csv', index=False)
                            sync_to_drive('pengguna.csv')
                            
                            # ========================================================
                            # 2. SINKRONISASI NAMA BARU KE SEMUA RIWAYAT & FEEDBACK
                            # ========================================================
                            if nama_lama != nama_baru:
                                # Update di Riwayat Pencarian
                                file_riwayat = 'riwayat_pencarian.csv'
                                if os.path.exists(file_riwayat):
                                    df_r = pd.read_csv(file_riwayat, dtype=str)
                                    if 'nama' in df_r.columns:
                                        df_r.loc[df_r['nama'] == nama_lama, 'nama'] = nama_baru
                                        df_r.to_csv(file_riwayat, index=False)
                                        sync_to_drive(file_riwayat)
                                        
                                # Update di Feedback AI
                                file_feedback = 'feedback_ai.csv'
                                if os.path.exists(file_feedback):
                                    df_f = pd.read_csv(file_feedback, dtype=str)
                                    if 'nama' in df_f.columns:
                                        df_f.loc[df_f['nama'] == nama_lama, 'nama'] = nama_baru
                                        df_f.to_csv(file_feedback, index=False)
                                        sync_to_drive(file_feedback)
                            # ========================================================
                            
                            # 3. Update Session State agar tampilan langsung berubah
                            st.session_state['nama'] = nama_baru
                            st.session_state['role'] = role_pilihan
                            
                            st.markdown('<div style="background:#D1FAE5; border-left:4px solid #10B981; padding:12px; border-radius:8px; display:flex; align-items:center; gap:8px; margin-top:10px;"><span class="material-symbols-rounded" style="color:#10B981;">check_circle</span><span style="color:#064E3B; font-size:0.9rem;">Profil berhasil diperbarui!</span></div>', unsafe_allow_html=True)
                            st.rerun()
        

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")

# --- 5. PENGATUR HALAMAN (ROUTER) ---
if not st.session_state.get('logged_in', False):
    halaman_login()
else:
    halaman_utama()
