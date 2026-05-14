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

# --- FUNGSI RIWAYAT PERMANEN (CSV) ---
def simpan_riwayat_csv(nama_user, pencarian):
    file_riwayat = 'riwayat_pencarian.csv'
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_baru = pd.DataFrame({'waktu': [waktu], 'nama': [nama_user], 'pencarian': [pencarian]})
    
    # PERBAIKAN: Baca, Gabungkan, lalu Simpan (Anti Hilang Header)
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
            # Pastikan kolom 'nama' dan 'pencarian' ada untuk mencegah error
            if 'nama' in df_riwayat.columns and 'pencarian' in df_riwayat.columns:
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

# 1. Menarik API Key dengan aman (Bisa jalan di lokal maupun di Streamlit Cloud)
try:
    # Membaca dari Streamlit Secrets jika di Cloud
    api_key = st.secrets["GROQ_API_KEY"]
except:
    # Masukkan API Key manual HANYA untuk tes di laptop lokal (Hapus sebelum di-push ke GitHub!)
    api_key = "MASUKKAN_API_KEY_GROQ_DI_SINI_UNTUK_TES_LOKAL" 

client = Groq(api_key=api_key)

# ====================================================
# 1.5. Fungsi "Sapu Jagat" (Validasi Pasca-Ekstraksi)
# ====================================================
KATA_KERJA_TERLARANG = {
    "menghadiri", "melaksanakan", "menyampaikan", "memohon",
    "mengundang", "melaporkan", "menindaklanjuti", "mengusulkan",
    "mengikuti", "membahas", "mengadakan", "meminta"
}

def sapu_kata_kerja_bocor(teks: str) -> str:
    """Fungsi untuk otomatis menghapus kata kerja yang lolos dari AI"""
    token = teks.lower().split()
    bocor = set(token) & KATA_KERJA_TERLARANG
    
    if bocor:
        # Tampilkan pesan ke layar agar Anda tahu AI-nya sedang bandel
        st.warning(f"🧹 Sistem SIKAP menyapu kata kerja yang lolos dari AI: {bocor}")
        
        # Otomatis hapus kata terlarang itu tanpa menghentikan aplikasi
        token_bersih = [kata for kata in token if kata not in KATA_KERJA_TERLARANG]
        return " ".join(token_bersih)
        
    return teks

from thefuzz import process

# ====================================================
# BANK DATA ANDA (Sudah dikonversi otomatis ke format pintar)
# ====================================================
BANK_CONTOH = [
    # [PENDIDIKAN]
    ("Pemotongan dana bantuan operasional sekolah (BOS) dan akreditasi puskesmas", "bantuan operasional sekolah, bos, akreditasi puskesmas"),
    ("Penetapan kuota dan formasi penerimaan peserta didik baru (PPDB) jenjang SD dan SMP", "penerimaan peserta didik baru, ppdb"),
    ("Pengajuan tunjangan profesi guru (TPG) dan tunjangan khusus guru daerah terpencil", "tunjangan profesi guru, tunjangan khusus guru"),
    ("Verifikasi dan validasi data pokok pendidikan (Dapodik) satuan pendidikan negeri", "data pokok pendidikan, dapodik, satuan pendidikan"),
    ("Penetapan standar pelayanan minimal (SPM) bidang pendidikan dasar dan menengah", "standar pelayanan minimal, pendidikan dasar menengah"),
    ("Pengembangan kurikulum muatan lokal dan program kegiatan ekstrakurikuler", "kurikulum muatan lokal, ekstrakurikuler"),
    ("Penyaluran beasiswa bagi siswa kurang mampu dan berprestasi tingkat kabupaten", "beasiswa siswa, kurang mampu berprestasi"),
    ("Pembangunan ruang kelas baru (RKB) dan pengadaan meubelair sekolah dasar", "pembangunan ruang kelas, pengadaan meubelair sekolah"),
    # [KESEHATAN]
    ("Pelaksanaan program Jaminan Kesehatan Nasional (JKN) dan National Health Account (NHA)", "jaminan kesehatan nasional, national health account"),
    ("Laporan progres pembangunan gedung perpustakaan daerah dan rehab puskesmas", "pembangunan gedung, rehab bangunan"),
    ("Penyelenggaraan imunisasi dasar lengkap dan surveilans penyakit menular", "imunisasi dasar lengkap, surveilans penyakit menular"),
    ("Penanganan kasus gizi buruk dan stunting balita di wilayah terpencil", "gizi buruk, stunting balita"),
    ("Pengawasan peredaran obat tradisional dan kosmetik ilegal di pasar daerah", "pengawasan obat tradisional, kosmetik ilegal"),
    ("Akreditasi fasilitas kesehatan tingkat pertama (FKTP) dan rumah sakit daerah", "akreditasi fasilitas kesehatan, fktp, rumah sakit daerah"),
    ("Pelaksanaan program keluarga berencana (KB) dan kesehatan reproduksi remaja", "keluarga berencana, kesehatan reproduksi remaja"),
    ("Pengadaan alat kesehatan dan obat-obatan untuk puskesmas pembantu (pustu)", "pengadaan alat kesehatan, obat obatan, puskesmas pembantu"),
    ("Penanggulangan kejadian luar biasa (KLB) demam berdarah dengue dan malaria", "kejadian luar biasa, demam berdarah dengue, malaria"),
    ("Rekrutmen dan penempatan tenaga kesehatan di daerah terpencil dan kepulauan", "tenaga kesehatan, daerah terpencil kepulauan"),
    # [PEKERJAAN UMUM & PENATAAN RUANG]
    ("Laporan progres pemeliharaan jalan bebas hambatan dan pengelolaan irigasi rawa", "pemeliharaan jalan bebas hambatan, pengelolaan irigasi rawa"),
    ("Pengajuan Rencana Detail Tata Ruang (RDTR) dan Rencana Tata Bangunan dan Lingkungan (RTBL)", "rencana detail tata ruang, rencana tata bangunan dan lingkungan"),
    ("Persetujuan penataan bangunan dan pengelolaan gedung rumah negara", "penataan bangunan, pengelolaan rumah negara"),
    ("Penanganan jalan rusak berat dan pembangunan jembatan penghubung desa terisolir", "penanganan jalan rusak, pembangunan jembatan desa"),
    ("Pengelolaan sistem penyediaan air minum (SPAM) regional dan jaringan perpipaan", "sistem penyediaan air minum, spam, jaringan perpipaan"),
    ("Penertiban bangunan tanpa izin mendirikan bangunan (IMB) di kawasan strategis", "penertiban bangunan, izin mendirikan bangunan, imb"),
    ("Pengelolaan drainase perkotaan dan penanggulangan genangan banjir permukiman", "drainase perkotaan, penanggulangan banjir permukiman"),
    ("Revisi Rencana Tata Ruang Wilayah (RTRW) dan penetapan kawasan lindung daerah", "rencana tata ruang wilayah, rtrw, kawasan lindung"),
    # [PERUMAHAN RAKYAT & KAWASAN PERMUKIMAN]
    ("Penyediaan rumah susun sederhana sewa (rusunawa) bagi masyarakat berpenghasilan rendah", "rumah susun sederhana sewa, rusunawa, masyarakat berpenghasilan rendah"),
    ("Verifikasi dan penetapan penerima bantuan stimulan perumahan swadaya (BSPS)", "bantuan stimulan perumahan swadaya, bsps"),
    ("Penataan kawasan kumuh perkotaan dan peningkatan kualitas permukiman nelayan", "penataan kawasan kumuh, permukiman nelayan"),
    ("Penanganan rumah tidak layak huni (RTLH) dan penyediaan sanitasi berbasis masyarakat", "rumah tidak layak huni, rtlh, sanitasi berbasis masyarakat"),
    ("Penetapan lokasi perumahan dan permukiman serta prasarana kawasan transmigrasi lokal", "lokasi perumahan permukiman, kawasan transmigrasi"),
    # [KETENTERAMAN, KETERTIBAN UMUM & PERLINDUNGAN MASYARAKAT]
    ("Penertiban pedagang kaki lima (PKL) dan penegakan peraturan daerah di kawasan terlarang", "penertiban pedagang kaki lima, penegakan peraturan daerah"),
    ("Penyelenggaraan pelatihan bela negara dan kesiapsiagaan satuan perlindungan masyarakat", "pelatihan bela negara, kesiapsiagaan perlindungan masyarakat"),
    ("Penanganan gangguan ketertiban umum akibat konflik sosial antarkelompok warga", "gangguan ketertiban umum, konflik sosial"),
    ("Pengamanan aset daerah dan penertiban penghuni liar bangunan milik pemerintah", "pengamanan aset daerah, penertiban penghuni liar bangunan"),
    ("Peningkatan kapasitas Satuan Polisi Pamong Praja (Satpol PP) dan Pemadam Kebakaran", "kapasitas satuan polisi pamong praja, pemadam kebakaran"),
    # [SOSIAL]
    ("Verifikasi dan validasi data terpadu kesejahteraan sosial (DTKS) fakir miskin", "data terpadu kesejahteraan sosial, dtks, fakir miskin"),
    ("Penyaluran bantuan sosial tunai (BST) dan program keluarga harapan (PKH)", "bantuan sosial tunai, bst, program keluarga harapan, pkh"),
    ("Penyelenggaraan panti sosial dan rehabilitasi sosial gelandangan dan pengemis", "rehabilitasi sosial, gelandangan pengemis"),
    ("Penanganan penyandang disabilitas dan lanjut usia terlantar melalui program asistensi", "penyandang disabilitas, lanjut usia terlantar, asistensi sosial"),
    ("Penanggulangan kemiskinan dan pemberdayaan fakir miskin melalui kelompok usaha bersama", "penanggulangan kemiskinan, kelompok usaha bersama"),
    ("Rekomendasi pengumpulan sumbangan sosial dan undian gratis berhadiah", "pengumpulan sumbangan sosial, undian gratis berhadiah"),
    # [KETENAGAKERJAAN]
    ("Penyelenggaraan pelatihan berbasis kompetensi di balai latihan kerja (BLK) daerah", "pelatihan berbasis kompetensi, balai latihan kerja"),
    ("Penetapan upah minimum kabupaten/kota (UMK) dan upah minimum sektoral", "upah minimum kabupaten, umk, upah minimum sektoral"),
    ("Penanganan perselisihan hubungan industrial dan mediasi pemutusan hubungan kerja", "perselisihan hubungan industrial, mediasi pemutusan hubungan kerja"),
    ("Pengawasan pelaksanaan norma kerja dan keselamatan kesehatan kerja (K3)", "norma kerja, keselamatan kesehatan kerja, k3"),
    ("Penempatan tenaga kerja lokal dan penerbitan izin penggunaan tenaga kerja asing (TKA)", "penempatan tenaga kerja, izin tenaga kerja asing, tka"),
    ("Penyusunan data dan informasi pasar kerja serta bursa kerja online daerah", "informasi pasar kerja, bursa kerja online"),
    ("Pembinaan lembaga pelatihan kerja swasta dan sertifikasi kompetensi kerja nasional", "lembaga pelatihan kerja, sertifikasi kompetensi kerja"),
    ("Pelaksanaan program magang dalam negeri bagi pencari kerja pemuda", "program magang dalam negeri, pencari kerja pemuda"),
    # [PEMBERDAYAAN PEREMPUAN & PERLINDUNGAN ANAK]
    ("Penanganan kasus kekerasan dalam rumah tangga (KDRT) dan perlindungan perempuan", "kekerasan dalam rumah tangga, kdrt, perlindungan perempuan"),
    ("Penyelenggaraan layanan terpadu perlindungan perempuan dan anak (P2A)", "layanan terpadu perlindungan perempuan, perlindungan anak"),
    ("Pemberdayaan ekonomi perempuan kepala keluarga melalui usaha mikro kecil", "pemberdayaan ekonomi perempuan, usaha mikro kecil"),
    ("Pencegahan pernikahan dini dan penanganan anak putus sekolah", "pencegahan pernikahan dini, anak putus sekolah"),
    ("Pemenuhan hak anak dan penilaian kabupaten/kota layak anak (KLA)", "hak anak, kabupaten kota layak anak, kla"),
    ("Penanganan anak berhadapan dengan hukum (ABH) dan anak terlantar", "anak berhadapan hukum, abh, anak terlantar"),
    # [PANGAN]
    ("Penyelenggaraan cadangan pangan pemerintah daerah dan lumbung pangan masyarakat", "cadangan pangan pemerintah daerah, lumbung pangan masyarakat"),
    ("Pemantauan harga dan pasokan komoditas pangan strategis di tingkat konsumen", "harga pasokan komoditas pangan strategis"),
    ("Penanganan kerawanan pangan dan penyaluran bantuan pangan non tunai (BPNT)", "kerawanan pangan, bantuan pangan non tunai, bpnt"),
    ("Pengawasan keamanan pangan segar asal tumbuhan dan hewan di pasar tradisional", "keamanan pangan segar, tumbuhan hewan, pasar tradisional"),
    ("Penguatan ketahanan pangan daerah melalui diversifikasi konsumsi pangan lokal", "ketahanan pangan, diversifikasi konsumsi pangan lokal"),
    # [PERTANAHAN]
    ("Fasilitasi penetapan dan penyelesaian sengketa tanah garapan dan tanah ulayat", "sengketa tanah garapan, tanah ulayat"),
    ("Inventarisasi dan penatagunaan tanah untuk kepentingan umum dan proyek strategis", "penatagunaan tanah, kepentingan umum, proyek strategis"),
    ("Penertiban penguasaan, pemilikan, penggunaan dan pemanfaatan tanah (P4T)", "penguasaan pemilikan penggunaan tanah, p4t"),
    ("Pengadaan tanah bagi pembangunan jalan tol dan infrastruktur publik daerah", "pengadaan tanah, pembangunan infrastruktur publik"),
    ("Redistribusi tanah objek landreform dan pensertifikatan tanah transmigrasi", "redistribusi tanah objek landreform, pensertifikatan tanah transmigrasi"),
    # [LINGKUNGAN HIDUP]
    ("Pembahasan dokumen analisis mengenai dampak lingkungan (AMDAL) dan UKL-UPL", "analisis mengenai dampak lingkungan, amdal, ukl upl"),
    ("Penerbitan izin pembuangan air limbah dan izin penyimpanan sementara limbah B3", "izin pembuangan air limbah, izin penyimpanan limbah b3"),
    ("Pemantauan kualitas udara ambien dan pengendalian pencemaran udara industri", "kualitas udara ambien, pencemaran udara industri"),
    ("Penilaian proper dan penghargaan lingkungan hidup perusahaan di daerah", "penilaian proper, penghargaan lingkungan hidup"),
    ("Pengelolaan tempat pemrosesan akhir (TPA) sampah dan fasilitas daur ulang", "tempat pemrosesan akhir, tpa sampah, daur ulang"),
    ("Rehabilitasi mangrove dan terumbu karang di kawasan pesisir terdampak abrasi", "rehabilitasi mangrove, terumbu karang, kawasan pesisir"),
    # [ADMINISTRASI KEPENDUDUKAN & CATATAN SIPIL]
    ("Laporan pelaksanaan Sistem Informasi Administrasi Kependudukan (SIAK) dan pencatatan sipil", "sistem informasi administrasi kependudukan, pencatatan sipil"),
    ("Percepatan perekaman KTP elektronik (KTP-el) dan penerbitan kartu identitas anak (KIA)", "perekaman ktp elektronik, kartu identitas anak, kia"),
    ("Penerbitan akta kelahiran, akta kematian, dan akta perkawinan bagi penduduk rentan", "akta kelahiran, akta kematian, akta perkawinan"),
    ("Pembersihan data ganda dan data anomali dalam database kependudukan daerah", "data ganda, data anomali, database kependudukan"),
    ("Sosialisasi pemutakhiran data mandiri penduduk melalui aplikasi layanan adminduk", "pemutakhiran data penduduk, layanan administrasi kependudukan"),
    # [PEMBERDAYAAN MASYARAKAT & DESA]
    ("Penyaluran dana desa dan alokasi dana desa (ADD) tahap pertama tahun anggaran berjalan", "dana desa, alokasi dana desa, add"),
    ("Pembinaan dan pengawasan pengelolaan keuangan badan usaha milik desa (BUMDes)", "pengelolaan keuangan, badan usaha milik desa, bumdes"),
    ("Fasilitasi musyawarah desa (musdes) penetapan rencana pembangunan jangka menengah desa", "musyawarah desa, rencana pembangunan jangka menengah desa, rpjmdes"),
    ("Verifikasi laporan pertanggungjawaban (LPJ) penggunaan dana desa oleh pemerintah desa", "laporan pertanggungjawaban, penggunaan dana desa"),
    ("Pembentukan dan pembinaan kader pemberdayaan masyarakat desa (KPMD)", "kader pemberdayaan masyarakat desa, kpmd"),
    ("Pengembangan kawasan perdesaan dan pembangunan desa mandiri melalui program unggulan", "kawasan perdesaan, desa mandiri"),
    # [PENGENDALIAN PENDUDUK & KB]
    ("Distribusi alokon (alat kontrasepsi) dan obat-obatan KB ke fasilitas kesehatan", "distribusi alat kontrasepsi, alokon, obat kb"),
    ("Peningkatan kapasitas penyuluh keluarga berencana (PKB) dan kader posyandu", "penyuluh keluarga berencana, pkb, kader posyandu"),
    ("Penyelenggaraan kampung keluarga berkualitas dan program genre remaja", "kampung keluarga berkualitas, program genre remaja"),
    ("Pendataan keluarga dan pemutakhiran basis data keluarga Indonesia (BDKI)", "pendataan keluarga, basis data keluarga indonesia, bdki"),
    # [PERHUBUNGAN]
    ("Sertifikasi uji tipe kendaraan bermotor dan pengesahan kualifikasi petugas terminal", "sertifikasi uji tipe kendaraan bermotor, kualifikasi petugas terminal"),
    ("Penetapan tarif angkutan umum dalam trayek perkotaan dan perdesaan", "tarif angkutan umum, trayek perkotaan perdesaan"),
    ("Pengujian berkala kendaraan bermotor (KIR) dan penertiban kendaraan over dimensi", "pengujian berkala kendaraan bermotor, kir, kendaraan over dimensi"),
    ("Penetapan lokasi terminal tipe C dan pembangunan halte bus rapid transit (BRT)", "terminal tipe c, halte bus rapid transit, brt"),
    ("Pengawasan keselamatan pelayaran di alur sungai dan pengelolaan pelabuhan sungai", "keselamatan pelayaran, alur sungai, pelabuhan sungai"),
    ("Penerbitan izin trayek angkutan kota dan perpanjangan kartu pengawasan kendaraan", "izin trayek angkutan kota, kartu pengawasan kendaraan"),
    # [KOMUNIKASI & INFORMATIKA]
    ("Permohonan layanan sertifikasi elektronik dan evaluasi tata kelola e-government", "sertifikasi elektronik, e government"),
    ("Pemantauan layanan jaringan telekomunikasi dan pengawasan keamanan informasi", "jaringan telekomunikasi, keamanan informasi"),
    ("Pengelolaan sistem penghubung layanan pemerintah (SPLP) dan interoperabilitas data", "sistem penghubung layanan pemerintah, splp, interoperabilitas data"),
    ("Pengembangan aplikasi SPBE dan audit keamanan sistem informasi pemerintah daerah", "aplikasi spbe, audit keamanan sistem informasi"),
    ("Diseminasi informasi publik melalui media center dan media sosial resmi pemerintah", "diseminasi informasi publik, media center"),
    ("Pengelolaan nama domain subdomain pemerintah daerah dan hosting website OPD", "nama domain subdomain, hosting website opd"),
    ("Pembangunan infrastruktur jaringan fiber optik pemerintah daerah dan wi-fi publik", "infrastruktur jaringan fiber optik, wi fi publik"),
    # [KOPERASI, USAHA KECIL & MENENGAH]
    ("Penerbitan izin usaha simpan pinjam (USP) koperasi primer dan pembubaran koperasi", "izin usaha simpan pinjam, koperasi primer"),
    ("Pembinaan dan pemeringkatan koperasi aktif dan koperasi berprestasi tingkat kabupaten", "pemeringkatan koperasi aktif, koperasi berprestasi"),
    ("Fasilitasi akses pembiayaan KUR (kredit usaha rakyat) bagi pelaku UMKM daerah", "kredit usaha rakyat, kur, pembiayaan umkm"),
    ("Penyelenggaraan pelatihan kewirausahaan dan pendampingan usaha mikro naik kelas", "pelatihan kewirausahaan, pendampingan usaha mikro"),
    ("Pengembangan sentra UMKM dan klaster produk unggulan daerah berbasis potensi lokal", "sentra umkm, klaster produk unggulan daerah"),
    ("Penerbitan nomor induk berusaha (NIB) dan sertifikat halal produk usaha mikro", "nomor induk berusaha, nib, sertifikat halal produk usaha mikro"),
    ("Revitalisasi koperasi unit desa (KUD) dan penguatan modal usaha kelompok tani", "koperasi unit desa, kud, modal usaha kelompok tani"),
    # [PENANAMAN MODAL]
    ("Fasilitasi penyelesaian masalah pencabutan pembatalan perizinan penanaman modal asing", "pencabutan pembatalan perizinan penanaman modal"),
    ("Penyelenggaraan promosi investasi daerah dan forum temu investor dalam negeri", "promosi investasi daerah, forum temu investor"),
    ("Penerbitan izin prinsip penanaman modal dalam negeri (PMDN) sektor pariwisata", "izin prinsip penanaman modal dalam negeri, pmdn"),
    ("Pemantauan dan pengawasan kepatuhan pelaksanaan kegiatan usaha penanaman modal", "kepatuhan pelaksanaan kegiatan usaha, penanaman modal"),
    ("Penyusunan peta potensi investasi daerah dan profil peluang usaha unggulan", "peta potensi investasi, peluang usaha unggulan"),
    ("Pengelolaan sistem pelayanan perizinan berusaha terintegrasi secara elektronik (OSS)", "perizinan berusaha terintegrasi, oss, elektronik"),
    # [KEPEMUDAAN & OLAHRAGA]
    ("Pembinaan organisasi kepemudaan dan seleksi peserta pertukaran pemuda antar negara", "organisasi kepemudaan, pertukaran pemuda antar negara"),
    ("Penyelenggaraan pekan olahraga daerah (PORDA) dan pekan olahraga pelajar daerah", "pekan olahraga daerah, porda, pekan olahraga pelajar"),
    ("Pembangunan dan rehabilitasi sarana prasarana olahraga stadion dan GOR daerah", "rehabilitasi sarana prasarana olahraga, stadion gor"),
    ("Pemberian penghargaan prestasi olahraga dan bantuan atlet berprestasi internasional", "penghargaan prestasi olahraga, bantuan atlet berprestasi"),
    ("Pengembangan sentra pembinaan olahraga prestasi dan pusat pendidikan latihan pelajar", "sentra pembinaan olahraga prestasi, pusat pendidikan latihan pelajar"),
    ("Fasilitasi kegiatan kepramukaan dan pembinaan komunitas pemuda kreatif daerah", "kegiatan kepramukaan, komunitas pemuda kreatif"),
    # [STATISTIK]
    ("Penyusunan profil daerah dan publikasi daerah dalam angka kabupaten tahun berjalan", "profil daerah, daerah dalam angka"),
    ("Pengelolaan dan pemutakhiran data statistik sektoral OPD melalui portal satu data", "data statistik sektoral, portal satu data"),
    ("Koordinasi penyelenggaraan sensus penduduk dan survei sosial ekonomi nasional", "sensus penduduk, survei sosial ekonomi nasional"),
    ("Penyusunan indeks pembangunan manusia (IPM) dan indeks kesenjangan kemiskinan daerah", "indeks pembangunan manusia, ipm, indeks kesenjangan kemiskinan"),
    # [PERSANDIAN & KEAMANAN INFORMASI]
    ("Pengelolaan sertifikat digital dan infrastruktur kunci publik (IKP) pemerintah daerah", "sertifikat digital, infrastruktur kunci publik, ikp"),
    ("Pengamanan komunikasi sandi antar instansi pemerintah dan klasifikasi informasi rahasia", "komunikasi sandi antar instansi, informasi rahasia"),
    ("Audit keamanan informasi dan penanganan insiden siber pada sistem pemerintah daerah", "audit keamanan informasi, insiden siber"),
    # [KEBUDAYAAN]
    ("Penetapan warisan budaya tak benda (WBTB) dan cagar budaya peringkat kabupaten", "warisan budaya tak benda, wbtb, cagar budaya"),
    ("Penyelenggaraan festival kesenian daerah dan pagelaran seni pertunjukan tradisional", "festival kesenian daerah, seni pertunjukan tradisional"),
    ("Pembinaan sanggar seni budaya dan pelestarian bahasa daerah yang terancam punah", "sanggar seni budaya, pelestarian bahasa daerah"),
    ("Pendataan dan registrasi cagar budaya serta penerbitan izin membawa benda cagar budaya", "registrasi cagar budaya, izin benda cagar budaya"),
    ("Revitalisasi museum daerah dan pengelolaan koleksi benda bersejarah peninggalan lokal", "revitalisasi museum daerah, koleksi benda bersejarah"),
    # [PERPUSTAKAAN]
    ("Penyelenggaraan pameran perpustakaan keliling dan donasi buku", "perpustakaan keliling, donasi buku"),
    ("Pengembangan koleksi bahan pustaka dan layanan perpustakaan digital daerah", "koleksi bahan pustaka, perpustakaan digital daerah"),
    ("Akreditasi perpustakaan desa/kelurahan dan perpustakaan sekolah tingkat dasar", "akreditasi perpustakaan desa, perpustakaan sekolah"),
    ("Peningkatan minat baca masyarakat melalui gerakan literasi dan taman baca masyarakat", "minat baca masyarakat, gerakan literasi, taman baca"),
    ("Pengelolaan deposit karya cetak dan karya rekam daerah sebagai koleksi wajib simpan", "deposit karya cetak karya rekam, koleksi wajib simpan"),
    # [KEARSIPAN]
    ("Persetujuan draf jadwal retensi arsip dan pemusnahan arsip inaktif", "jadwal retensi arsip, pemusnahan arsip inaktif"),
    ("Penyelamatan arsip statis bernilai sejarah dan alih media arsip konvensional", "arsip statis bernilai sejarah, alih media arsip"),
    ("Akreditasi lembaga kearsipan daerah dan sertifikasi kompetensi pengelola arsip", "akreditasi lembaga kearsipan, sertifikasi kompetensi arsiparis"),
    ("Penerapan sistem informasi kearsipan daerah (SIKD) dan pengelolaan arsip elektronik", "sistem informasi kearsipan daerah, sikd, arsip elektronik"),
    ("Penyerahan arsip statis dari OPD kepada lembaga kearsipan daerah kabupaten", "penyerahan arsip statis, lembaga kearsipan daerah"),
    # [KELAUTAN & PERIKANAN]
    ("Penerbitan izin usaha perikanan tangkap di bawah 12 mil dan izin budidaya ikan", "izin usaha perikanan tangkap, izin budidaya ikan"),
    ("Pengawasan sumber daya kelautan dan penanganan illegal fishing di perairan daerah", "sumber daya kelautan, illegal fishing, perairan daerah"),
    ("Pengembangan kawasan budidaya ikan air tawar dan pembinaan kelompok pembudidaya ikan", "budidaya ikan air tawar, kelompok pembudidaya ikan"),
    ("Pengelolaan tempat pelelangan ikan (TPI) dan pembinaan nelayan kecil pesisir", "tempat pelelangan ikan, tpi, nelayan kecil pesisir"),
    ("Pembangunan cold storage dan sarana penanganan ikan segar di sentra perikanan", "cold storage, sarana penanganan ikan, sentra perikanan"),
    ("Rehabilitasi terumbu karang dan ekosistem padang lamun di kawasan konservasi laut", "rehabilitasi terumbu karang, ekosistem padang lamun, konservasi laut"),
    # [PARIWISATA]
    ("Penyusunan rencana induk pembangunan kepariwisataan daerah (RIPPAR) kabupaten", "rencana induk kepariwisataan daerah, rippar"),
    ("Pengembangan desa wisata dan pemberdayaan masyarakat sadar wisata (pokdarwis)", "desa wisata, masyarakat sadar wisata, pokdarwis"),
    ("Penerbitan tanda daftar usaha pariwisata (TDUP) hotel, restoran, dan agen perjalanan", "tanda daftar usaha pariwisata, tdup, hotel restoran agen perjalanan"),
    ("Penyelenggaraan event pariwisata internasional dan promosi destinasi wisata unggulan", "event pariwisata internasional, promosi destinasi wisata"),
    ("Peningkatan kapasitas pemandu wisata lokal dan sertifikasi usaha pariwisata", "pemandu wisata lokal, sertifikasi usaha pariwisata"),
    ("Pengembangan infrastruktur kawasan wisata alam dan ekowisata pesisir", "infrastruktur kawasan wisata alam, ekowisata pesisir"),
    ("Pengelolaan sistem informasi pariwisata daerah dan kalender event wisata tahunan", "sistem informasi pariwisata, kalender event wisata"),
    # [PERTANIAN]
    ("Pengendalian Organisme Pengganggu Tumbuhan (OPT) dan Pengendalian Hama Terpadu (PHT)", "organisme pengganggu tumbuhan, pengendalian hama terpadu"),
    ("Penyaluran pupuk bersubsidi dan benih unggul bersertifikat kepada kelompok tani", "pupuk bersubsidi, benih unggul bersertifikat, kelompok tani"),
    ("Pengembangan kawasan pertanian komoditas unggulan dan agribisnis hortikultura", "kawasan pertanian komoditas unggulan, agribisnis hortikultura"),
    ("Penerbitan izin usaha pertanian dan rekomendasi impor komoditas pertanian strategis", "izin usaha pertanian, impor komoditas pertanian"),
    ("Asuransi usaha tani padi (AUTP) dan asuransi usaha ternak sapi (AUTS)", "asuransi usaha tani padi, autp, asuransi usaha ternak sapi"),
    ("Pembangunan jaringan irigasi tersier dan rehabilitasi embung pertanian desa", "jaringan irigasi tersier, rehabilitasi embung pertanian"),
    ("Pengembangan mekanisasi pertanian dan pengadaan alsintan bagi kelompok tani", "mekanisasi pertanian, alsintan, kelompok tani"),
    # [KEHUTANAN & PERKEBUNAN]
    ("Penerbitan izin usaha pemanfaatan hasil hutan kayu pada hutan hak/hutan rakyat", "izin usaha pemanfaatan hasil hutan kayu, hutan rakyat"),
    ("Rehabilitasi hutan dan lahan kritis melalui program penghijauan dan reklamasi", "rehabilitasi hutan lahan kritis, penghijauan reklamasi"),
    ("Pengawasan peredaran hasil hutan dan penanganan kasus perambahan kawasan hutan", "peredaran hasil hutan, perambahan kawasan hutan"),
    ("Pengembangan perkebunan kelapa sawit rakyat dan penerbitan sertifikat RSPO", "perkebunan kelapa sawit rakyat, sertifikat rspo"),
    ("Pengendalian kebakaran hutan dan lahan (karhutla) serta pembentukan MPA desa", "kebakaran hutan lahan, karhutla, masyarakat peduli api"),
    # [ENERGI & SUMBER DAYA MINERAL]
    ("Penerbitan Sertifikat Laik Operasi (SLO) dan Izin Usaha Pertambangan (IUP) Batubara", "sertifikat laik operasi, izin usaha pertambangan batubara"),
    ("Pengelolaan air tanah dan penerbitan surat izin pengambilan air tanah (SIPA)", "pengelolaan air tanah, surat izin pengambilan air tanah, sipa"),
    ("Pembangunan jaringan listrik pedesaan dan elektrifikasi daerah terpencil 3T", "jaringan listrik pedesaan, elektrifikasi daerah terpencil"),
    ("Pengembangan energi baru terbarukan (EBT) dan pembangkit listrik tenaga surya", "energi baru terbarukan, ebt, pembangkit listrik tenaga surya"),
    ("Pengawasan keselamatan pengusahaan tambang mineral bukan logam dan batuan", "keselamatan pengusahaan tambang, mineral bukan logam, batuan"),
    # [PERDAGANGAN]
    ("Penerbitan surat izin usaha perdagangan (SIUP) dan tanda daftar perusahaan (TDP)", "surat izin usaha perdagangan, siup, tanda daftar perusahaan"),
    ("Pengawasan ketersediaan stok dan distribusi barang kebutuhan pokok di pasar", "stok distribusi barang kebutuhan pokok, pasar"),
    ("Penyelenggaraan pasar lelang daerah dan pameran produk ekspor unggulan", "pasar lelang daerah, pameran produk ekspor"),
    ("Penerbitan rekomendasi izin usaha pergudangan dan pengendalian distribusi pupuk", "izin usaha pergudangan, distribusi pupuk"),
    ("Pembangunan dan revitalisasi pasar rakyat tradisional dan pasar seni daerah", "revitalisasi pasar rakyat tradisional, pasar seni"),
    ("Perlindungan konsumen dan pengawasan barang beredar tidak memenuhi standar SNI", "perlindungan konsumen, barang beredar, standar sni"),
    # [PERINDUSTRIAN]
    ("Penyusunan rencana pembangunan industri kabupaten/kota (RPIK) sektor unggulan", "rencana pembangunan industri kabupaten, rpik"),
    ("Penerbitan izin usaha industri (IUI) dan izin perluasan industri kecil menengah", "izin usaha industri, iui, industri kecil menengah"),
    ("Pengembangan kawasan industri kecil menengah (KIKIM) dan sentra industri logam", "kawasan industri kecil menengah, kikim, sentra industri"),
    ("Pembinaan industri kreatif berbasis kearifan lokal dan sertifikasi produk IKM", "industri kreatif kearifan lokal, sertifikasi produk ikm"),
    ("Fasilitasi penerapan SNI wajib produk industri dan pengujian mutu di laboratorium", "penerapan sni wajib, pengujian mutu produk industri"),
    # [TRANSMIGRASI]
    ("Penempatan transmigran umum dan transmigran swakarsa mandiri ke kawasan transmigrasi", "penempatan transmigran umum, transmigrasi swakarsa mandiri"),
    ("Pembangunan infrastruktur permukiman transmigrasi dan fasilitas umum kawasan SP", "infrastruktur permukiman transmigrasi, fasilitas umum satuan permukiman"),
    ("Pembinaan masyarakat transmigran dan pengembangan usaha ekonomi di kawasan transmigrasi", "pembinaan masyarakat transmigran, usaha ekonomi kawasan transmigrasi"),
    # [KEUANGAN, ANGGARAN & ASET]
    ("Penyampaian dokumen rencana kerja anggaran (RKA) dan dokumen pelaksanaan anggaran (DPA)", "rencana kerja anggaran, dpa"),
    ("Permohonan penerbitan surat perintah pencairan dana (SP2D) dan SPPR", "pencairan dana, sp2d, sppr"),
    ("Rekonsiliasi dan laporan barang milik daerah (BMD)", "rekonsiliasi barang milik daerah, aset daerah"),
    ("Penetapan status penggunaan barang milik daerah", "status penggunaan barang milik daerah, aset"),
    ("Pengelolaan kas daerah dan penempatan investasi jangka pendek pemerintah daerah", "pengelolaan kas daerah, investasi jangka pendek"),
    ("Penyusunan laporan keuangan pemerintah daerah (LKPD) dan neraca aset", "laporan keuangan pemerintah daerah, lkpd, neraca aset"),
    ("Penyelesaian tuntutan perbendaharaan dan tuntutan ganti rugi (TP-TGR) keuangan", "tuntutan perbendaharaan, tuntutan ganti rugi, tp tgr"),
    ("Penganggaran hibah dan bantuan sosial dalam APBD serta pertanggungjawabannya", "hibah bantuan sosial, apbd, pertanggungjawaban"),
    ("Pengelolaan utang dan piutang daerah serta penerbitan obligasi daerah", "utang piutang daerah, obligasi daerah"),
    # [PENGAWASAN, KEPEGAWAIAN & HUKUM]
    ("Tindak lanjut temuan laporan hasil pemeriksaan (LHP) dan Laporan Auditor Independen BPK", "tindak lanjut temuan, laporan hasil pemeriksaan, laporan auditor independen"),
    ("Laporan hasil audit investigasi (LHAI) yang mengandung unsur tindak pidana korupsi", "laporan hasil audit investigasi, tindak pidana korupsi"),
    ("Usulan penetapan angka kredit (PAK) jabatan fungsional arsiparis tingkat ahli", "penetapan angka kredit, jabatan fungsional"),
    ("Penyusunan formasi ASN dan usulan pengadaan CPNS serta PPPK daerah", "formasi asn, pengadaan cpns, pppk"),
    ("Proses kenaikan pangkat reguler, pilihan, dan pengabdian PNS daerah", "kenaikan pangkat pns, pangkat reguler pilihan pengabdian"),
    ("Penyelesaian kasus pelanggaran disiplin PNS dan penjatuhan hukuman disiplin", "pelanggaran disiplin pns, hukuman disiplin"),
    ("Pemberhentian PNS atas permintaan sendiri dan pemberhentian tidak dengan hormat", "pemberhentian pns, pemberhentian tidak dengan hormat"),
    ("Pelaksanaan seleksi terbuka jabatan pimpinan tinggi pratama dan madya", "seleksi terbuka jabatan pimpinan tinggi, pratama madya"),
    ("Penyusunan produk hukum daerah raperda dan rancangan peraturan bupati/walikota", "produk hukum daerah, raperda, peraturan bupati"),
    # [PEMILU, KESBANGPOL & KETERTIBAN]
    ("Penyampaian daftar pemilih sementara (DPS) dan daftar penduduk potensial pemilih (DP4)", "daftar pemilih sementara, daftar penduduk potensial pemilih"),
    ("Penanganan konflik sosial berbasis SARA dan deteksi dini potensi kerawanan daerah", "konflik sosial sara, deteksi dini kerawanan daerah"),
    ("Pemantauan dan pembinaan organisasi kemasyarakatan (ormas) di daerah", "pembinaan organisasi kemasyarakatan, ormas"),
    ("Fasilitasi pendidikan politik masyarakat dan peningkatan partisipasi pemilih", "pendidikan politik masyarakat, partisipasi pemilih"),
    ("Penanganan paham radikalisme dan terorisme melalui forum kewaspadaan dini masyarakat", "paham radikalisme terorisme, forum kewaspadaan dini masyarakat"),
    # [PERENCANAAN, LITBANG & INOVASI]
    ("Penyusunan rencana pembangunan jangka menengah daerah (RPJMD) dan RKPD", "rencana pembangunan jangka menengah daerah, rpjmd, rkpd"),
    ("Musyawarah perencanaan pembangunan (musrenbang) kabupaten dan provinsi", "musyawarah perencanaan pembangunan, musrenbang"),
    ("Evaluasi capaian indikator kinerja utama (IKU) dan indikator kinerja program (IKP)", "capaian indikator kinerja utama, iku, indikator kinerja program"),
    ("Penyelenggaraan riset inovasi daerah dan lomba inovasi pelayanan publik", "riset inovasi daerah, inovasi pelayanan publik"),
    ("Penyusunan kajian kebijakan strategis dan naskah akademik rancangan perda", "kajian kebijakan strategis, naskah akademik rancangan perda"),
    # [PEMERINTAHAN UMUM & KEPROTOKOLAN]
    ("Penyampaian laporan hasil perjalanan dinas ke Arsip Nasional", "perjalanan dinas"),
    ("Penyelenggaraan upacara kedinasan hari besar nasional dan daerah", "upacara kedinasan, hari besar nasional daerah"),
    ("Penyusunan laporan penyelenggaraan pemerintahan daerah (LPPD) tahunan", "laporan penyelenggaraan pemerintahan daerah, lppd"),
    ("Pengelolaan pengaduan masyarakat dan tindak lanjut laporan SP4N-LAPOR", "pengaduan masyarakat, sp4n lapor"),
    ("Penyelenggaraan penerimaan kunjungan tamu negara dan delegasi luar negeri", "kunjungan tamu negara, delegasi luar negeri"),
    ("Koordinasi pelaksanaan tata upacara sipil dan protokol acara kenegaraan daerah", "tata upacara sipil, protokol kenegaraan daerah"),
    # [BENCANA, SAR & PEMADAM KEBAKARAN]
    ("Laporan operasi pencarian dan pertolongan (SAR) korban banjir bandang", "operasi pencarian pertolongan, sar, korban banjir"),
    ("Penyusunan dokumen rencana penanggulangan bencana (RPB) dan rencana kontigensi", "rencana penanggulangan bencana, rpb, rencana kontigensi"),
    ("Distribusi logistik dan peralatan kebencanaan ke daerah terdampak bencana alam", "logistik peralatan kebencanaan, daerah terdampak bencana"),
    ("Pemulihan pascabencana dan relokasi korban bencana alam ke hunian tetap", "pemulihan pascabencana, relokasi korban bencana, hunian tetap"),
    ("Penanganan kebakaran permukiman padat dan pemeriksaan instalasi proteksi kebakaran", "kebakaran permukiman, instalasi proteksi kebakaran")
]

# ====================================================
# FUNGSI ASISTEN PINTAR (Diperbaiki dari Bug Claude)
# ====================================================
def ambil_contoh_relevan(teks_user: str, top_n: int = 12) -> str:
    semua_input = [c[0] for c in BANK_CONTOH]
    
    # thefuzz hanya mengembalikan 2 nilai: teks_cocok dan skor
    hasil = process.extract(teks_user, semua_input, limit=top_n)
    
    contoh_terpilih = ""
    # Kita ubah dari 3 variabel menjadi 2 variabel saja di sini
    for teks_cocok, skor in hasil:
        # Kita cari manual index-nya agar tidak error
        idx = semua_input.index(teks_cocok)
        contoh_terpilih += f'Input: "{BANK_CONTOH[idx][0]}"\nOutput: {BANK_CONTOH[idx][1]}\n'
        
    return contoh_terpilih

# ====================================================
# 2. Fungsi "Otak Ekstraktor" (VERSI X-RAY DEBUGGING)
# ====================================================
# SEMENTARA CACHE DIMATIKAN DULU AGAR DEBUG SELALU MUNCUL
# @st.cache_data(show_spinner=False, ttl=3600) 
def ekstrak_inti_surat(teks_user: str) -> str | None:
    
    instruksi_sistem = """
    Anda adalah Sistem AI Ahli Kearsipan Pemerintahan Daerah.
    Tugas Anda: menganalisis perihal surat dan mengekstrak "Inti Substansi"
    (maksimal 2-3 frasa nominal) untuk mesin pencari klasifikasi TF-IDF.

    GUNAKAN LOGIKA BERPIKIR BERIKUT SECARA BERURUTAN:
    1. HAPUS KATA PENGANTAR: Buang kata basa-basi (penyampaian, permohonan,
       undangan, laporan, tindak lanjut, usulan, hal, mengenai, draf, rancangan,
       penerbitan, fasilitasi, perihal, rekomendasi, sosialisasi).
    2. HAPUS ENTITAS & LOKASI: Buang nama instansi (Dinas, Badan, Kementerian,
       KPU, Bawaslu, RSUD), nama tempat (Provinsi, Kabupaten, Desa), nama orang,
       jabatan (Bupati, Kadis, Kades), dan tahun/tanggal.
    3. CARI SUBSTANSI UTAMA: Temukan urusan aslinya.
    4. RESOLUSI JEBAKAN "ARSIP":
       - JANGAN jadikan "arsip" sebagai inti jika hanya lokasi/tujuan
         (misal: "Bimtek kearsipan" -> intinya "bimbingan teknis").
       - GUNAKAN "arsip" JIKA teknis murni (misal: "jadwal retensi arsip").
    5. RESOLUSI JEBAKAN ASET/BANGUNAN:
       - Kata "Barang Milik Daerah" atau "Aset" WAJIB DIPERTAHANKAN jika
         urusan berkaitan dengan BMD.
       - JIKA konteks FISIK (Pembangunan/Rehab/Keamanan Gedung): buang nama
         tempatnya, ambil inti "pembangunan gedung" / "rehab bangunan".
       - JIKA konteks OPERASIONAL/LAYANAN (Pembinaan perpustakaan, Akreditasi
         puskesmas, Dana BOS Sekolah): kata "Perpustakaan", "Puskesmas",
         "Sekolah" WAJIB DIPERTAHANKAN.

    LARANGAN KERAS — PELANGGARAN INI MERUSAK SISTEM:
    - DILARANG menulis kata kerja dalam output (menghadiri, melaksanakan, dll)
    - DILARANG menyertakan nama orang, instansi, atau lokasi
    - DILARANG menulis lebih dari 3 frasa
    - DILARANG menulis kalimat lengkap ber-subjek-predikat
    - DILARANG menulis penjelasan, preamble, atau apapun sebelum/sesudah output
    - OUTPUT HANYA 1 BARIS: frasa dipisah koma, huruf kecil semua
    """

    # ─── TAHAP 1: ASISTEN PINTAR (THEFUZZ) ───
    contoh_relevan = ambil_contoh_relevan(teks_user)
    
    with st.expander("🛠️ DEBUG TAHAP 1: Asisten Pencari (thefuzz)"):
        st.write("Apakah 12 contoh di bawah ini nyambung dengan input surat?")
        st.code(contoh_relevan, language="text")
    
    prompt_user = f"""
    Pelajari BANK DATA pola pikir berikut, lalu kerjakan input di bawahnya
    dengan pola yang SAMA PERSIS:
    
{contoh_relevan}

    SEKARANG KERJAKAN:
    Input: "{teks_user}"
    Output:
    """

    try:
        # ─── TAHAP 2: OUTPUT MENTAH LLM (GROQ 70B) ───
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": instruksi_sistem},
                {"role": "user",   "content": prompt_user},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=80,
        )

        inti_teks_mentah = chat_completion.choices[0].message.content.strip()

        with st.expander("🛠️ DEBUG TAHAP 2: Output Mentah Groq 70B"):
            st.write("Apakah Groq mematuhi aturan 1 baris, atau malah nulis novel?")
            st.info(inti_teks_mentah)

        # ─── TAHAP 3: PEMBERSIHAN ARTEFAK ───
        daftar_baris = [b.strip() for b in inti_teks_mentah.split('\n') if b.strip()]

        if not daftar_baris:
            st.warning("⚠️ GROQ mengembalikan respons kosong.")
            return None

        inti_teks_bersih = daftar_baris[-1]

        inti_teks_bersih = (
            inti_teks_bersih
            .replace('**', '')
            .replace('"', '')
            .replace("'", '')
            .replace('Output:', '')
            .strip()
        )

        with st.expander("🛠️ DEBUG TAHAP 3: Setelah Pembersihan Artefak"):
            st.write("Hasil setelah bintang, tanda kutip, dan kata 'Output:' dibuang:")
            st.warning(inti_teks_bersih)

        if not inti_teks_bersih:
            st.warning("⚠️ Hasil ekstraksi kosong setelah cleaning.")
            return None
            
        # ─── TAHAP 4: SAPU JAGAT ───
        inti_teks_final = sapu_kata_kerja_bocor(inti_teks_bersih)

        with st.expander("🛠️ DEBUG TAHAP 4: Hasil Final (Dikirim ke Mesin TF-IDF)"):
            st.write("Ini adalah kata kunci yang dipakai untuk mencari Kode Arsip:")
            st.success(inti_teks_final)

        return inti_teks_final if inti_teks_final.strip() else None

    except Exception as e:
        st.error(f"🚨 ERROR GROQ (Tahap Ekstraksi): {e}")
        return None
        
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

# --- MEGA KAMUS JARGON & SINGKATAN BIROKRASI PEMDA (VERSI FINAL) ---
kamus_birokrasi = {
    # ========================================================
    # BAGIAN 1: SINGKATAN RESMI PEMERINTAHAN (DATABASE UTAMA)
    # ========================================================
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
    "ppm": "program pengembangan dan pemberdayaan masyarakat",
    "sirup": "sistem informasi rencana umum pengadaan",
    "rup": "rencana umum pengadaan",

    # ========================================================
    # BAGIAN 2: MEGA TRANSLASI LOGIKA KHUSUS (200+ JARGON)
    # ========================================================
    
    # [A] RUTINITAS, KEBUGARAN & SEREMONIAL PEGAWAI
    "apel": "upacara kedinasan pegawai",
    "apel pagi": "upacara kedinasan pegawai",
    "apel gabungan": "upacara kedinasan pegawai",
    "apel gelar pasukan": "upacara kedinasan pengamanan",
    "senam": "olahraga kebugaran pegawai",
    "jumat sehat": "olahraga kebugaran pegawai",
    "jumat bersih": "kebersihan gotong royong lingkungan",
    "kerja bakti": "kebersihan gotong royong lingkungan",
    "gotong royong": "kebersihan lingkungan",
    "outbound": "pengembangan kapasitas sumber daya manusia",
    "family gathering": "pembinaan mental keakraban pegawai",
    "capacity building": "pendidikan dan pelatihan pegawai",

    # [B] PERJALANAN DINAS, TAMU & STUDI
    "kunker": "kunjungan kerja",
    "studi banding": "kunjungan kerja peninjauan",
    "studi tiru": "kunjungan kerja peninjauan",
    "benchmarking": "kunjungan kerja peninjauan",
    "kaji tiru": "kunjungan kerja peninjauan",
    "kaji banding": "kunjungan kerja peninjauan",
    "audiensi": "penerimaan tamu daerah",
    "silaturahmi": "kunjungan sosial kemasyarakatan",
    "penerimaan tamu": "kunjungan kerja tamu daerah",

    # [C] RAPAT, DISKUSI & EVALUASI
    "rakor": "rapat koordinasi",
    "coffee morning": "rapat koordinasi pimpinan",
    "ngopi bareng": "rapat koordinasi pimpinan",
    "konsinyering": "rapat kerja teknis",
    "zoom meeting": "rapat koordinasi virtual",
    "vidcon": "rapat koordinasi virtual",
    "webinar": "bimbingan teknis virtual",
    "blusukan": "pemantauan tinjauan lapangan",
    "turba": "pemantauan tinjauan lapangan",
    "sidak": "inspeksi mendadak pengawasan",
    "rapat dadakan": "rapat koordinasi insidentil",
    "rapat paripurna istimewa": "sidang dewan perwakilan rakyat daerah",
    "temu wicara": "forum diskusi masyarakat",
    "temu kader": "pembinaan kader masyarakat",

    # [D] KARIR, PELATIHAN & JABATAN PEGAWAI
    "aktualisasi": "laporan pelatihan dasar pegawai",
    "habituasi": "laporan pelatihan dasar pegawai",
    "rolling": "mutasi pemindahan jabatan pegawai",
    "nonjob": "pemberhentian dari jabatan",
    "non job": "pemberhentian dari jabatan",
    "dicopot": "pemberhentian dari jabatan",
    "naik pangkat": "peningkatan golongan pegawai",
    "kenaikan pangkat": "peningkatan golongan pegawai",
    "pengambilan sumpah": "pengangkatan jabatan pegawai",
    "purna tugas": "pemberhentian pegawai pensiun",
    "lepas sambut": "mutasi pengangkatan jabatan",
    "pisah sambut": "mutasi pengangkatan jabatan",
    "inhouse training": "bimbingan teknis pegawai",
    "ujian dinas": "penilaian kompetensi kenaikan pangkat",
    "penyesuaian ijazah": "penilaian kompetensi jenjang pendidikan",
    "izin belajar": "persetujuan pendidikan lanjutan pegawai",
    "tugas belajar": "penugasan pendidikan lanjutan pegawai",
    "bebas tugas": "pemberhentian sementara jabatan pegawai",
    "cuti melahirkan": "hak istirahat bersalin pegawai",
    "cuti alasan penting": "hak istirahat urusan keluarga pegawai",
    "cuti besar": "hak istirahat panjang pegawai",
    "sakit keras": "izin tidak masuk kerja pegawai",
    "baju keki": "pakaian dinas harian pegawai",
    "seragam korpri": "pakaian sipil resmi pegawai",
    "pakai hitam putih": "pakaian dinas hitam putih",

    # [E] KEUANGAN, HONOR & PERTANGGUNGJAWABAN
    "honorarium": "pembayaran jasa kegiatan",
    "uang lelah": "honorarium tim kerja",
    "uang saku": "biaya perjalanan dinas",
    "uang jalan": "biaya perjalanan dinas",
    "uang ketik": "honorarium jasa narasumber",
    "uang duka": "bantuan sosial kematian",
    "uang transport": "biaya perjalanan dinas",
    "potong pajak": "pemungutan pajak penghasilan",
    "gaji ke 13": "tunjangan hari raya kesejahteraan",
    "gaji ke-13": "tunjangan hari raya kesejahteraan",
    "tukin": "tambahan penghasilan pegawai",
    "dana sisa": "sisa lebih perhitungan anggaran",
    "dana talangan": "pinjaman sementara keuangan daerah",
    "kas bon": "pinjaman sementara pegawai",
    "bon sementara": "pinjaman sementara pegawai",
    "tutup buku": "laporan akhir tahun keuangan",
    "buka buku": "laporan awal tahun keuangan",
    "ketok palu": "pengesahan anggaran pendapatan belanja",
    "pagu indikatif": "rencana batas tertinggi anggaran",
    "pagu definitif": "alokasi anggaran final",
    "luncuran kegiatan": "kegiatan lanjutan tahun anggaran berjalan",

    # [F] ACARA DAERAH, FESTIVAL & PERINGATAN
    "festival daerah": "pameran kebudayaan pariwisata",
    "festival": "pameran kebudayaan",
    "hari jadi": "peringatan hari besar daerah",
    "hut pemda": "peringatan hari besar daerah",
    "tujuh belasan": "peringatan hari besar nasional",
    "upacara bendera": "peringatan hari besar nasional",
    "halal bihalal": "pembinaan mental pegawai keagamaan",
    "buka puasa bersama": "pembinaan mental pegawai keagamaan",
    "bukber": "pembinaan mental pegawai keagamaan",
    "safari ramadhan": "kunjungan sosial keagamaan",
    "safari jumat": "kunjungan sosial keagamaan",
    "anjangsana": "kunjungan sosial kemasyarakatan",
    "takbiran": "peringatan hari besar keagamaan",
    "tarawih keliling": "kunjungan sosial keagamaan",
    "pasar murah": "operasi pasar perekonomian",
    "jalan sehat": "olahraga rekreasi masyarakat",
    "car free day": "hari bebas kendaraan bermotor",
    "doorprize": "kegiatan hiburan masyarakat",

    # [G] LOMBA, PENDIDIKAN & KESEHATAN
    "rembuk stunting": "koordinasi penanganan gizi buruk",
    "stunting": "penanggulangan gizi buruk anak",
    "odgj": "penanganan masalah kejiwaan",
    "fogging": "pengendalian penyakit menular",
    "bedah rumah": "rehabilitasi rumah tidak layak huni",

    # [H] PENGADAAN, INFRASTRUKTUR & ASET
    "lelang rongsokan": "penghapusan aset rusak berat",
    "dum kendaraan": "penjualan lelang aset daerah",
    "tarik mobil dinas": "penarikan aset kendaraan dinas",
    "mobil dinas": "kendaraan dinas operasional",
    "plat merah": "kendaraan dinas operasional",
    "pinjam pakai gedung": "penggunaan sementara aset daerah",
    "sewa tenda": "penyewaan perlengkapan acara",
    "konsumsi rapat": "belanja makanan dan minuman",
    "snack box": "belanja makanan dan minuman ringan",
    "nasi kotak": "belanja makanan dan minuman berat",
    "belanja atk": "pengadaan alat tulis kantor",
    "kertas hvs": "pengadaan barang habis pakai",
    "tinta printer": "pengadaan barang habis pakai",
    "gagal lelang": "pembatalan pengadaan barang jasa",
    "tender ulang": "pengadaan ulang barang jasa",
    "harga borongan": "nilai kontrak kerja pengadaan",
    "peletakan batu pertama": "peresmian dimulainya pembangunan fisik",
    "ground breaking": "peresmian dimulainya pembangunan fisik",
    "gunting pita": "peresmian selesainya pembangunan fisik",
    "potong tumpeng": "acara syukuran peresmian",
    "teken prasasti": "penandatanganan peresmian infrastruktur",
    "jalan berlubang": "perbaikan infrastruktur jalan raya",
    "tambal sulam jalan": "pemeliharaan rutin jalan raya",
    "normalisasi sungai": "pemeliharaan sumber daya air",
    "pengerukan sungai": "pemeliharaan sumber daya air",
    "pembebasan lahan": "pengadaan tanah kepentingan umum",

    # [I] PELAYANAN PUBLIK & PENERTIBAN
    "jemput bola": "pelayanan publik keliling terpadu",
    "layanan keliling": "pelayanan publik keliling terpadu",
    "sim keliling": "pelayanan izin mengemudi keliling",
    "ktp keliling": "pelayanan administrasi kependudukan keliling",
    "operasi yustisi": "razia penegakan peraturan daerah",
    "razia ktp": "pemeriksaan identitas kependudukan",
    "razia pekat": "penertiban penyakit masyarakat",
    "razia masker": "penegakan protokol kesehatan masyarakat",
    "penertiban pkl": "relokasi pedagang kaki lima",
    "razia gepeng": "penertiban gelandangan dan pengemis",
    "tilang": "penindakan pelanggaran lalu lintas",
    "operasi pasar": "pengendalian harga kebutuhan pokok",
    "gusur lahan": "penertiban lahan pemerintah daerah",
    "segel bangunan": "penertiban izin mendirikan bangunan",

    # [J] KEARSIPAN & PERPUSTAKAAN (URAT NADI SIKAP)
    "bakar arsip": "pemusnahan arsip inaktif",
    "kiloan arsip": "penjualan kertas limbah arsip",
    "gudang arsip": "pengelolaan depo arsip inaktif",
    "arsip numpuk": "penataan kembali arsip kacau",
    "arsip hilang": "laporan kehilangan dokumen negara",
    "arsip basah": "restorasi perbaikan arsip rusak",
    "rayap arsip": "fumigasi pemeliharaan fisik arsip",
    "kunjungan perpus": "layanan sirkulasi perpustakaan daerah",
    "pinjam buku": "layanan sirkulasi perpustakaan daerah",
    "kembalikan buku": "layanan sirkulasi perpustakaan daerah",
    "buku denda": "sanksi keterlambatan pengembalian buku",
    "perpusling": "layanan perpustakaan keliling daerah",
    "buku hibah": "penerimaan sumbangan bahan pustaka",
    "bedah buku": "diskusi literasi perpustakaan",
    "duta baca": "kampanye budaya gemar membaca",

    # [K] SOSIAL, BENCANA & KEAGAMAAN
    "siraman rohani": "pembinaan mental keagamaan pegawai",
    "kultum": "ceramah pembinaan keagamaan",
    "tarling": "kunjungan safari tarawih keliling",
    "jumling": "kunjungan safari jumat keliling",
    "takjil gratis": "pembagian makanan berbuka puasa",
    "sahur on the road": "kegiatan sosial bulan ramadhan",
    "qurban": "pemotongan hewan kurban keagamaan",
    "zakat fitrah": "pengumpulan amal zakat keagamaan",
    "naik haji": "fasilitasi pemberangkatan jamaah haji",
    "umroh": "fasilitasi ibadah umrah pegawai",
    "kebakaran rumah": "bantuan tanggap darurat bencana",
    "banjir bandang": "penanganan tanggap darurat bencana",
    "tanah longsor": "penanggulangan bencana alam",
    "pohon tumbang": "penanganan darurat infrastruktur",
    "puting beliung": "penanggulangan bencana alam angin",
    "gempa bumi": "penanganan tanggap darurat bencana alam",
    "kekeringan": "bantuan pasokan air bersih",
    "wabah penyakit": "penanganan kejadian luar biasa",

    # [L] PEMERINTAHAN DESA & KEMASYARAKATAN
    "pemilihan kades": "penyelenggaraan pemilihan kepala desa",
    "sengketa lahan": "penyelesaian konflik pertanahan",
    "tapal batas": "penetapan batas wilayah administrasi",
    "pemekaran desa": "pembentukan wilayah administrasi baru",
    "dana siluman": "penyelewengan pengelolaan keuangan",
    "demo warga": "penanganan unjuk rasa masyarakat",
    "unjuk rasa": "penyampaian aspirasi masyarakat",
    "mogok kerja": "penyelesaian perselisihan hubungan industrial",
    "surat kaleng": "pengaduan masyarakat tanpa identitas",
    "pengaduan warga": "penanganan keluhan pelayanan publik",
    "kotak saran": "evaluasi indeks kepuasan masyarakat",

    # [M] LEGISLATIF & DOKUMEN HUKUM
    "pokok pikiran": "usulan dewan perwakilan rakyat daerah",
    "surat sakti": "surat rekomendasi pimpinan daerah",
    "surat pengantar": "nota penyampaian dokumen resmi",
    "surat teguran": "peringatan pelanggaran disiplin",
    "surat peringatan": "sanksi pelanggaran administrasi",
    "surat tugas": "perintah pelaksanaan perjalanan dinas",
    "surat kuasa": "pelimpahan wewenang hukum",
    "surat keterangan": "pernyataan resmi pemerintah daerah",
    "surat pernyataan": "deklarasi pertanggungjawaban mutlak",
    "nota kesepahaman": "perjanjian kerja sama antar lembaga",
    "perjanjian kerja": "kontrak kerja pelaksanaan tugas",
    "kontrak kerja": "perjanjian pengadaan barang jasa",
    "berita acara": "laporan pengesahan kegiatan resmi",
    "buku tamu": "registrasi kunjungan pihak luar",
    "buku ekspedisi": "catatan pengiriman penerimaan surat",
    "lembar disposisi": "catatan arahan pimpinan daerah",
    "buku register": "pencatatan nomor surat keluar",
    "cap dinas": "stempel resmi instansi pemerintah",
    "stempel basah": "pengesahan dokumen resmi negara",

    # [N] EVENT & PUBLIKASI
    "pameran pembangunan": "publikasi hasil kinerja pemerintah",
    "expo daerah": "promosi potensi pariwisata daerah",
    "pawai pembangunan": "kegiatan perayaan masyarakat daerah",
    "karnaval budaya": "pagelaran seni budaya daerah",
    "lomba desa": "penilaian prestasi pemerintahan desa",
    "lomba kelurahan": "penilaian prestasi pemerintahan kelurahan",
    "lomba kebersihan": "penilaian adipura lingkungan hidup",
    "penghargaan adipura": "prestasi kebersihan lingkungan hidup",
    "penghargaan wtp": "prestasi akuntabilitas keuangan daerah",
    "iklan koran": "publikasi media cetak daerah",
    "ucapan selamat": "karangan bunga publikasi humas",
    "karangan bunga": "bantuan sosial ucapan simpati",
    "baliho": "pemasangan media luar ruang",
    "spanduk": "pemasangan alat peraga kampanye",
    "umbul umbul": "dekorasi perayaan acara daerah",
    "brosur": "penyebaran materi informasi publik",

    # [O] ISTILAH KHAS PEJABAT PEMDA
    "pejabat teras": "pimpinan tinggi pratama daerah",
    "pejabat eselon": "pejabat struktural pemerintah daerah",
    "staf ahli": "penasihat khusus kepala daerah",
    "asisten sekda": "koordinator bidang sekretariat daerah",
    "ajudan": "staf pendamping kepala daerah",
    "walpri": "pengawal pribadi kepala daerah",
    "sopir bupati": "petugas transportasi kepala daerah",
    "protokoler": "petugas tata acara keprotokolan"
}

# --- FUNGSI PENERJEMAH SINGKATAN ---
def terjemahkan_singkatan(text):
    text_lower = str(text).lower()
    # Cek frasa multi-kata dulu (urutkan dari yang terpanjang)
    kunci_terurut = sorted(kamus_birokrasi.keys(), key=len, reverse=True)
    for frasa in kunci_terurut:
        if frasa in text_lower:
            text_lower = text_lower.replace(frasa, kamus_birokrasi[frasa])
    return text_lower

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

# --- FUNGSI PEMBUAT BADGE UNTUK TAB 1 (ANTI MELUBER DI HP) ---
def get_badge_html(kode, uraian, level):
    levels_name = ["Primer", "Sekunder", "Tersier", "Kuartier", "Kuintier"]
    label = levels_name[level] if level < len(levels_name) else f"Level {level+1}"
    
    warna_level = ["#EF4444", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]
    warna_bg = warna_level[level] if level < len(warna_level) else "#424242"
    
    # JURUS RESPONSIVE: Gunakan CSS 'clamp' agar jarak margin mengecil otomatis di HP
    indent = f"clamp(0px, {level * 5}vw, {level * 30}px)" 
    
    return f"<div style='margin-left: {indent}; margin-bottom: 8px; width: 100%; box-sizing: border-box; padding-right: 10px;'>" \
           f"<div style='background-color: {warna_bg}; color: #ffffff; padding: 10px 15px; border-radius: 12px; font-weight: normal; font-size: 0.9rem !important; display: flex; align-items: flex-start; width: 100%; box-shadow: 0px 3px 6px rgba(0,0,0,0.15);'>" \
           f"<div style='display: flex; align-items: flex-start; flex-shrink: 0; margin-right: 8px;'>" \
           f"<span class='material-symbols-rounded' style='font-size: 1.15rem; margin-right: 4px; margin-top: 2px;'>folder</span>" \
           f"<strong style='margin-top: 2px;'>{kode}</strong>" \
           f"<span style='margin: 2px 4px 0 4px; opacity: 0.6;'>|</span>" \
           f"</div>" \
           f"<div style='flex-grow: 1; text-align: left; line-height: 1.4; margin-top: 2px; word-wrap: break-word; overflow-wrap: break-word;'>" \
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
@st.cache_data(show_spinner=False)
def smart_classify(user_input, df, top_n=3):
    # 1. Biarkan LLM mengekstrak "inti" dari uraian panjang user
    inti_dari_llm = ekstrak_inti_surat(user_input)

    # 🛡️ PASANG GEMBOK KEAMANAN DI SINI 🛡️
    if not inti_dari_llm:
        st.error("⚠️ SIKAP gagal mengekstrak inti surat karena kendala teks atau server. Silakan coba lagi atau perbaiki kalimat Anda.")
        return []  # Langsung hentikan fungsi dan kembalikan list kosong
        
    # st.info(f"🧠 SIKAP menangkap inti surat Anda sebagai: **{inti_dari_llm}**")
    
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
            /* 1. Ubah Background Sidebar Jadi Biru Gelap Gradasi */
            [data-testid="stSidebar"] {
                background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
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
            /* 1. Bersihkan Kotak Expander (Hasil Pencarian) */
            div[data-testid="stExpander"] {
                border: 2px solid var(--input-border) !important; /* <--- OUTLINE BIRU KEKINIAN */
                border-radius: 16px !important;
                box-shadow: 0 4px 15px rgba(0, 157, 255, 0.1) !important; /* <--- GLOW BIRU HALUS */
                background: var(--card-bg) !important; /* <--- UBAH DI SINI */
                margin-bottom: 15px !important;
                overflow: hidden !important;
            }
            
            /* 2. PERBAIKAN TOTAL: BUNUH TEKS HANTU "arrow" ("arr") */
            div[data-testid="stExpander"] summary {
                padding: 15px 20px !important;
                background: var(--expander-bg) !important; /* <--- UBAH DI SINI */
                border-bottom: 1px solid var(--card-border) !important; /* <--- UBAH DI SINI */
                border-radius: 16px 16px 0 0 !important;
                
                /* JURUS PAMUNGKAS: Jadikan semua ukuran teks di sini 0 agar teks hantu lenyap */
                font-size: 0px !important; 
                color: transparent !important;
            }
            
            /* Bangkitkan HANYA teks judul (p) kita agar tampil sempurna */
            div[data-testid="stExpander"] summary p {
                font-weight: 700 !important;
                color: var(--text-title) !important; /* <--- UBAH DI SINI */
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
                    results = smart_classify(user_input, df)
                    
                    if results:
                        # BANNER SUKSES KUSTOM
                        st.markdown("""
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
                
                if st.button("Hapus Riwayat"):
                    st.session_state.search_history = []
                    # (Opsional: Kamu bisa tambahkan fungsi hapus file CSV di Drive juga di sini)
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
            st.markdown('<div class="section-title">📖 Petunjuk Penggunaan SIKAP</div>', unsafe_allow_html=True)
            st.info("Halaman panduan dan tata cara penggunaan aplikasi akan segera hadir di sini. Harap bersabar ya!")

        # --- HALAMAN 7: TENTANG SIKAP ---
        elif st.session_state.page == 'Tentang':
            st.markdown('<div class="section-title">ℹ️ Tentang SIKAP</div>', unsafe_allow_html=True)
            st.write("SIKAP (Sistem Informasi Klasifikasi Arsip Pintar) adalah inovasi digital cerdas yang dirancang untuk membantu aparatur pemerintah daerah dalam mengklasifikasikan dan menemukan kode arsip secara presisi, cepat, dan modern.")
            
            st.markdown("""
            <div style="margin-top: 20px; padding: 15px; border-radius: 8px; background-color: #FFFFFF; border-left: 4px solid #009DFF; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
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

            # Statistik (Tetap dipertahankan di bawah)
            st.markdown('<div style="display:flex; align-items:center; gap:8px; margin-top:40px; margin-bottom:10px;"><span class="material-symbols-rounded" style="color:#475569; font-size:1.4rem;">trending_up</span><span style="font-weight:700; color:#0F172A; font-size:1.1rem; font-family:\'Poppins\';">Statistik Penggunaan</span></div>', unsafe_allow_html=True)
            riwayat_pribadi = baca_riwayat_csv(st.session_state['nama'])
            
            st.markdown(f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 20px; border-radius: 16px; display:flex; align-items:center; gap:15px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); max-width: 400px;">
                <div style="background: #E0F2FE; width: 54px; height: 54px; border-radius: 14px; display:flex; align-items:center; justify-content:center;">
                    <span class="material-symbols-rounded" style="color:#009DFF; font-size:2rem;">manage_search</span>
                </div>
                <div>
                    <div style="color: #64748B; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom:4px;">Total Pencarian AI</div>
                    <div style="color: #0F172A; font-size: 1.6rem; font-weight: 800; line-height: 1;">{len(riwayat_pribadi)} <span style="font-size: 1rem; color: #475569; font-weight: 600;">kali</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")

# --- 5. PENGATUR HALAMAN (ROUTER) ---
if not st.session_state.get('logged_in', False):
    halaman_login()
else:
    halaman_utama()
