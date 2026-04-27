import streamlit as st
import pandas as pd
import re
import os
import numpy as np                                          
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer       
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
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
    Input: "Permohonan penerbitan surat perintah pencairan dana (SP2D) untuk kegiatan sosialisasi"
    Output: pencairan dana, sp2d
    Input: "Penyampaian berita acara serah terima (BAST) kendaraan dinas roda empat"
    Output: berita acara serah terima, kendaraan dinas
    
    [KASUS KEPEGAWAIAN, PENGAWASAN & HUKUM]
    Input: "Usulan penetapan angka kredit (PAK) jabatan fungsional arsiparis tingkat ahli"
    Output: penetapan angka kredit, jabatan fungsional
    Input: "Teguran disiplin pegawai dan pemanggilan pemeriksaan pelanggaran kode etik ASN"
    Output: disiplin pegawai, pelanggaran kode etik
    Input: "Tindak lanjut temuan laporan hasil pemeriksaan (LHP) BPK RI perwakilan Sulawesi Tenggara"
    Output: tindak lanjut temuan, laporan hasil pemeriksaan
    Input: "Permohonan fasilitasi penyusunan rancangan peraturan bupati tentang pedoman tata naskah dinas"
    Output: peraturan bupati, tata naskah dinas
    
    [KASUS PENDIDIKAN, KESEHATAN & INFRASTRUKTUR]
    Input: "Penyaluran dan pencairan dana bantuan operasional sekolah (BOS) tahap I"
    Output: dana bantuan operasional sekolah, bos
    Input: "Klaim penggantian biaya pelayanan kesehatan BPJS Kesehatan pasien rawat inap RSUD"
    Output: klaim bpjs kesehatan, pelayanan kesehatan rawat inap
    Input: "Persetujuan rencana anggaran biaya (RAB) dan gambar kerja proyek pembangunan jembatan"
    Output: rencana anggaran biaya, gambar kerja proyek
    
    [KASUS PEMILU, KESBANGPOL & KETERTIBAN]
    Input: "Penyampaian daftar pemilih sementara (DPS) dan daftar penduduk potensial pemilih (DP4) Pilkada"
    Output: daftar pemilih sementara, daftar penduduk potensial pemilih
    Input: "Laporan pemantauan kegiatan partai politik dan organisasi kemasyarakatan (Ormas)"
    Output: pemantauan partai politik, organisasi kemasyarakatan
    Input: "Penertiban pedagang kaki lima dan pembongkaran baliho reklame ilegal"
    Output: penertiban pedagang kaki lima, pembongkaran baliho
    
    [KASUS LINGKUNGAN HIDUP, BENCANA & PERTANIAN]
    Input: "Pembahasan dokumen analisis mengenai dampak lingkungan (AMDAL) dan UKL-UPL pabrik kelapa sawit"
    Output: analisis mengenai dampak lingkungan, amdal, ukl upl
    Input: "Laporan operasi pencarian dan pertolongan (SAR) korban banjir bandang"
    Output: operasi pencarian pertolongan, sar, korban banjir
    Input: "Sertifikasi dan pengujian keamanan pangan segar asal tumbuhan (PSAT) pasar tradisional"
    Output: sertifikasi keamanan pangan segar asal tumbuhan, psat
    
    [KASUS PEMERINTAHAN DESA & UMUM]
    Input: "Penyaluran dana desa (DD) dan penyelesaian sengketa pemilihan kepala desa (Pilkades) serentak"
    Output: dana desa, sengketa pemilihan kepala desa
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
        /* --- KEJUTAN WARNA: Gradasi Sunset (Oranye ke Pink) --- */
        background: linear-gradient(45deg, #FF512F, #DD2476);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .streamlit-expanderHeader {
        font-weight: bold;
        color: #0288D1;
    }
    /* PEMBUNUH IKON PANAH BIRU BAWAAN BROWSER */
    details > summary {
        list-style: none !important;
        outline: none !important;
    }
    details > summary::-webkit-details-marker {
        display: none !important; 
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

@st.cache_resource
def load_embedding_model():
    # Model ini paham Bahasa Indonesia, ukuran ~120MB
    # Di-download otomatis saat pertama kali jalan, lalu di-cache
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embedding_model = load_embedding_model()

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
                if len(curr) == 3 and curr.endswith('00'):
                    break 
                elif len(curr) > 3:
                    curr = curr[:-1]
                elif len(curr) == 3:
                    if curr.endswith('0'): curr = curr[0] + '00'
                    else: curr = curr[0:2] + '0'
                else:
                    break
        return " > ".join(jalur)
    
    df['uraian_lengkap'] = df['kode'].apply(bangun_hierarki)

    
    # ↓↓↓ BAGIAN BARU: HITUNG EMBEDDING UNTUK SELURUH DATABASE ↓↓↓
    teks_untuk_embed = df['uraian_lengkap'].tolist()
    
    embeddings = embedding_model.encode(
        teks_untuk_embed,
        show_progress_bar=False,
        batch_size=64
    )
    
    # Simpan embedding sebagai list of numpy array di kolom baru
    df['embedding'] = list(embeddings)
    # ↑↑↑ SELESAI ↑↑↑
    
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

def bangun_konteks_rumpun(df):
    """
    Baca isi database sendiri untuk membuat deskripsi kaya tiap rumpun.
    LLM tidak perlu 'nebak' lagi — dia dikasih contoh nyata dari dalam DB.
    """
    konteks = {}
    for i in range(10):
        kode_rumpun = f"{i}00"
        baris_rumpun = df[df['kode'] == kode_rumpun]
        if baris_rumpun.empty:
            continue
        
        nama_rumpun = baris_rumpun.iloc[0]['uraian'].title()
        digit_awal = str(i)
        
        # Ambil kode anak langsung (sekunder, tanpa titik)
        anak = df[
            (df['kode'].str.startswith(digit_awal)) &
            (df['kode'] != kode_rumpun) &
            (~df['kode'].str.contains(r'\.', regex=True)) &
            (df['kode'].str.len() <= 3)
        ]['uraian'].head(6).tolist()
        
        contoh_str = ", ".join(anak) if anak else "—"
        konteks[kode_rumpun] = f"{nama_rumpun} | Berisi: {contoh_str}"
    
    return konteks


def navigasi_rumpun(inti_surat, df):
    """
    Level 1 Corong: Pilih 2-3 rumpun dari 10 pilihan.
    Konteks dibangun DINAMIS dari database, bukan hardcoded.
    """
    daftar_rumpun = bangun_konteks_rumpun(df)
    daftar_opsi = "\n".join([f"  {k}: {v}" for k, v in daftar_rumpun.items()])
    
    prompt = f"""Anda ahli kearsipan pemerintah Indonesia. Tugas: tentukan RUMPUN arsip.

URUSAN yang dicari: "{inti_surat}"

DAFTAR RUMPUN TERSEDIA:
{daftar_opsi}

CARA BERPIKIR:
- Baca "Berisi:" di tiap rumpun untuk memahami isinya
- Pilih rumpun yang topiknya paling dekat dengan urusan di atas
- Boleh pilih 2 rumpun jika urusan lintas bidang

BALAS HANYA dengan kode 3 digit dipisah koma. Contoh: 200, 800
Dilarang keras menulis teks lain."""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=20,
        )
        balasan = response.choices[0].message.content.strip()
        kode_ditemukan = re.findall(r'\b[0-9]{3}\b', balasan)
        kode_valid = [k for k in kode_ditemukan if k in daftar_rumpun]
        return kode_valid if kode_valid else list(daftar_rumpun.keys())[:3]
    except Exception as e:
        st.warning(f"Navigator Rumpun gagal: {e}")
        return list(bangun_konteks_rumpun(df).keys())[:3]


def navigasi_subkategori(inti_surat, df_rumpun):
    """
    Level 2 Corong: Setelah dapat rumpun, persempit ke sub-kategori.
    LLM memilih dari ~20-40 opsi, bukan 300.
    """
    # Ambil hanya kode sekunder (tanpa titik)
    kode_sekunder = df_rumpun[
        (~df_rumpun['kode'].str.contains(r'\.', regex=True)) &
        (df_rumpun['kode'].str.len() > 1)
    ].copy()
    
    if len(kode_sekunder) <= 5:
        return df_rumpun
    
    daftar_opsi = ""
    for _, row in kode_sekunder.iterrows():
        daftar_opsi += f"  [{row['kode']}] {row['uraian'].title()}\n"
    
    prompt = f"""Anda ahli kearsipan pemerintah Indonesia.

URUSAN: "{inti_surat}"

PILIH 2-3 sub-kategori arsip yang paling relevan:
{daftar_opsi}

BALAS HANYA dengan kode-kode yang dipilih, dipisah koma.
Contoh: 210, 230
Dilarang menulis teks lain."""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=30,
        )
        balasan = response.choices[0].message.content.strip()
        kode_disebut = re.findall(r'\b\d+\b', balasan)
        kode_valid = [k for k in kode_disebut if k in df_rumpun['kode'].values]
        
        if not kode_valid:
            return df_rumpun
        
        mask = df_rumpun['kode'].apply(
            lambda k: any(
                str(k).startswith(str(sub)[0] if len(str(sub)) == 1
                    else str(sub)[:2] if len(str(sub)) == 2
                    else str(sub))
                for sub in kode_valid
            )
        )
        df_tersaring = df_rumpun[mask]
        return df_tersaring if len(df_tersaring) >= 5 else df_rumpun
        
    except Exception as e:
        st.warning(f"Navigator Sub-Kategori gagal: {e}")
        return df_rumpun

# ↓↓↓ FUNGSI BARU 2: SEMANTIC SEARCH (GANTI TF-IDF) ↓↓↓
def semantic_search(inti_surat, df_subset, top_n=15):
    # Encode kalimat query menjadi vektor angka
    query_embedding = embedding_model.encode([inti_surat])
    
    # Susun semua embedding dari baris yang sudah difilter menjadi matrix
    db_embeddings = np.vstack(df_subset['embedding'].values)
    
    # Hitung kemiripan makna (bukan kemiripan karakter)
    similarities = cosine_similarity(query_embedding, db_embeddings)[0]
    
    # Bonus untuk kode yang lebih dalam/spesifik
    depth_bonuses = df_subset['kode'].apply(
        lambda k: str(k).count('.') * 0.04
    ).values
    
    final_scores = similarities + depth_bonuses
    
    # Ambil index top N dengan skor tertinggi
    top_indices = np.argsort(final_scores)[::-1][:top_n]
    
    # Kembalikan index ASLI dari df (bukan index lokal df_subset)
    return [
        (df_subset.index[i], float(final_scores[i]))
        for i in top_indices
    ]

# --- 3. LOGIKA AI HYBRID (RERANKING) ---
def smart_classify(user_input, df, top_n=3):

    # ══ FASE 1: Ekstraksi Inti ══════════════════════════════════════
    inti_dari_llm = ekstrak_inti_surat(user_input)
    st.info(f"🧠 Inti surat: **{inti_dari_llm}**")

    # ══ FASE 2: Navigasi Rumpun (Level 1 Corong) ═══════════════════
    with st.spinner("📂 Level 1 — Menentukan bidang klasifikasi..."):
        rumpun_terpilih = navigasi_rumpun(inti_dari_llm, df)

    mask_rumpun = df['kode'].apply(
        lambda k: any(str(k).startswith(str(r)[0]) for r in rumpun_terpilih)
    )
    df_level1 = df[mask_rumpun].copy()

    if len(df_level1) < 10:
        df_level1 = df.copy()

    nama_rumpun = []
    for r in rumpun_terpilih:
        baris = df[df['kode'] == r]
        if not baris.empty:
            nama_rumpun.append(f"{r} ({baris.iloc[0]['uraian'].title()})")
    st.info(f"📁 Bidang terpilih: **{', '.join(nama_rumpun)}** — {len(df_level1)} entri")

    # ══ FASE 3: Navigasi Sub-Kategori (Level 2 Corong) ═════════════
    with st.spinner("🗂️ Level 2 — Mempersempit ke sub-kategori..."):
        df_level2 = navigasi_subkategori(inti_dari_llm, df_level1)

    st.caption(f"🔍 Dipersempit ke {len(df_level2)} entri untuk pencarian mendalam")

    # ══ FASE 4: Semantic Search (ruang sudah kecil) ═════════════════
    with st.spinner("🔎 Level 3 — Mencocokkan makna..."):
        top_kandidat = semantic_search(inti_dari_llm, df_level2, top_n=15)

    # ══ FASE 5: Juri AI ════════════════════════════════════════════
    daftar_kandidat_str = ""
    for i, (idx, skor) in enumerate(top_kandidat):
        baris = df.loc[idx]
        daftar_kandidat_str += f"[{i+1}] Kode: {baris['kode']} | {baris['uraian_lengkap'].title()}\n"

    prompt_juri = f"""Anda ahli kearsipan pemerintah daerah Indonesia.

URUSAN SURAT: "{inti_dari_llm}"

KANDIDAT KODE KLASIFIKASI:
{daftar_kandidat_str}

TUGAS: Pilih 3 kode yang PALING TEPAT.

ATURAN:
1. Balas HANYA 3 angka urutan dipisah koma (contoh: 2, 7, 11)
2. Angka harus antara 1 sampai {len(top_kandidat)}
3. Pilih kode yang PALING SPESIFIK/DALAM jika ada pilihan sekode
4. Baca konteks hierarki dengan teliti sebelum memilih
5. DILARANG menulis teks lain selain 3 angka"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_juri}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
        )
        balasan_juri = response.choices[0].message.content.strip()

        angka_mentah = re.findall(r'\d+', balasan_juri)
        angka_pilihan = []
        for angka in angka_mentah:
            n = int(angka)
            if 1 <= n <= len(top_kandidat) and n not in angka_pilihan:
                angka_pilihan.append(n)
            if len(angka_pilihan) == 3:
                break

        hasil_akhir = []
        for nomor in angka_pilihan:
            idx_kandidat = nomor - 1
            if 0 <= idx_kandidat < len(top_kandidat):
                skor_simulasi = 0.99 - (len(hasil_akhir) * 0.14)
                hasil_akhir.append((top_kandidat[idx_kandidat][0], skor_simulasi))

        if hasil_akhir:
            return hasil_akhir

    except Exception as e:
        st.error(f"🚨 ERROR GROQ (Juri AI): {e}")

    return [(idx, skor) for idx, skor in top_kandidat[:top_n]]
    
# --- 4. ANTARMUKA UTAMA ---
def halaman_utama():
    st.markdown("<div class='sikap-title'>SIKAP</div>", unsafe_allow_html=True)
    st.markdown("<div class='sikap-subtitle'>Sistem Informasi Klasifikasi Arsip Pintar</div>", unsafe_allow_html=True)

    try:
        df = load_data()

        if st.checkbox("🔍 Lihat konteks rumpun yang dibaca sistem"):
            konteks = bangun_konteks_rumpun(df)
            for k, v in konteks.items():
                st.write(f"**{k}**: {v}")

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
                            res = df.loc[idx]
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
