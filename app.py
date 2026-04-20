import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from thefuzz import process, fuzz

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="SIKAP - Klasifikasi Arsip Pintar", page_icon="🗂️")

# --- INISIALISASI NLP (Sastrawi) ---
@st.cache_resource
def init_nlp():
    stemmer = StemmerFactory().create_stemmer()
    remover = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, remover

stemmer, remover = init_nlp()

# --- KAMUS JARGON & SINGKATAN BIROKRASI (VERSI FULL PERMENDAGRI 83/2022) ---
kamus_birokrasi = {
    # ==========================================
    # 1. KEUANGAN & ANGGARAN (KODE 900)
    # ==========================================
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

    # ==========================================
    # 2. KEPEGAWAIAN (KODE 800)
    # ==========================================
    "asn": "aparatur sipil negara",
    "pns": "pegawai negeri sipil",
    "cpns": "calon pegawai negeri sipil",
    "pppk": "pegawai pemerintah dengan perjanjian kerja",
    "p3k": "pegawai pemerintah dengan perjanjian kerja",
    "nip": "nomor induk pegawai",
    "bkn": "badan kepegawaian negara",
    "skp": "standar kinerja pegawai", # atau sasaran kinerja pegawai
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

    # ==========================================
    # 3. PENGAWASAN & HUKUM (KODE 700 & 100)
    # ==========================================
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

    # ==========================================
    # 4. PEMERINTAHAN, POLITIK & UMUM (KODE 000, 100, 200)
    # ==========================================
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

    # ==========================================
    # 5. KEARSIPAN & TEKNIS LAINNYA (KODE 000.5, 400, 500, 600)
    # ==========================================
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
    # Memecah kalimat menjadi kata-kata tunggal
    kata_kata = str(text).lower().split()
    # Mengecek satu per satu, apakah kata tersebut ada di kamus?
    kata_terjemahan = [kamus_birokrasi.get(kata, kata) for kata in kata_kata]
    # Menggabungkannya kembali menjadi kalimat utuh
    return " ".join(kata_terjemahan)

# --- FUNGSI PEMBERSIH UTAMA ---
def preprocess_text(text):
    text = str(text).lower()
    
    # 1. Terjemahkan singkatan terlebih dahulu! (Ini kuncinya)
    text = terjemahkan_singkatan(text)
    
    # 2. Kecilkan huruf & hapus karakter aneh/simbol
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 3. Hapus Stopwords (kata sambung seperti "di", "dan", "yang")
    text = remover.remove(text)
    
    # 4. Stemming (Potong imbuhan menggunakan Sastrawi)
    text = stemmer.stem(text)
    
    return text

def preprocess_text(text):
    # 1. Kecilkan huruf & hapus karakter aneh
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # 2. Hapus Stopwords (kata sambung)
    text = remover.remove(text)
    # 3. Stemming (Potong imbuhan)
    text = stemmer.stem(text)
    return text

# --- 1. MEMUAT DATABASE (Jurus Pamungkas) ---
@st.cache_data
def load_data():
    # 1. Coba buka file dengan koma, jika gagal pakai titik koma
    try:
        df = pd.read_csv('klasifikasi_arsip_emas.csv', sep=',', on_bad_lines='skip', dtype=str)
    except:
        df = pd.read_csv('klasifikasi_arsip_emas.csv', sep=';', on_bad_lines='skip', dtype=str)
    
    # 2. Jika datanya menyatu di satu kolom, paksa belah dua
    if len(df.columns) == 1:
        col_name = df.columns[0]
        df[['kode', 'uraian']] = df[col_name].str.split(r'[,;]', n=1, expand=True)
        df = df.drop(columns=[col_name])
        
    # 3. PAKSA GANTI NAMA KOLOM (Ini yang akan menghilangkan Error 'uraian')
    # Tidak peduli apa nama aslinya, kolom ke-1 pasti 'kode', ke-2 pasti 'uraian'
    if len(df.columns) >= 2:
        kolom_baru = list(df.columns)
        kolom_baru[0] = 'kode'
        kolom_baru[1] = 'uraian'
        df.columns = kolom_baru
    
    # 4. Bersihkan isi baris dari karakter titik koma (;) di ujung kalimat jika ada terbawa
    df['uraian'] = df['uraian'].astype(str).str.replace(r';$', '', regex=True).str.strip().fillna("")
    df['kode'] = df['kode'].astype(str).str.strip().fillna("000")
    
    # 5. Lakukan pembersihan NLP
    df['clean_uraian'] = df['uraian'].apply(preprocess_text)
    
    return df

# --- 2. FITUR HIERARKI ---
def get_hierarchy(kode_target, df):
    parts = str(kode_target).split('.')
    hierarchy_list = []
    current_code = ""
    levels = ["Primer", "Sekunder", "Tersier", "Kuartier", "Kuintier"]

    for i, part in enumerate(parts):
        current_code = (current_code + "." + part) if current_code else part
        match = df[df['kode'] == current_code]
        uraian = match.iloc[0]['uraian'].title() if not match.empty else "Detail Klasifikasi"
        label = levels[i] if i < len(levels) else f"Level {i+1}"
        hierarchy_list.append(f"└─ **{current_code}**: {uraian} *({label})*")
    return hierarchy_list

# --- 3. LOGIKA NLP & FUZZY MATCHING (Versi "Magnet Rumpun") ---
def smart_classify(user_input, df, top_n=3):
    # A. Preprocess input user
    clean_input = preprocess_text(user_input)
    
    # B. TF-IDF Matching (Akurasi Makna)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    all_docs = df['clean_uraian'].tolist() + [clean_input]
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    # C. Hitung Skor Murni Awal
    skor_awal = []
    for idx, score in enumerate(cosine_sim):
        fuzzy_score = fuzz.token_set_ratio(clean_input, df.iloc[idx]['clean_uraian']) / 100
        combined_score = (score * 0.70) + (fuzzy_score * 0.30)
        skor_awal.append({'idx': idx, 'skor': combined_score})
        
    # Urutkan untuk mengetahui siapa Juara 1 sesungguhnya
    skor_awal = sorted(skor_awal, key=lambda x: x['skor'], reverse=True)
    
    # D. FITUR MAGNET RUMPUN (Merapikan Juara 2 dan 3 agar tidak melenceng)
    if skor_awal:
        idx_juara_1 = skor_awal[0]['idx']
        kode_juara_1 = str(df.iloc[idx_juara_1]['kode'])
        
        # Ambil rumpun utama dari Juara 1 (Contoh: "900.1.3.1" -> "900.1")
        bagian_kode = kode_juara_1.split('.')
        rumpun_utama = ".".join(bagian_kode[:2]) if len(bagian_kode) >= 2 else bagian_kode[0]
        
        hasil_akhir = []
        for item in skor_awal:
            idx = item['idx']
            skor = item['skor']
            kode_item = str(df.iloc[idx]['kode'])
            
            # Berikan bonus skor 50% jika item ini satu rumpun dengan Juara 1
            if kode_item.startswith(rumpun_utama) and idx != idx_juara_1:
                skor = skor * 1.5 
                
            hasil_akhir.append((idx, skor))
            
        # Urutkan kembali setelah bonus diberikan
        hasil_akhir = sorted(hasil_akhir, key=lambda x: x[1], reverse=True)
        
        # Selalu tampilkan 3 teratas sesuai permintaan
        return hasil_akhir[:top_n]
    
    return []

# --- 4. ANTARMUKA ---
st.title("🗂️ SIKAP (Ultimate)")
st.caption("Asisten Arsiparis Cerdas - Pendekatan NLP & Fuzzy Logic")

try:
    df = load_data()
    user_input = st.text_input("Masukkan Perihal Surat:", placeholder="Contoh: keneikan pangkt pns...")

    if user_input:
        with st.spinner('Menganalisis bahasa...'):
            results = smart_classify(user_input, df)
            
            if results:
                st.write("---")
                for i, (idx, score) in enumerate(results):
                    res = df.iloc[idx]
                    with st.expander(f"Rekomendasi #{i+1}: Kode {res['kode']} (Keyakinan: {score:.1%})", expanded=(i==0)):
                        st.write("**Struktur Hierarki:**")
                        hierarki = get_hierarchy(res['kode'], df)
                        for h in hierarki:
                            st.markdown(h)
            else:
                st.warning("Tidak ditemukan klasifikasi yang cocok.")
except Exception as e:
    st.error(f"Error: {e}")
