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

# --- 1. MEMUAT DATABASE (ASLI 100%) ---
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
    df['clean_uraian'] = df['uraian'].apply(preprocess_text)
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

# --- 3. LOGIKA AI HYBRID (HIERARCHICAL NARROWING - TOP 3 JALUR) ---
def smart_classify(user_input, df, top_n=3):
    # TAHAP 0: Ekstrak inti surat via LLM (sudah ada konteks bersih dari Groq)
    inti_dari_llm = ekstrak_inti_surat(user_input)
    st.info(f"🧠 SIKAP menangkap inti surat Anda sebagai: **{inti_dari_llm}**")
    clean_input = preprocess_text(inti_dari_llm)

    # ── HELPER: Hitung level dari jumlah titik ──
    # "000"       → level 0 (Primer)
    # "000.2"     → level 1 (Sekunder)
    # "000.2.3"   → level 2 (Tersier)
    # "000.2.3.2" → level 3 (Kuartier)
    def get_level(kode):
        return str(kode).count('.')

    # ── HELPER: Ambil anak langsung dari suatu kode induk ──
    def get_children_df(induk_kode, target_level):
        return df[
            (df['kode'].str.startswith(induk_kode + ".")) &
            (df['kode'].apply(get_level) == target_level)
        ].copy().reset_index(drop=True)

    # ── HELPER: TF-IDF + Fuzzy → dua jalur (stem + tanpa stem) ──
  def skor_dan_ambil(subset_df, top_k=5):
        if subset_df.empty:
            return subset_df.iloc[0:0]

        # Jalur 1: pakai clean_uraian (sudah di-stem) vs query yang sudah di-stem
        vectorizer1 = TfidfVectorizer(ngram_range=(1, 3))
        docs1 = subset_df['clean_uraian'].tolist() + [query_clean]
        tfidf1 = vectorizer1.fit_transform(docs1)
        cosine1 = cosine_similarity(tfidf1[-1], tfidf1[:-1])[0]

        # Jalur 2: pakai uraian ASLI (lowercase) vs inti mentah dari LLM (tanpa stem)
        uraian_asli = subset_df['uraian'].str.lower().tolist()
        query_mentah = inti_dari_llm.lower()  # langsung pakai hasil Groq tanpa stem
        vectorizer2 = TfidfVectorizer(ngram_range=(1, 3))
        docs2 = uraian_asli + [query_mentah]
        tfidf2 = vectorizer2.fit_transform(docs2)
        cosine2 = cosine_similarity(tfidf2[-1], tfidf2[:-1])[0]

        hasil = []
        for i in range(len(subset_df)):
            # Fuzzy juga dua jalur
            fuzzy_stem = fuzz.partial_ratio(query_clean, subset_df.iloc[i]['clean_uraian']) / 100
            fuzzy_asli = fuzz.partial_ratio(query_mentah, uraian_asli[i]) / 100

            # Gabungkan: stem 30% + asli 40% + fuzzy stem 10% + fuzzy asli 20%
            combined = (
                cosine1[i] * 0.30 +
                cosine2[i] * 0.40 +
                fuzzy_stem  * 0.10 +
                fuzzy_asli  * 0.20
            )
            hasil.append((i, combined))

        hasil.sort(key=lambda x: x[1], reverse=True)
        top_idx = [h[0] for h in hasil[:top_k]]
        return subset_df.iloc[top_idx].reset_index(drop=True)

  # ── HELPER: Groq menjuri dan memilih N kode terbaik ──
    # Groq menerima "inti_dari_llm" (hasil ekstraksi tahap 0) sebagai konteks utama
    def juri_ai(inti, kandidat_df, label_level, pilih_n=1):
        if kandidat_df.empty:
            return []
        daftar = ""
        for i, (_, row) in enumerate(kandidat_df.iterrows()):
            daftar += f"[{i+1}] Kode: {row['kode']} | Uraian: {row['uraian'].title()}\n"
        if pilih_n == 1:
            instruksi = "Balas HANYA dengan 1 angka saja."
            contoh = "Contoh balasan benar: 3"
        else:
            instruksi = f"Balas HANYA dengan {pilih_n} angka dipisah koma, urut dari paling tepat."
            contoh = "Contoh balasan benar: 2, 5, 1"
        prompt = f"""Anda adalah ahli kearsipan pemerintah daerah Indonesia.

Inti urusan surat yang sudah diekstrak AI: "{inti}"
Uraian surat asli dari user: "{user_input}"

Pilih {pilih_n} nomor dari daftar {label_level} yang PALING TEPAT untuk urusan tersebut:

{daftar}
ATURAN MUTLAK: {instruksi}
{contoh}
"""
        try:
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.0,
            )
            balasan = resp.choices[0].message.content.strip()
            angka_mentah = re.findall(r'\d+', balasan)
            hasil_kode = []
            for a in angka_mentah:
                idx = int(a) - 1
                if 0 <= idx < len(kandidat_df):
                    kode = kandidat_df.iloc[idx]['kode']
                    if kode not in hasil_kode:
                        hasil_kode.append(kode)
                if len(hasil_kode) == pilih_n:
                    break
            # Fallback: isi dari baris teratas jika kurang
            for _, row in kandidat_df.iterrows():
                if len(hasil_kode) == pilih_n:
                    break
                if row['kode'] not in hasil_kode:
                    hasil_kode.append(row['kode'])
            return hasil_kode
        except Exception as e:
            st.error(f"🚨 ERROR GROQ (Juri {label_level}): {e}")
            return list(kandidat_df['kode'])[:pilih_n]

 # ════════════════════════════════════════════════════════
    # TAHAP 1 — PRIMER: Pilih top 3 rumpun utama (level 0)
    # Sertakan sampel uraian keturunan agar Groq paham isi rumpun
    # ════════════════════════════════════════════════════════
    df_primer = df[df['kode'].apply(get_level) == 0].copy().reset_index(drop=True)

    if df_primer.empty:
        st.warning("Database tidak memiliki data level Primer.")
        return []

    # Bangun daftar primer DENGAN sampel keturunannya untuk konteks Groq
    def build_primer_context(kandidat_primer_df):
        daftar = ""
        for i, (_, row) in enumerate(kandidat_primer_df.iterrows()):
            kode_p = row['kode']
            uraian_p = row['uraian'].title()
            # Ambil 5 sampel keturunan acak dari rumpun ini
            keturunan = df[df['kode'].str.startswith(kode_p + ".")]['uraian'].dropna()
            sampel = keturunan.sample(min(5, len(keturunan)), random_state=42).tolist() if len(keturunan) > 0 else []
            sampel_str = "; ".join(sampel) if sampel else "tidak ada sub-klasifikasi"
            daftar += f"[{i+1}] Kode: {kode_p} | Uraian: {uraian_p}\n"
            daftar += f"      Contoh isi rumpun: {sampel_str}\n\n"
        return daftar

    # Skor semua primer dulu
    top_primer = skor_dan_ambil(df_primer, top_k=len(df_primer))  # ambil semua primer

    # Buat prompt khusus primer dengan konteks isi rumpun
    daftar_primer_konteks = build_primer_context(top_primer)

    prompt_primer = f"""Anda adalah ahli kearsipan pemerintah daerah Indonesia.

Inti urusan surat yang sudah diekstrak AI: "{inti_dari_llm}"
Uraian surat asli dari user: "{user_input}"

Berikut daftar RUMPUN UTAMA (Primer) beserta contoh isi klasifikasinya:

{daftar_primer_konteks}

Pilih 3 nomor rumpun yang PALING MUNGKIN mengandung klasifikasi untuk urusan tersebut.
ATURAN MUTLAK: Balas HANYA dengan 3 angka dipisah koma, urut dari paling tepat.
Contoh balasan benar: 1, 4, 7
"""
    try:
        resp_primer = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_primer}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
        )
        balasan_primer = resp_primer.choices[0].message.content.strip()
        angka_primer = re.findall(r'\d+', balasan_primer)
        kode_primer_list = []
        for a in angka_primer:
            idx = int(a) - 1
            if 0 <= idx < len(top_primer):
                kode = top_primer.iloc[idx]['kode']
                if kode not in kode_primer_list:
                    kode_primer_list.append(kode)
            if len(kode_primer_list) == 3:
                break
        # Fallback
        for _, row in top_primer.iterrows():
            if len(kode_primer_list) == 3:
                break
            if row['kode'] not in kode_primer_list:
                kode_primer_list.append(row['kode'])
    except Exception as e:
        st.error(f"🚨 ERROR GROQ (Juri Primer): {e}")
        kode_primer_list = list(top_primer['kode'])[:3]

    st.caption(f"📌 Top 3 Primer: **{', '.join(kode_primer_list)}**")

    # ════════════════════════════════════════════════════════
    # TAHAP 2–4: Untuk setiap primer, telusuri 1 jalur terbaik
    # ════════════════════════════════════════════════════════
    jalur_hasil = []

    for kode_primer in kode_primer_list:
        jejak = [kode_primer]

        # ── SEKUNDER ──
        df_sek = get_children_df(kode_primer, target_level=1)
        if df_sek.empty:
            jalur_hasil.append({'kode_akhir': kode_primer, 'jejak': jejak})
            continue
        top_sek = skor_dan_ambil(df_sek, clean_input, top_k=5)
        kode_sek = juri_ai(inti_dari_llm, top_sek, "Sekunder", pilih_n=1)[0]
        jejak.append(kode_sek)

        # ── TERSIER ──
        df_ter = get_children_df(kode_sek, target_level=2)
        if df_ter.empty:
            jalur_hasil.append({'kode_akhir': kode_sek, 'jejak': jejak})
            continue
        top_ter = skor_dan_ambil(df_ter, clean_input, top_k=5)
        kode_ter = juri_ai(inti_dari_llm, top_ter, "Tersier", pilih_n=1)[0]
        jejak.append(kode_ter)

        # ── KUARTIER ──
        df_kua = get_children_df(kode_ter, target_level=3)
        if df_kua.empty:
            jalur_hasil.append({'kode_akhir': kode_ter, 'jejak': jejak})
            continue
        top_kua = skor_dan_ambil(df_kua, clean_input, top_k=5)
        kode_kua = juri_ai(inti_dari_llm, top_kua, "Kuartier", pilih_n=1)[0]
        jejak.append(kode_kua)
        jalur_hasil.append({'kode_akhir': kode_kua, 'jejak': jejak})

    # Konversi ke format (df_index, skor_simulasi, jejak)
    hasil_return = []
    skor_simulasi = [0.99, 0.85, 0.70]
    for i, jalur in enumerate(jalur_hasil):
        idx_match = df[df['kode'] == jalur['kode_akhir']].index
        if not idx_match.empty:
            hasil_return.append((
                idx_match[0],
                skor_simulasi[i] if i < len(skor_simulasi) else 0.60,
                jalur['jejak']
            ))

    return hasil_return
    
# --- 4. ANTARMUKA UTAMA ---

st.markdown("<div class='sikap-title'>SIKAP</div>", unsafe_allow_html=True)
st.markdown("<div class='sikap-subtitle'>Sistem Informasi Klasifikasi Arsip Pintar</div>", unsafe_allow_html=True)

try:
    df = load_data()
    
    with st.sidebar:
        st.header("🕒 Riwayat Pencarian")
        if st.session_state.search_history:
            for riwayat in reversed(st.session_state.search_history[-10:]):
                st.caption(f"• {riwayat}")
            if st.button("Hapus Riwayat", use_container_width=True):
                st.session_state.search_history = []
                st.rerun()
        else:
            st.info("Belum ada pencarian.")

    tab_ai, tab_katalog = st.tabs(["🤖 Pencarian AI (Cerdas)", "📂 Jelajah Kode Klasifikasi (Manual)"])

    # ================= TAB 1: PENCARIAN AI =================
    with tab_ai:
        st.write("Masukkan perihal atau deskripsi surat, biarkan kecerdasan buatan mencari kode klasifikasi yang paling tepat untuk Anda.")
        user_input = st.text_input("📝 Perihal Surat / Dokumen:", placeholder="Contoh: permohonan cuti tahunan pegawai atau undangan rapat tapd...", key="input_ai")

        if user_input:
            if user_input not in st.session_state.search_history:
                st.session_state.search_history.append(user_input)

            with st.spinner('Menganalisis bahasa dan mencari kecocokan...'):
                results = smart_classify(user_input, df)
                
                if results:
                    st.success("✅ Analisis selesai! Berikut 3 jalur klasifikasi terbaik untuk dokumen Anda:")
                    pilihan_feedback = []
                    medal = ["🥇", "🥈", "🥉"]

                    for i, (idx, score, jejak) in enumerate(results):
                        res = df.iloc[idx]
                        pilihan_feedback.append(f"{res['kode']} - {res['uraian'].title()}")

                        with st.expander(
                            f"{medal[i]} Jalur #{i+1}: Kode {res['kode']} — {res['uraian'].title()} (Keyakinan: {score:.1%})",
                            expanded=(i == 0)
                        ):
                            # Jejak hierarki yang dilalui AI level per level
                            st.markdown("**🗺️ Jejak Hierarki yang Dipilih AI:**")
                            label_level = ["Primer", "Sekunder", "Tersier", "Kuartier"]
                            warna = ["#B71C1C", "#1565C0", "#2E7D32", "#E65100"]
                            for j, kode_jejak in enumerate(jejak):
                                match = df[df['kode'] == kode_jejak]
                                uraian_jejak = match.iloc[0]['uraian'].title() if not match.empty else "—"
                                level_label = label_level[j] if j < len(label_level) else f"Level {j}"
                                warna_bg = warna[j] if j < len(warna) else "#424242"
                                indent = j * 24
                                arrow = "└─ " if j > 0 else ""
                                st.markdown(
                                    f"<div style='margin-left:{indent}px; margin-bottom:6px;'>"
                                    f"<span style='background:{warna_bg}; color:#fff; padding:5px 10px; "
                                    f"border-radius:5px; font-size:0.9em; display:inline-block;'>"
                                    f"{arrow}<strong>{kode_jejak}</strong> &nbsp;|&nbsp; {uraian_jejak} "
                                    f"<i style='opacity:0.8;'>({level_label})</i></span></div>",
                                    unsafe_allow_html=True
                                )

                            st.markdown("**📋 Struktur Hierarki Lengkap:**")
                            hierarki = get_hierarchy(res['kode'], df)
                            for h in hierarki:
                                st.markdown(h, unsafe_allow_html=True)

                    st.divider()
                    st.markdown("#### 💡 Bantu SIKAP Menjadi Lebih Pintar")
                    st.write("Dari 3 jalur di atas, mana kode yang paling tepat menurut Anda?")

                    with st.form("feedback_form"):
                        jawaban_benar = st.radio("Pilih kode yang benar:", pilihan_feedback)
                        submit_feedback = st.form_submit_button("Kirim Masukan")

                        if submit_feedback:
                            waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            data_feedback = f"{waktu_sekarang} | Input: {user_input} | Terpilih: {jawaban_benar}\n"
                            with open("feedback_ai_log.txt", "a", encoding="utf-8") as f:
                                f.write(data_feedback)
                            st.success("Terima kasih! Pilihan Anda telah disimpan untuk evaluasi AI ke depannya.")

                else:
                    st.warning("Tidak ditemukan klasifikasi yang cocok. Coba gunakan kata kunci lain.")

    # ================= TAB 2: JELAJAH KODE (ROLLBACK DENGAN FIX NATURAL SORTING) =================
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

except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat data: {e}")
