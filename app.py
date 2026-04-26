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
    Output: <HASIL>rencana kerja anggaran, dpa</HASIL>
    Input: "Permohonan penerbitan surat perintah pencairan dana (SP2D) untuk kegiatan sosialisasi"
    Output: <HASIL>pencairan dana, sp2d</HASIL>
    Input: "Penyampaian berita acara serah terima (BAST) kendaraan dinas roda empat"
    Output: <HASIL>berita acara serah terima, kendaraan dinas</HASIL>
    
    [KASUS KEPEGAWAIAN, PENGAWASAN & HUKUM]
    Input: "Usulan penetapan angka kredit (PAK) jabatan fungsional arsiparis tingkat ahli"
    Output: <HASIL>penetapan angka kredit, jabatan fungsional</HASIL>
    Input: "Teguran disiplin pegawai dan pemanggilan pemeriksaan pelanggaran kode etik ASN"
    Output: <HASIL>disiplin pegawai, pelanggaran kode etik</HASIL>
    Input: "Tindak lanjut temuan laporan hasil pemeriksaan (LHP) BPK RI perwakilan Sulawesi Tenggara"
    Output: <HASIL>tindak lanjut temuan, laporan hasil pemeriksaan</HASIL>
    Input: "Permohonan fasilitasi penyusunan rancangan peraturan bupati tentang pedoman tata naskah dinas"
    Output: <HASIL>peraturan bupati, tata naskah dinas</HASIL>
    
    [KASUS PENDIDIKAN, KESEHATAN & INFRASTRUKTUR]
    Input: "Penyaluran dan pencairan dana bantuan operasional sekolah (BOS) tahap I"
    Output: <HASIL>dana bantuan operasional sekolah, bos</HASIL>
    Input: "Klaim penggantian biaya pelayanan kesehatan BPJS Kesehatan pasien rawat inap RSUD"
    Output: <HASIL>klaim bpjs kesehatan, pelayanan kesehatan rawat inap</HASIL>
    Input: "Persetujuan rencana anggaran biaya (RAB) dan gambar kerja proyek pembangunan jembatan"
    Output: <HASIL>rencana anggaran biaya, gambar kerja proyek</HASIL>
    
    [KASUS PEMILU, KESBANGPOL & KETERTIBAN]
    Input: "Penyampaian daftar pemilih sementara (DPS) dan daftar penduduk potensial pemilih (DP4) Pilkada"
    Output: <HASIL>daftar pemilih sementara, daftar penduduk potensial pemilih</HASIL>
    Input: "Laporan pemantauan kegiatan partai politik dan organisasi kemasyarakatan (Ormas)"
    Output: <HASIL>pemantauan partai politik, organisasi kemasyarakatan</HASIL>
    Input: "Penertiban pedagang kaki lima dan pembongkaran baliho reklame ilegal"
    Output: <HASIL>penertiban pedagang kaki lima, pembongkaran baliho</HASIL>
    
    [KASUS LINGKUNGAN HIDUP, BENCANA & PERTANIAN]
    Input: "Pembahasan dokumen analisis mengenai dampak lingkungan (AMDAL) dan UKL-UPL pabrik kelapa sawit"
    Output: <HASIL>analisis mengenai dampak lingkungan, amdal, ukl upl</HASIL>
    Input: "Laporan operasi pencarian dan pertolongan (SAR) korban banjir bandang"
    Output: <HASIL>operasi pencarian pertolongan, sar, korban banjir</HASIL>
    Input: "Sertifikasi dan pengujian keamanan pangan segar asal tumbuhan (PSAT) pasar tradisional"
    Output: <HASIL>sertifikasi keamanan pangan segar asal tumbuhan, psat</HASIL>
    
    [KASUS PEMERINTAHAN DESA & UMUM]
    Input: "Penyaluran dana desa (DD) dan penyelesaian sengketa pemilihan kepala desa (Pilkades) serentak"
    Output: <HASIL>dana desa, sengketa pemilihan kepala desa</HASIL>
    Input: "Penyampaian laporan hasil perjalanan dinas ke Arsip Nasional"
    Output: <HASIL>perjalanan dinas</HASIL>
    Input: "Persetujuan draf jadwal retensi arsip dan pemusnahan arsip inaktif"
    Output: <HASIL>jadwal retensi arsip, pemusnahan arsip inaktif</HASIL>
    
    SEKARANG, KERJAKAN DENGAN POLA LOGIKA YANG SAMA.
    ATURAN MUTLAK: TULIS HASIL AKHIRMU DI DALAM TAG <HASIL> dan </HASIL>.
    JANGAN ADA TEKS LAIN DI LUAR TAG ITU!

    Input: "{teks_user}"
    Output:
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant", # Model terbaru, pengganti llama3-8b
            temperature=0.0, # 0.0 membuat AI tidak berhalusinasi/kreatif, murni mengekstrak
        )
        # Mengambil balasan mentah dari Groq
        inti_teks_mentah = chat_completion.choices[0].message.content.strip()
        
        # PISAU BEDAH PYTHON: Ekstrak paksa isi tag <HASIL>
        import re
        match = re.search(r'<HASIL>(.*?)</HASIL>', inti_teks_mentah, re.IGNORECASE | re.DOTALL)
        if match:
            inti_teks = match.group(1).strip()
        else:
            # Fallback kalau Llama lupa tag
            baris = [b for b in inti_teks_mentah.split('\n') if b.strip()]
            inti_teks = baris[-1].replace('**', '').strip()
            
        # Membersihkan tanda kutip jika LLM iseng menambahkannya
        inti_teks = inti_teks.replace('"', '').replace("'", "")
        return inti_teks
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
        # Fuzzy Matching menggunakan partial_ratio agar lebih fleksibel terhadap typo
        fuzzy_score = fuzz.partial_ratio(clean_input, df.iloc[idx]['clean_uraian']) / 100
        combined_score = (score * 0.70) + (fuzzy_score * 0.30) 
        skor_awal.append({'idx': idx, 'skor': combined_score})
        
    # Ambil 10 besar nominasi untuk dinilai ulang oleh AI
    top_10_kandidat = sorted(skor_awal, key=lambda x: x['skor'], reverse=True)[:10]
    
   # 4. FASE JURI AI (Llama-3 memilih 3 terbaik dari 10 nominasi matematis)
    daftar_kandidat = ""
    for i, item in enumerate(top_10_kandidat):
        baris = df.iloc[item['idx']]
        daftar_kandidat += f"[{i+1}] Kode: {baris['kode']} | Uraian: {baris['uraian'].title()}\n"
        
    prompt_juri = f"""
    Pilih 3 nomor urut opsi yang paling tepat untuk urusan: "{inti_dari_llm}"
    
    Daftar Opsi:
    {daftar_kandidat}
    
    ATURAN MUTLAK:
    Kamu HANYA BOLEH membalas dengan 3 angka urutan (antara 1 sampai 10) yang dipisah koma.
    JANGAN tulis kodenya. JANGAN ada teks apapun selain 3 angka.
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
