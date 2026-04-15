import streamlit as st
import pandas as pd
import google.generativeai as genai
from thefuzz import fuzz, process
import os

# =================================================================
# 1. KONFIGURASI HALAMAN
# =================================================================
st.set_page_config(page_title="AI Archivist Pro", page_icon="📁", layout="centered")

st.markdown("""
<style>
    .main-title { text-align: center; font-size: 30px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { text-align: center; color: #64748B; margin-bottom: 30px; }
    .stButton>button { width: 100%; border-radius: 10px; height: 50px; background-color: #0F766E; color: white; font-weight: bold; border:none;}
    .result-card { padding: 20px; border-radius: 12px; background-color: #F0FDF4; border-left: 6px solid #16A34A; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

# =================================================================
# 2. INISIALISASI DATA & AI
# =================================================================
@st.cache_data
def inisialisasi_sistem_lokal():
    try:
        df = pd.read_csv("klasifikasi_arsip.csv", encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        df['uraian_ai'] = df['uraian_ai'].fillna("")
        return df
    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")
        st.stop()

df_data = inisialisasi_sistem_lokal()

# Setup Gemini untuk Tahap 2
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    try:
        model_ai = genai.GenerativeModel("gemini-1.5-flash")
    except:
        model_ai = genai.GenerativeModel("gemini-pro")
else:
    model_ai = None

# =================================================================
# 3. MESIN PENCARI HYBRID (KHUSUS BAHASA INDONESIA)
# =================================================================
def cari_kandidat_hybrid(query, df, limit=6):
    query_clean = str(query).lower().strip()
    query_words = set(query_clean.split())
    
    # Kata kunci prioritas yang memaksa mesin menargetkan kelompok tertentu
    keyword_map = {
        "cuti": "800", "gaji": "900", "mutasi": "800", "pensiun": "800", "cpns": "800",
        "keuangan": "900", "spp": "900", "spm": "900", "sp2d": "900", "apbd": "900",
        "arsip": "040", "kearsipan": "040", "sosialisasi": "000", "rapat": "000"
    }
    
    # Deteksi prioritas kelompok (misal: ada kata "cuti", langsung fokus ke 800)
    target_prefix = None
    for word in query_words:
        for key, prefix in keyword_map.items():
            if key in word:
                target_prefix = prefix
                break
        if target_prefix: break

    results = []
    for idx, row in df.iterrows():
        uraian_asli = str(row['uraian']).lower()
        uraian_ai = str(row['uraian_ai']).lower()
        kode = str(row['kode']).strip()
        
        # 1. Skor Token Set (Cocok logis antar kata)
        score_set_ai = fuzz.token_set_ratio(query_clean, uraian_ai)
        score_set_asli = fuzz.token_set_ratio(query_clean, uraian_asli)
        
        # 2. Skor Token Sort (Kecocokan murni teks)
        score_sort = fuzz.token_sort_ratio(query_clean, uraian_ai)
        
        # 3. Pembobotan Hibrida
        # Uraian asli lebih penting (bobot 2x) daripada silsilah AI untuk mencegah bias
        total_score = (score_set_asli * 2) + score_set_ai + score_sort
        
        # 4. Injeksi Prioritas (Jika sistem mendeteksi kata kunci khusus)
        if target_prefix and kode.startswith(target_prefix):
            total_score += 150 # Beri bonus besar untuk kode yang sesuai target
        
        if total_score > 100: # Batas toleransi relevansi
            results.append({
                "Kode": kode,
                "Uraian": str(row['uraian']).strip(),
                "Konteks": str(row['uraian_ai']).strip(),
                "Score": total_score
            })
            
    # Urutkan dan ambil yang terbaik
    results = sorted(results, key=lambda x: x['Score'], reverse=True)[:limit]
    
    # Format nilai agar mudah dibaca user
    for r in results:
        r['Relevansi'] = "Tinggi" if r['Score'] > 250 else ("Sedang" if r['Score'] > 150 else "Rendah")
        
    return results

# =================================================================
# 4. ANTARMUKA (UI)
# =================================================================
st.markdown('<div class="main-title">📁 AI Klasifikasi Arsip</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Mesin Pencari Hybrid (Optimasi Tata Naskah Indonesia)</div>', unsafe_allow_html=True)

uraian_user = st.text_input("🔍 Masukkan Perihal Surat / Uraian Arsip:", placeholder="Contoh: Cuti tahunan pegawai")

if uraian_user:
    if st.button("🚀 EKSEKUSI KLASIFIKASI"):
        
        # --- TAHAP 1: PENCARIAN HYBRID LOKAL ---
        kandidat = cari_kandidat_hybrid(uraian_user, df_data, limit=6)
        
        st.markdown("### 🔎 Tahap 1: Hasil Pencarian Lokal")
        if not kandidat:
            st.warning("Tidak ditemukan kecocokan di database.")
        else:
            st.table(pd.DataFrame(kandidat)[["Kode", "Uraian", "Relevansi"]])
            
            # --- TAHAP 2: VALIDASI AI ---
            if model_ai:
                st.markdown("### 🤖 Tahap 2: Keputusan Akhir AI")
                with st.spinner("AI sedang mengunci fungsi utama..."):
                    kandidat_str = "\n".join([f"- Kode {c['Kode']} | {c['Uraian']} (Konteks: {c['Konteks']})" for c in kandidat])
                    
                    prompt = f"""
                    Anda adalah Ahli Tata Naskah Dinas Pemerintah Indonesia.
                    
                    URAIAN PENGGUNA: "{uraian_user}"
                    
                    KANDIDAT KODE:
                    {kandidat_str}
                    
                    PANDUAN MUTLAK:
                    - Surat tentang Cuti/Mutasi/Nasib Pegawai WAJIB masuk kelompok Kepegawaian (800).
                    - Surat pencairan uang WAJIB masuk Keuangan (900).
                    - Sosialisasi/Bimtek tentang kearsipan WAJIB masuk Kearsipan (040/000.5).
                    
                    Pilih HANYA 1 KODE TERBAIK dari kandidat di atas.
                    
                    FORMAT HASIL:
                    **BAGIAN INTI:** [Nama Urusan, misal: Kepegawaian]
                    **KODE FINAL:** [Kode] - [Uraian]
                    **ANALISIS:** [Jelaskan alasan pemilihannya]
                    """
                    try:
                        response = model_ai.generate_content(prompt)
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.write(response.text)
                        st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Gagal menghubungi AI: {e}")
            else:
                st.info("API Gemini tidak aktif. Silakan pilih dari tabel di atas.")

st.markdown("---")
st.caption("EKlasifikasi Pro | Sistem Pencari Berbasis Kata Kunci Prioritas")
