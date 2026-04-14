import streamlit as st
import pandas as pd
import re
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document

# ========================
# CONFIG
# ========================
st.set_page_config(page_title="EKlasifikasi Arsip PRO", page_icon="📁")
st.title("📁 EKlasifikasi Arsip PRO")
st.caption("Ultimate Version: Smart + AI + Kamus Istilah")

# ========================
# GEMINI (OPSIONAL)
# ========================
model = None
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            model = genai.GenerativeModel(m.name)
            break
except:
    model = None

# ========================
# LOAD DATA (AUTO CLEAN)
# ========================
@st.cache_data
def load_data():
    df = pd.read_csv("klasifikasi_arsip.csv", encoding="latin1")

    df = df.iloc[:, 0]
    df = df.str.split(",", n=1, expand=True)
    df.columns = ["kode", "uraian"]

    df["kode"] = df["kode"].str.strip()
    df["uraian"] = df["uraian"].str.strip()

    df = df.dropna()
    df = df[df["uraian"].str.len() > 3]
    df = df.drop_duplicates()

    return df

data = load_data()

# ========================
# KAMUS ISTILAH 🔥
# ========================
keyword_map = {
    "cuti": "cuti pegawai",
    "izin": "cuti pegawai",
    "libur": "cuti pegawai",
    "gaji": "penggajian pegawai",
    "honor": "penggajian pegawai",
    "tunjangan": "penggajian pegawai",
    "rapat": "kegiatan rapat dinas",
    "perjalanan": "perjalanan dinas",
    "dinas luar": "perjalanan dinas"
}

# ========================
# STOPWORDS
# ========================
stopwords_id = [
    "dan","yang","di","ke","dari","untuk","pada","dengan",
    "adalah","itu","ini","dalam","oleh","atau","sebagai"
]

# ========================
# PREPROCESS
# ========================
def preprocess(text):
    text = text.lower()

    for k,v in keyword_map.items():
        text = text.replace(k, v)

    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    words = text.split()
    words = [w for w in words if w not in stopwords_id]

    return " ".join(words)

data["uraian_clean"] = data["uraian"].astype(str).apply(preprocess)

# ========================
# DOCX SMART
# ========================
def read_docx_smart(file):
    try:
        doc = Document(file)
        paragraphs = [p.text.strip() for p in doc.paragraphs if len(p.text.strip()) > 30]

        if not paragraphs:
            return ""

        full_text = " ".join(paragraphs)

        sentences = full_text.split(".")
        sentences = [s.strip() for s in sentences if len(s.strip()) > 40]

        sentences = sorted(sentences, key=len, reverse=True)[:5]

        return ". ".join(sentences)

    except:
        return ""

# ========================
# INPUT
# ========================
uploaded = st.file_uploader("📂 Upload Word (.docx)", type=["docx"])

if uploaded:
    uraian_user = read_docx_smart(uploaded)
    if uraian_user:
        st.success("Dokumen dipahami (inti diambil)")
    else:
        st.warning("Dokumen tidak terbaca")
else:
    uraian_user = st.text_input("🔍 Masukkan uraian / kode:")

# ========================
# DETEKSI KODE LANGSUNG 🔥
# ========================
if uraian_user:
    match = data[data["kode"] == uraian_user.strip()]
    if not match.empty:
        row = match.iloc[0]
        st.success("🎯 Kode ditemukan langsung")
        st.write(f"{row['kode']} - {row['uraian']}")
        st.stop()

# ========================
# TF-IDF + BOOST
# ========================
def tfidf_boost(df, text, top_n=5):
    vec = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, max_features=5000)

    tfidf = vec.fit_transform(df["uraian_clean"])
    user_vec = vec.transform([preprocess(text)])

    sim = cosine_similarity(user_vec, tfidf)[0]

    # 🔥 BOOST jika kata muncul
    for i, uraian in enumerate(df["uraian_clean"]):
        for word in text.lower().split():
            if word in uraian:
                sim[i] += 0.1

    idx = sim.argsort()[-top_n:][::-1]

    hasil = df.iloc[idx].copy()
    hasil["skor"] = sim[idx]

    return hasil

# ========================
# PROSES
# ========================
if uraian_user:

    # AI RINGKAS
    if model and len(uraian_user) > 300:
        try:
            res = model.generate_content(f"Ringkas inti dokumen:\n{uraian_user}")
            if hasattr(res, "text"):
                uraian_user = res.text
                st.info("🧠 Ringkasan AI aktif")
        except:
            pass

    kandidat = tfidf_boost(data, uraian_user, 5)

    kandidat["panjang"] = kandidat["kode"].apply(len)
    kandidat = kandidat.sort_values(by=["skor","panjang"], ascending=False)

    # ========================
    # TAMPILKAN
    # ========================
    st.info("💡 Kandidat terbaik")

    kandidat_text = ""
    for _, row in kandidat.iterrows():
        st.write(f"{row['kode']} - {row['uraian']} ({row['skor']*100:.2f}%)")
        kandidat_text += f"{row['kode']} - {row['uraian']}\n"

    terbaik = kandidat.iloc[0]

    st.success("🎯 Rekomendasi Sistem")
    st.write(f"**{terbaik['kode']} - {terbaik['uraian']}**")

    # ========================
    # AI FINAL
    # ========================
    if model:
        if st.button("🚀 Validasi dengan AI"):
            try:
                prompt = f"""
Pilih 1 klasifikasi paling tepat.

Kandidat:
{kandidat_text}

Uraian:
{uraian_user}

Jawaban:
Kode - Uraian
Alasan
"""
                res = model.generate_content(prompt)
                st.success("🎯 Hasil AI")
                st.write(res.text)
            except:
                st.warning("AI gagal → gunakan hasil sistem")

# ========================
# FOOTER
# ========================
st.markdown("---")
st.caption("EKlasifikasi Arsip PRO © 2026 | by Heryanto S.Pd")
