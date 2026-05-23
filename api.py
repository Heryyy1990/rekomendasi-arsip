from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# KUNCI RAHASIA: Kita "meminjam" fungsi pintar dari app.py Anda TANPA mengubah isinya!
from app import smart_classify, load_data

app = FastAPI(title="SIKAP API")

# Izinkan akses dari aplikasi Android
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Muat database arsip saat server menyala
df_klasifikasi = load_data()

# Struktur data yang akan dikirim oleh HP Android
class SuratInput(BaseModel):
    teks: str

@app.post("/api/klasifikasi")
def proses_klasifikasi(surat: SuratInput):
    # Memanggil otak SIKAP dari app.py
    rekomendasi, inti = smart_classify(surat.teks, df_klasifikasi)
    
    # Mengemas hasil ke dalam JSON untuk dikirim ke Android
    hasil_bersih = []
    for idx, skor in rekomendasi:
        baris = df_klasifikasi.iloc[idx]
        hasil_bersih.append({
            "kode": str(baris['kode']),
            "uraian": str(baris['uraian']).title(),
            "hierarki": str(baris['uraian_lengkap']).title(),
            "skor_kecocokan": round(skor * 100, 2)
        })
        
    return {
        "inti_surat": inti,
        "hasil": hasil_bersih
    }
