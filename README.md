# Streamlit App Prediksi Status Mahasiswa

Aplikasi web untuk memprediksi status mahasiswa (Dropout, Graduate, Enrolled) menggunakan model Random Forest yang sudah dilatih.

---

## Persiapan

Pastikan kamu sudah memiliki semua file berikut di dalam satu folder proyek yang sama dengan `app.py`:

- `model_rf.pkl` — model Random Forest yang sudah dilatih
- `scaler.pkl` — scaler fitur untuk preprocessing
- `selected_features_mask.pkl` — mask untuk seleksi fitur
- `fitur_awal.pkl` — daftar nama fitur asli sebelum seleksi
- `label_mapping.pkl` — mapping label target ke nama status

---

## Instalasi dependencies

1. Pastikan kamu menggunakan Python versi 3.11 atau versi kompatibel.
2. Buat file `requirements.txt` dengan isi berikut:

    ```
    streamlit
    pandas
    numpy
    scikit-learn
    joblib
    imblearn
    ```

3. Jalankan perintah ini di terminal untuk menginstal semua paket:

    ```bash
    pip install -r requirements.txt
    ```

---

## Menjalankan aplikasi Streamlit

Jalankan perintah berikut di terminal pada folder proyek:

```bash
streamlit run app.py

---

## Cara menggunakan aplikasi

1. Isi form data mahasiswa sesuai input yang diminta.
2. Klik tombol **Prediksi**.
3. Hasil prediksi status mahasiswa (Dropout, Graduate, atau Enrolled) akan muncul di layar.
