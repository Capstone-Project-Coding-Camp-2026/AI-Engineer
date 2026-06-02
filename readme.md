
📖 FINTIME MACHINE LEARNING MODELS - README


Repositori ini memuat pipeline pemodelan Machine Learning untuk aplikasi FinTime. 
Terdapat tiga model utama yang dibangun menggunakan TensorFlow/Keras, di mana 
masing-masing menangani klasifikasi transaksi, prediksi pengeluaran, dan simulasi 
keputusan finansial pengguna (What-If Analysis). Sebagian besar model didesain 
agar kompatibel dan dapat diekspor ke format TensorFlow.js untuk deployment di frontend.

--------------------------------------------------------------------------------
🧠 PENJELASAN SINGKAT MODEL
--------------------------------------------------------------------------------

1. Model A: NLP Transaction Classifier (FinTime_Modelling_A_Clasification.ipynb)
   * Fungsi: Mengkategorikan riwayat transaksi secara otomatis berdasarkan teks 
     deskripsinya (misal: "MCDONALD" menjadi Makanan & Minuman).
   * Arsitektur: Menggunakan pendekatan hibrida. Lapis pertama berupa 
     rule-based keyword matching. Lapis kedua menggunakan model Keras Sequential 
     (dengan pendekatan char-ngram hashing + Dense softmax). Model akan otomatis 
     melempar transaksi ke kategori "lainnya" jika confidence level berada di 
     bawah 0.70.
   * Output Ekspor: File berformat `.keras` beserta file `metadata.json` yang 
     memuat konfigurasi vocabulary dan indeks.

2. Model B: Time-Series Forecasting (FinTime_Modelling_B_Forecasting.ipynb)
   * Fungsi: Memprediksi pengeluaran total (total_expense) pengguna pada bulan 
     berikutnya berdasarkan pola finansial historis mereka.
   * Arsitektur: Model diregresi menggunakan Keras Sequential yang mengevaluasi 
     parameter lag features (data pengeluaran bulan N-1, N-2, N-3) serta rolling 
     mean. Evaluasi dipisah secara temporal per user_id untuk menghindari 
     kebocoran data (data leakage).
   * Output Ekspor: Dkonversi khusus via tfjs.converters.save_keras_model() 
     sehingga menghasilkan `model.json` + `*.bin` untuk browser, beserta file 
     parameter `scaler_config.json`.

3. Model C: What-If Analysis (FinTime_Modelling_C_WhatIf.ipynb)
   * Fungsi: Menjalankan simulasi rencana pembelian untuk memberikan rekomendasi 
     klasifikasi cerdas (misal: buy_careful, dont_buy, dll).
   * Arsitektur: Model klasifikasi Keras yang memproses input 38 kolom fitur 
     dinamis yang telah dilakukan normalisasi dan dummifikasi.
   * Output Ekspor: File `whatif_model.keras` dan `metadata.json` (memuat informasi 
     rata-rata scaler, skala scaler, nama kelas, dan nama-nama kolom fitur).

--------------------------------------------------------------------------------
⚙️ PERSYARATAN SISTEM (PREREQUISITES)
--------------------------------------------------------------------------------

Pastikan environment lokal Anda memiliki perangkat lunak berikut sebelum mengeksekusi notebook:
* Python: Versi 3.10 atau di atasnya (sangat disarankan Python 3.12).
* Environment: Jupyter Notebook atau JupyterLab.
* Library Python Utama: pandas, numpy, scikit-learn, tensorflow, tensorflowjs, joblib, matplotlib, seaborn.

--------------------------------------------------------------------------------
🚀 TUTORIAL MENJALANKAN MODEL
--------------------------------------------------------------------------------

Ikuti langkah-langkah di bawah ini untuk menjalankan kode sumber, melatih ulang 
(retraining), atau menghasilkan file ekspor model baru:

Langkah 1: Siapkan Virtual Environment
Direkomendasikan menggunakan virtual environment yang terisolasi agar versi 
library (terutama TensorFlow) tidak bertabrakan dengan project lain.
   
   Command (Terminal/Prompt):
   python -m venv .venv

   Aktivasi Environment:
   * Windows: .venv\Scripts\activate
   * Mac/Linux: source .venv/bin/activate

Langkah 2: Instalasi Kebutuhan Library
Instal semua modul yang diperlukan (Anda juga dapat menggunakan requirements.txt jika ada).
   
   Command:
   pip install -r requirements.txt

   Atau install secara manual:
   pip install pandas numpy scikit-learn tensorflow tensorflowjs joblib matplotlib seaborn jupyter

Langkah 3: Jalankan Jupyter Notebook
Posisikan terminal Anda pada direktori tempat file .ipynb tersebut berada, lalu ketik:
   
   Command:
   jupyter notebook

Langkah 4: Melatih dan Mengekspor Model
Buka antarmuka Jupyter Notebook di browser Anda dan buka salah satu file berikut 
sesuai kebutuhan:
1. Buka 'FinTime_Modelling_A_Clasification.ipynb' lalu klik "Run All Cells" untuk 
   melatih ulang model NLP klasifikasi transaksi.
2. Buka 'FinTime_Modelling_B_Forecasting.ipynb' lalu klik "Run All Cells" untuk 
   merangkai fitur lag temporal dan melatih model prediksinya.
3. Buka 'FinTime_Modelling_C_WhatIf.ipynb' lalu klik "Run All Cells" untuk 
   mengegsekusi model simulasi klasifikasi skenario pengeluaran.

Langkah 5: Periksa Hasil Output
Jika proses Run All Cells berhasil sampai ke bagian paling akhir, perhatikan 
folder lokal tempat notebook tersebut berada. Semua file model (ekstensi .keras, 
model.json, atau .bin) dan parameter pendukung (file .json, vectorizer.pkl) akan 
ter-generate dan siap ditransfer menuju repositori Frontend atau Backend Anda!
================================================================================
