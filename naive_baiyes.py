import pandas as pd
import numpy as np

# Fungsi untuk memuat data dari file EXCEL
def muat_data(jalur_file):
    data = pd.ExcelFile(jalur_file)
    df = data.parse(data.sheet_names[0])
    return df

# Fungsi untuk membagi data menjadi data latih dan data uji
def bagi_data(df, rasio_latih=0.83):
    ukuran_latih = int(len(df) * rasio_latih)
    data_latih = df[:ukuran_latih]
    data_uji = df[ukuran_latih:]
    return data_latih, data_uji

# Fungsi untuk menghitung probabilitas prior untuk setiap kelas
def hitung_probabilitas_prior(data_latih):
    jumlah_kelas = data_latih['Jenis Rumput'].value_counts()
    total_latih = len(data_latih)
    return jumlah_kelas / total_latih

# Fungsi untuk menghitung probabilitas kondisional untuk setiap atribut
def hitung_probabilitas_kondisional(data_latih):
    jumlah_kelas = data_latih['Jenis Rumput'].value_counts()
    probabilitas_kondisional = {}
    for kolom in data_latih.columns[:-1]:  # Hindari kolom 'Jenis Rumput'
        probabilitas_kondisional[kolom] = (
            data_latih.groupby(['Jenis Rumput', kolom]).size()
            .div(jumlah_kelas, level=0)
            .unstack(fill_value=1e-6)
        )
    return probabilitas_kondisional

# Fungsi untuk menghitung likelihood setiap kelas
def hitung_likelihood(input_pengguna, probabilitas_prior, probabilitas_kondisional):
    likelihoods = {}

    for kelas_tujuan in probabilitas_prior.index:
        likelihood = probabilitas_prior[kelas_tujuan]

        for kolom, nilai in input_pengguna.items():
            probabilitas = probabilitas_kondisional[kolom].get(nilai, {}).get(kelas_tujuan, 1e-6)
            likelihood *= probabilitas

        likelihoods[kelas_tujuan] = likelihood

    return likelihoods

# Fungsi untuk menghitung likelihood relatif
def hitung_likelihood_relatif(likelihoods):
    likelihood_relatif = {}
    total_likelihood = sum(likelihoods.values())

    for kelas, likelihood in likelihoods.items():
        likelihood_lain = total_likelihood - likelihood
        likelihood_relatif[kelas] = likelihood / likelihood_lain if likelihood_lain > 0 else 0

    return likelihood_relatif

# Fungsi untuk membuat prediksi berdasarkan input pengguna
def prediksi_input(input_pengguna, probabilitas_prior, probabilitas_kondisional):
    probabilitas_posterior = {}
    probabilitas_atribut = {}

    for kelas_tujuan in probabilitas_prior.index:
        posterior = np.log(probabilitas_prior[kelas_tujuan])
        probabilitas_atribut[kelas_tujuan] = {}

        for kolom, nilai in input_pengguna.items():
            probabilitas = probabilitas_kondisional[kolom].get(nilai, {}).get(kelas_tujuan, 1e-6)
            posterior += np.log(probabilitas)
            probabilitas_atribut[kelas_tujuan][kolom] = probabilitas

        probabilitas_posterior[kelas_tujuan] = np.exp(posterior)

    return probabilitas_posterior, probabilitas_atribut

# Fungsi untuk mengevaluasi akurasi model
def hitung_akurasi(data_uji, probabilitas_prior, probabilitas_kondisional):
    prediksi_benar = 0
    for _, baris in data_uji.iterrows():
        input_pengguna = baris[:-1].to_dict()
        probabilitas_posterior, _ = prediksi_input(input_pengguna, probabilitas_prior, probabilitas_kondisional)
        kelas_prediksi = max(probabilitas_posterior, key=probabilitas_posterior.get)
        if kelas_prediksi == baris['Jenis Rumput']:
            prediksi_benar += 1
    return prediksi_benar / len(data_uji)

# Fungsi utama untuk menjalankan program
def main(jalur_file):
    # Memuat dan membagi data
    df = muat_data(jalur_file)

    # Memastikan kolom sesuai
    kolom_utama = ['Suhu Udara', 'Curah Hujan', 'Kelembapan Udara', 'Harga Pasaran', 'Jenis Rumput']
    df = df[kolom_utama]
    df = df.apply(lambda x: x.str.strip().str.lower() if x.dtype == "object" else x)

    data_latih, data_uji = bagi_data(df)

    # Menghitung probabilitas prior dan kondisional
    probabilitas_prior = hitung_probabilitas_prior(data_latih)
    probabilitas_kondisional = hitung_probabilitas_kondisional(data_latih)

    # Menghitung akurasi model
    akurasi = hitung_akurasi(data_uji, probabilitas_prior, probabilitas_kondisional)
    print(f"Akurasi Model: {akurasi:.2%}")

    # Input manual untuk prediksi
    print("\n=== Masukkan Data untuk Prediksi ===")
    input_pengguna = {}
    for kolom in data_latih.columns[:-1]:
        opsi = sorted(data_latih[kolom].unique())
        print(f"\nPilihan untuk {kolom}:")
        for i, nilai in enumerate(opsi):
            print(f"  {chr(65 + i)}. {nilai}")
        while True:
            pilihan = input(f"Pilih opsi untuk {kolom} (A-{chr(64 + len(opsi))}): ").upper()
            if pilihan in [chr(65 + i) for i in range(len(opsi))]:
                input_pengguna[kolom] = opsi[ord(pilihan) - 65]
                break
            else:
                print("Pilihan tidak valid. Coba lagi.")

    # Menghitung likelihood
    likelihoods = hitung_likelihood(input_pengguna, probabilitas_prior, probabilitas_kondisional)

    # Menghitung likelihood relatif
    likelihood_relatif = hitung_likelihood_relatif(likelihoods)

    # Membuat prediksi
    probabilitas_posterior, probabilitas_atribut = prediksi_input(input_pengguna, probabilitas_prior, probabilitas_kondisional)
    kelas_prediksi = max(probabilitas_posterior, key=probabilitas_posterior.get)

    # Menampilkan hasil
    print("\n=== Ranking Likelihood ===")
    likelihood_terurut = sorted(likelihoods.items(), key=lambda x: x[1], reverse=True)
    for urutan, (kelas, likelihood) in enumerate(likelihood_terurut, start=1):
        print(f"{urutan}. {kelas}: {likelihood:.6f}")

    print("\n=== Likelihood Relatif ===")
    likelihood_relatif_terurut = sorted(likelihood_relatif.items(), key=lambda x: x[1], reverse=True)
    for urutan, (kelas, rel_likelihood) in enumerate(likelihood_relatif_terurut, start=1):
        print(f"{urutan}. {kelas}: {rel_likelihood:.6f}")

    print("\n=== Probabilitas untuk Setiap Atribut ===")
    for kelas_tujuan, atribut in probabilitas_atribut.items():
        print(f"\nKelas: {kelas_tujuan}")
        for atribut, probabilitas in atribut.items():
            print(f"  {atribut}: {probabilitas:.6f}")

    print("\n=== Probabilitas Kelas ===")
    for kelas, probabilitas in probabilitas_posterior.items():
        print(f"{kelas}: {probabilitas:.6f}")

    print(f"\nKelas yang Diprediksi: {kelas_prediksi}")

# Jalankan program
jalur_file = 'dataset_62.xlsx'  # Sesuaikan lokasi file
main(jalur_file)
