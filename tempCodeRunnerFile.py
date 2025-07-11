from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# Fungsi untuk memuat data dari file EXCEL
def load_data(file_path):
    data = pd.ExcelFile(file_path)
    df = data.parse(data.sheet_names[0])
    return df

# Fungsi untuk membagi data menjadi data latih dan data uji
def split_data(df, train_ratio=0.83):
    train_size = int(len(df) * train_ratio)
    train_data = df[:train_size]
    test_data = df[train_size:]
    return train_data, test_data

# Fungsi untuk menghitung probabilitas prior untuk setiap kelas
def calculate_prior_probabilities(train_data):
    class_counts = train_data['Jenis Rumput'].value_counts()
    total_train = len(train_data)
    return class_counts / total_train

# Fungsi untuk menghitung probabilitas kondisional untuk setiap atribut
def calculate_conditional_probabilities(train_data):
    class_counts = train_data['Jenis Rumput'].value_counts()
    conditional_probs = {}
    for column in train_data.columns[:-1]:  # Hindari kolom 'Jenis Rumput'
        conditional_probs[column] = (
            train_data.groupby(['Jenis Rumput', column]).size()
            .div(class_counts, level=0)
            .unstack(fill_value=1e-6)
        )
    return conditional_probs

# Fungsi untuk menghitung likelihood setiap kelas
def calculate_likelihood(user_input, prior_probs, conditional_probs):
    likelihoods = {}
    for target_class in prior_probs.index:
        likelihood = prior_probs[target_class]
        for column, value in user_input.items():
            prob = conditional_probs[column].get(value, {}).get(target_class, 1e-6)  # Probabilitas kondisional
            likelihood *= prob  # Kalkulasi likelihood
        likelihoods[target_class] = likelihood
    return likelihoods

# Fungsi utama Flask
@app.route('/', methods=['GET', 'POST'])
def index():
    file_path = 'dataset_62.xlsx'
    df = load_data(file_path)

    # Pastikan kolom sesuai
    relevant_columns = ['Suhu Udara', 'Curah Hujan', 'Kelembapan Udara', 'Harga Pasaran', 'Jenis Rumput']
    df = df[relevant_columns]
    df = df.apply(lambda x: x.str.strip().str.lower() if x.dtype == "object" else x)  # Standarisasi teks

    train_data, _ = split_data(df)
    prior_probs = calculate_prior_probabilities(train_data)
    conditional_probs = calculate_conditional_probabilities(train_data)

    if request.method == 'POST':
        user_input = {
            'Suhu Udara': request.form.get('suhu_udara').lower(),
            'Curah Hujan': request.form.get('curah_hujan').lower(),
            'Kelembapan Udara': request.form.get('kelembapan_udara').lower(),
            'Harga Pasaran': request.form.get('harga_pasaran').lower()
        }

        likelihoods = calculate_likelihood(user_input, prior_probs, conditional_probs)
        predicted_class = max(likelihoods, key=likelihoods.get)

        # Konversi kode ke nama
        class_mapping = {
            "r1": "Gajah Mini",
            "r2": "Gajah Mini Variegata",
            "r3": "Rumput Paeking",
            "r4": "Rumput Paitan",
            "r5": "Rumput Jepang",
            "r6": "Rumput Swiss",
            "r7": "Rumput Golf"
        }
        predicted_name = class_mapping.get(predicted_class, "Unknown")

        return render_template('result.html', prediction=predicted_name, likelihoods=likelihoods)

    # Mengambil opsi untuk input dari data
    options = {
        column: sorted(train_data[column].unique())
        for column in train_data.columns[:-1]
    }
    return render_template('index.html', options=options)

if __name__ == '__main__':
    app.run(debug=True)
