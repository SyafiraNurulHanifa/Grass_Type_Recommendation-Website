from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = 'kunci_rahasia_anda'

def load_data(file_path):
    try:
        data = pd.ExcelFile(file_path)
        df = data.parse(data.sheet_names[0])
        return df
    except Exception as e:
        raise FileNotFoundError(f"Kesalahan saat memuat data: {e}")

def split_data(df, train_ratio=0.83):
    train_size = int(len(df) * train_ratio)
    train_data = df[:train_size]
    test_data = df[train_size:]
    return train_data, test_data

def calculate_prior_probabilities(train_data):
    class_counts = train_data['Jenis Rumput'].value_counts()
    total_train = len(train_data)
    return class_counts / total_train

def calculate_conditional_probabilities(train_data):
    class_counts = train_data['Jenis Rumput'].value_counts()
    conditional_probs = {}
    for column in train_data.columns[:-1]:
        conditional_probs[column] = (
            train_data.groupby(['Jenis Rumput', column]).size()
            .div(class_counts, level=0)
            .unstack(fill_value=1e-6)
        )
    return conditional_probs

def predict_single_input(user_input, prior_probs, conditional_probs):
    posterior_probs = {}
    attribute_probs = {}
    for target_class in prior_probs.index:
        posterior = np.log(prior_probs[target_class])
        attribute_probs[target_class] = {}
        for column, value in user_input.items():
            prob = conditional_probs[column].get(value, {}).get(target_class, 1e-6)
            posterior += np.log(prob)
            attribute_probs[target_class][column] = prob
        posterior_probs[target_class] = np.exp(posterior)
    return posterior_probs, attribute_probs

def calculate_accuracy(test_data, prior_probs, conditional_probs):
    correct_predictions = 0
    for _, row in test_data.iterrows():
        user_input = row[:-1].to_dict()
        posterior_probs, _ = predict_single_input(user_input, prior_probs, conditional_probs)
        predicted_class = max(posterior_probs, key=posterior_probs.get)
        if predicted_class == row['Jenis Rumput']:
            correct_predictions += 1
    return correct_predictions / len(test_data)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        file_path = 'dataset.xlsx'
        df = load_data(file_path)
    except FileNotFoundError as e:
        return f"Kesalahan: {e}", 500

    relevant_columns = ['Suhu Udara', 'Curah Hujan', 'Kelembapan Udara', 'Harga Pasaran', 'Jenis Rumput']
    df = df[relevant_columns]
    df = df.apply(lambda x: x.str.strip().str.lower() if x.dtype == "object" else x)

    train_data, test_data = split_data(df)

    prior_probs = calculate_prior_probabilities(train_data)
    conditional_probs = calculate_conditional_probabilities(train_data)
    accuracy = calculate_accuracy(test_data, prior_probs, conditional_probs)

    if request.method == 'POST':
        user_input = {column: request.form[column] for column in train_data.columns[:-1]}
        posterior_probs, attribute_probs = predict_single_input(user_input, prior_probs, conditional_probs)
        predicted_class = max(posterior_probs, key=posterior_probs.get)

        # Simpan hasil prediksi ke session
        session['user_input'] = user_input
        session['posterior_probs'] = dict(sorted(posterior_probs.items(), key=lambda x: x[1], reverse=True))
        session['predicted_class'] = predicted_class
        session['accuracy'] = accuracy
        session['attribute_probs'] = attribute_probs

        return redirect(url_for('result'))

    options = {column: sorted(train_data[column].unique()) for column in train_data.columns[:-1]}
    return render_template('index.html', options=options)

@app.route('/result')
def result():
    # Ambil data dari session
    user_input = session.get('user_input')
    posterior_probs = session.get('posterior_probs')
    predicted_class = session.get('predicted_class')
    accuracy = session.get('accuracy')

    if not (user_input and posterior_probs and predicted_class and accuracy is not None):
        return redirect(url_for('index'))

    return render_template(
        'result.html',
        user_input=user_input,
        posterior_probs=posterior_probs,
        predicted_class=predicted_class,
        accuracy=accuracy
    )

@app.route('/attribute_probs')
def attribute_probs():
    attribute_probs = session.get('attribute_probs')
    if not attribute_probs:
        return redirect(url_for('index'))

    return render_template('attribute_probs.html', attribute_probs=attribute_probs)

if __name__ == '__main__':
    app.run(debug=True)
