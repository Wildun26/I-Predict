from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load model dan scaler
model = load_model('my_model.h5')
scaler = joblib.load('scaler.pkl')

# Fungsi untuk membuat data time series
def create_time_series(data, time_steps):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
    return np.array(X)

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    data = json_data['data']

    # Preprocessing data
    data = np.array(data).reshape(-1, 1)
    data_scaled = scaler.transform(data)
    time_steps = 30
    X = create_time_series(data_scaled, time_steps)

    # Predict
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
