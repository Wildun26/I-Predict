from tensorflow.keras.models import load_model
import numpy as np

# Fungsi untuk memuat model
def load_my_model():
    global model
    model = load_model('my_model.h5')  # Ganti dengan nama model yang benar

# Fungsi untuk melakukan prediksi
def predict(input_data):
    input_data = np.array(input_data).reshape(1, -1)  # Sesuaikan dengan format input yang diperlukan
    prediction = model.predict(input_data)
    return prediction.tolist()

# Load model saat aplikasi dimulai
load_my_model()

# Contoh penggunaan prediksi
input_data = [0.1, 0.2, 0.3]  # Ganti dengan input yang sesuai
result = predict(input_data)
print(f'Prediction result: {result}')
