import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import uuid

app = Flask(__name__)

# =============================================
# LOAD MODEL & LABEL
# =============================================
MODEL_PATH = 'model/weather_model.h5'
LABEL_PATH = 'model/class_labels.json'
UPLOAD_FOLDER = 'static/uploads'
IMG_SIZE = (128, 128)

print("⏳ Loading model...")
model = load_model(MODEL_PATH)

with open(LABEL_PATH, 'r') as f:
    class_labels = json.load(f)

# Label dalam Bahasa Indonesia
label_indonesia = {
    'dew': 'Embun',
    'fogsmog': 'Kabut/Polusi',
    'frost': 'Frost (Beku)',
    'glaze': 'Glaze (Es Tipis)',
    'hail': 'Hujan Es',
    'lightning': 'Petir',
    'rain': 'Hujan',
    'rainbow': 'Pelangi',
    'rime': 'Rime (Es Kristal)',
    'sandstorm': 'Badai Pasir',
    'snow': 'Salju'
}

label_emoji = {
    'dew': '💧',
    'fogsmog': '🌫️',
    'frost': '🧊',
    'glaze': '🌨️',
    'hail': '🌧️',
    'lightning': '⚡',
    'rain': '🌧️',
    'rainbow': '🌈',
    'rime': '❄️',
    'sandstorm': '🌪️',
    'snow': '❄️'
}

print("✅ Model loaded!")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(img_path):
    img = Image.open(img_path).resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_idx = str(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100
    predicted_label = class_labels[predicted_idx]
    
    # Top 3 prediksi
    top3_idx = np.argsort(predictions[0])[::-1][:3]
    top3 = []
    for idx in top3_idx:
        lbl = class_labels[str(idx)]
        top3.append({
            'label': lbl,
            'label_id': label_indonesia.get(lbl, lbl),
            'emoji': label_emoji.get(lbl, '🌤️'),
            'confidence': round(float(predictions[0][idx]) * 100, 2)
        })
    
    return {
        'label': predicted_label,
        'label_id': label_indonesia.get(predicted_label, predicted_label),
        'emoji': label_emoji.get(predicted_label, '🌤️'),
        'confidence': round(confidence, 2),
        'top3': top3
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diupload'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file dipilih'}), 400
    
    # Simpan file
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Prediksi
    result = predict_image(filepath)
    result['image_url'] = f"/static/uploads/{filename}"
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)