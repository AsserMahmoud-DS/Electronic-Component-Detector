from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load your model
model = load_model("Electronic-Component-Detector2.keras")

class_names = {
    0: 'arduino',
    1: 'battery',
    2: 'DCmotor',
    3: 'DHT-11',
    4: 'ESP8266',
    5: 'LCD',
    6: 'Loadcell',
    7: 'RFID',
    8: 'Tiva',
    9: 'Ultrasonic',
}

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize((224, 224))
    data = np.asarray(img) / 255.0
    return np.expand_dims(data, axis=0)

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image'].read()
    processed_image = preprocess_image(image)
    probs = model.predict(processed_image)
    top_prob = probs.max()
    top_pred = class_names[np.argmax(probs)]
    return jsonify({'prediction': f'this is a {top_pred}', 'probability': float(top_prob)})

# Export the app as a module-level variable named "app"
app = app