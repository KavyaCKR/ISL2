"""
Flask API for ISL Gesture Recognition
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Configuration
IMG_SIZE = 224
MODEL_PATH = 'isl_model.h5'
CLASSES_PATH = 'class_names.json'
CONFIDENCE_THRESHOLD = 0.65

# Load model and classes
print("[Loading model...]")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✓ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

try:
    with open(CLASSES_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"✓ Classes loaded: {len(class_names)} gestures")
except Exception as e:
    print(f"✗ Error loading classes: {e}")
    class_names = []


def preprocess_image(img):
    """Preprocess image for MobileNetV2"""
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def get_top_predictions(predictions, k=5):
    """Get top-k predictions with confidence scores"""
    top_indices = np.argsort(predictions[0])[-k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'class': class_names[idx] if idx < len(class_names) else f"Unknown_{idx}",
            'confidence': float(predictions[0][idx])
        })
    return results


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model': 'ISL Gesture Recognition',
        'classes': len(class_names)
    }), 200


@app.route('/info', methods=['GET'])
def info():
    """Model information endpoint"""
    return jsonify({
        'model': 'ISL Gesture Recognition',
        'architecture': 'MobileNetV2',
        'classes': len(class_names),
        'input_size': IMG_SIZE,
        'confidence_threshold': CONFIDENCE_THRESHOLD
    }), 200


@app.route('/classes', methods=['GET'])
def get_classes():
    """List all gesture classes"""
    return jsonify({
        'classes': class_names,
        'total': len(class_names)
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Predict gesture from image"""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get image from request
        if 'image' in request.files:
            img_file = request.files['image']
            img = Image.open(img_file.stream).convert('RGB')
        elif 'image_base64' in request.json:
            img_base64 = request.json['image_base64']
            img_data = base64.b64decode(img_base64)
            img = Image.open(BytesIO(img_data)).convert('RGB')
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess
        x = preprocess_image(img)
        
        # Predict
        predictions = model.predict(x, verbose=0)
        class_id = np.argmax(predictions[0])
        confidence = float(predictions[0][class_id])
        
        # Determine prediction
        if confidence >= CONFIDENCE_THRESHOLD:
            predicted_class = class_names[class_id] if class_id < len(class_names) else "Unknown"
        else:
            predicted_class = "Uncertain"
        
        return jsonify({
            'gesture': predicted_class,
            'confidence': confidence,
            'top_5': get_top_predictions(predictions, k=5)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=False, host='localhost', port=5000)
