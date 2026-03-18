from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
import requests
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
IMG_SIZE = 128
MODEL_PATH = "bone_fracture_cnn_model.h5"
MODEL_URL = "https://huggingface.co/Sricharan08/bone_fracture_detection/resolve/main/bone_fracture_cnn_model.h5"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Download model if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from HuggingFace...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Model downloaded successfully!")

# Initialize model on startup
download_model()
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_bytes):
    # Convert bytes to numpy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    # Decode image
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to match model's expected input
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    # Normalize pixel values
    image = image / 255.0
    # Reshape for model input (1, 128, 128, 1)
    image = image.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return image

# Route for the homepage (serves the HTML UI)
@app.route('/')
def home():
    return render_template('index.html')

# API Route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    # 1. Check if file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    # 2. Check if user actually selected a file
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # 3. Validate file type and process
    if file and allowed_file(file.filename):
        try:
            # Read image bytes directly from memory
            image_bytes = file.read()
            
            # Preprocess the image
            processed_image = preprocess_image(image_bytes)
            
            # Make prediction
            prediction = model.predict(processed_image)[0][0]

            # Format the result based on your model's threshold
            if prediction > 0.5:
                result_text = "FRACTURED BONE DETECTED"
                status = "fractured"
            else:
                result_text = "NORMAL BONE"
                status = "normal"

            # Return JSON response to frontend
            return jsonify({
                'prediction': float(prediction),
                'result': result_text,
                'status': status
            })

        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload JPG or PNG.'}), 400

if __name__ == '__main__':
    # Run the Flask development server
    app.run(debug=True, port=5000)
