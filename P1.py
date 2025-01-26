# app.py

import base64
import io
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Match the order used in training
CLASS_NAMES = ['triangle', 'circle', 'square', 'hexagon', 'octagon']

# Load the model
model = tf.keras.models.load_model("shape_model.h5")

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('image', None)
    if not data:
        return jsonify({"error": "No image data provided"}), 400
    
    # Typically "data:image/png;base64,..." 
    if "," in data:
        data = data.split(",")[1]
    image_data = base64.b64decode(data)
    
    # Convert to grayscale
    img = Image.open(io.BytesIO(image_data)).convert('L')
    # Resize to 28x28 if that’s what your model expects
    img = img.resize((28, 28))
    
    img_arr = np.array(img).astype('float32') / 255.0
    # Expand dims => (1, 28, 28, 1)
    img_arr = np.expand_dims(img_arr, axis=(0, -1))
    
    preds = model.predict(img_arr)
    class_idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))
    predicted_class = CLASS_NAMES[class_idx]
    
    return jsonify({
        "prediction": predicted_class,
        "confidence": confidence
    })

if __name__ == '__main__':
    # Start Flask
    app.run(debug=True)