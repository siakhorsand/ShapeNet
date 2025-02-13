import base64
import io
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import tensorflow as tf

app = Flask(__name__, static_folder='static')

# Match the order used in training
CLASS_NAMES = ['circle', 'square', 'triangle', 'hexagon', 'octagon']

# Load the model with custom compile to ensure all metrics are properly loaded
model = tf.keras.models.load_model("shapes_model.keras", compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('image', None)
        if not data:
            return jsonify({"error": "No image data provided"}), 400
        
        if "," in data:
            data = data.split(",")[1]
        image_data = base64.b64decode(data)
        
        # Load and preprocess image
        img = Image.open(io.BytesIO(image_data)).convert('L')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_arr = np.array(img).astype('float32') / 255.0
        
        # Invert colors and ensure proper normalization
        img_arr = 1.0 - img_arr
        
        # Reshape and add channel dimension
        img_arr = img_arr.reshape(1, 28, 28, 1)

        # Make prediction with higher confidence threshold
        preds = model.predict(img_arr, verbose=0)
        
        # Get top 2 predictions for confidence comparison
        top_2_idx = np.argsort(preds[0])[-2:][::-1]
        
        class_idx = top_2_idx[0]
        confidence = float(preds[0][class_idx])
        predicted_class = CLASS_NAMES[class_idx]
        
        # Get all probabilities for display
        probabilities = {name: float(prob) for name, prob in zip(CLASS_NAMES, preds[0])}
        
        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        })
    
    except Exception as e:
        print("Error making prediction:", str(e))
        return jsonify({"error": "Error making prediction: " + str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)