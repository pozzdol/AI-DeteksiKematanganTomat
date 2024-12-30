from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Path to the saved model (Keras format)
MODEL_PATH = 'model/model_tomato.keras'

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Class labels
CLASSES = ['Mentah', 'Setengah Matang', 'Matang']

# Folder for uploaded files
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess the image
def preprocess_image(image):
    """
    Preprocess the uploaded image for prediction.
    """
    try:
        image = image.resize((224, 224))  # Resize to match model input size
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")

@app.route('/')
def index():
    """
    Renders the homepage.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles prediction requests.
    """
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No file selected')

    if not file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        return render_template('index.html', error='Invalid file format. Only PNG, JPG, and JPEG are supported.')

    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Open and preprocess the image
        image = Image.open(file).convert('RGB')  # Ensure image is in RGB format
        processed_image = preprocess_image(image)
        
        # Make predictions
        predictions = model.predict(processed_image)
        predicted_class = CLASSES[np.argmax(predictions)]
        
        # Transform the prediction result
        if predicted_class == 'Mentah':
            final_result = 'Matang'
        elif predicted_class == 'Setengah Matang':
            final_result = 'Mentah'
        else:
            final_result = predicted_class

        # Render template with result and uploaded image path
        return render_template('index.html', result=final_result, image_url=file_path)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    # Run Flask app
    app.run(debug=True)
