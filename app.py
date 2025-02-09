from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import json
from tensorflow import keras
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model dan label kelas
model = keras.models.load_model("pepaya_ripeness_model.h5")

with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# Konversi indeks ke nama kelas
index_to_class = {v: k for k, v in class_labels.items()}

# Preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        file.save(file_path)
        img = preprocess_image(file_path)
        prediction = model.predict(img)[0]
        predicted_class = index_to_class[np.argmax(prediction)]
        confidence = round(np.max(prediction) * 100, 2)

        # Bandingkan prediksi dengan label asli berdasarkan nama folder dataset
        actual_class = os.path.basename(os.path.dirname(file_path))
        is_correct = "Benar" if predicted_class == actual_class else "Salah"

        return render_template('index.html', filename=filename, result=predicted_class, accuracy=confidence, correctness=is_correct, actual_class=actual_class)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
