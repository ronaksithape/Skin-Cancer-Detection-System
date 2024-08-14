from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model("skin_model.keras")

# Define the mapping of class indices to cancer types
class_mapping = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc"
}

def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_skin_cancer(image_path):
    try:
        processed_image = preprocess_image(image_path)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        cancer_type = class_mapping[predicted_class]
        return {"result": "Skin cancer detected", "cancer_type": cancer_type}
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    try:
        img_path = 'uploaded_image.jpg'
        file.save(img_path)
        result = predict_skin_cancer(img_path)
        os.remove(img_path)  # Remove the uploaded image after prediction
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
