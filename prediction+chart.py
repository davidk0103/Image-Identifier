from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)
CORS(app)

model_path = 'C:/Users/david/Desktop/IC/image-classifier/image-classifier-app/model.h5'
model = load_model(model_path)
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['file']
    img_path = 'uploads/' + img_file.filename
    img_file.save(img_path)

    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    prediction = class_names[np.argmax(predictions[0])]
    confidence_scores = {class_names[i]: float(predictions[0][i]) for i in range(10)}

    return jsonify({
        'prediction': prediction,
        'confidence_scores': confidence_scores
    })

if __name__ == '__main__':
    app.run(debug=True)
