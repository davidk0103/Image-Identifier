import os
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from fpdf import FPDF

app = Flask(__name__)
CORS(app)

# Path to the model
model_path = 'C:\\Users\\david\\Desktop\\IC\\image-classifier\\image-classifier-app\\model_augmented.h5'
if not os.path.exists(model_path):
    print(f"Model file does not exist at path: {model_path}")
else:
    print(f"Loading model from: {model_path}")

model = load_model(model_path)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def prepare_image(img):
    img = img.resize((32, 32))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def generate_report(image_path, prediction, confidence_scores):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Prediction Report", ln=True, align="C")
    
    # Original Image
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Original Image", ln=True)
    pdf.image(image_path, x=10, y=30, w=90)
    
    # Prediction
    pdf.set_font("Arial", size=10)
    pdf.set_xy(110, 30)
    pdf.cell(200, 10, txt="Prediction: {}".format(prediction), ln=True)
    
    # Confidence Scores
    pdf.set_xy(110, 40)
    pdf.cell(200, 10, txt="Confidence Scores:", ln=True)
    for i, (class_name, score) in enumerate(confidence_scores.items()):
        pdf.set_xy(110, 50 + (i * 10))
        pdf.cell(200, 10, txt="{}: {:.2f}".format(class_name, score), ln=True)
    
    # Save PDF
    report_path = os.path.join(os.getcwd(), 'prediction_report.pdf')
    pdf.output(report_path)
    
    return report_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Prediction request received")
        file = request.files['file']
        img = Image.open(file)
        processed_image = prepare_image(img)
        
        print(f"Image processed: {processed_image.shape}")

        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence_scores = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}

        print(f"Predictions: {predictions}")
        print(f"Predicted class: {class_names[predicted_class]}")
        print(f"Confidence scores: {confidence_scores}")

        # Save the original image
        original_image_path = os.path.join(os.getcwd(), 'uploaded_image.png')
        img.save(original_image_path)

        # Generate a detailed report
        report_path = generate_report(original_image_path, class_names[predicted_class], confidence_scores)

        # Read the report and encode it in base64
        with open(report_path, "rb") as pdf_file:
            encoded_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')

        return jsonify({
            'prediction': class_names[predicted_class],
            'report': encoded_pdf
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
