import os
import json

import torch
from flask import Flask, render_template, request, jsonify
from torchvision import transforms
from PIL import Image
import numpy as np
from model import InceptionSS
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load class_indict
json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'class_indices.json'))

assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
with open(json_path, "r") as f:
    class_indict = json.load(f)

# Create model
model = InceptionSS(num_classes=4).to(device)

# Load model weights
weights_path = "./model.pth"
assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def homepage():
    return render_template('home_page.html')

@app.route('/learnMore')
def learnMore():
    return render_template('learnMore.html')

@app.route('/detect_page', methods=['GET', 'POST'])
def detect_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('detect_page.html', message='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('detect_page.html', message='No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Load image
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = Image.open(img_path)
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)

            # Predict class
            with torch.no_grad():
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            result = "Class: {}   Probability: {:.3}".format(class_indict[str(predict_cla)],
                                                             predict[predict_cla].numpy())

            return render_template('detect_page.html', message='File uploaded successfully!', prediction=result)

    return render_template('detect_page.html')


# 新增路由用于接收预测请求
@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request.")
    if 'imageFile' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['imageFile']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file to a specific folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Perform prediction using the saved file path
    img = Image.open(file_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # Predict class
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    result = "Class: {}   Probability: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())

    # Return prediction result as JSON response
    return jsonify({'prediction': result}), 200

if __name__ == '__main__':
    app.run(debug=True)
