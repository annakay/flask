import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static/uploads/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_people(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Failed to load image at {image_path}")

    net = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg', 'yolov4-tiny.weights')
    
    layer_names = net.getLayerNames()

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward([layer_names[65], layer_names[77]])

    people_count = 0
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.35 and class_id == 0:  # class_id 0 is for 'person'
                people_count += 1

    return people_count

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        people_count = detect_people(file_path)
        result = {
            "filename": f"{filename}", 
            "message": f"{people_count}人"
        }
        return render_template('result.html', **result)
    return redirect(request.url)

@app.errorhandler(Exception)
def handle_exception(e):
    return str(e), 10000

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=int(os.getenv('PORT', 10000)))
