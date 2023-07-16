import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from cvlib.object_detection import draw_bbox
import cv2
import cvlib as cv

app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_people(image_path):
    image = cv2.imread(image_path)
    bbox, label, conf = cv.detect_common_objects(image)
    output_image = draw_bbox(image, bbox, label, conf)
    cv2.imwrite(image_path, output_image)
    return label.count('person')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
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

if __name__ == "__main__":
    app.run(port=5000, debug=True)
