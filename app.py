import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import cv2
import cvlib as cv
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

#UPLOAD_FOLDER = './static/uploads/'
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def draw_bbox_cv2(img, bbox, labels, confidences):
    for i, box in enumerate(bbox):
        start_x, start_y, end_x, end_y = box
        label = labels[i]
        confidence = confidences[i]

        # バウンディングボックスを描画
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        # ラベルと確信度を描画
        text = "{}: {:.4f}".format(label, confidence)
        cv2.putText(img, text, (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    return img # 画像を返す

def detect_people(image_path):
    image = cv2.imread(image_path)
    bbox, label, conf = cv.detect_common_objects(image)
    output_image = draw_bbox_cv2(image, bbox, label, conf)  # 処理後の画像を受け取る
    cv2.imwrite(image_path, output_image) # 画像を保存
    return label.count('person')

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
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
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
    app.run(host='0.0.0.0', debug=True , port=int(os.getenv('PORT', 10000)))
