import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import cv2
import cvlib as cv
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static/uploads/')
#UPLOAD_FOLDER = './static/uploads/'
#UPLOAD_FOLDER = 'static/uploads/'
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
    if image is None:
        raise Exception(f"Failed to load image at {image_path}")

    # YOLOv4-tinyの設定と重みを指定
    config_path = 'https://github.com/annakay/flask/blob/main/yolov4-tiny.cfg'
    weights_path = 'https://github.com/annakay/flask/blob/main/yolov4-tiny.weights'
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # 画像の前処理
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    people_count = 0
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # class_id 0 is for 'person'
                people_count += 1
                # ここでバウンディングボックスを描画することもできます

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
        file.save(file_path)  # ←ここでファイルを保存
        people_count = detect_people(file_path)  # ←保存したファイルを物体検出に使用
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
