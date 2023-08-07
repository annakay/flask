import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import cv2
import cvlib as cv
import numpy as np
from werkzeug.utils import secure_filename
import urllib.request

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
    
    # Load YOLO v4
    net = cv2.dnn.readNet("https://drive.google.com/file/d/1xlDlBk79H6psU9Uto6kpdJ7kwJAMrazS/view?usp=drive_link", "yolov4.cfg")

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), [0,0,0], 1, crop=False)

    # Sets the blob as the input of the network
    net.setInput(blob)

    # Get the names of the output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Perform a forward pass through the network
    outs = net.forward(output_layers)

    # Initialization
    class_ids = []
    confidences = []
    boxes = []
    height, width = image.shape[:2]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Object detected
            if confidence > 0.5:
                # Center coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    bbox, label, conf = [], [], []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label.append(str(class_ids[i])) # add class label
            conf.append(confidences[i]) # add confidence score
            bbox.append([x, y, x+w, y+h]) # add bounding box coordinates

    output_image = draw_bbox_cv2(image, bbox, label, conf)  # 処理後の画像を受け取る
    print(f"image_path: {image_path}") # 保存先のパスを出力
    print(f"output_image type: {type(output_image)}") # output_imageの型を出力
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
    weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
    weights_filepath = "./yolov4.weights"

    if not os.path.exists(weights_filepath):
        urllib.request.urlretrieve(weights_url, weights_filepath)
    app.run(host='0.0.0.0', debug=True , port=int(os.getenv('PORT', 10000)))
