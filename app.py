# -*- coding: utf-8 -*-
"""キカガク_アプリケーション開発-YOLO5(安中嘉彦).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ydALoHO1kN-TRj4zvZfogFFBNE6-Tesm
"""

import torch
from flask import Flask
app = Flask(__name__)
import subprocess
import os

# YOLOv5のリポジトリをクローン
subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])

# 必要なパッケージをインストール
subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"])


def load_model(weights_path):
    # モデルのロード
    model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=weights_path)
    return model

def detect_faces(image_path, model):
    # 画像から物体検出
    results = model(image_path)
    # 検出されたオブジェクトの情報を取得
    detections = results.pandas().xyxy[0]

    # 人数をカウントするための変数
    face_count = 0

    # 各オブジェクトについて
    for _, row in detections.iterrows():
        # オブジェクトのクラスが「person」であるかどうかをチェック
        if row['name'] == 'person':
            face_count += 1

    return face_count

# Commented out IPython magic to ensure Python compatibility.
# 1. YOLOv5のリポジトリをクローン
#git clone https://github.com/ultralytics/yolov5.git

# 2. 必要なパッケージをインストール
#!pip install -r yolov5/requirements.txt
os.system('pip install -r yolov5/requirements.txt')

# 3. データセットの準備
# 自分で作成したデータセットをアップロードし、適切なフォルダに配置します。
# データセットは画像とアノテーションがペアになったフォーマットである必要があります。

# 4. YOLOv5の訓練
# %cd /content/yolov5/
#!python train.py --img 640 --batch 16 --epochs 100 --data dataset.yaml --weights yolov5s.pt --name my_model
os.system('python train.py --img 640 --batch 16 --epochs 100 --data dataset.yaml --weights yolov5s.pt --name my_model')

# 学習済みの重みのパスを指定
weights_path = 'best.pt'

# モデルのロード
#!pip install --upgrade torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 画像のパスを指定
#image_path = '/content/Inside_of_the_Bus-1.jpg'
#image_path = 'https://github.com/annakay/flask/blob/main/static/uploads/Inside_of_the_Bus-1.jpg'
image_path = 'https://github.com/annakay/flask/blob/main/Inside_of_the_Bus-1.jpg'

face_count = detect_faces(image_path, model)

print(f"人数: {face_count}")

