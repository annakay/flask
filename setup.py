import os

# YOLOv5のリポジトリをクローン
os.system('git clone https://github.com/ultralytics/yolov5.git')

# 必要なパッケージをインストール
os.system('pip install -r yolov5/requirements.txt')

# YOLOv5の訓練（各自のデータセットと設定に応じて変更する必要があります）
os.system('python yolov5/train.py --img 640 --batch 16 --epochs 100 --data dataset.yaml --weights yolov5s.pt --name my_model')

