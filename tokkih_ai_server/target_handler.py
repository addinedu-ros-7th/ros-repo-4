import torch
import os
import numpy as np
import io
import cv2
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from ultralytics import YOLO  # YOLOv8 모델 로드

class YOLOv8Handler(BaseHandler):
    def initialize(self, context):
        """ 모델 초기화 """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = context.system_properties.get("model_dir")
        model_path = os.path.join(model_dir, "yolov8n.pt")

        # YOLOv8 모델 로드
        self.model = YOLO(model_path).to(self.device)
        self.model.eval()
        print("✅ YOLOv8 Model loaded successfully")

    def preprocess(self, data):
        """ 영상 전처리 """
        try:
            video_bytes = data[0].get("body", None)
            if video_bytes is None:
                raise ValueError("❌ Received empty video data")

            # OpenCV로 읽을 수 있도록 파일로 저장 후 다시 로드
            video_path = "/tmp/temp_video.mp4"
            with open(video_path, "wb") as f:
                f.write(video_bytes)

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise ValueError("❌ VideoCapture failed to open video file")

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 🚀 YOLOv8이 허용하는 크기로 리사이징 (Stride 32의 배수)
                target_size = (320, 256)  # 32의 배수인 크기로 조정
                frame = cv2.resize(frame, target_size)
                frame = torch.from_numpy(frame).permute(2, 0, 1).float().to(self.device) / 255.0
                frames.append(frame)

            cap.release()

            if len(frames) == 0:
                raise ValueError("❌ No valid frames found in video")

            print(f"📌 Total Frames Extracted: {len(frames)}")
            return {"frames": frames}
        
        except Exception as e:
            print(f"🔥 Preprocessing error: {e}")
            return None

    def inference(self, data):
        """ YOLOv8 모델 추론 """
        with torch.no_grad():
            results = [self.model(frame.unsqueeze(0), verbose=False)[0] for frame in data["frames"]]
        return results

    def postprocess(self, data):
        """ YOLOv8 결과 후처리 (여러 프레임 지원) """
        predictions = []
        for result in data:
            boxes = result.boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표
            scores = result.boxes.conf.cpu().numpy()  # 신뢰도 점수
            classes = result.boxes.cls.cpu().numpy()  # 클래스 ID

            frame_predictions = []
            for i in range(len(boxes)):
                x_min, y_min, x_max, y_max = boxes[i]

                frame_predictions.append({
                    "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
                    "confidence": float(scores[i]),
                    "class_id": int(classes[i])
                })

            predictions.append(frame_predictions)

        print(f"📌 Total Predictions: {len(predictions)} Frames")
        return predictions
