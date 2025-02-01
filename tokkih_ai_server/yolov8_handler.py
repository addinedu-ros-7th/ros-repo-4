import torch
import os
import numpy as np
import io
from PIL import Image
import cv2
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
        """ 이미지 전처리 """
        try:
            image_bytes = data[0].get("body", None)
            if image_bytes is None:
                raise ValueError("Received empty image data")

            # 원본 이미지 로드 및 크기 저장
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            original_size = image.size  # (W, H) 저장
            image = np.array(image)

            # YOLOv8은 (1, 3, H, W) 형식의 Tensor를 사용해야 함
            image = cv2.resize(image, (640, 640))  # ✅ YOLOv8에 맞게 리사이징
            image = torch.from_numpy(image).permute(2, 0, 1).float().to(self.device) / 255.0
            image = image.unsqueeze(0)  # (H, W, C) → (1, C, H, W)

            return {"image": image, "original_size": original_size}
        except Exception as e:
            print(f"🔥 Preprocessing error: {e}")
            return None

    def inference(self, data):
        """ YOLOv8 모델 추론 """
        with torch.no_grad():
            results = self.model(data["image"], verbose=False)
        return {"results": results, "original_size": data["original_size"]}

    def postprocess(self, data):
        """ YOLOv8 결과 후처리 (여러 객체 탐지 + 원본 크기로 변환) """
        predictions = []
        results = data["results"]
        original_size = data["original_size"]  # 원본 이미지 크기 (W, H)

        # 원본 이미지 크기로 바운딩 박스 좌표를 변환하는 스케일 계산
        scale_x = original_size[0] / 640  # W / 640
        scale_y = original_size[1] / 640  # H / 640

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표 (640x640 기준)
            scores = result.boxes.conf.cpu().numpy()  # 신뢰도 점수
            classes = result.boxes.cls.cpu().numpy()  # 클래스 ID

            batch_predictions = []
            for i in range(len(boxes)):
                x_min, y_min, x_max, y_max = boxes[i]

                # ✅ 원본 이미지 크기로 바운딩 박스 좌표 변환
                x_min = float(x_min * scale_x)
                x_max = float(x_max * scale_x)
                y_min = float(y_min * scale_y)
                y_max = float(y_max * scale_y)

                batch_predictions.append({
                    "bbox": [x_min, y_min, x_max, y_max],  # 변환된 좌표 반환
                    "confidence": float(scores[i]),  # ✅ float 변환 추가
                    "class_id": int(classes[i])  # ✅ int 변환 추가
                })

            predictions.append(batch_predictions)  # 여러 객체 지원

        print(f"📌 Total Predictions: {len(predictions)}")  # 디버깅용
        return predictions
