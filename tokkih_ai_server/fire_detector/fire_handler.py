import torch
import os
import numpy as np
import io
import cv2
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from ultralytics import YOLO

class FireDetectionHandler(BaseHandler):
    def initialize(self, context):
        """ 🔥 모델 초기화 """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = context.system_properties.get("model_dir")
        model_path = os.path.join(model_dir, "fire.pt")

        # YOLOv8 모델 로드
        self.model = YOLO(model_path).to(self.device)
        self.model.eval()

        self.confidence_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.4))

        print(f"✅ Fire Detection Model (YOLOv8) Loaded Successfully with threshold {self.confidence_threshold}")

    def preprocess(self, data):
        """ 🔥 영상 전처리 (동영상 → 프레임) """
        try:
            video_bytes = data[0].get("body", None)
            if video_bytes is None:
                raise ValueError("❌ Received empty video data")

            # OpenCV에서 읽을 수 있도록 임시 파일로 저장
            video_path = "/tmp/fire_video.mp4"
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

                # 🚀 YOLOv8 입력 사이즈로 리사이징 (32의 배수)
                target_size = (320, 256)  # YOLO 모델의 최적 입력 크기
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
        """ 🔥 YOLOv8 기반 불/연기 탐지 """
        with torch.no_grad():
            results = [self.model(frame.unsqueeze(0), verbose=False)[0] for frame in data["frames"]]
        return results

    def postprocess(self, detections):
        """ 🔥 탐지된 결과 후처리 (프레임별 탐지 정보) """
        predictions = []
        
        for result in detections:
            boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes, 'xyxy') else []
            scores = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else []
            classes = result.boxes.cls.cpu().numpy() if hasattr(result.boxes, 'cls') else []

            frame_predictions = []
            for i in range(len(boxes)):
                x_min, y_min, x_max, y_max = boxes[i]
                
                confidence = float(scores[i])

                # ✅ Confidence Threshold 적용 (기본값: 0.5)
                if confidence < self.confidence_threshold:
                    continue


                class_mapping = {0: 100, 1: 101}  # 0: 불 → 100, 1: 연기 → 101
                adjusted_class_id = class_mapping.get(int(classes[i]), 199)

                frame_predictions.append({
                    "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
                    "confidence": float(scores[i]),
                    "class_id": adjusted_class_id
                })

            # 🔥 탐지된 객체가 있는 경우만 predictions에 추가
            if frame_predictions:
                predictions.append(frame_predictions)

        return predictions

    def handle(self, data, context):
        """ 🔥 요청 처리 (전처리 → 추론 → 후처리) """
        preprocessed_data = self.preprocess(data)
        if preprocessed_data is None:
            return [{"results": []}]  # ✅ 빈 응답도 리스트로 감싸서 반환

        detections = self.inference(preprocessed_data)
        response = self.postprocess(detections)

        # 🔥 TorchServe 호환 응답 반환
        return [{"results": response}]  # ✅ 반드시 리스트로 감싸서 반환
