import torch
import os
import numpy as np
import io
import cv2
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

class FireDetectionHandler(BaseHandler):
    def initialize(self, context):
        """ 🔥 모델 초기화 """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = context.system_properties.get("model_dir")
        model_path = os.path.join(model_dir, "fire.pt")

        # YOLOv8 모델 로드
        self.model = YOLO(model_path).to(self.device)
        self.model.eval()

        self.confidence_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.2))

        print(f"✅ Fire Detection Model (YOLOv8) Loaded Successfully with threshold {self.confidence_threshold}")

    def preprocess_single_video(self, video_idx, video_bytes):
        """ 🔥 개별 영상 전처리 """
        try:
            video_path = f"/tmp/fire_video_{video_idx}.mp4"
            with open(video_path, "wb") as f:
                f.write(video_bytes)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"❌ VideoCapture failed to open video file {video_idx}")

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
                raise ValueError(f"❌ No valid frames found in video {video_idx}")

            print(f"📌 Video {video_idx} - Total Frames Extracted: {len(frames)}")
            return {"frames": frames}

        except Exception as e:
            print(f"🔥 Preprocessing error in Video {video_idx}: {e}")
            return None

    def preprocess(self, data, num_workers=2):
        """ 🔥 다중 영상 병렬 전처리 """
        videos = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.preprocess_single_video, idx, video_data.get("body", None))
                       for idx, video_data in enumerate(data)]
            
            for future in futures:
                result = future.result()
                if result:
                    videos.append(result)
                else:
                    videos.append({"frames": []})  # 실패한 경우 빈 리스트 반환

        return {"videos": videos}

    def inference_single_video(self, video_idx, frames):
        """ 🔥 개별 영상 추론 """
        if len(frames) == 0:
            return []

        with torch.no_grad():
            results = [self.model(frame.unsqueeze(0), verbose=False)[0] for frame in frames]
        return results

    def inference(self, data, num_workers=2):
        """ 🔥 다중 영상 병렬 추론 """
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.inference_single_video, idx, video_data["frames"])
                       for idx, video_data in enumerate(data["videos"])]

            for future in futures:
                results.append(future.result())

        return results

    def postprocess_single_video(self, video_idx, detections):
        """ 🔥 개별 영상 후처리 """
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

    def postprocess(self, detections, num_workers=2):
        """ 🔥 다중 영상 병렬 후처리 """
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.postprocess_single_video, idx, detections[idx])
                       for idx in range(len(detections))]

            for future in futures:
                results.append(future.result())

        return results

    def handle(self, data, context):
        """ 📌 TorchServe 배치 처리 대응 """
        
        batch_size = len(data)  # ✅ 입력 배치 크기 확인
        preprocessed_data = [self.preprocess([d]) for d in data]  # ✅ 개별 전처리

        # 🔹 전처리 실패한 경우 빈 딕셔너리 유지
        preprocessed_data = [p if p else {} for p in preprocessed_data]

        detections = [self.inference(p) for p in preprocessed_data]
        responses = [self.postprocess(d) for d in detections]

        # ✅ 항상 입력 개수와 동일한 리스트 반환
        return responses if len(responses) == batch_size else [[]] * batch_size




