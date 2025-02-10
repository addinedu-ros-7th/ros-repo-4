import torch
import os
import numpy as np
import io
import cv2
from collections import defaultdict, deque
from ts.torch_handler.base_handler import BaseHandler
from ultralytics import YOLO
import torch.nn as nn
from transformer_model import TransformerPoseClassifier  # ✅ Transformer 모델 불러오기


class PoseEstimatorHandler(BaseHandler):
    def __init__(self):
        """핸들러 초기화"""
        super().__init__()
        self.device = None
        self.pose_model = None
        self.transformer_model = None
        self.class_names = ["running", "walking", "sitting", "lying"]
        self.sequence_buffers = None
        self.initialized = False

    def initialize(self, context):
        """모델 초기화"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 디렉토리 경로 확인
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        if not model_dir:
            raise ValueError("❌ Model directory not specified in context")

        # 모델 파일 경로 확인
        yolo_path = os.path.join(model_dir, "yolov8n-pose.pt")
        transformer_path = os.path.join(model_dir, "pose_estimator.pth")

        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"❌ YOLO model not found at {yolo_path}")
        if not os.path.exists(transformer_path):
            raise FileNotFoundError(f"❌ Transformer model not found at {transformer_path}")

        # ✅ YOLO 모델 로드
        try:
            self.pose_model = YOLO(yolo_path).to(self.device)
            self.pose_model.eval()
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load YOLO model: {str(e)}")

        # ✅ Transformer 모델 로드
        try:
            checkpoint = torch.load(transformer_path, map_location=self.device)
            self.transformer_model = TransformerPoseClassifier(
                input_dim=34,  # 17개 랜드마크 * 2 (x, y)
                output_dim=len(self.class_names),
                num_heads=4,
                num_layers=4,
                hidden_dim=128,
                dropout=0.5,
                max_len=96
            ).to(self.device)

            self.transformer_model.load_state_dict(checkpoint, strict=False)
            self.transformer_model.eval()
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load Transformer model: {str(e)}")

        # ✅ 시퀀스 버퍼 초기화 (최대 5명 추적)
        self.sequence_buffers = defaultdict(lambda: deque(maxlen=96))
        self.initialized = True

        print("✅ Pose Estimator Models loaded successfully")
        print(f"📌 Model directory: {model_dir}")
        print(f"📌 Device: {self.device}")

    def preprocess(self, data):
        """ 영상을 프레임 단위로 변환 (MP4 → Frame) """
        try:
            video_bytes = data[0].get("body", None)
            if video_bytes is None:
                raise ValueError("❌ Received empty video data")

            # MP4 파일 저장 후 OpenCV로 로드
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

                # ✅ YOLOv8 Pose 입력 크기로 조정
                target_size = (320, 256)
                frame = cv2.resize(frame, target_size)
                frames.append(frame)

            cap.release()

            if len(frames) == 0:
                raise ValueError("❌ No valid frames found in video")

            print(f"📌 Total Frames Extracted: {len(frames)}")
            return {"frames": frames}

        except Exception as e:
            print(f"🔥 Preprocessing error: {e}")
            return None

    def extract_landmarks(self, frame):
        """ YOLOv8 Pose를 사용해 랜드마크 추출 """
        results = self.pose_model(frame)
        height, width, _ = frame.shape
        people_data = []

        if len(results) > 0 and hasattr(results[0], 'keypoints'):
            keypoints = results[0].keypoints.xy.cpu().numpy()  # (N, 17, 2)
            for person_id, landmarks in enumerate(keypoints):
                keypoints_normalized = landmarks / [width, height]  # 정규화
                people_data.append((person_id, keypoints_normalized))
        return people_data

    def predict_class(self, sequence):
        """ Transformer 모델을 사용하여 동작(class) 예측 """
        if len(sequence) < 96:
            return "Waiting for data..."

        input_tensor = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.transformer_model(input_tensor)
            _, predicted = torch.max(output, 1)
        return self.class_names[predicted.item()]

    def inference(self, data):
        """ YOLO Pose → Transformer 동작 인식 실행 """
        frames = data["frames"]
        results = []

        for frame in frames:
            # ✅ YOLO Pose 실행 (바운딩 박스 X, 랜드마크만)
            people_data = self.extract_landmarks(frame)

            if not people_data:
                results.append({"frame_id": len(results), "pose": "No person detected"})
                continue

            frame_results = []
            for person_id, landmarks in people_data:
                self.sequence_buffers[person_id].append(landmarks.flatten())  # (34,)

                predicted_class = self.predict_class(self.sequence_buffers[person_id])
                frame_results.append({"person_id": person_id, "pose": predicted_class})

            results.append({"frame_id": len(results), "poses": frame_results})

        return results

    def postprocess(self, inference_output):
        """ TorchServe 응답 형식으로 변환 """
        return inference_output
