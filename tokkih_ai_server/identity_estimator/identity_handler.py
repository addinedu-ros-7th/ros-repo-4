import torch
import os
import io
import cv2
import json
import numpy as np
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from identity_model import ImprovedGenderAgeModel, AgeNormalizer
import torchvision.transforms as transforms

class IdentityEstimatorHandler(BaseHandler):
    def initialize(self, context):
        """ 모델 초기화 """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = context.system_properties.get("model_dir")
        model_path = os.path.join(model_dir, "identity_estimator.pth")  

        # 모델 로드
        self.model = ImprovedGenderAgeModel(pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 데이터 전처리 정의
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),         
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 나이 정규화 객체
        self.age_normalizer = AgeNormalizer()

        print("✅ Identity Estimator Model loaded successfully")

    def preprocess(self, data):
        """ 영상 전처리 (비디오 → 여러 프레임) """
        try:
            video_bytes = data[0].get("body", None)
            if video_bytes is None:
                raise ValueError("❌ Received empty video data")

            # 임시 파일 저장 후 OpenCV로 로드
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

                # OpenCV BGR → RGB 변환 후 PIL 이미지 변환
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # 🚀 전처리 후 Tensor 변환
                input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
                frames.append(input_tensor)

            cap.release()

            if len(frames) == 0:
                raise ValueError("❌ No valid frames found in video")

            print(f"📌 Total Frames Extracted: {len(frames)}")
            return {"frames": frames}

        except Exception as e:
            print(f"🔥 Preprocessing error: {e}")
            return None

    def inference(self, data):
        """ 모델 추론 (여러 프레임 지원) """
        predictions = []
        with torch.no_grad():
            for frame in data["frames"]:
                age_output, gender_output = self.model(frame)
                gender_prob = torch.sigmoid(gender_output).item()
                age = self.age_normalizer.denormalize(age_output.item())

                gender = "Female" if gender_prob > 0.5 else "Male"
                confidence = max(gender_prob, 1 - gender_prob) * 100

                predictions.append({"age": round(age, 2), "gender": gender, "confidence": round(confidence, 2)})

        return predictions

    def postprocess(self, data):
        """ 후처리: JSON 형식으로 반환 """
        try:
            return [{"age": d["age"], "gender": d["gender"], "confidence": d["confidence"]} for d in data]  # ✅ 올바른 형식
        except Exception as e:
            print(f"🔥 Postprocessing error: {e}")
            return [{"error": str(e)}]
