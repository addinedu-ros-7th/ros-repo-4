import torch
import os
import numpy as np
import cv2
import io
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from ultralytics import YOLO

class PositionHandler(BaseHandler):
    def initialize(self, context):
        """ 📌 모델 초기화 """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = context.system_properties.get("model_dir")
        model_path = os.path.join(model_dir, "position_adjuster.pt")

        # YOLOv8 모델 로드
        self.model = YOLO(model_path).to(self.device)
        self.model.eval()

        print(f"✅ Position Detection Model (YOLOv8) Loaded Successfully!")

        # 카메라 캘리브레이션 파라미터
        self.calibration_matrix = np.array([
            [267.79650171, 0., 140.13033557],
            [0., 271.46411043, 147.62847192],
            [0., 0., 1.]
        ])
        self.distortion_coefficients = np.array([-0.51294983, 3.29922758, -0.01057857, -0.01969346, -6.78451324])

        # 객체 크기 정보 (단위: m)
        self.object_real_size = {
            "green": (0.098, 0.098),  # 비상구
            "silver": (0.098, 0.136)  # 소화전
        }

        # 월드 좌표 (객체의 고정된 위치)
        self.object_world_coords = {
            "green": np.array([-0.23252665996551514, -0.0031995137687772512, -0.001434326171875]),
            "silver": np.array([0.8195663690567017, 1.2595571279525757, 0.002471923828125])
        }

    def preprocess(self, data):
        """ 📌 입력 이미지 전처리 """
        try:
            image_bytes = data[0].get("body", None)
            if image_bytes is None:
                raise ValueError("❌ Received empty image data")

            image = np.array(Image.open(io.BytesIO(image_bytes)))

            # 카메라 왜곡 보정
            h, w = image.shape[:2]
            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                self.calibration_matrix, self.distortion_coefficients, (w, h), 1, (w, h)
            )
            image = cv2.undistort(image, self.calibration_matrix, self.distortion_coefficients, None, new_camera_matrix)

            # Torch 변환
            image = torch.from_numpy(image).permute(2, 0, 1).float().to(self.device) / 255.0
            return {"image": image}

        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            return None

    def inference(self, data):
        """ 📌 YOLOv8 기반 객체 탐지 """
        with torch.no_grad():
            results = self.model(data["image"].unsqueeze(0), verbose=False)[0]
        return results

    def postprocess(self, detection):
        """ 📌 위치 및 방향 분석 """
        predictions = []
        
        if hasattr(detection, "masks") and detection.masks is not None:
            for i, mask in enumerate(detection.masks.xy):
                # numpy array를 일반 list로 변환
                mask = np.array(mask, dtype=np.int32).tolist()

                # 클래스 ID 가져오기 (tensor를 int로 변환)
                class_id = int(detection.boxes.cls[i].cpu().item())
                class_name = self.model.names[class_id] if class_id in self.model.names else "Unknown"

                # 마스크를 numpy array로 변환 (계산용)
                mask_np = np.array(mask, dtype=np.int32)

                # 마스크 중심점 계산
                M = cv2.moments(mask_np)
                if M["m00"] == 0:
                    continue
                u = int(M["m10"] / M["m00"])
                v = int(M["m01"] / M["m00"])

                # 거리(Z) 계산
                if class_name in self.object_real_size:
                    W_real, H_real = self.object_real_size[class_name]
                    Z = float((self.calibration_matrix[1, 1] * H_real) / cv2.boundingRect(mask_np)[3])

                    # 픽셀 좌표 → 카메라 좌표 변환
                    pixel_coords = np.array([u, v, 1])
                    cam_coords = Z * np.linalg.inv(self.calibration_matrix) @ pixel_coords
                    X_cam, Y_cam, Z_cam = [float(x) for x in cam_coords.flatten()]

                    # 카메라 좌표 → 월드 좌표 변환
                    if class_name in self.object_world_coords:
                        X_obj, Y_obj, Z_obj = [float(x) for x in self.object_world_coords[class_name]]

                        # 로봇 위치 보정
                        X_robot = float(X_obj - X_cam)
                        Y_robot = float(Y_obj - Y_cam)
                        Z_robot = float(Z_obj - Z_cam)

                        # 방향 분석 (PCA 기반)
                        mean, eigenvectors = cv2.PCACompute(mask_np.astype(np.float32), mean=None)
                        yaw_angle = float(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))

                        # 방향 값 변환
                        Z_orientation = float(np.sin(yaw_angle))
                        W_orientation = float(abs(np.cos(yaw_angle)))

                        predictions.append({
                            "class": class_name,
                            "position": {
                                "X": round(float(X_robot), 6),
                                "Y": round(float(Y_robot), 6),
                                "Z": round(float(Z_robot), 6)
                            },
                            "orientation": {
                                "Z": round(float(Z_orientation), 4),
                                "W": round(float(W_orientation), 4)
                            }
                        })

        return predictions

    def handle(self, data, context):
        """ 📌 TorchServe 배치 처리 대응 """
        
        batch_size = len(data)  # ✅ 입력 배치 크기 확인
        preprocessed_data = [self.preprocess([d]) for d in data]  # ✅ 개별 전처리

        # 🔹 전처리 실패한 경우 빈 리스트 유지
        preprocessed_data = [p if p else {} for p in preprocessed_data]

        detections = [self.inference(p) for p in preprocessed_data]
        responses = [self.postprocess(d) for d in detections]

        # ✅ 항상 입력 개수와 동일한 리스트 반환
        return responses if len(responses) == batch_size else [[]] * batch_size



