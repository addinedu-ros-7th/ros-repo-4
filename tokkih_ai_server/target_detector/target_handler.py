import torch
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ts.torch_handler.base_handler import BaseHandler
from ultralytics import YOLO  # YOLOv8 모델 로드
from torchvision import transforms
from PIL import Image

# ✅ 서버 실행 시 한 번만 MiDaS 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔄 Loading MiDaS model (server initialization)...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
midas.eval()
print("✅ MiDaS Model loaded successfully (server-wide)")

# ✅ MiDaS 입력 변환 설정 (서버 전역 사용)
midas_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # MiDaS Small 모델 해상도
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class YOLOv8Handler(BaseHandler):
    def initialize(self, context):
        """ 모델 초기화 """
        model_dir = context.system_properties.get("model_dir")

        # ✅ YOLOv8 모델 로드
        model_path = os.path.join(model_dir, "yolov8n.pt")
        self.model = YOLO(model_path).to(device)
        self.model.eval()
        print("✅ YOLOv8 Model loaded successfully")

        # ✅ MiDaS는 서버 전역 변수 사용
        self.midas = midas
        self.midas_transform = midas_transform

    def run_midas(self, frame):
        """ MiDaS로 깊이 맵 추출 """
        try:
            # 1. 입력 프레임 정보 출력
            print(f"Input frame - Shape: {frame.shape}, Type: {frame.dtype}, Range: [{frame.min()}, {frame.max()}]")
            
            # 2. 입력 검증
            if len(frame.shape) != 3 or frame.shape[-1] != 3:
                raise ValueError(f"Invalid frame shape: {frame.shape}, expected (H, W, 3)")
            
            # 3. float32 -> uint8 변환 (범위 조정)
            if frame.dtype == np.float32:
                # [-1,1] 또는 [0,1] 범위를 [0,255]로 변환
                if frame.min() < 0:  # [-1,1] 범위인 경우
                    frame = ((frame + 1.0) * 127.5).astype(np.uint8)
                else:  # [0,1] 범위인 경우
                    frame = (frame * 255).astype(np.uint8)
            
            # 4. uint8 타입 확인
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # 5. BGR -> RGB 변환 (uint8 형식으로)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"After conversion - Shape: {img_rgb.shape}, Type: {img_rgb.dtype}, Range: [{img_rgb.min()}, {img_rgb.max()}]")
            
            # 6. PIL Image 변환
            img_pil = Image.fromarray(img_rgb, mode='RGB')
            
            # 7. MiDaS 변환 및 추론
            img_tensor = self.midas_transform(img_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                depth_map = self.midas(img_tensor)
            
            # 8. 깊이 맵 후처리
            depth_map = depth_map.squeeze().cpu().numpy()
            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            return depth_map
            
        except Exception as e:
            print(f"MiDaS processing error: {str(e)}")
            print(f"Frame info at error: Shape={frame.shape}, Type={frame.dtype}")
            raise


    def preprocess(self, data, num_workers=2):
        """ 병렬 영상 전처리 (단일/다중 영상 지원) """
        def process_video(idx, video_bytes):
            video_path = f"/tmp/temp_video_{idx}.mp4"
            with open(video_path, "wb") as f:
                f.write(video_bytes)

            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (320, 256))
                frame = torch.from_numpy(frame).permute(2, 0, 1).float().to(self.device) / 255.0
                frames.append(frame)
            cap.release()
            return frames

        videos = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_video, idx, video_data.get("body", None)) for idx, video_data in enumerate(data)]
            for future in futures:
                videos.append(future.result())

        return {"videos": videos}

    def inference(self, data, batch_size=1):
        """ YOLOv8 + MiDaS 배치 추론 """
        results = []
        with torch.no_grad():
            for video_idx, frames in enumerate(data["videos"]):
                video_results = []
                for i in range(0, len(frames), batch_size):
                    batch = torch.stack(frames[i:i+batch_size]) if len(frames[i:i+batch_size]) > 1 else frames[i:i+batch_size][0].unsqueeze(0)
                    
                    # YOLOv8 객체 탐지 실행
                    batch_results = self.model(batch, verbose=False)
                    
                    # YOLO 감지 결과를 확인하고 MiDaS 실행
                    for j, result in enumerate(batch_results):
                        if result.boxes is not None and len(result.boxes.xyxy) > 0:
                            # torch tensor -> numpy array 변환 (값 범위 보존)
                            frame = frames[i + j].permute(1, 2, 0).cpu().numpy()
                            print(f"Frame before MiDaS - Shape: {frame.shape}, Type: {frame.dtype}")
                            
                            try:
                                depth_map = self.run_midas(frame)
                                result.depth_map = depth_map
                            except Exception as e:
                                print(f"MiDaS 처리 실패: {str(e)}")
                                result.depth_map = None
                        else:
                            result.depth_map = None
                    
                    video_results.extend(batch_results)
                
                results.append(video_results)
                
        return results


    def run_midas(self, frame):
        """ MiDaS로 깊이 맵 추출 """
        try:
            print("🔄 Starting MiDaS processing...")
            
            # 1. 입력 프레임 정보
            print(f"Input frame - Shape: {frame.shape}, Type: {frame.dtype}, Range: [{frame.min()}, {frame.max()}]")
            
            # 2. uint8 변환
            if frame.dtype == np.float32:
                frame = (frame * 255).astype(np.uint8)
            
            # 3. RGB 변환
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"After RGB conversion - Shape: {img_rgb.shape}, Type: {img_rgb.dtype}")
            
            # 4. PIL 변환
            img_pil = Image.fromarray(img_rgb, 'RGB')
            
            # 5. MiDaS 입력 준비
            img_tensor = self.midas_transform(img_pil).unsqueeze(0).to(device)
            print(f"Tensor ready for MiDaS - Shape: {img_tensor.shape}")
            
            # 6. MiDaS 추론
            with torch.no_grad():
                depth_map = self.midas(img_tensor)
            
            # 7. 후처리
            depth_map = depth_map.squeeze().cpu().numpy()
            print(f"Raw depth map - Shape: {depth_map.shape}, Range: [{depth_map.min()}, {depth_map.max()}]")
            
            # 8. 정규화
            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            print(f"Normalized depth map - Shape: {depth_map.shape}, Range: [{depth_map.min()}, {depth_map.max()}]")
            
            return depth_map
            
        except Exception as e:
            print(f"❌ MiDaS error: {str(e)}")
            print(f"Frame info at error: Shape={frame.shape}, Type={frame.dtype}")
            raise

    def postprocess(self, data):
        """ YOLOv8 + MiDaS 후처리 """
        all_predictions = []
        for video_idx, video_results in enumerate(data):
            video_predictions = []
            for result in video_results:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                depth_map = result.depth_map  # MiDaS 깊이 맵

                frame_predictions = []
                for box, score, cls in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(cls)

                    # 깊이 계산 디버깅
                    if depth_map is not None:
                        try:
                            # 바운딩 박스 좌표가 유효한지 확인
                            h, w = depth_map.shape
                            x1 = max(0, min(x1, w-1))
                            x2 = max(0, min(x2, w-1))
                            y1 = max(0, min(y1, h-1))
                            y2 = max(0, min(y2, h-1))
                            
                            # ROI 추출
                            roi = depth_map[y1:y2, x1:x2]
                            
                            if roi.size > 0:
                                # ROI 통계 출력 (디버깅용)
                                print(f"ROI shape: {roi.shape}")
                                print(f"ROI range: [{roi.min()}, {roi.max()}]")
                                print(f"ROI mean before round: {np.mean(roi)}")
                                
                                # 평균 깊이 계산 및 반올림
                                Z = int(round(np.mean(roi)))
                                print(f"Final depth value: {Z}")
                            else:
                                print("❌ ROI is empty")
                                Z = 0
                        except Exception as e:
                            print(f"❌ Depth calculation error: {str(e)}")
                            Z = 0
                    else:
                        print("❌ Depth map is None")
                        Z = 0

                    frame_predictions.append({
                        "bbox": box.tolist(),
                        "confidence": float(score),
                        "class_id": class_id,
                        "class_name": self.model.names[class_id],
                        "depth": Z
                    })

                video_predictions.append(frame_predictions)
                print(f"📊 Frame predictions: {frame_predictions}")  # 전체 예측 결과 출력

            all_predictions.append(video_predictions)
            print(f"📌 Video {video_idx}: Total Predictions {len(video_predictions)} Frames")
        return all_predictions
