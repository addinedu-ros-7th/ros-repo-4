import torch
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ts.torch_handler.base_handler import BaseHandler
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

# ✅ Global MiDaS model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔄 Loading MiDaS model (server initialization)...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
midas.eval()
print("✅ MiDaS Model loaded successfully (server-wide)")

# ✅ Global MiDaS transform configuration
midas_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class YOLOv8Handler(BaseHandler):
    def initialize(self, context):
        """🔥 모델 초기화"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = context.system_properties.get("model_dir")
        model_path = os.path.join(model_dir, "yolov8n.pt")

        # ✅ Load YOLOv8
        self.model = YOLO(model_path).to(self.device)
        self.model.eval()
        print("✅ YOLOv8 Model loaded successfully")

        # ✅ Use global MiDaS model
        self.midas = midas
        self.midas_transform = midas_transform

    def preprocess(self, data, num_workers=2):
        """🔥 다중 영상 병렬 전처리"""
        def process_single_video(idx, video_bytes):
            video_path = f"/tmp/temp_video_{idx}.mp4"
            frames = []
            
            try:
                # ✅ Save video bytes to temporary file
                with open(video_path, "wb") as f:
                    f.write(video_bytes)

                # ✅ Process video frames
                cap = cv2.VideoCapture(video_path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.resize(frame, (320, 256))
                    frame = torch.from_numpy(frame).permute(2, 0, 1).float().to(self.device) / 255.0
                    frames.append(frame)

                    # ✅ Frame limit for memory management
                    if len(frames) >= 300:  
                        print(f"⚠️ Warning: Frame limit reached for video {idx}")
                        break

                cap.release()

            except Exception as e:
                print(f"❌ Error processing video {idx}: {str(e)}")
                raise
            finally:
                if os.path.exists(video_path):
                    os.remove(video_path)

            return frames

        videos = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_video, idx, video_data["body"])
                       for idx, video_data in enumerate(data) if "body" in video_data]
            
            for future in futures:
                try:
                    videos.append(future.result())
                except Exception as e:
                    print(f"❌ Error in video processing: {str(e)}")
                    videos.append([])

        return {"videos": videos}

    def run_midas(self, frame):
        """🔥 MiDaS 깊이 추정"""
        try:
            if not isinstance(frame, np.ndarray):
                raise ValueError(f"Invalid frame type: {type(frame)}")
            
            if frame.dtype == np.float32:
                frame = (frame * 255).astype(np.uint8)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb, 'RGB')
            img_tensor = self.midas_transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                depth_map = self.midas(img_tensor)

            depth_map = depth_map.squeeze().cpu().numpy()
            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            return depth_map
            
        except Exception as e:
            print(f"❌ MiDaS error: {str(e)}")
            raise

    def inference(self, data, batch_size=1):
        """🔥 YOLOv8 + MiDaS 추론"""
        results = []
        
        try:
            with torch.no_grad():
                for video_idx, frames in enumerate(data["videos"]):
                    video_results = []

                    for i in range(0, len(frames), batch_size):
                        batch = torch.stack(frames[i:i+batch_size]) if len(frames[i:i+batch_size]) > 1 else frames[i:i+batch_size][0].unsqueeze(0)
                        
                        # ✅ YOLOv8 detection
                        batch_results = self.model(batch, verbose=False)

                        for j, result in enumerate(batch_results):
                            if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes.xyxy) > 0:
                                try:
                                    frame = frames[i + j].permute(1, 2, 0).cpu().numpy()
                                    depth_map = self.run_midas(frame)
                                    result.depth_map = depth_map
                                except Exception as e:
                                    print(f"❌ MiDaS processing failed: {str(e)}")
                                    result.depth_map = None
                            else:
                                result.depth_map = None
                        
                        video_results.extend(batch_results)
                    
                    results.append(video_results)
                    
        except Exception as e:
            print(f"❌ Inference error: {str(e)}")
            raise
            
        return results

    def postprocess(self, inference_output):
        """🔥 YOLOv8 + MiDaS 후처리"""
        all_predictions = []

        target_classes = {0, 16, 15, 56, 57, 60}  

        try:
            for video_idx, video_results in enumerate(inference_output):
                video_predictions = []

                for result in video_results:
                    frame_predictions = []

                    if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes.xyxy) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        scores = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        depth_map = result.depth_map

                        for box, score, cls in zip(boxes, scores, classes):
                            try:
                                class_id = int(cls)
                                if class_id not in target_classes:
                                    continue

                                x1, y1, x2, y2 = map(int, box)
                                Z = 0
                                if depth_map is not None:
                                    roi = depth_map[y1:y2, x1:x2]
                                    Z = int(round(np.mean(roi))) if roi.size > 0 else 0

                                frame_predictions.append({
                                    "bbox": box.tolist(),
                                    "confidence": float(score),
                                    "class_id": class_id,
                                    "class_name": self.model.names[class_id] if class_id in self.model.names else "Unknown",
                                    "depth": Z
                                })

                            except Exception as e:
                                print(f"❌ Error processing detection: {str(e)}")
                                continue

                    video_predictions.append(frame_predictions)
                    print(f"📊 Frame predictions: {frame_predictions}")

                all_predictions.append(video_predictions)
                print(f"📌 Video {video_idx}: Total Predictions {len(video_predictions)} Frames")

        except Exception as e:
            print(f"❌ Post-processing error: {str(e)}")
            raise

        return all_predictions
