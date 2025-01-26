from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import requests

# FastAPI 서버 초기화
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLO 모델 로드
try:
    object_model = YOLO("yolov8n.pt")
    print("YOLO 모델 로드 성공")
except Exception as e:
    print(f"YOLO 모델 로드 실패: {e}")

# 클라이언트 서버 URL
CLIENT_SERVER_URL = "http://192.168.35.207:8001/receive-results"

def send_results_to_client_server(frame_results):
    """
    클라이언트 서버로 추론 결과 전송
    """
    try:
        response = requests.post(
            CLIENT_SERVER_URL,
            json={"frame_results": frame_results},
            timeout=5
        )
        if response.status_code == 200:
            print("클라이언트 서버로 데이터 전송 성공")
        else:
            print(f"클라이언트 서버 오류: {response.status_code}")
    except Exception as e:
        print(f"클라이언트 서버로 데이터 전송 실패: {e}")

@app.post("/inference")
async def inference(video: UploadFile = File(...)):
    """
    클라이언트 서버로부터 영상을 수신받아 YOLO 모델로 추론 수행
    """
    try:
        # 영상 데이터를 읽고 프레임 변환
        video_bytes = await video.read()
        np_arr = np.frombuffer(video_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # YOLO 추론
        results = object_model(frame, verbose=False)
        frame_results = []

        if len(results) > 0 and hasattr(results[0], 'boxes'):
            boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
            classes = results[0].boxes.cls.cpu().numpy()  # (N,)
            for box, cls in zip(boxes, classes):
                frame_results.append({
                    "class": int(cls),
                    "box": box.tolist()
                })

        # 결과를 클라이언트 서버로 전송
        send_results_to_client_server(frame_results)

        return JSONResponse(content={"status": "Processed", "results": frame_results})

    except Exception as e:
        print(f"추론 중 오류 발생: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def read_root():
    return {"message": "AI Inference Server is Running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.35.207", port=8000)
