import cv2
import requests
from fastapi import FastAPI, Request

# FastAPI 클라이언트 서버 초기화
app = FastAPI()

# AI 서버 URL
AI_SERVER_URL = "http://192.168.35.233:8000/inference"

def send_frame_to_ai_server(frame):
    """
    웹캠 프레임을 AI 서버로 전송
    """
    # 프레임을 JPEG 형식으로 인코딩
    _, buffer = cv2.imencode(".jpg", frame)
    response = requests.post(
        AI_SERVER_URL,
        files={"video": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
    )
    return response.json() if response.status_code == 200 else {"error": response.status_code}

@app.post("/receive-results")
async def receive_results(request: Request):
    """
    AI 서버로부터 결과 데이터 수신
    """
    data = await request.json()
    print("AI 서버로부터 받은 데이터:", data)
    return {"status": "Received"}

# 웹캠 초기화
cap = cv2.VideoCapture(0)  # 기본 웹캠 (0번 장치)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # AI 서버로 프레임 전송
        response_data = send_frame_to_ai_server(frame)
        print("응답 데이터:", response_data)

        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
