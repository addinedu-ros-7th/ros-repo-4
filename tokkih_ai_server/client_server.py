import cv2
import requests
from fastapi import FastAPI, Request
import uvicorn

# FastAPI 클라이언트 서버 초기화
app = FastAPI()

# AI 서버 URL
AI_SERVER_URL = "http://192.168.35.207:8000/inference"

def send_frame_to_ai_server(frame):
    """
    웹캠 프레임을 AI 서버로 전송
    """
    try:
        _, buffer = cv2.imencode(".jpg", frame)
        response = requests.post(
            AI_SERVER_URL,
            files={"video": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
        )
        return response.json() if response.status_code == 200 else {"error": response.status_code}
    except Exception as e:
        print(f"AI 서버로 데이터 전송 실패: {e}")
        return {"error": str(e)}

@app.post("/receive-results")
async def receive_results(request: Request):
    """
    AI 서버로부터 결과 데이터 수신
    """
    try:
        data = await request.json()
        print("AI 서버로부터 받은 데이터:", data)
        return {"status": "Received"}
    except Exception as e:
        print(f"결과 수신 중 오류 발생: {e}")
        return {"status": "Failed", "error": str(e)}

def run_webcam_loop():
    """
    웹캠에서 프레임을 읽어 AI 서버로 전송
    """
    cap = cv2.VideoCapture(1)  # 기본 웹캠

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

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

if __name__ == "__main__":
    # FastAPI 서버 실행 (백그라운드 작업 가능)
    import threading

    # FastAPI 서버를 별도의 스레드에서 실행
    def run_server():
        uvicorn.run(app, host="192.168.35.207", port=8001)

    # 서버 실행
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # 웹캠 루프 실행
    run_webcam_loop()
