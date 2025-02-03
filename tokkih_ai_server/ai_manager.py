import socket
import select
import numpy as np
import cv2
import requests
import json

# ✅ UDP 설정
UDP_IP = "192.168.0.9"
UDP_PORT = 9506

# ✅ TCP 설정
MAIN_ADDR = "192.168.0.8"
MAIN_PORT = 8081

# ✅ TorchServe 설정 (두 모델 동시 실행)
MODEL_URLS = {
    "target": "http://localhost:8080/predictions/target_detector",
    "fire": "http://localhost:8080/predictions/fire_detector"
}

HEADERS = {
    "Authorization": "Bearer borSUIIE",  
    "Content-Type": "image/jpeg"
}

def send_tcp_message(messages):
    """ 🔥 TCP로 탐지 결과 전송 """
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((MAIN_ADDR, MAIN_PORT))
        client_socket.send(f"{'&&'.join(messages)}".encode('utf-8'))

        ready = select.select([client_socket], [], [], 2)
        if ready[0]:
            response = client_socket.recv(2048).decode('utf-8')
        else:
            response = "Timeout"
    except Exception as e:
        response = f"Error: {str(e)}"
    finally:
        client_socket.close()
    return response

# ✅ UDP 소켓 초기화
c_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
c_sock.bind((UDP_IP, UDP_PORT))
socketList = [c_sock]

while True:
    read_socket, _, _ = select.select(socketList, [], [], 1)
    
    for sock in read_socket:
        data, addr = sock.recvfrom(57601)

        if data:
            encodete_img = np.frombuffer(data[1:], dtype=np.uint8)
            frame = cv2.imdecode(encodete_img, cv2.IMREAD_COLOR)

            # ✅ YOLOv8 입력 크기로 리사이징 (stride 32의 배수)
            resized_frame = cv2.resize(frame, (320, 256))

            # ✅ JPEG으로 인코딩
            _, img_encoded = cv2.imencode(".jpg", resized_frame)
            img_bytes = img_encoded.tobytes()

            responses = {}
            for model_name, model_url in MODEL_URLS.items():
                try:
                    # 🔥 TorchServe 요청
                    response = requests.post(model_url, headers=HEADERS, data=img_bytes)

                    if response.status_code == 200:
                        responses[model_name] = response.json()
                    else:
                        print(f"❌ {model_name.upper()} 서버 오류: {response.status_code}, {response.text}")
                        responses[model_name] = []
                
                except Exception as e:
                    print(f"❌ {model_name.upper()} 요청 실패: {str(e)}")
                    responses[model_name] = []

            # 🔹 탐지 결과 확인
            print("🎯 Target Detection:", responses["target"])
            print("🔥 Fire Detection:", responses["fire"])

            # ✅ TCP로 탐지 결과 전송
            detection_json = json.dumps(responses)
            tcp_response = send_tcp_message([detection_json])
            print(f"📡 TCP Response: {tcp_response}")

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ 종료
c_sock.close()
