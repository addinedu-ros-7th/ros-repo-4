import socket
import select
import numpy as np
import cv2
import requests
import time

# UDP 설정
UDP_IP = "192.168.0.9"
UDP_PORT = 9506

# TorchServe 설정
TORCHSERVE_URL = "http://localhost:8080/predictions/target_detector"
HEADERS = {
    "Authorization": "Bearer gdwW71wG",
    "Content-Type": "image/jpeg"
}

# UDP 소켓 초기화
c_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
c_sock.bind((UDP_IP, UDP_PORT))
socketList = [c_sock]

# 버퍼 초기화 (4개의 패킷을 저장)
s = [b'\x00' * 57600 for _ in range(4)]

while True:
    picture = b''
    read_socket, _, _ = select.select(socketList, [], [], 1)
    
    for sock in read_socket:
        data, addr = sock.recvfrom(57601)
        s[data[0]] = data[1:]
        
        if data[0] == 3:  # 마지막 패킷을 받았을 때
            # 전체 이미지 데이터 조합
            for i in range(4):
                picture += s[i]
            
            # numpy 배열로 변환
            frame = np.frombuffer(picture, dtype=np.uint8)
            frame = frame.reshape(240, 320, 3)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # YOLOv8 입력 크기로 리사이징 (stride 32의 배수)
            resized_frame = cv2.resize(frame, (320, 256))
            
            # JPEG으로 인코딩
            _, img_encoded = cv2.imencode(".jpg", resized_frame)
            img_bytes = img_encoded.tobytes()
            
            try:
                # TorchServe로 전송
                response = requests.post(TORCHSERVE_URL, headers=HEADERS, data=img_bytes)
                
                if response.status_code == 200:
                    detections = response.json()
                    print("🎯 Detection Results:", detections)
                    
                    # 바운딩 박스 그리기
                    for obj in detections:
                        x_min, y_min, x_max, y_max = map(int, obj["bbox"])
                        conf = obj["confidence"]
                        class_id = obj["class_id"]
                        
                        # 바운딩 박스 & 라벨 표시
                        label = f"Class {class_id}: {conf:.2f}"
                        cv2.rectangle(resized_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(resized_frame, label, (x_min, y_min - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    print(f"❌ 서버 오류: {response.status_code}, {response.text}")
                    
            except Exception as e:
                print(f"❌ 요청 실패: {str(e)}")
            
            # 결과 영상 출력
            cv2.imshow("YOLOv8 UDP Stream Inference", resized_frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cv2.destroyAllWindows()
c_sock.close()