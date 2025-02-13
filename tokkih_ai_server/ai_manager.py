import socket
import select
import numpy as np
import cv2
import requests
import json
import threading
import logging
from typing import Dict, List
from dataclasses import dataclass
import os
import signal
import time
import threading

# Configuration
@dataclass
class Config:
    UDP_IP: str = os.getenv("UDP_IP", "192.168.0.9")
    UDP_PORT: int = int(os.getenv("UDP_PORT", "9506"))
    MAIN_ADDR: str = os.getenv("MAIN_ADDR", "192.168.0.10")
    MAIN_PORT: int = int(os.getenv("MAIN_PORT", "8081"))
    BUFFER_SIZE: int = 57601
    TCP_TIMEOUT: int = 5
    AUTH_TOKEN: str = os.getenv("AUTH_TOKEN", "B_Yg3ZpZ")

    MODEL_URLS: Dict[str, str] = None
    
    def __post_init__(self):
        base_url = "http://localhost:8080/predictions"
        self.MODEL_URLS = {
            "target": f"{base_url}/target_detector",
            "fire": f"{base_url}/fire_detector",
            "pose": f"{base_url}/pose_estimator",
            "identity": f"{base_url}/identity_estimator",
            "position": f"{base_url}/position_adjuster"
        }

config = Config()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_processing.log')
    ]
)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.running = threading.Event()
        self.running.set()
        self.video_threads = {}
        self.video_threads_lock = threading.Lock()
        self.setup_socket()
        self.setup_signal_handlers()
        self.last_sent_time = {}  # 각 주소별 마지막 전송 시간 기록
        self.send_intervals = 0.1  # 0.1초 주기 전송
        self.tcp_senders = {}  #  비동기 전송 관리


    def send_periodic_tcp(self, pyri_id, responses, source_addr):
        """0.1초마다 일정한 JSON 형식으로 TCP 메시지 전송"""
        if source_addr in self.tcp_senders and self.tcp_senders[source_addr].is_alive():
            return  # 이미 실행 중이면 중복 실행 방지

        def periodic_task():
            while self.running.is_set():
                time.sleep(self.send_intervals)
                self.send_tcp_message(str(pyri_id), responses, source_addr)  # ✅ 올바르게 인자 전달
                logger.debug(f"📡 Periodic TCP Response for {source_addr}")

            logger.info(f"🛑 Stopped periodic TCP sending for {source_addr}")
            self.tcp_senders.pop(source_addr, None)  # 스레드 종료 시 정리

        sender_thread = threading.Thread(target=periodic_task, daemon=True)
        self.tcp_senders[source_addr] = sender_thread
        sender_thread.start()

    def setup_socket(self):
        """Initialize UDP socket"""
        try:
            self.c_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.c_sock.bind((config.UDP_IP, config.UDP_PORT))
            self.socketList = [self.c_sock]
            logger.info(f"UDP socket initialized on {config.UDP_IP}:{config.UDP_PORT}")
        except Exception as e:
            logger.error(f"Socket initialization failed: {e}")
            raise

    def setup_signal_handlers(self):
        """Setup handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received. Cleaning up...")
        self.cleanup()

    def send_tcp_message(self, pyri_id: str, responses: dict, source_addr: tuple):
        """TCP로 일정한 형식의 JSON 데이터를 전송"""

        def get_json_value(data, key, default_value):
            """JSON에서 key 값을 가져오고, 없거나 형식이 다르면 default_value 반환"""
            value = data.get(key, default_value)

            # 리스트 확인 (리스트가 아닌 경우에도 리스트로 변환)
            if isinstance(default_value, list) and not isinstance(value, list):
                return [value] if value else []

            # 딕셔너리 확인
            if isinstance(default_value, dict) and not isinstance(value, dict):
                return default_value

            return value

        # ✅ **항상 동일한 JSON 형식 유지**
        formatted_json = {
            "target": get_json_value(responses, "target", []),
            "fire": get_json_value(responses, "fire", []),
            "position": get_json_value(responses, "position", []),
            "pose": get_json_value(responses, "pose", []),
            "identity": get_json_value(responses, "identity", [])
        }

        # ✅ **JSON 직렬화**
        detection_json = json.dumps(formatted_json)

        # ✅ **TCP 메시지 전송 로직 추가**
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(config.TCP_TIMEOUT)
            client_socket.connect((config.MAIN_ADDR, config.MAIN_PORT))
            client_socket.send(f"{str(pyri_id)}&&{detection_json}".encode('utf-8'))

            response = client_socket.recv(2048).decode('utf-8')
            logger.info(f"📡 Sent TCP message to {source_addr}: {detection_json}")
            return response

        except socket.timeout:
            logger.warning("TCP connection timeout")
            return "Timeout"

        except Exception as e:
            logger.error(f"TCP communication error: {e}")
            return f"Error: {str(e)}"

        finally:
            client_socket.close()


    def process_model_request(self, model_name: str, img_bytes: bytes) -> dict:
        """Process individual model request"""
        headers = {
            "Authorization": f"Bearer {config.AUTH_TOKEN}",
            "Content-Type": "image/jpeg"
        }
        
        try:
            response = requests.post(
                config.MODEL_URLS[model_name],
                headers=headers,
                data=img_bytes
            )
            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from {model_name} response")
                    return []
            else:
                logger.error(f"Model {model_name} error: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Model {model_name} request failed: {e}")
            return []

    def process_frame(self, frame: np.ndarray, source_addr: tuple) -> dict:
        """한 프레임을 분석하고 일정한 JSON 포맷 유지"""
        # Resize frame
        resized_frame = cv2.resize(frame, (320, 256))
        _, img_encoded = cv2.imencode(".jpg", resized_frame)
        img_bytes = img_encoded.tobytes()

        # ✅ 기본 모델 실행
        responses = {}
        initial_models = ["target", "fire", "position"]
        for model in initial_models:
            model_response = self.process_model_request(model, img_bytes)
            
            # JSON 문자열이면 변환
            if isinstance(model_response, str):
                try:
                    model_response = json.loads(model_response)
                except json.JSONDecodeError:
                    model_response = []
            
            responses[model] = model_response

        # ✅ **다중 객체 감지 확인**
        target_detections = responses.get("target", [])
        target_detected = False
        
        if isinstance(target_detections, list):
            # 중첩 리스트 처리
            flattened_detections = target_detections[0] if target_detections and isinstance(target_detections[0], list) else target_detections
            target_detected = any(isinstance(d, dict) and d.get("class_id") == 0 for d in flattened_detections)

        if target_detected:
            logger.info(f"🎯 Target detected for {source_addr}")
            responses["pose"] = self.process_model_request("pose", img_bytes)
            responses["identity"] = self.process_model_request("identity", img_bytes)

        return responses


    def process_video_stream(self, source_addr: tuple):
        """비디오 스트림을 처리하고 최초 감지 시 즉시 전송"""
        logger.info(f"🎥 Starting video stream processing for {source_addr}")

        try:
            while self.running.is_set():
                data, addr = self.c_sock.recvfrom(config.BUFFER_SIZE)
                if not data or addr != source_addr:
                    continue

                # Convert UDP data to frame
                pyri_id = data[0]
                encoded_img = np.frombuffer(data[1:], dtype=np.uint8)
                frame = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # 🔹 실제 모델을 통한 분석 수행
                responses = self.process_frame(frame, source_addr)

                # ✅ **TCP 메시지 전송**
                self.send_tcp_message(pyri_id, responses, source_addr)

                # ✅ **0.1초마다 반복 전송 시작**
                self.send_periodic_tcp(pyri_id, responses, source_addr)

        except Exception as e:
            logger.error(f"❌ Stream processing error for {source_addr}: {e}")
        finally:
            self.remove_video_thread(source_addr)




    def add_video_thread(self, addr: tuple):
        """Add new video processing thread"""
        with self.video_threads_lock:
            if addr not in self.video_threads:
                thread = threading.Thread(
                    target=self.process_video_stream,
                    args=(addr,)
                )
                self.video_threads[addr] = thread
                thread.start()

    def remove_video_thread(self, addr: tuple):
        """비디오 스레드 제거 및 주기적 TCP 전송 중지"""
        with self.video_threads_lock:
            if addr in self.video_threads:
                del self.video_threads[addr]
                logger.info(f"🛑 Removed video thread for {addr}")

            # TCP 전송 스레드 종료
            sender_thread = self.tcp_senders.pop(addr, None)
            if sender_thread and sender_thread.is_alive():
                logger.info(f"🛑 Stopping periodic TCP sending for {addr}")
                sender_thread.join()  # 완전 종료 보장



    def cleanup(self):
        """Cleanup resources"""
        self.running.clear()
        logger.info("Waiting for threads to finish...")
        with self.video_threads_lock:
            for thread in self.video_threads.values():
                thread.join()
        self.c_sock.close()
        logger.info("Cleanup completed")

    def run(self):
        """Main run loop"""
        logger.info("Starting video processor...")
        try:
            while self.running.is_set():
                read_socket, _, _ = select.select(self.socketList, [], [], 1)
                
                for sock in read_socket:
                    data, addr = sock.recvfrom(config.BUFFER_SIZE)
                    self.add_video_thread(addr)

        except Exception as e:
            logger.error(f"Main loop error: {e}")
        finally:
            self.cleanup()

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.run()