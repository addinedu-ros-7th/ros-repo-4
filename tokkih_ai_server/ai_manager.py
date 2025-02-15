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
    AUTH_TOKEN: str = os.getenv("AUTH_TOKEN", "ZMZoQEMi")

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
        self.latest_responses = {}  # 각 source_addr별 최신 응답 저장
        self.responses_lock = threading.Lock()  # 스레드 안전성을 위한 락
        self.setup_socket()
        self.setup_signal_handlers()
        self.send_intervals = 0.1  # 0.1초 주기 전송
        self.tcp_senders = {}  # 비동기 전송 관리

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

    def send_periodic_tcp(self, pyri_id, source_addr):
        """0.1초마다 최신 responses를 전송"""
        if source_addr in self.tcp_senders and self.tcp_senders[source_addr].is_alive():
            return

        def periodic_task():
            while self.running.is_set():
                time.sleep(self.send_intervals)
                with self.responses_lock:
                    current_responses = self.latest_responses.get(source_addr, {})
                self.send_tcp_message(str(pyri_id), current_responses, source_addr)
                logger.debug(f"📡 Periodic TCP Response for {source_addr}")

            logger.info(f"🛑 Stopped periodic TCP sending for {source_addr}")
            self.tcp_senders.pop(source_addr, None)

        sender_thread = threading.Thread(target=periodic_task, daemon=True)
        self.tcp_senders[source_addr] = sender_thread
        sender_thread.start()

    def send_tcp_message(self, pyri_id: str, responses: dict, source_addr: tuple):
        """TCP로 일정한 형식의 JSON 데이터를 전송"""
        def get_json_value(data, key, default_value):
            """JSON에서 key 값을 가져오고, 없거나 형식이 다르면 default_value 반환"""
            value = data.get(key, default_value)

            if isinstance(default_value, list) and not isinstance(value, list):
                return [value] if value else []

            if isinstance(default_value, dict) and not isinstance(value, dict):
                return default_value

            return value

        formatted_json = {
            "target": get_json_value(responses, "target", []),
            "fire": get_json_value(responses, "fire", []),
            "position": get_json_value(responses, "position", []),
            "pose": get_json_value(responses, "pose", []),
            "identity": get_json_value(responses, "identity", [])
        }

        detection_json = json.dumps(formatted_json)
        message = f"{str(pyri_id)}&&{detection_json}"
        
        # Print the message being sent
        print(f"\n[TCP Message to {config.MAIN_ADDR}:{config.MAIN_PORT}]")
        print(f"PyRI ID: {pyri_id}")
        print(f"Message Content: {json.dumps(formatted_json, indent=2)}")
        print("-" * 50)

        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(config.TCP_TIMEOUT)
            client_socket.connect((config.MAIN_ADDR, config.MAIN_PORT))
            client_socket.send(message.encode('utf-8'))

            response = client_socket.recv(2048).decode('utf-8')
            print(f"Server Response: {response}\n")
            logger.debug(f"📡 Sent TCP message to {source_addr}")
            return response

        except socket.timeout:
            print("Connection timeout")
            logger.warning("TCP connection timeout")
            return "Timeout"

        except Exception as e:
            print(f"Error sending message: {str(e)}")
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

    def is_valid_frame(self, frame: np.ndarray) -> bool:
        """프레임이 유효한지 검사 (완전히 검은색이거나 움직임이 없는 경우 제외)"""
        if frame is None:
            return False
            
        # 프레임이 거의 검은색인지 확인
        mean_value = np.mean(frame)
        if mean_value < 5:  # 거의 검은색
            return False
            
        return True

    def process_frame(self, frame: np.ndarray, source_addr: tuple) -> dict:
        """한 프레임을 분석하고 일정한 JSON 포맷 유지"""
        # 프레임 유효성 검사
        if not self.is_valid_frame(frame):
            return {
                "target": [],
                "fire": [],
                "position": [],
                "pose": [],
                "identity": []
            }

        # Resize frame
        resized_frame = cv2.resize(frame, (320, 256))
        _, img_encoded = cv2.imencode(".jpg", resized_frame)
        img_bytes = img_encoded.tobytes()

        # 기본 모델 실행
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

        # 다중 객체 감지 확인
        target_detections = responses.get("target", [])
        target_detected = False
        
        if isinstance(target_detections, list):
            flattened_detections = target_detections[0] if target_detections and isinstance(target_detections[0], list) else target_detections
            target_detected = any(isinstance(d, dict) and d.get("class_id") == 0 for d in flattened_detections)

        if target_detected:
            logger.info(f"🎯 Target detected for {source_addr}")
            responses["pose"] = self.process_model_request("pose", img_bytes)
            responses["identity"] = self.process_model_request("identity", img_bytes)
        else:
            responses["pose"] = []
            responses["identity"] = []

        return responses

    def process_video_stream(self, source_addr: tuple):
        """비디오 스트림을 처리하고 최신 결과만 전송"""
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

                # 프레임 분석 수행
                responses = self.process_frame(frame, source_addr)
                
                # 최신 responses 업데이트
                with self.responses_lock:
                    self.latest_responses[source_addr] = responses

                # TCP 주기적 전송이 실행 중이 아니면 시작
                self.send_periodic_tcp(pyri_id, source_addr)

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
        """비디오 스레드 제거 및 관련 데이터 정리"""
        with self.video_threads_lock:
            if addr in self.video_threads:
                del self.video_threads[addr]
                logger.info(f"🛑 Removed video thread for {addr}")

            # TCP 전송 스레드 종료
            sender_thread = self.tcp_senders.pop(addr, None)
            if sender_thread and sender_thread.is_alive():
                logger.info(f"🛑 Stopping periodic TCP sending for {addr}")
                sender_thread.join()

        # responses 데이터 정리
        with self.responses_lock:
            self.latest_responses.pop(addr, None)

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