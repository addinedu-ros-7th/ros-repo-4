
# TCP용 import
import threading
import socket
import select

# ROS용 import
import rclpy as rp
from turtlesim.msg import Pose
from turtlesim.msg import Color
from std_msgs.msg import Float32
from sensor_msgs.msg import MagneticField

# 기타 import
import time
import signal
import sys

# SolCareGUI 수신 소켓 설정
host0 = "192.168.0.48"  # 서버의 IP 주소 또는 도메인 이름
port0 = 8081       # 포트 번호
server_socket0 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket0.bind((host0, port0))
server_socket0.listen()
userListen_thread_flag = True

# 전역 변수
pose_data = Pose()
color_data = Color()


def TCPListener(server_socket, set_time):
    global userListen_thread_flag, pose_data, color_data
    print("hello thread")

    cnt = 0
    socketList = [server_socket]
    while userListen_thread_flag:
        
        if not set_time == -1:
            cnt += 1
            print("cnt:", cnt, "/", set_time)
            if cnt >= set_time:
                print("end thread by cnt")
                userListen_thread_flag = False
                break

        print("Wait accept")
        read_socket, write_socket, error_socket = select.select(socketList, [], [], 1)
        for sock in read_socket :
            if not userListen_thread_flag:
                break
            elif sock == server_socket :
                print("나 아닌데")
                client_socket, client_address = sock.accept()
                socketList.append(client_socket)
            else :
                try:
                    print("Wait recive")
                    # 클라이언트로부터 요청 받기
                    data = sock.recv(1024).decode("utf-8")
                    if not data:
                        print("No Data")
                        continue

                    # 요청 파싱
                    parts = data.split("&&")
                    print(parts[:3])
                    if len(parts) != 0:
                        if parts[0] == "test_client":
                            print("Hello~")

                            messages = ["test_client"]
                            messages.append("Hello client, this is ROS server")
                            sock.send("&&".join(messages).encode('utf-8'))

                        elif parts[0] == "RequestPose":
                            print("Hello~")

                            messages = ["RequestPose"]
                            messages.append(str(pose_data))
                            sock.send("&&".join(messages).encode('utf-8'))
                        elif parts[0] == "RequestColor":
                            print("Hello~")

                            messages = ["RequestColor"]
                            messages.append(str(color_data))
                            sock.send("&&".join(messages).encode('utf-8'))
                        else:
                            print("뭐 이상한 거 왔어요")
                            print(parts)
                except Exception as e:
                    print(f"오류 발생: {e}")

                finally:
                    # 클라이언트 소켓 닫기
                    # print("클라이언트 연결종료")
                    sock.close()
                    socketList.remove(sock)

    print("Server End")
    server_socket.close()
    # cv2.destroyAllWindows()

### 아직 어떻게 할 지 생각 중
########################################################################################################
class DataStructure:
    def __init__(self):
        self.pose_data = Pose()
        self.color_data = Color()


def pose_callback(data):
    global pose_data
    pose_data = data

def color_callback(data):
    global color_data
    color_data = data

rp.init()
server_node =  rp.create_node('server_node')
server_node.create_subscription(Pose, 'turtle1/pose', pose_callback, 10)
server_node.create_subscription(Color, 'turtle1/color_sensor', color_callback, 10)

########################################################################################################


def signal_handler(signal, frame):
    global userListen_thread_flag

    print('You pressed Ctrl+C!', signal)
    userListen_thread_flag = False

    server_socket0.close()
    # remote.close()
    server_node.destroy_node()

    time.sleep(1)
    print("Good Bye")
    sys.exit(0)


if __name__=="__main__":
    signal.signal(signal.SIGINT, signal_handler)
    userListenTread = threading.Thread(target=TCPListener, args=(server_socket0, -1))
    
    userListenTread.start()
    rp.spin(server_node)
