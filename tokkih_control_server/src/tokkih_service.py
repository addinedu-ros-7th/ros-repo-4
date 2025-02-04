
# TCP용 import
import threading
import socket
import select
import json

# ROS용 import
import rclpy as rp
from turtlesim.msg import Pose
from turtlesim.msg import Color
from std_msgs.msg import Float32
from sensor_msgs.msg import MagneticField

from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped

# 기타 import
import time
import signal
import sys
import numpy as np
import cv2

MY_IP = "192.168.0.8"

# from Tokkih GUI 수신 소켓 설정
TCP_PORT0 = 8081       # 포트 번호
server_socket0 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket0.bind((MY_IP, TCP_PORT0))
server_socket0.listen()
userListen_thread_flag = True

# Pyri로 부터의 영상 udp 수신 소켓 설정
UDP_PORT0 = 9505
udp_listnener_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_listnener_sock.bind((MY_IP, UDP_PORT0))
imgListen_thread_flag = True

# 영상을 다시 AI서버로 보내줄 소켓 seoljeong
AI_IP = "192.168.0.9"
UDP_PORT1 = 9506
udp_talker_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 영상을 다시 GUI로 보내줄 소켓 seoljeong
GUI_IP = "192.168.0.8"
UDP_PORT2 = 9506

# 전역 변수
pose_data = Pose()
color_data = Color()
tp_data = [0.0, 1.1, 2.2]
tf_data = [1.1, 2.2, 3.3, 4.4]

ai_result_flag = False
received_json = dict()
ai_result_name = "aiResult"


temp_img = np.zeros((240,320,3), dtype=np.int8)
_, buffer = cv2.imencode('.jpg', temp_img)
temp_img_b = b'\x02' + buffer.tobytes()

def IMGListener(udp_socket0, udp_socket1, set_time):
    # udp_socket0 = Listnener
    # udp_socket1 = Talker to AI Server
    # udp_socket1 = Talker to GUI
    global imgListen_thread_flag

    data = temp_img_b
    socketList = [udp_socket0]

    while imgListen_thread_flag:
        read_socket, _, _ = select.select(socketList, [], [], 1)
        for sock in read_socket:
            data, _ = sock.recvfrom(65535)
            udp_socket1.sendto(data, (AI_IP, UDP_PORT1))
            udp_socket1.sendto(data, (GUI_IP, UDP_PORT2))



def TCPListener(server_socket, set_time):
    global userListen_thread_flag, pose_data, color_data, ai_result_flag
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

                signal_handler(None, None)
                break

        # print("Wait accept")
        read_socket, write_socket, error_socket = select.select(socketList, [], [], 1)
        for sock in read_socket :
            if not userListen_thread_flag:
                break
            elif sock == server_socket :
                client_socket, client_address = sock.accept()
                socketList.append(client_socket)
            else :
                try:
                    # print("Wait recive")
                    # 클라이언트로부터 요청 받기
                    data = sock.recv(1024).decode("utf-8")
                    if not data:
                        print("No Data")
                        continue

                    # data='[{"bbox": [138.38656616210938, 116.25907897949219, 221.21823120117188, 205.5818634033203], "confidence": 0.522793173789978, "class_id": 56}, {"bbox": [0.0, 176.331298828125, 41.553306579589844, 206.30224609375], "confidence": 0.5043512582778931, "class_id": 45}, {"bbox": [96.24269104003906, 75.76333618164062, 127.29405212402344, 131.20449829101562], "confidence": 0.26554909348487854, "class_id": 41}]'

                    try:
                        received_json = json.loads(data)[0]
                        ai_result_flag = True
                    except:
                        # 요청 파싱
                        parts = data.split("&&")
                        # print(parts[:3])
                        if len(parts) != 0:
                            if parts[0] == "test_client":
                                print("Hello~")

                                messages = ["test_client"]
                                messages.append("Hello client, this is ROS server")
                                sock.send("&&".join(messages).encode('utf-8'))

                            elif parts[0] == "RequestPose":

                                messages = ["RequestPose"]
                                messages.append(str(pose_data))
                                sock.send("&&".join(messages).encode('utf-8'))
                            elif parts[0] == "RequestColor":

                                messages = ["RequestColor"]
                                messages.append(str(color_data))
                                sock.send("&&".join(messages).encode('utf-8'))
                            elif parts[0] == "SyncData":

                                messages = ["SyncData"]

                                if ai_result_flag:
                                    messages.append(ai_result_name)
                                    messages.append(str(received_json))

                                    ai_result_flag = False
                                else:
                                    messages.append("ideal")
                                    for val in tf_data:
                                        messages.append(str(val))

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


def pose_callback(data):
    global pose_data
    pose_data = data

def color_callback(data):
    global color_data
    color_data = data

def tf_callback(data):
    global tf_data
    # tf_data = data

    map_frame = data.transforms[0]
    odom_frame = data.transforms[1]

    tf_data = [float(map_frame.transform.translation.x), 
                float(map_frame.transform.translation.y),
                float(odom_frame.transform.translation.x),
                float(odom_frame.transform.translation.x)]
    # print(map_frame.header.frame_id, "//", map_frame.child_frame_id)
    # print("time ==> ", map_frame.header.stamp)
    # print("pose ==> ", map_frame.transform.translation.x, map_frame.transform.translation.y)
    # print("-----------")
    # print(odom_frame.header.frame_id, "//", odom_frame.child_frame_id)
    # print("time ==> ", odom_frame.header.stamp)
    # print("pose ==> ", odom_frame.transform.translation.x, odom_frame.transform.translation.y)

def tp_callback(data):
    # print(data)

    global tp_data
    pose = data.pose.position

    tp_data = [float(pose.x),
                float(pose.y),
                float(pose.z)]

    print(tp_data)


rp.init()
server_node =  rp.create_node('server_node')
server_node.create_subscription(TFMessage, '/tf', tf_callback, 10)
server_node.create_subscription(PoseStamped, '/tracked_pose', tp_callback, 10)
rosListen_thread_flag = True

server_node

def ROSListener(s_node):
    global rosListen_thread_flag

    while rosListen_thread_flag:
        print("test1")
        rp.spin_once(s_node)
        print("test2")

    s_node.destroy_node()
    print("rosListner close")


########################################################################################################


def signal_handler(signal, frame):
    global userListen_thread_flag, imgListen_thread_flag, rosListen_thread_flag

    print('\nsignal_handler called', signal)
    userListen_thread_flag = False
    imgListen_thread_flag = False
    rosListen_thread_flag = False
    
    cv2.destroyAllWindows()
    time.sleep(1)

    for th in threading.enumerate():
        print(th.name)

    server_socket0.close()
    rp.shutdown()
    # remote.close()

    print("Good Bye")
    time.sleep(1)
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)



if __name__=="__main__":
    print("signal setting")
    userListenTread = threading.Thread(target=TCPListener, 
                                        args=(server_socket0, -1))
    imgListenTread = threading.Thread(target=IMGListener, 
                                        args=(udp_listnener_sock, udp_talker_sock, -1))
    rosListenTread = threading.Thread(target=ROSListener, 
                                        args=(server_node,))
    
    userListenTread.start()
    imgListenTread.start()
    rosListenTread.start()

    userListenTread.join()
    imgListenTread.join()
    rosListenTread.join()


