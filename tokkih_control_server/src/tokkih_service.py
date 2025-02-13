# TCP용 import
import threading
import socket
import select
import json

# ROS용 import
import rclpy as rp
from rclpy.node import Node
from turtlesim.msg import Pose
from turtlesim.msg import Color
from std_msgs.msg import Float32
from sensor_msgs.msg import MagneticField
from geometry_msgs.msg import Twist

from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped

# 기타 import
import time
import signal
import sys
import numpy as np
import cv2
import math
from scipy.spatial import distance
import heapq
import ast

import gaejjeonda

MY_IP = "192.168.0.10"

# from Tokkih GUI 수신 소켓 설정
TCP_PORT0 = 8081       # 포트 번호
server_socket0 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket0.bind((MY_IP, TCP_PORT0))
server_socket0.listen()
userListen_thread_flag = True

# Pyri로 부터의 영상 udp 수신 소켓 seoljeong
UDP_PORT0 = 9505
udp_listnener_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_listnener_sock.bind((MY_IP, UDP_PORT0))
imgListen_thread_flag = True

# 영상을 다시 AI서버로 보내줄 소켓 seoljeong
AI_IP = "192.168.0.9"
UDP_PORT1 = 9506
udp_talker_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 영상을 다시 GUI로 보내줄 소켓 seoljeong
GUI_IP = "192.168.0.10"
UDP_PORT2 = 9506

# 전역 변수
pose_data = Pose()
color_data = Color()
tp_data = [0.0, 1.1, 2.2]

pyri1_tf_data = [1.2, 3.4, 0.0, 1.0]
pyri2_tf_data = [0.0, 0.0, 0.0, 1.0]
pyri1_tp_data = [1.2, 3.4, 0.0, 1.0]
pyri2_tp_data = [0.0, 0.0, 0.0, 1.0]
pyri1_battery = 92
pyri2_battery = 57

main_thread_flag = True
pyri1_operating_flag = False
pyri2_operating_flag = False

ai_result_flag = False
ai_result_name = "aiResult"
ai_target_cnt = 0
ai_target_data = None

region_risc_list = []
pyri1_curr_risc = 0
pyri2_curr_risc = 0



temp_img = np.zeros((240,320,3), dtype=np.int8)
_, buffer = cv2.imencode('.jpg', temp_img)
temp_img_b = b'\x02' + buffer.tobytes()



# pyri2 이동 테스트 리스트
test_ls = [(0, 0, 0, 10)]
for _ in range(70):
    test_ls.append((test_ls[-1][0] + 1, test_ls[-1][1], test_ls[-1][2], test_ls[-1][3]))

for _ in range(340):
    test_ls.append((test_ls[-1][0], test_ls[-1][1] + 1, test_ls[-1][2], test_ls[-1][3]))

for _ in range(80):
    test_ls.append((test_ls[-1][0] + 1, test_ls[-1][1], test_ls[-1][2], test_ls[-1][3]))
test_flag = False
test_i = 0
test_increase_flag = True

test_circle_ls = []
for i, val in enumerate(test_ls):
    test_ls[i] = (test_ls[i][0] / 100, test_ls[i][1] / 100, test_ls[i][2] / 10, test_ls[i][3] / 10)

for locs in test_ls:
    # print(locs[0], locs[1])
    continue_flag = False
    for locs2 in test_circle_ls:
        d = ((locs[0] - locs2[0])**2 + (locs[1] - locs2[1])**2) ** (1/2)
        if d < 0.2:
            continue_flag = True
            break
    if continue_flag:
        continue
    test_circle_ls.append((locs[0], locs[1], 0))

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

def get_json_value(data, key, default_value):
    """JSON에서 key 값을 가져오고, 없거나 형식이 다르면 기본값 반환"""
    value = data.get(key, default_value)

    # 리스트인지 확인
    if isinstance(default_value, list) and not isinstance(value, list):
        return [value] if value else []

    # 딕셔너리인지 확인
    if isinstance(default_value, dict) and not isinstance(value, dict):
        return default_value

    return value

def is_tokkih_empty(val):
    if val == []:
        return True
    if val == [[]]:
        return True
    if val == {'results': []}:
        return True
    if val == {'results': [[]]}:
        return True
    if val == [{'results': []}]:
        return True
    if val == [{'results': [[]]}]:
        return True
    if val == [[{'results': []}]]:
        return True
    if val == [[{'results': [[]]}]]:
        return True
    
    return False

def TCPListener(server_socket, set_time):
    global userListen_thread_flag, test_flag, test_i, test_increase_flag
    global pyri1_tp_data, pyri2_tp_data
    global ai_result_flag, ai_result_name, ai_target_cnt, ai_target_data
    global pyri1_curr_risc, pyri2_curr_risc
    
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
                    data = sock.recv(65535).decode("utf-8")
                    if not data:
                        print("No Data")
                        continue

                    # data='[{"bbox": [138.38656616210938, 116.25907897949219, 221.21823120117188, 205.5818634033203], "confidence": 0.522793173789978, "class_id": 56}, {"bbox": [0.0, 176.331298828125, 41.553306579589844, 206.30224609375], "confidence": 0.5043512582778931, "class_id": 45}, {"bbox": [96.24269104003906, 75.76333618164062, 127.29405212402344, 131.20449829101562], "confidence": 0.26554909348487854, "class_id": 41}]'

                    try:
                        parts = data.split("&&")
                        pyri_id = parts[0]
                        if pyri_id not in ["1", "2"]:
                            raise
                        # else:

                        ai_result = ast.literal_eval(parts[-1])

                        ai_target_list = get_json_value(ai_result, "target", [])
                        ai_fire_list = get_json_value(ai_result, "fire", [])
                        ai_position_list = get_json_value(ai_result, "position", [])
                        ai_pose_list = get_json_value(ai_result, "pose", [])
                        ai_identity_list = get_json_value(ai_result, "identity", [])

                        # print()
                        # print("is good?")
                        # print(ai_target_list)
                        # print(ai_fire_list)
                        # print(ai_position_list)
                        # print(ai_pose_list)
                        # print(ai_identity_list)
                        # print()

                        if pyri_id == "1":
                            pyri1_curr_risc = 0
                        else:
                            pyri2_curr_risc = 0
                        if not is_tokkih_empty(ai_fire_list):
                            if ai_fire_list[0][0]["confidence"] >= 0.0:
                                if pyri_id == "1":
                                    pyri1_curr_risc = 1 if ai_fire_list[0][0]["class_id"] == 101 else 2
                                else:
                                    pyri2_curr_risc = 1 if ai_fire_list[0][0]["class_id"] == 101 else 2

                        if not is_tokkih_empty(ai_target_list):
                            if ai_target_list[0][0]["class_name"] == "person" and ai_target_list[0][0]["confidence"] >= 0.0:
                                bbox = ai_target_list[0][0]["bbox"]
                                depth = ai_target_list[0][0]["depth"]

                                pose = None
                                identity = None

                                if not is_tokkih_empty(ai_pose_list):
                                    pose = ai_pose_list[0]["poses"][0]["pose"]
                                if not is_tokkih_empty(ai_identity_list):
                                    age = ai_identity_list[0]["age"]
                                    gender = ai_identity_list[0]["gender"]
                                    identity = (age, gender)

                                # 타겟 발견 신호 뿌리기.
                                ai_result_flag = True
                                ai_result_name = "targetDetect"
                                ai_target_data = (pyri_id, bbox, depth, pose, identity)
                                # continue
                                
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

                            elif parts[0] == "SyncData":
                                messages = ["SyncData"]
                                if test_flag:
                                    pyri1_tp_data = test_ls[test_i]
                                    pyri2_tp_data = test_ls[test_i]
                                    test_i += 1
                                    if test_i >= len(test_ls):
                                        test_i = 0
                                    break_flag = False
                                    loc_x = pyri1_tp_data[0]
                                    loc_y = pyri1_tp_data[1]
                                    for locs2 in region_risc_list:
                                        d = ((loc_x - locs2[0])**2 + (loc_y - locs2[1])**2) ** (1/2)
                                        if d < 0.3 and locs2[2] >= pyri1_curr_risc:
                                            break_flag = True
                                            break
                                    if not break_flag:
                                        region_risc_list.append((loc_x, loc_y, pyri1_curr_risc))
                                    
                                    break_flag = False
                                    loc_x = pyri2_tp_data[0]
                                    loc_y = pyri2_tp_data[1]
                                    for locs2 in region_risc_list:
                                        d = ((loc_x - locs2[0])**2 + (loc_y - locs2[1])**2) ** (1/2)
                                        if d < 0.3 and locs2[2] >= pyri2_curr_risc:
                                            break_flag = True
                                            break
                                    if not break_flag:
                                        region_risc_list.append((loc_x, loc_y, pyri2_curr_risc))
                                    
                                if ai_result_flag:
                                    messages.append(ai_result_name)
                                    messages.append(str(ai_target_data))
                                    ai_target_data = None

                                    ai_result_flag = False
                                else:
                                    messages.append("ideal")
                                    for val in pyri1_tp_data:
                                        messages.append(str(val))
                                    messages.append(str(pyri1_battery))
                                    for val in pyri2_tp_data:
                                        messages.append(str(val))
                                    messages.append(str(pyri2_battery))
                                    messages.append(str(region_risc_list))
                                    

                                sock.send("&&".join(messages).encode('utf-8'))
                            elif parts[0] == "Test2Clicked":

                                test_flag = not test_flag

                                messages = ["Test2Clicked"]
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

def pyri1_tf_callback(data):
    global pyri1_tf_data, region_risc_list, pyri1_operating_flag
    # tf_data = data

    pyri1_operating_flag = True

    map_frame = data.transforms[0]
    odom_frame = data.transforms[1]

    

    loc_x = round(float(odom_frame.transform.translation.x), 2)
    if loc_x > 1.2:
        loc_x = 1.2
    elif loc_x < 0:
        loc_x = 0

    loc_y = round(float(odom_frame.transform.translation.y), 2)
    if loc_y > 3.4:
        loc_y = 3.4
    elif loc_y < 0:
        loc_y = 0
  
    pyri1_tf_data = [loc_x, 
                    loc_y,
                    float(odom_frame.transform.rotation.z),
                    float(odom_frame.transform.rotation.w)]
    
    break_flag = False
    for locs2 in region_risc_list:
        d = ((loc_x - locs2[0])**2 + (loc_y - locs2[1])**2) ** (1/2)
        if d < 0.3 and locs2[2] < pyri1_curr_risc:
            break_flag = True
            break
    if break_flag:
        return
    region_risc_list.append((loc_x, loc_y, pyri1_curr_risc))
    
def pyri2_tf_callback(data):
    global pyri2_tf_data, region_risc_list, pyri2_operating_flag
    # tf_data = data

    pyri2_operating_flag = True

    map_frame = data.transforms[0]
    odom_frame = data.transforms[1]

    loc_x = round(float(odom_frame.transform.translation.x), 2)
    if loc_x > 1.2:
        loc_x = 1.2
    elif loc_x < 0:
        loc_x = 0

    loc_y = round(float(odom_frame.transform.translation.y), 2)
    if loc_y > 3.4:
        loc_y = 3.4
    elif loc_y < 0:
        loc_y = 0
  

    pyri2_tf_data = [loc_x, 
                    loc_y,
                    float(odom_frame.transform.rotation.z),
                    float(odom_frame.transform.rotation.w)]

    break_flag = False
    for locs2 in region_risc_list:
        d = ((loc_x - locs2[0])**2 + (loc_y - locs2[1])**2) ** (1/2)
        if d < 0.3 and locs2[2] < pyri2_curr_risc:
            break_flag = True
            break
    if break_flag:
        return
    region_risc_list.append((loc_x, loc_y, pyri2_curr_risc))

def pyri1_tp_callback(data):
    # print(data)
    global pyri1_tp_data, region_risc_list, pyri1_operating_flag
    pyri1_operating_flag = True

    position = data.pose.position
    orientation = data.pose.orientation

    loc_x = round(float(position.x), 2)
    if loc_x > 1.2:
        loc_x = 1.2
    elif loc_x < 0:
        loc_x = 0
    loc_y = round(float(position.y), 2)
    if loc_y > 3.4:
        loc_y = 3.4
    elif loc_y < 0:
        loc_y = 0

    pyri1_tp_data = [loc_x,
                loc_y,
                float(orientation.z),
                float(orientation.w)]
    
    break_flag = False
    for locs2 in region_risc_list:
        d = ((loc_x - locs2[0])**2 + (loc_y - locs2[1])**2) ** (1/2)
        if d < 0.3 and locs2[2] <= pyri1_curr_risc:
            break_flag = True
            break
    if break_flag:
        return
    region_risc_list.append((loc_x, loc_y, pyri1_curr_risc))
    
def pyri2_tp_callback(data):
    # print(data)
    global pyri2_tp_data, region_risc_list, pyri2_operating_flag
    pyri2_operating_flag = True

    position = data.pose.position
    orientation = data.pose.orientation

    loc_x = round(float(position.x), 2)
    if loc_x > 1.2:
        loc_x = 1.2
    elif loc_x < 0:
        loc_x = 0
    loc_y = round(float(position.y), 2)
    if loc_y > 3.4:
        loc_y = 3.4
    elif loc_y < 0:
        loc_y = 0

    pyri2_tp_data = [loc_x,
                loc_y,
                float(orientation.z),
                float(orientation.w)]
    
    break_flag = False
    for locs2 in region_risc_list:
        d = ((loc_x - locs2[0])**2 + (loc_y - locs2[1])**2) ** (1/2)
        if d < 0.3 and locs2[2] <= pyri2_curr_risc:
            break_flag = True
            break
    if break_flag:
        return
    region_risc_list.append((loc_x, loc_y, pyri2_curr_risc))

def pyri1_battery_callback(data):
    global pyri1_battery
    pyri1_battery = int(data.data)

def pyri2_battery_callback(data):
    global pyri2_battery
    pyri2_battery = int(data.data)

rp.init()
server_node =  rp.create_node('server_node')
# server_node.create_subscription(TFMessage, '/pyri1_tf', pyri1_tf_callback, 10)
server_node.create_subscription(PoseStamped, '/pyri1_tracked_pose', pyri1_tp_callback, 10)
server_node.create_subscription(Float32, '/pyri1_battery', pyri1_battery_callback, 10)

# server_node.create_subscription(TFMessage, '/pyri2_tf', pyri2_tf_callback, 10)
server_node.create_subscription(PoseStamped, '/pyri2_tracked_pose', pyri2_tp_callback, 10)
server_node.create_subscription(Float32, '/pyri2_battery', pyri2_battery_callback, 10)
rosListen_thread_flag = True


def ROSListener(s_node):
    global rosListen_thread_flag

    while rosListen_thread_flag:
        rp.spin_once(s_node, timeout_sec=1)
    s_node.destroy_node()
    print("rosListner close")


########################################################################################################


def signal_handler(signal, frame):
    global userListen_thread_flag, imgListen_thread_flag, rosListen_thread_flag, main_thread_flag

    print('\nsignal_handler called', signal)
    userListen_thread_flag = False
    imgListen_thread_flag = False
    rosListen_thread_flag = False
    main_thread_flag = False
    
    cv2.destroyAllWindows()
    time.sleep(1)

    for th in threading.enumerate()[1:]:
        print(th.name)
        th.join()

    server_socket0.close()
    rp.shutdown()
    # remote.close()

    print("Good Bye")
    time.sleep(1)
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


########################################################################################################
def load_map(image_path):
    loaded_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # start_point = (26, 67)  # 시작점 좌표 (x, y)
    # end_point = (36, 67)  # 끝점 좌표 (x, y)
    # color = 255  # 하얀색 (GRAYSCALE에서는 255가 하얀색)
    # thickness = 1  # 선 두께
    # loaded_image = cv2.line(loaded_image, start_point, end_point, color, thickness)

    loaded_image = np.flipud(loaded_image)  
    map_data = np.zeros_like(loaded_image)
    map_data[loaded_image == 255] = 1  
    map_data[loaded_image == 127] = 2  
    return map_data, loaded_image

def extract_path_coordinates(map_data):
    path_coordinates = []
    for row_idx, row in enumerate(map_data):
        for col_idx, cell in enumerate(row):
            if cell == 2:
                path_coordinates.append((row_idx, col_idx))
    return path_coordinates

origin_min = (6, 6)  
origin_max = (75, 31)
target_min = (0, 0)
target_max = (1.2, 3.4)

scale_x = (target_max[0] - target_min[0]) / (origin_max[1] - origin_min[1])
scale_y = (target_max[1] - target_min[1]) / (origin_max[0] - origin_min[0])

def transform_coordinates(pixel):
    row, col = pixel
    new_x = (col - origin_min[1]) * scale_x  
    new_y = (row - origin_min[0]) * scale_y  
    return round(new_x, 2), round(new_y, 2)

def inverse_transform_coordinates(coord):
    new_x, new_y = coord
    row = round((new_y - target_min[1]) / scale_y + origin_min[0])
    col = round((new_x - target_min[0]) / scale_x + origin_min[1])
    return row, col

def parse_coordinates(input_str):
    try:
        x_str, y_str = input_str.split(",")
        return float(x_str.strip()), float(y_str.strip())
    except ValueError:
        raise ValueError(f"Invalid coordinate format: {input_str}")

def is_wall(coord, map_data):
    row, col = coord
    return not (0 <= row < map_data.shape[0] and 0 <= col < map_data.shape[1]) or map_data[row, col] == 1

def is_path_clear(coord1, coord2, map_data, threshold):
    if distance.cityblock(coord1, coord2) > threshold:
        return False
    r1, c1 = coord1
    r2, c2 = coord2
    num_steps = max(abs(r2 - r1), abs(c2 - c1))
    for step in range(1, num_steps + 1):
        r = int(r1 + step * (r2 - r1) / num_steps)
        c = int(c1 + step * (c2 - c1) / num_steps)
        if is_wall((r, c), map_data):
            return False
    return True
# A*
def find_shortest_path(start, goal, path_coordinates, map_data, threshold=15):
    # print("start", start, "goal", goal, "path", path_coordinates)
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {coord: float("inf") for coord in path_coordinates}
    g_score[start] = 0
    f_score = {coord: float("inf") for coord in path_coordinates}
    f_score[start] = distance.cityblock(start, goal)

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [goal]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for neighbor in path_coordinates:
            if neighbor == current or not is_path_clear(current, neighbor, map_data, threshold):
                continue
            
            tentative_g_score = g_score[current] + distance.cityblock(current, neighbor)
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + distance.cityblock(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []
'''
아 와타시노 코이와 미나미노 
1. 목적지 설정
2. 목적지까지의 Waypoint 설정

3. goal pose Publish
4. tf 확인하며, goal_pose가 제대로 안 간 것 같으면 재발행해줄 것
5. 맵 상태 확인(장애물 유무)하며 이상이 있을 경우 Waypoint 재설정

6. 목적지 도달 후 다음 목적지 설정
'''
class TokkihNavigator(Node):
    def __init__(self):
        super().__init__("tokkih_navigator")
        self.pyri1_goal_pose = self.create_publisher(PoseStamped, "/pyri1_goal_pose", 10)
        self.pyri2_goal_pose = self.create_publisher(PoseStamped, "/pyri2_goal_pose", 10)

        self.pyri1_cmd_vel = self.create_publisher(Twist, "/pyri1_cmd_vel", 10)
        self.pyri2_cmd_vel = self.create_publisher(Twist, "/pyri2_cmd_vel", 10)

        image_path = "/home/hdk/ws/final_project/data/load_map.png"
        self.map_data, self.loaded_image = load_map(image_path)
        # self.path_coordinates = extract_path_coordinates(self.map_data)

        self.nodes_list, self.global_path, self.global_path_list, self.path_coordinates, self.local_end, self.branchs, self.graph = gaejjeonda.gaejjeonda_main(self.map_data, self.loaded_image)
        # self.path_coordinates[self.result_path.index('X')]
        self.node_check_list = [False for _ in self.nodes_list]
        self.node_check_list[0] = True # 시작점 A노드는 이미 방문함

        self.pyri1_prev_dst = 'A'
        self.pyri1_curr_dst = 'A'
        self.pyri1_curr_node = 'A'

        self.pyri2_prev_dst = 0
        self.pyri2_curr_dst = 0
        self.pyri2_curr_node = 0

        self.test_cnt = 0

    def generate_gaejjeonda_waypoint1(self, pyri_prev_dst, pyri_curr_dst):
        print(pyri_prev_dst, pyri_curr_dst)
        waypoints_char, _ = gaejjeonda.astar(self.graph, pyri_prev_dst, pyri_curr_dst)

        waypoints_pixel = []
        for c in waypoints_char:
            waypoints_pixel.append(self.path_coordinates[self.nodes_list.index(c)])

        waypoint_coord = []
        for pixel in waypoints_pixel:
            waypoint_coord.append(transform_coordinates(pixel))

        if not waypoint_coord:
            return None
        waypoints = []
        for i in range(len(waypoint_coord)-1):
            x1, y1 = waypoint_coord[i]
            x2, y2 = waypoint_coord[i + 1]
            if x2 > x1:
                direction = "F"
                if y2 > y1:
                    direction = "L"
                elif y2 < y1:
                    direction = "R"  
            elif x2 < x1:
                direction = "B"
                if y2 > y1:
                    direction = "L"
                elif y2 < y1:
                    direction = "R"      
            else:
                if y2 > y1:
                    direction = "L"
                elif y2 < y1:
                    direction = "R"  
            waypoints.append([y1, x1, direction])

        final_x, final_y = waypoint_coord[-1]
        if waypoints:
            last_x, last_y, last_direction = waypoints[-1]
            waypoints.append([final_y, final_x, last_direction])
        else:
            waypoints.append([final_y, final_x, "F"])
        return waypoints
    
    def generate_gaejjeonda_waypoint2(self, pyri_prev_dst, middle_dst, pyri_curr_dst):
        start_input = transform_coordinates(self.path_coordinates[self.nodes_list.index(pyri_prev_dst)])
        goal_input = transform_coordinates(self.path_coordinates[self.nodes_list.index(middle_dst)])
        waypoint1 = self.generate_waypoints(start_input, goal_input)

        waypoint2 = self.generate_gaejjeonda_waypoint1(middle_dst, pyri_curr_dst)

        return waypoint1 + waypoint2

    def generate_destination(self, pyri_num):
        if pyri_num == 1:
            # 모든 노드를 확인했다면, 시작점으로 복main_thread_flag귀
            if not False in self.node_check_list:
                self.pyri1_curr_dst = 0
                s = (pyri1_tp_data[0], pyri1_tp_data[1])
                waypoint = self.generate_waypoints(start_input=s, goal_input=(0.0, 0.0))
                return waypoint
            
            # 다음 목적 노드 설정
            """
            a* == generate_waypoint
            a*1 == 주원님이 짠 a*
            a*2 == 내가 짠 a*

              1. 현재 노드가 local_end인지, branchs인지 확인
              
              현재 노드가 branchs라면 
              2-1. global_path에서 현재 노드 다음에 잇는 곳을 모두 방문했는 지 확인, 안한 곳이 있다면 그곳이 목적지(a*2)
              2-2. 연결된 노드를 모두 방문했다면,
              2-3. global_path에서 방문하지 않은 다음 branch의 이전 브랜치까지의 경로(a*1) + 해당 branch까지의 경로 (a*2)
              
              현재 노드가 local_end라면
              3-1. 이전 branch에 연결된 local_end를 모두 방문했는 지 확인, 안된 곳이 있다면, 이전 branch로 이동(a*1)
              3-2. 모두 확인 됐다면, global_path에서 다음 방문하지 않은 branch를 확인, 

              3-4. 방문하지 않은 branch를 확인했다면 이전 branch에서 해당 branch사이의 (global_path_list에서 branch-branch를 확인) 노드를 모두 방문했는 지 확인,
              3-5. 모두 방문했다면 해당 branch로 바로 이동(a*1)
              3-6. 모두 방문한 게 아니라면, 해당 브랜치의 이전 브랜치까지의 경로(a1) + 해당 브랜치까지의 경로 (a*2)

              3-3. 없다면 모두 그 사이에 모두 방문 됐다는 거임, start node로 이동(a*1)
            """
            curr_node_char = self.nodes_list[self.pyri1_curr_node]

            # 현재 노드가 branchs라면 
            if curr_node_char in self.branchs:
                # 2-1. global_path에서 현재 노드 다음에 잇는 곳을 모두 방문했는 지 확인, 안한 곳이 있다면 그곳이 목적지(a*2)
                for i, node in enumerate(self.global_path):
                    if node == curr_node_char:
                        # 현 브랜치 다음에 잇는 곳 확인
                        # 안한 곳이 있다면 그곳이 목적지(a*2)
                        if not self.node_check_list[self.nodes_list.index(self.global_path[i+1])]:
                            self.pyri1_prev_dst = self.pyri1_curr_dst
                            self.pyri1_curr_dst = self.nodes_list.index(self.global_path[i+1])
                            # a*2 and return waypoints
                            waypoint = self.generate_gaejjeonda_waypoint1(self.pyri1_prev_dst, self.pyri1_curr_dst)
                            return waypoint


                # 2-2. 연결된 노드를 모두 방문했다면,
                # 2-3. global_path에서 방문하지 않은 다음 branch의 이전 브랜치까지의 경로(a*1) + 해당 branch까지의 경로 (a*2)
                prev_branch = curr_node_char
                for node in self.global_path:
                    # 방문하지 않은 다음 branch
                    if node in self.branch:
                        prev_branch = node
                        if not self.node_check_list[self.nodes_list.index(node)]:
                            self.pyri1_prev_dst = self.pyri1_curr_dst
                            self.pyri1_curr_dst = self.nodes_list.index(node)
                            prev_branch = self.nodes_list.index(prev_branch)
                            # (prev_dst2prev_branch)a*1 + (prev_branch2curr_dst)a*2 and return waypoints
                            waypoint = self.generate_gaejjeonda_waypoint2(self.pyri1_prev_dst, prev_branch, self.pyri1_curr_dst)
                            return waypoint

            # 현재 노드가 local_end라면
            elif self.nodes_list[self.pyri1_curr_node] in self.local_end:
                prev_branch = None
                if curr_node_char != 'A':
                    # 이전 브랜치 확인
                    # 이전 branch가 있는 지를 먼저 확인
                    # 3-1. 이전 branch에 연결된 local_end를 모두 방문했는 지 확인, 안된 곳이 있다면, 이전 branch로 이동(a*1)
                    for i, node in enumerate(self.global_path):
                        if node == curr_node_char:
                            prev_branch == self.global_path[i-1]
                    if prev_branch is None:
                        print("뭔 소리일까 또")
                        return None

                    # 이전 branch에 연결된 local_end를 모두 방문했는 지 확인
                    for i, node in enumerate(self.global_path):
                      if node == prev_branch:
                          # global_path에서 다음에 잇는 곳 확인, 다음에 있는 곳이 local_end이면서 방문하지 않았는 지를 확인
                          if self.global_path[i+1] in self.local_end:
                              if not self.node_check_list[self.nodes_list.index(self.global_path[i+1])]:
                                # 안된 곳이 있다면, 이전 branch로 바로 이동(a*1)
                                self.pyri1_prev_dst = self.pyri1_curr_dst
                                self.pyri1_curr_dst = prev_branch
                                start_input = transform_coordinates(self.path_coordinates[self.nodes_list.index(self.pyri1_prev_dst)])
                                goal_input = transform_coordinates(self.path_coordinates[self.nodes_list.index(self.pyri1_curr_dst)])
                                waypoint = self.generate_waypoints(start_input=start_input, goal_input=goal_input)
                                return waypoint

                    # 3-2. 모두 확인 됐다면, global_path에서 다음 방문하지 않은 branch를 확인
                    for i, node in enumerate(self.global_path):
                        # 방문하지 않은 다음 branch
                        if node in self.branch:
                            if not self.node_check_list[self.nodes_list.index(node)]:
                                # 3-4. 방문하지 않은 branch를 확인했다면 
                                # 이전 branch에서 해당 branch사이의 (global_path_list에서 branch-branch를 확인) 
                                # 노드를 모두 방문했는 지 확인,
                                dst_branch = node
                                for path in self.global_path_list:
                                    if path[0] == prev_branch and path[-1] in self.branchs:
                                        for node_in_path in path:
                                            if not self.node_check_list[self.nodes_list.index(node_in_path)]:
                                                # 3-6. 모두 방문한 게 아니라면, 해당 브랜치의 이전 브랜치까지의 경로(a1) + 해당 브랜치까지의 경로 (a*2)
                                                self.pyri1_prev_dst = self.pyri1_curr_dst
                                                self.pyri1_curr_dst = dst_branch
                                                middle_dst = self.branchs[self.branchs.inex(dst_branch) - 1]
                                                # (prev_dst2prev_branch)a*1 + (prev_branch2curr_dst)a*2 and return waypoints
                                                waypoint = self.generate_gaejjeonda_waypoint2(self.pyri1_prev_dst, middle_dst, self.pyri1_curr_dst)
                                                return waypoint

                                    if path[-1] == dst_branch:
                                        break
                                # 3-5. 모두 방문했다면 해당 branch로 이동(a*1)
                                s = (pyri1_tp_data[0], pyri1_tp_data[1])
                                self.pyri1_prev_dst = self.pyri1_curr_dst
                                self.pyri1_curr_dst = dst_branch
                                start_input = transform_coordinates(self.path_coordinates[self.nodes_list.index(self.pyri1_prev_dst)])
                                goal_input = transform_coordinates(self.path_coordinates[self.nodes_list.index(self.pyri1_curr_dst)])
                                waypoint = self.generate_waypoints(start_input=start_input, goal_input=goal_input)
                                return waypoint
                else:
                    self.pyri1_prev_dst = self.pyri1_curr_dst
                    self.pyri1_curr_dst = 'B'
                    waypoint = self.generate_gaejjeonda_waypoint1(self.pyri1_prev_dst, self.pyri1_curr_dst)
                    return waypoint
            else:
                print("curr_node Error")


            self.pyri1_curr_dst = 0
            s = (pyri1_tp_data[0], pyri1_tp_data[1])
            waypoint = self.generate_waypoints(start_input=s, goal_input=(0.0, 0.0))
            return waypoint


        elif pyri_num == 2:
            if not False in self.node_check_list:
                self.pyri2_curr_dst = 0
                s = (pyri2_tp_data[0], pyri2_tp_data[1])
                waypoint = self.generate_waypoints(start_input=s, goal_input=(0.0, 0.0))
                return waypoint
            curr_node_char = self.nodes_list[self.pyri2_curr_node]

            if curr_node_char in self.branchs:
                for i, node in enumerate(self.global_path):
                    if node == curr_node_char:
                        if not self.node_check_list[self.nodes_list.index(self.global_path[i+1])]:
                            self.pyri2_prev_dst = self.pyri2_curr_dst
                            self.pyri2_curr_dst = self.nodes_list.index(self.global_path[i+1])
                            waypoint = self.generate_gaejjeonda_waypoint1(self.pyri2_prev_dst, self.pyri2_curr_dst)
                            return waypoint

                prev_branch = curr_node_char
                for node in self.global_path:
                    # 방문하지 않은 다음 branch
                    if node in self.branch:
                        prev_branch = node
                        if not self.node_check_list[self.nodes_list.index(node)]:
                            self.pyri2_prev_dst = self.pyri2_curr_dst
                            self.pyri2_curr_dst = self.nodes_list.index(node)
                            prev_branch = self.nodes_list.index(prev_branch)
                            waypoint = self.generate_gaejjeonda_waypoint2(self.pyri2_prev_dst, prev_branch, self.pyri2_curr_dst)
                            return waypoint

            elif self.nodes_list[self.pyri2_curr_node] in self.local_end:
                prev_branch = None
                if curr_node_char != 'A':
                    for i, node in enumerate(self.global_path):
                        if node == curr_node_char:
                            prev_branch == self.global_path[i-1]
                    if prev_branch is None:
                        print("뭔 소리일까 또")
                        return None
                    
                    for i, node in enumerate(self.global_path):
                      if node == prev_branch:
                          if self.global_path[i+1] in self.local_end:
                              if not self.node_check_list[self.nodes_list.index(self.global_path[i+1])]:
                                self.pyri2_prev_dst = self.pyri2_curr_dst
                                self.pyri2_curr_dst = prev_branch
                                start_input = transform_coordinates(self.path_coordinates[self.nodes_list.index(self.pyri2_prev_dst)])
                                goal_input = transform_coordinates(self.path_coordinates[self.nodes_list.index(self.pyri2_curr_dst)])
                                waypoint = self.generate_waypoints(start_input=start_input, goal_input=goal_input)
                                return waypoint

                    for i, node in enumerate(self.global_path):
                        if node in self.branch:
                            if not self.node_check_list[self.nodes_list.index(node)]:
                                dst_branch = node
                                for path in self.global_path_list:
                                    if path[0] == prev_branch and path[-1] in self.branchs:
                                        for node_in_path in path:
                                            if not self.node_check_list[self.nodes_list.index(node_in_path)]:
                                                self.pyri2_prev_dst = self.pyri2_curr_dst
                                                self.pyri2_curr_dst = dst_branch
                                                middle_dst = self.branchs[self.branchs.inex(dst_branch) - 1]
                                                waypoint = self.generate_gaejjeonda_waypoint2(self.pyri2_prev_dst, middle_dst, self.pyri2_curr_dst)
                                                return waypoint

                                    if path[-1] == dst_branch:
                                        break
                                s = (pyri2_tp_data[0], pyri2_tp_data[1])
                                self.pyri2_prev_dst = self.pyri2_curr_dst
                                self.pyri2_curr_dst = dst_branch
                                start_input = transform_coordinates(self.path_coordinates[self.nodes_list.index(self.pyri2_prev_dst)])
                                goal_input = transform_coordinates(self.path_coordinates[self.nodes_list.index(self.pyri2_curr_dst)])
                                waypoint = self.generate_waypoints(start_input=start_input, goal_input=goal_input)
                                return waypoint
                else:
                    self.pyri2_prev_dst = self.pyri2_curr_dst
                    self.pyri2_curr_dst = 'B'
                    waypoint = self.generate_gaejjeonda_waypoint1(self.pyri2_prev_dst, self.pyri2_curr_dst)
                    return waypoint
            else:
                print("curr_node Error")


            self.pyri2_curr_dst = 0
            s = (pyri2_tp_data[0], pyri2_tp_data[1])
            waypoint = self.generate_waypoints(start_input=s, goal_input=(0.0, 0.0))
            return waypoint

    def generate_waypoints(self, start_input, goal_input):
        start = inverse_transform_coordinates(start_input)
        goal = inverse_transform_coordinates(goal_input)

        threshold_value = 22
        optimal_path = find_shortest_path(start, goal, self.path_coordinates, self.map_data, threshold_value)
        # print("hdk", optimal_path)
        if not optimal_path:
            return None
        waypoints = []
        for i in range(len(optimal_path) - 1):
            x1, y1 = transform_coordinates(optimal_path[i])
            x2, y2 = transform_coordinates(optimal_path[i + 1])
            if x2 > x1:
                direction = "F"
                if y2 > y1:
                    direction = "L"
                elif y2 < y1:
                    direction = "R"  
            elif x2 < x1:
                direction = "B"
                if y2 > y1:
                    direction = "L"
                elif y2 < y1:
                    direction = "R"      
            else:
                if y2 > y1:
                    direction = "L"
                elif y2 < y1:
                    direction = "R"  
            waypoints.append([x1, y1, direction])

        final_x, final_y = transform_coordinates(goal)
        if waypoints:
            last_x, last_y, last_direction = waypoints[-1]
            waypoints.append([final_x, final_y, last_direction])
        else:
            waypoints.append([final_x, final_y, "F"])


        return waypoints

    def check_waypoints(self, waypoints):
        if waypoints is None:
            return True
        
        return False

    def check_moving(self, pyri_num):
        return True
        pass

    def check_arrive_waypoint(self, pyri_num, waypoints, waypoint_i, tolerance = 0.3):
        curr_loc = ()

        dist = 0
        if pyri_num == 1:
            curr_loc = ((pyri1_tp_data[0], pyri1_tp_data[1]))
        elif pyri_num == 2:
            curr_loc = ((pyri2_tp_data[0], pyri2_tp_data[1]))
        dst_loc = (waypoints[waypoint_i][0], waypoints[waypoint_i][1])
        dist = math.dist(curr_loc, dst_loc)

        if dist <= tolerance:
            # print(f"Reached goal: {waypoints[waypoint_i]}")
            return True, dist
        return False, dist
    
    def move_robot(self, linear_x, cmd_vel_pub, duration=2.0):
        twist = Twist()
        twist.linear.x = linear_x
        start_time = time.time()
        while time.time() - start_time < duration:
            cmd_vel_pub.publish(twist)
            time.sleep(0.1)  # 0.1초마다 속도 명령을 반복 전송
        # self.get_logger().info(f":로켓: Moving {'forward' if linear_x > 0 else 'backward'}...")
        self.stop_robot(cmd_vel_pub)  # 이동이 끝나면 멈춤
    def stop_robot(self, cmd_vel_pub):
        twist = Twist()
        start_time = time.time()
        stop_duration = 4
        while time.time() - start_time < stop_duration:
            cmd_vel_pub.publish(twist)
            time.sleep(0.1)
        # self.get_logger().info(":팔각형_기호: Robot stopped.")

    def send_goal(self, pyri_num, pyri_waypoints, pyri_waypoint_i):
        pyri_goal_pose = self.pyri1_goal_pose if pyri_num == 1 else self.pyri2_goal_pose

        pose = PoseStamped()
        pose.header.frame_id = "map"
        print(pyri_waypoints, pyri_waypoint_i)
        pose.pose.position.x = pyri_waypoints[pyri_waypoint_i][0]
        pose.pose.position.y = pyri_waypoints[pyri_waypoint_i][1]
        orientations = {
            "F": (0.0, 1.0),
            "B": (-1.0, 0.0),
            "L": (0.7, 0.7),
            "R": (-0.7, 0.7),
        }
        pose.pose.orientation.z, pose.pose.orientation.w = orientations.get(pyri_waypoints[pyri_waypoint_i][2], (0.0, 1.0))

        pyri_goal_pose.publish(pose)
        
        # self.wait_for_goal_reach(pyri_num, pyri_waypoints, pyri_goal_pose, (pyri_waypoints[pyri_waypoint_i][0], pyri_waypoints[pyri_waypoint_i][1]))

def ROSMainThread():
    global main_thread_flag, pyri1_tp_data, pyri2_tp_data
    tokkih_nav = TokkihNavigator()

    PYRI1 = 1
    PYRI2 = 2

    pyri1_destination = None
    pyri2_destination = None

    pyri1_waypoints = None
    pyri2_waypoints = None

    pyri1_waypoint_i = 0
    pyri2_waypoint_i = 0

    cnt = 0

    pyri1_prev_dist = None
    pyri1_prev_time = time.time()
    pyri1_notmove_cnt = 0
    pyri1_goback_flag = False
    pyri1_arrive_wait_flag = False
    pyri1_first_moving_flag = True

    pyri2_prev_dist = None
    pyri2_prev_time = time.time()
    pyri2_notmove_cnt = 0
    pyri2_goback_flag = False
    pyri2_arrive_wait_flag = False
    pyri2_first_moving_flag = True

    print("여긴 왔는데")

    while main_thread_flag:
        # print("나 살아있다.", main_thread_flag, pyri1_operating_flag,pyri2_operating_flag)
        if pyri1_operating_flag:
            pyri1_curr_time = time.time()
            # 다음 목적지가 없다면 목적지 생성
            if pyri1_waypoints is None:
                pyri1_waypoints = tokkih_nav.generate_destination(PYRI1)
                pyri1_waypoint_i = 1
                if pyri1_waypoints:
                    print("파이리1 목적지 생성!")
                    print(pyri1_waypoints)
                    # tokkih_nav.send_goal(PYRI1, pyri1_waypoints, pyri1_waypoint_i)
                else:
                    print("파이리1 waypoint 생성 Error")
                continue

            # waypoint들에 도착할 때마다 다음 waypoint 전달 or
            # 목적지 도착 확인 후 변수 초기화
            pyri1_arrive_check_flag, pyri1_dist = tokkih_nav.check_arrive_waypoint(PYRI1, pyri1_waypoints, pyri1_waypoint_i)
            if pyri1_arrive_check_flag:
                if pyri1_arrive_wait_flag:
                    if pyri1_curr_time - pyri1_arrive_wait_prev_time >= 0.1:
                        print("파이리1가 waypoint에 도착했습니다.")
                        pyri1_prev_dist = None
                        pyri1_notmove_cnt = 0
                        pyri1_prev_time = pyri1_curr_time
                        pyri1_waypoint_i += 1
                        pyri1_arrive_wait_flag = False
                        if pyri1_waypoint_i >= len(pyri1_waypoints):
                            print("파이리1 목적지 도착")
                            pyri1_destination = None
                            pyri1_waypoints = None
                            pyri1_waypoint_i = 0
                        else:
                            print("파이리1에 다음 목적지를 전달합니다.", pyri1_waypoint_i)
                            # tokkih_nav.send_goal(PYRI1, pyri1_waypoints, pyri1_waypoint_i)
                else:
                    # print("일단 멈추게는 해볼게", pyri1_curr_time)
                    pyri1_arrive_wait_flag = True
                    pyri1_arrive_wait_prev_time  = pyri1_curr_time
                    pyri1_notmove_cnt = 0
            # else:
            #     # 파이리가 움직이지 않고 있다면
            #     if pyri1_curr_time - pyri1_prev_time > 3.0:
            #         pyri1_prev_time = pyri1_curr_time
            #         if pyri1_prev_dist is not None:
            #             dist_change = abs(pyri1_prev_dist - pyri1_dist)
            #             min_movement = 0.01
            #             if dist_change < min_movement:
            #                 if pyri1_notmove_cnt < 5 or pyri1_first_moving_flag:
            #                     pyri1_notmove_cnt += 1
            #                     print("goal_pose 재발행", pyri1_waypoints[pyri1_waypoint_i], pyri1_prev_dist, pyri1_dist)

            #                     print(pyri1_waypoints[pyri1_waypoint_i])
            #                     tokkih_nav.send_goal(PYRI1, pyri1_waypoints, pyri1_waypoint_i)
            #                 else:
                                
            #                     if pyri1_goback_flag:
            #                         print("이래도 못가네;; 후진 주행")
            #                         tokkih_nav.move_robot(-0.2, tokkih_nav.pyri1_cmd_vel, duration=3.0)  # 후진 후 멈춤
            #                         pyri1_notmove_cnt = 0
            #                         pyri1_goback_flag = False
            #                     else:
            #                         print("일단 멈추게 해")
            #                         start = inverse_transform_coordinates((pyri1_tp_data[0], pyri1_tp_data[1]))
            #                         x1, y1 = transform_coordinates(start)
            #                         print(x1, y1)
            #                         tokkih_nav.send_goal(PYRI1, [(x1, y1, "L")], 0)
            #                         pyri1_goback_flag = True
            #             else:
            #                 pyri1_first_moving_flag = False

            #         pyri1_prev_dist = pyri1_dist

##############################################################################################################

        if pyri2_operating_flag:
            pyri2_curr_time = time.time()
            # 다음 목적지가 없다면 목적지 생성
            if pyri2_waypoints is None:
                pyri2_waypoints = tokkih_nav.generate_destination(PYRI2)

                s = (pyri1_tp_data[0], pyri1_tp_data[1])
                pyri2_waypoints = tokkih_nav.generate_waypoints(start_input=s, goal_input=(0.0, 1.04))
                pyri2_waypoint_i = 1

                if pyri2_waypoints:
                    print("파이리2 목적지 생성!")
                    print(pyri2_waypoints)
                    # tokkih_nav.send_goal(PYRI2, pyri2_waypoints, pyri2_waypoint_i)
                else:
                    print("waypoint 생성 Error")
                continue

            # waypoint들에 도착할 때마다 다음 waypoint 전달 or
            # 목적지 도착 확인 후 변수 초기화
            pyri2_arrive_check_flag, pyri2_dist = tokkih_nav.check_arrive_waypoint(PYRI2, pyri2_waypoints, pyri2_waypoint_i)
            if pyri2_arrive_check_flag:
                if pyri2_arrive_wait_flag:
                    if pyri2_curr_time - pyri2_arrive_wait_prev_time >= 0.1:
                        print("파이리2가 waypoint에 도착했습니다.")
                        pyri2_prev_dist = None
                        pyri2_notmove_cnt = 0
                        pyri2_prev_time = pyri2_curr_time
                        pyri2_waypoint_i += 1
                        pyri2_arrive_wait_flag = False
                        pyri2_goback_flag = False
                        if pyri2_waypoint_i >= len(pyri2_waypoints):
                            print("파이리2 목적지 도착")
                            pyri2_destination = None
                            pyri2_waypoints = None
                            pyri2_waypoint_i = 0
                        else:
                            print("파이리2에 다음 목적지를 전달합니다.", pyri2_waypoint_i)
                            # tokkih_nav.send_goal(PYRI2, pyri2_waypoints, pyri2_waypoint_i)
                else:
                    # print("일단 멈추게는 해볼게", pyri2_curr_time)
                    pyri2_arrive_wait_flag = True
                    pyri2_arrive_wait_prev_time  = pyri2_curr_time
                    pyri2_notmove_cnt = 0
            # else:
            #     # 파이리가 움직이지 않고 있다면
            #     if pyri2_curr_time - pyri2_prev_time > 3.0:
            #         pyri2_prev_time = pyri2_curr_time
            #         if pyri2_prev_dist is not None:
            #             dist_change = abs(pyri2_prev_dist - pyri2_dist)
            #             min_movement = 0.01
            #             if dist_change < min_movement:
            #                 if pyri2_notmove_cnt < 5 or pyri2_first_moving_flag:
            #                     pyri2_notmove_cnt += 1
            #                     print("goal_pose 재발행", pyri2_waypoints[pyri2_waypoint_i], pyri2_prev_dist, pyri2_dist)

            #                     print(pyri2_waypoints[pyri2_waypoint_i])
            #                     tokkih_nav.send_goal(PYRI2, pyri2_waypoints, pyri2_waypoint_i)
            #                 else:
                                
            #                     if pyri2_goback_flag:
            #                         print("이래도 못가네;; 후진 주행")
            #                         tokkih_nav.move_robot(-0.2, tokkih_nav.pyri2_cmd_vel, duration=3.0)  # 후진 후 멈춤
            #                         pyri2_notmove_cnt = 0
            #                         pyri2_goback_flag = False
            #                     else:
            #                         print("일단 멈추게 해")
            #                         start = inverse_transform_coordinates((pyri2_tp_data[0], pyri2_tp_data[1]))
            #                         x1, y1 = transform_coordinates(start)
            #                         print(x1, y1)
            #                         tokkih_nav.send_goal(PYRI2, [(x1, y1, "L")], 0)
            #                         pyri2_goback_flag = True
            #             else:
            #                 pyri2_first_moving_flag = False

            #         pyri2_prev_dist = pyri2_dist


                

    print("good bye pyris")
    del tokkih_nav

                    
            

########################################################################################################

if __name__=="__main__":
    print("signal setting")
    userListenTread = threading.Thread(target=TCPListener, 
                                        args=(server_socket0, -1))
    imgListenTread = threading.Thread(target=IMGListener, 
                                        args=(udp_listnener_sock, udp_talker_sock, -1))
    rosListenTread = threading.Thread(target=ROSListener, 
                                        args=(server_node,))
    
    rosMainThread = threading.Thread(target=ROSMainThread)
    
    userListenTread.start()
    imgListenTread.start()
    rosListenTread.start()
    rosMainThread.start()

    userListenTread.join()
    imgListenTread.join()
    rosListenTread.join()
    rosMainThread.join()

    print("Hello~~Hdk")


