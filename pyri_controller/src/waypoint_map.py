import cv2
import numpy as np
import heapq
from scipy.spatial import distance
import rclpy
from rclpy.node import Node
from nav2_msgs.action import FollowWaypoints
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
import re


# ========== 1. 이미지 로드 및 좌표 변환 ==========
def load_map(image_path):
    loaded_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    loaded_image = np.flipud(loaded_image)  # 좌우 반전 (y축 유지)
    map_data = np.zeros_like(loaded_image)
    map_data[loaded_image == 255] = 1  # 벽
    map_data[loaded_image == 127] = 2  # 이동 경로
    return map_data

def extract_path_coordinates(map_data):
    path_coordinates = []
    for row_idx, row in enumerate(map_data):
        for col_idx, cell in enumerate(row):
            if cell == 2:
                path_coordinates.append((row_idx, col_idx))
    return path_coordinates

# 좌표 변환 설정
origin_min = (6, 7)  
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
    match = re.match(r"\(([-+]?[0-9]*\.?[0-9]+),\s*([-+]?[0-9]*\.?[0-9]+)\)", input_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    else:
        raise ValueError("Invalid coordinate format. Please use (x,y)")

# ========== 2. 최적 경로 탐색 ==========
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

def find_shortest_path(start, goal, path_coordinates, map_data, threshold=15):
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

# ========== 3. ROS2 주행 명령 전송 ==========
class WaypointNavigator(Node):
    def __init__(self, waypoints):
        super().__init__("follow_waypoints_client")
        self.client = ActionClient(self, FollowWaypoints, "/follow_waypoints")
        self.goal_msg = FollowWaypoints.Goal()
        self.goal_msg.poses = [self.create_pose(wp) for wp in waypoints]

    def create_pose(self, waypoint):
        x, y, direction = waypoint
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = x
        pose.pose.position.y = y

        orientations = {
            "F": (0.0, 1.0),
            "B": (-1.0, 0.0),
            "L": (0.7, 0.7),
            "R": (-0.7, 0.7),
        }
        pose.pose.orientation.z, pose.pose.orientation.w = orientations.get(direction, (0.0, 1.0))
        return pose

    def send_goal(self):
        self.client.wait_for_server()
        self.future = self.client.send_goal_async(self.goal_msg)
        self.future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            print("Goal rejected")
            return
        print("Goal accepted")


# ========== 실행 코드 ==========
if __name__ == "__main__":
    # 사용자 입력
    start_input = parse_coordinates(input("start point: "))
    goal_input = parse_coordinates(input("goal point: "))

    # 맵 데이터 로드
    image_path = "/home/zoo/project/pyri/src/test/load_test2.png"
    map_data = load_map(image_path)
    path_coordinates = extract_path_coordinates(map_data)

    # 좌표 변환
    start = inverse_transform_coordinates(start_input)
    goal = inverse_transform_coordinates(goal_input)

    # 최적 경로 계산
    threshold_value = 22
    optimal_path = find_shortest_path(start, goal, path_coordinates, map_data, threshold_value)

    # 최적 경로 변환
    direction_list = {"F": "B", "B": "F", "L": "R", "R": "L"}
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

    # 마지막 좌표가 goal과 일치하도록 추가
    final_x, final_y = transform_coordinates(goal)
    if waypoints:
        last_x, last_y, last_direction = waypoints[-1]
        waypoints.append([final_x, final_y, direction_list[last_direction]])
    else:
        waypoints.append([final_x, final_y, "F"])

    rclpy.init()
    navigator = WaypointNavigator(waypoints)
    navigator.send_goal()

    for waypoint in waypoints:
        print(f"waypoint: {waypoint}")

    rclpy.spin(navigator)