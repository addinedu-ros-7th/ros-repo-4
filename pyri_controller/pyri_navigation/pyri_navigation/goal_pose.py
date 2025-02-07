import cv2
import numpy as np
import heapq
from scipy.spatial import distance
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage
import transforms3d.euler as euler
import time
import math
import re
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor

def load_map(image_path):
    loaded_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    loaded_image = np.flipud(loaded_image)  
    map_data = np.zeros_like(loaded_image)
    map_data[loaded_image == 255] = 1  
    map_data[loaded_image == 127] = 2  
    return map_data

def extract_path_coordinates(map_data):
    path_coordinates = []
    for row_idx, row in enumerate(map_data):
        for col_idx, cell in enumerate(row):
            if cell == 2:
                path_coordinates.append((row_idx, col_idx))
    return path_coordinates

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


class WaypointNavigator(Node):
    def __init__(self, waypoints):
        super().__init__("waypoint_navigator")
        self.publisher = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.subscription = self.create_subscription(TFMessage, "/tf", self.tf_callback, 10)
        self.current_pose = None
        self.waypoints = waypoints

    def tf_callback(self, data: TFMessage):
        if len(data.transforms) < 2:
            return
        
        odom_frame = data.transforms[1]
        self.current_pose = (
            round(odom_frame.transform.translation.x, 2),
            round(odom_frame.transform.translation.y, 2)
        )

    def wait_for_goal_reach(self, goal, tolerance=0.2, max_retries=3, stable_check_time=3, min_movement=0.05):
        last_log_time = time.time()  # 마지막 로그 출력 시간
        retries = 0  # 목표 도달 실패 횟수
        start_time = time.time()  # 목표 도달 시작 시간
        last_dist = None  # 마지막 거리 값 저장
        stable_start_time = None  # 움직임이 거의 없는 상태의 시작 시간

        while rclpy.ok():
            current_time = time.time()

            if self.current_pose is None:
                if current_time - last_log_time >= 1.0:  # 1초마다 로그 출력
                    self.get_logger().warn("Waiting for TF data...")
                    last_log_time = current_time
                rclpy.spin_once(self, timeout_sec=0.1)
                continue  

            dist = math.dist(self.current_pose, goal)

            if current_time - last_log_time >= 1.0:  # 1초마다 거리 출력
                self.get_logger().info(f"Distance to goal: {dist}")
                last_log_time = current_time

            # 목표 도착 확인
            if dist <= tolerance:
                self.get_logger().info(f"Reached goal: {goal}")
                time.sleep(2)  # 목표 도착 후 2초 대기
                return

            # 움직임이 거의 없는 상태 감지
            if last_dist is not None:
                dist_change = abs(last_dist - dist)
                if dist_change < min_movement:  # 변화량이 min_movement 미만이면
                    if stable_start_time is None:
                        stable_start_time = current_time  # 처음으로 정체 상태 감지
                    elif current_time - stable_start_time >= stable_check_time:
                        if retries < max_retries:
                            self.get_logger().warn(f"Goal not reached, retrying... ({retries + 1}/{max_retries})")

                            # 목표 다시 발행
                            pose = PoseStamped()
                            pose.header.frame_id = "map"
                            pose.pose.position.x = goal[0]
                            pose.pose.position.y = goal[1]
                            pose.pose.orientation.w = 1.0  
                            self.publisher.publish(pose)

                            retries += 1
                            stable_start_time = None  
                            start_time = time.time()  
                        else:
                            self.get_logger().error("Failed to reach goal after multiple retries.")
                            return
                else:
                    stable_start_time = None  

            last_dist = dist  
            rclpy.spin_once(self, timeout_sec=0.1)


    def send_goals(self):
        for waypoint in self.waypoints:
            x, y, direction = waypoint
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg() 
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

            self.publisher.publish(pose)
            self.get_logger().info(f"Sent waypoint: {waypoint}")
            self.get_logger().info(f"Pose: x={x}, y={y}, z={pose.pose.orientation.z}, w={pose.pose.orientation.w}")

            self.wait_for_goal_reach((x, y))

class TFListener(Node):
    def __init__(self):
        super().__init__('tf_listener_once')
        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10
        )
        self.tf_data = None

    def tf_callback(self, data: TFMessage):
        if len(data.transforms) < 2:
            return
        
        odom_frame = data.transforms[1]
        self.tf_data = [
            math.trunc(odom_frame.transform.translation.x * 10) / 10, 
            math.trunc(odom_frame.transform.translation.y * 10) / 10
        ]
        
        self.destroy_node()

def main():
    if not rclpy.utilities.ok():
        rclpy.init()

    tf_listener = TFListener()
    rclpy.spin_once(tf_listener)  

    if tf_listener.tf_data is None:
        print("[ERROR] Failed to retrieve TF data.")
        exit(1)

    start_input = tf_listener.tf_data    
    start_param = f"{start_input[0]},{start_input[1]}"

    node = rclpy.create_node("waypoint_loader")
    goal_param = node.declare_parameter("goal", "0.0,0.0").value  
    node.destroy_node()

    print(f"start: {start_param}")
    print(f"goal: {goal_param}")

    try:
        start_input, goal_input = parse_coordinates(start_param), parse_coordinates(goal_param)
    except ValueError as e:
        print(f"[ERROR] {e}")
        exit(1)

    image_path = "/home/zoo/project/pyri/src/test/load_vel.png"
    map_data = load_map(image_path)
    path_coordinates = extract_path_coordinates(map_data)

    start = inverse_transform_coordinates(start_input)
    goal = inverse_transform_coordinates(goal_input)

    threshold_value = 22
    optimal_path = find_shortest_path(start, goal, path_coordinates, map_data, threshold_value)

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

    final_x, final_y = transform_coordinates(goal)
    if waypoints:
        last_x, last_y, last_direction = waypoints[-1]
        waypoints.append([final_x, final_y, last_direction])
    else:
        waypoints.append([final_x, final_y, "F"])

    navigator = WaypointNavigator(waypoints)
    navigator.send_goals()

    for waypoint in waypoints:
        print(f"waypoint: {waypoint}")

    rclpy.spin(navigator)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
