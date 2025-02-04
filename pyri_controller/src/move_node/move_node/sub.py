import rclpy
from rclpy.node import Node
from move_msgs.msg import DestinationMove
from move_msgs.msg import StartGoal
import argparse
import subprocess
import ast

def parse_tuple(s):
    try:
        return ast.literal_eval(s)
    except:
        raise argparse.ArgumentTypeError(f"Invalid tuple: {s}")

class ArgSubscriber(Node):
    def __init__(self):
        super().__init__('arg_subscriber')
        # self.subscription = self.create_subscription(
        #     DestinationMove,
        #     '/destination_move',
        #     self.listener_callback,
        #     10
        # )
        # self.subscription
        
        self.subscription = self.create_subscription(
            StartGoal,
            '/start_goal',
            self.startgoal_callback,
            10
        )

    def startgoal_callback(self, msg):
        # start_x, start_y = msg.start.x, msg.start.y
        # goal_x, goal_y = msg.goal.x, msg.goal.y

        # print(start_x, start_y, goal_x, goal_y)
        print(type(msg.start_x))

        parser = argparse.ArgumentParser(description='Move the robot')
        parser.add_argument('--start_x', type=float, default=msg.start_x)
        parser.add_argument('--start_y', type=float, default=msg.start_y)
        parser.add_argument('--goal_x', type=float, default=msg.goal_x)
        parser.add_argument('--goal_y', type=float, default=msg.goal_y)
        args = parser.parse_args()

        print("args :", args)

        try:
            print("try~")
            result = subprocess.run(
                [
                    'python3',
                    "./src/move_node/move_node/waypoints.py",
                    '--start_x', str(args.start_x),
                    '--start_y', str(args.start_y),
                    '--goal_x', str(args.goal_x),
                    '--goal_y', str(args.goal_y)
                ],
                    check=True,
                    capture_output=True,
                    text=True
            )
            print("result :", result)
            print("🔹 실행 결과:")
            print(result.stdout)
        except Exception as e:
            print(f"⚠ 오류 발생: {e}")
            print(e.stderr)

    def listener_callback(self, msg):
        parser = argparse.ArgumentParser(description='Move the robot')
        # parser.add_argument("waypoints.py")
        parser.add_argument('--x', type=float, default=msg.dest_x)
        parser.add_argument('--y', type=float, default=msg.dest_y)
        parser.add_argument('--z', type=float, default=msg.dest_z)
        parser.add_argument('--angle', type=str, default=msg.angle)
        args = parser.parse_args()

        print(args)

        try:
            result = subprocess.run(
                [
                    'python3',
                    "./src/move_node/move_node/waypoints.py",
                    '--x', str(args.x),
                    '--y', str(args.y),
                    '--z', str(args.z),
                    '--angle', args.angle
                ],
                    capture_output=True,
                    text=True
            )
            print(result)
            print("🔹 실행 결과:")
            print(result.stdout)
        except Exception as e:
            print(f"⚠ 오류 발생: {e}")
            print(e.stderr)

def main(args=None):
    rclpy.init(args=args)
    arg_subscriber = ArgSubscriber()
    rclpy.spin(arg_subscriber)
    arg_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()