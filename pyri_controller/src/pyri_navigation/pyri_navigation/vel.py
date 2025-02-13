import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor

class StraightBackward(Node):
    def __init__(self):
        super().__init__('straight_backward')
        self.declare_parameter('time', 0.0).value
        self.time_value = float(self.get_parameter('time').value)
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(self.time_value, self.run)
        self.movement_stage = 0  
        
    def move_robot(self, linear_x):
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info(f"Moving {'forward' if linear_x > 0 else 'backward'}")

    def stop_robot(self):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info("Robot stopped.")

    def run(self):
        if self.movement_stage == 0:
            self.move_robot(0.2)
            self.movement_stage = 1
        elif self.movement_stage == 1:
            self.move_robot(-0.2)
            self.movement_stage = 2
        else:
            self.stop_robot()
            self.destroy_node()

def main():
    if not rclpy.utilities.ok():
        rclpy.init()
    node = StraightBackward()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
