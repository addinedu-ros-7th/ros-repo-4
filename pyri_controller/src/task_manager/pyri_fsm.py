import rclpy
from rclpy.node import Node
from transitions import Machine

from std_msgs.msg import String
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped

PYRI_ID = 1

class TaskManager(Node):
    # states = ['idle', 'running', 'success', 'failure', 'cancelled']
    states = ['launching', 'checking', 'failure', 'idle', 'working', 
              'fire_door_checking', 'door_closing', 
              'obstacle_avoidance', 
              'target_approach', 'delivering', 'waiting_command',
              'evacuation_guide', 'rescue',
              'state_check']

    def __init__(self):
        super().__init__('task_manager')
        self.machine = Machine(model=self, states=self.states, initial=self.states[0])
        
        # add_transition(trigger, source, dest)

        # 서비스 시작 > 상태점검
        self.machine.add_transition('launch', 'launching', 'checking')
        # 정상 상태 > 작업 대기
        self.machine.add_transition('status_check', 'checking', 'idle')
        # 상태 불량 > 불량 경고
        self.machine.add_transition('status_fail', 'checking', 'failure')
        # 작업 지시 > 위험도 분석 및 구조 탐지 주행
        self.machine.add_transition('start', 'idle', 'working')

        # 방화문 지날 시 > 방화문 상태 확인
        self.machine.add_transition('fire_door_point', 'working', 'fire_door_checking')
        # 방화문 열림 > 방화문 처리
        self.machine.add_transition('fire_door_open', 'fire_door_checking', 'door_closing')
        # 방화문 처리 완료 > 방화문 상태 확인
        self.machine.add_transition('door_closing_done', 'door_closing', 'fire_door_checking')
        # 방화문 닫힘 > 위험도 분석 및 구조 탐지 주행
        self.machine.add_transition('fire_door_closed', 'fire_door_checking', 'working')

        # 장애물 감지 > 장애물 회피
        self.machine.add_transition('obstacle_detected', 'working', 'obstacle_avoidance')
        # 회피 완료 > 위험도 분석 및 구조 탐지 주행
        self.machine.add_transition('obstacle_avoidance_done', 'obstacle_avoidance', 'working')

        # 구조 대상 탐지 > (구조 작업) 타겟 접근 주행
        self.machine.add_transition('target_detected', 'working', 'target_approach')
        # 목표 도달 > 구호 물품 전달
        self.machine.add_transition('target_arrived', 'target_approach', 'delivering')
        # 전달 완료 > 명령 대기
        self.machine.add_transition('delivering_done', 'delivering', 'waiting_command')

        # 대피 안내 명령 > 대피 안내 주행
        self.machine.add_transition('command_arrived', 'waiting_command', 'evacuation_guide')
        # 인계 완료 > 상태 정비
        self.machine.add_transition('evacuation_done', 'evacuation_guide', 'state_check')
        # 정비 완료, 마지막 작업 위치 도달 > 위험도 분석 및 구조 탐지 주행
        self.machine.add_transition('state_check_done', 'state_check', 'working')

        # 구조 명령 > 구조 주행
        self.machine.add_transition('rescuee_command', 'waiting_command', 'rescue')
        # 인계 완료 > 상태 정비
        self.machine.add_transition('rescue_done', 'rescue', 'state_check')
        # 정비 완료, 마지막 작업 위치 도달 > 위험도 분석 및 구조 탐지 주행
        self.machine.add_transition('state_check_done', 'state_check', 'working')

        # 탐색 재개 명령 > 위험도 분석 및 구조 탐지 주행
        self.machine.add_transition('continue_work', 'waiting_command', 'working')

        # 대기 시간 초과
        self.machine.add_transition('timeout', 'waiting_command', 'rescue')

        # 작업 완료 > 작업 대기
        self.machine.add_transition('work_done', 'working', 'idle')

        self.timer = self.create_timer(1.0, self.execute_task)

        self.subscribers()
        self.test_data = None
        self.prev_state = None

        # load last_state / default: launching
        self.last_state = self.declare_parameter('task_state', self.machine.initial).value
        if self.last_state in self.states:
            self.machine.set_state(self.last_state)

    def subscribers(self):
        self.sub_battery = self.create_subscription(
            Float32,
            '/battery',
            self.battery_callback,
            10
        )

        self.sub_test = self.create_subscription(
            String,
            '/test_task',
            self.listener_callback,
            10
        )

        self.sub_goal = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )

    def listener_callback(self, msg):
        print("")
        self.get_logger().info(f"Received message: {msg.data}")
        self.test_data = msg.data
        print("[listener_callback]", self.test_data)
        # return self.test_data

    def battery_callback(self, msg):
        self.get_logger().info(f"Pyri battery: {msg.data}")
        self.battery = msg.data
        # return msg.data

    def goal_callback(self, msg):
        self.get_logger().info(f"Pyri goal: {msg.pose}")
        self.goal = msg

    def update_state(self, new_state):
        """ 
        status will be updated as ros parameter
        ros2 param get /task_manager task_state
        """
        if new_state in self.states:
            self.set_parameters(
                [rclpy.parameter.Parameter(
                    'task_state', 
                    rclpy.Parameter.Type.STRING, 
                    new_state
                    )])
            self.machine.set_state(new_state)
            self.get_logger().info(f"Task state updated to: {self.state}.")

    def execute_task(self):
        if self.state == 'launching':
            self.start()
            self.get_logger().info("Pyri is launched. Loading services...")
            self.prev_state = self.state
            self.update_state('checking')

        if self.state == 'checking':
            self.status_check()
            self.get_logger().info("Pyri is ready to work.")
            self.prev_state = self.state
            self.update_state('idle')
            """
            TODO:
            - battery check
            - /tracked_pose topic sub

            else > led alert
            """

        if self.state == 'idle':
            self.working()
            self.get_logger().info("Pyri is working.")
            self.prev_state = self.state
            self.update_state('working')

        if self.state == 'working':
            """
            lack of priority
            """
            # if self.obstacle_detected():
            #     self.get_logger().info("Pyri will be avoiding obstacle.")
            #     self.update_state('obstacle_avoidance')

            # if self.target_detected():
            #     self.get_logger().info("Pyri will be rescue target.")
            #     self.update_state('target_approach')

            # if self.fire_door_point():
            #     self.get_logger().info("Pyri will be checking fire door.")
            #     self.update_state('fire_door_checking')

            # if self.work_done():
            #     self.get_logger().info("Pyri is finished work.")
            #     self.update_state('idle')

            if self.event() == 'obstacle_detected':
                self.get_logger().info("Pyri will be avoiding obstacle.")
                self.update_state('obstacle_avoidance')

            if self.event() == 'target_detected':
                self.get_logger().info("Pyri will be rescue target.")
                self.update_state('target_approach')

            if self.event() == 'fire_door_point':
                self.get_logger().info("Pyri will be checking fire door.")
                self.update_state('fire_door_checking')

            if self.work_done():
                self.get_logger().info("Pyri is finished work.")
                self.update_state('idle')

            self.prev_state = self.state
            self.test_data = None

        # fire door
        if self.state == 'fire_door_checking':
            self.get_logger().info("Pyri is checking fire door.")
            if self.fire_door_opened():
                self.get_logger().info("Fire door is opened. Pyri will close the door.")
                self.update_state('door_closing')
            else:
                self.get_logger().info("Fire door is not opened.")
                self.update_state('working')

        if self.state == 'door_closing':
            self.door_closing_done()
            self.get_logger().info("Pyri is closing fire door.")
            self.update_state('fire_door_checking')

        if self.state == 'fire_door_closed':
            self.fire_door_closed()
            self.get_logger().info("Fire door is successfully closed.")
            self.update_state('working')

        # obstacle
        if self.state == 'obstacle_avoidance':
            self.obstacle_avoidance_done()
            self.get_logger().info("Pyri is successfully avoiding obstacle.")
            self.update_state('working')

        # target rescue
        if self.state == 'target_approach':
            self.rescuee_detected()
            self.get_logger().info("Target has been detected. Pyri is approaching rescue target.")
            self.update_state('target_arrived')
            print(self.state)

        if self.state == 'target_arrived':
            print("where are uuuuuu")
            self.delivering_done()
            self.get_logger().info("Pyri is deleivering supplies.")
            self.update_state('waiting_command')

        # command
        if self.state == 'waiting_command':
            self.get_logger().info("Pyri is waiting for command.")
            self.command = self.command_arrived()

            if self.command == 'continue_work':
                self.get_logger().info("Pyri will continue working.")
                self.update_state('working')

            elif self.command == 'evacuation':
                self.get_logger().info("Pyri will start evacuation.")
                self.update_state('evacuation_guide')

            elif self.command == 'rescue':
                self.get_logger().info("Pyri will start rescue.")
                self.update_state('rescue')

            else:
                self.get_logger().error("Wrong command.")

        if self.state == 'evacuation_guide':
            self.get_logger().info("Pyri is on evacuation guide mode.")
            self.evacuation_done()
            self.update_state('state_check')

        if self.state == 'rescue':
            self.rescue_done()
            self.get_logger().info("Pyri is on rescue mode.")
            self.update_state('state_check')

        if self.state == 'state_check':
            self.get_logger().info("Pyri is checking status.")
            self.state_check_done() # 배터리 체크, 마지막 작업 위치 받기
            self.update_state('working')
        
        # if self.state == 'running':
        #     if self.check_success():
        #         self.succeed()
        #     else:
        #         self.fail()

    def update_state(self, new_state):
        """ 
        status will be updated as ros parameter
        ros2 param get /task_manager task_state
        """
        if new_state in self.states:
            self.set_parameters([rclpy.parameter.Parameter('task_state', rclpy.Parameter.Type.STRING, new_state)])
            self.machine.set_state(new_state)
            self.get_logger().info(f"Task state updated to: {self.state}.")    

    def start(self):
        print(f"pyri{PYRI_ID} has launched.")
            
    def status_check(self):
        print("///배터리 체크, 포즈값 받아오면 넘어가기///")
        if self.battery > 40:
            # print(f"pyri{PYRI_ID} battery level: {self.battery}")
            return True 
        else:
            self.get_logger().error(f"pyri{PYRI_ID} battery level is too low.")
    
    def working(self):
        print(f"pyri{PYRI_ID} start working...")
        print(self.state, self.prev_state)
    
    def event(self):
        if self.prev_state != self.state:
            print(f"[ Pyri{PYRI_ID} state is ",self.state, "]\n[ Pyri{PYRI_ID} prev state is ",self.prev_state, "]")
        else:
            print(".", sep="", end="", flush=True)
        return self.test_data
    
    def work_done(self):
        if self.test_data == 'done':
            print(f"pyri{PYRI_ID} work done. data: {self.test_data}")
            return True
        else:
            return False
        
    def fier_door_opened(self):
        print("fire door opened!")
        return True
    
    def door_closing_done(self):
        print("door closing done")
        return True
    
    def fire_door_closed(self):
        print("fire door closed")
        return True
    
    def obstacle_avoidance_done(self):
        print("obstacle avoidance done")
        return True
    
    def rescuee_detected(self):
        print("rescuee detected")
        print(self.state, self.prev_state)
        return True 
    
    def delivering_done(self):
        print("delivering done")
        return True

    def command_arrived(self):
        # ok this will be service
        print("command arrived.")
        return self.test_data
    
    def evacuation_done(self):
        print("evacuation done")
        return True
    
    def rescue_done(self):
        print("rescue done")
        return True
    
    def state_check_done(self):
        print("state check done")
        return True

def main():
    rclpy.init()
    node = TaskManager()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
