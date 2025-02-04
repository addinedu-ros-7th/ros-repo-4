import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QThread, QTimer
from PyQt5 import uic
import cv2

import socket
import select
import threading
import numpy as np

import time

# Main Server
MAIN_SERVER = 0
MAIN_ADDR = "192.168.0.8"  # 서버의 IP 주소 또는 도메인 이름
MAIN_PORT = 8081       # 포트 번호

AI_IP = "192.168.0.8"
UDP_PORT1 = 9506

udp_listnener_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_listnener_sock.bind((AI_IP, UDP_PORT1))
imgListen_thread_flag = True

def IMGListener(window, udp_socket0, set_time):
    # udp_socket0 = Listnener
    global imgListen_thread_flag

    socketList = [udp_socket0]
    while imgListen_thread_flag:

        read_socket, _, _ = select.select(socketList, [], [], 1)
        for sock in read_socket:
            data, _ = sock.recvfrom(65535)

            pyri_id = data[0]
            img_data = data[1:]
            encoded_img = np.frombuffer(img_data, dtype = np.uint8)
            frame = cv2.imdecode(encoded_img,  cv2.IMREAD_COLOR_BGR)

            h,w,c = frame.shape
            qimage = QImage(frame.data, w,h,w*c, QImage.Format_RGB888)

            window.pixmap = window.pixmap.fromImage(qimage)

            if pyri_id == 1:
                lbl = window.lbl_cam1
            elif pyri_id == 2:
                lbl = window.lbl_cam2

            window.pixmap = window.pixmap.scaled(lbl.width(), lbl.height())
            lbl.setPixmap(window.pixmap)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     pass

def requestTCP(messages):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((MAIN_ADDR, MAIN_PORT))

    client_socket.send(f"{'&&'.join(messages)}".encode('utf-8'))

    response = "Error" # Default
    ready = select.select([client_socket], [], [], 2)
    if ready[0]:
        response = client_socket.recv(2048).decode('utf-8')
    
    client_socket.close()
    return response

def draw_transparent_circle(img, center_xy):
    """
        img 이미지 파일
        center_xy: 원의 중심 좌표, tuple
    """

    radius = 30
    # 초록색 (Green) with Alpha (투명도)
    color = (0, 255, 0, 128)  # 마지막 128은 Alpha 값으로, 0.5 =  255 / 2 ~= 128
    alpha = 0.5

    try:
        # RGBA 이미지를 생성합니다.
        if img.shape[2] == 3:  # 만약 이미지가 RGB 이미지라면
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # BGR에서 BGRA로 변경합니다.
        overlay = img.copy()
        cv2.circle(overlay, center_xy, radius, color, -1)
        # Opacity 적용
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        return img

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
#######################################GUI PART########################################################
main_ui = uic.loadUiType("/home/hdk/ws/final_project/data/test_main.ui")[0]

class WindowClass(QDialog, main_ui):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # self.synctimer = SyncTimer(self)
        # self.synctimer.daemon = True
        # self.synctimer.update.connect(self.updateSyncData)
        self.timer_1sec = QTimer()
        self.timer_1sec.setInterval(500)
        self.timer_1sec.timeout.connect(self.updateSyncData)
        self.timer_1sec.start()

        self.pixmap = QPixmap()

        self.img = cv2.imread("/home/hdk/ws/final_project/data/pyri_map_final.png")
        self.img = cv2.rotate(self.img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cvt_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        h,w,c = cvt_img.shape
        qimage = QImage(cvt_img.data, w,h,w*c, QImage.Format_RGB888)

        self.pixmap = self.pixmap.fromImage(qimage)
        self.pixmap = self.pixmap.scaled(self.lbl_map.width(), self.lbl_map.height())
        self.lbl_map.setPixmap(self.pixmap)
        self.btn_test1.clicked.connect(self.btn_test1Clicked)

    def updateSyncData(self):
        # 시간 업데이트
        now = time.localtime()
        self.lbl_time.setText(time.strftime("%Y-%m-%d %H:%M:%S", now))
        # 서버로부터 정보 동기화
        messages = ["SyncData"]
        response = requestTCP(messages)
        parts = response.split("&&")

        if parts[0] == "SyncData":
            if parts[1] == "ideal":
                pose_x = float(parts[2])
                pose_y = float(parts[3])
                odom_x = float(parts[4])
                odom_y = float(parts[5])

                test_text = f"""pose_x: {pose_x}\npose_y: {pose_y}\nodom_x: {odom_x}\nodom_y: {odom_y}"""
                self.lbl_test1.setText(test_text)
            elif parts[1] == "aiResult":
                self.lbl_test2.setText(parts[2])

        else:
            print("Wrong Response: ", parts)

    def btn_test1Clicked(self):
        # self.img = cv2.imread("/home/hdk/ws/final_project/data/pyri_map_final.png")
        # self.img = cv2.rotate(self.img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        center_xy = (int(self.le_test1.text()), int(self.le_test2.text()))

        self.img = draw_transparent_circle(self.img, center_xy)

        cvt_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        h,w,c = cvt_img.shape
        qimage = QImage(cvt_img.data, w,h,w*c, QImage.Format_RGB888)

        self.pixmap = self.pixmap.fromImage(qimage)
        self.pixmap = self.pixmap.scaled(self.lbl_map.width(), self.lbl_map.height())
        self.lbl_map.setPixmap(self.pixmap)


######################################################################################################

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindows = WindowClass()
    imgListenTread = threading.Thread(target=IMGListener, 
                                        args=(myWindows, udp_listnener_sock, 300))
    myWindows.show()

    imgListenTread.start()


    print("hello")
    # imgListenTread.join()
    sys.exit(app.exec_())