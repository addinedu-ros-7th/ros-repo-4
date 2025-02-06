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
import math
import ast

import time

# Main Server
MAIN_SERVER = 0
MAIN_ADDR = "192.168.0.10"  # 서버의 IP 주소 또는 도메인 이름
MAIN_PORT = 8081       # 포트 번호

AI_IP = "192.168.0.10"
UDP_PORT1 = 9506

udp_listnener_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_listnener_sock.bind((MAIN_ADDR, UDP_PORT1))
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

def draw_transparent_circle(img_origin, center_xy):
    """
        img 이미지 파일
        center_xy: 원의 중심 좌표, tuple
    """

    img = img_origin.copy()
    radius = 30
    # 초록색 (Green) with Alpha (투명도)
    color = (0, 255, 0, 128)  # 마지막 128은 Alpha 값으로, 0.5 =  255 / 2 ~= 128
    alpha = 0.5

    hdk_flag = False

    try:
        # RGBA 이미지를 생성합니다.
        if img.shape[2] == 3:  # 만약 이미지가 RGB 이미지라면
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # BGR에서 BGRA로 변경합니다.
            hdk_flag = True
        overlay = img.copy()
        cv2.circle(overlay, center_xy, radius, color, -1)
        # Opacity 적용
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        if hdk_flag:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # BGRA에서 BGR로 변경합니다.
            
        return img

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def extract_mask(img):
    """
    흰색 배경의 캐릭터 이미지에서 마스크를 추출하는 함수 (중간 흰색 영역 포함).

    Args:
        image_path: 캐릭터 이미지 파일 경로.

    Returns:
        마스크 이미지 (numpy array, 0 또는 255 값을 가짐).
        마스크 추출에 실패한 경우 None 반환.
    """

    try:
        # 이미지 읽기 (BGR)
        image = img.copy()

        # 이미지를 HSV 색 공간으로 변환
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 흰색 영역 정의 (HSV 값 범위)
        lower_white = np.array([0, 0, 200])  # 낮은 흰색 범위
        upper_white = np.array([180, 20, 255])  # 높은 흰색 범위

        # 흰색 영역 마스크 생성
        mask = cv2.inRange(hsv_image, lower_white, upper_white)

        # 마스크 반전 (캐릭터 영역을 흰색으로)
        mask = 255 - mask

        # 마스크 다듬기 (선택 사항)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # Opening 연산 (노이즈 제거)

        # 마스크 채우기 (중간 흰색 영역 포함)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        
        mask1 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask = 255 - mask
        mask2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        return mask1, mask2

    except Exception as e:
        print(f"Error extracting mask: {e}")
        return None

def overlay_image(img, small_img, x_offset, y_offset):
    dst = img.copy()
    s_img = small_img.copy()
    mask1, mask2 =  extract_mask(s_img)

    roi = dst[y_offset:y_offset + s_img.shape[1], x_offset:x_offset + s_img.shape[0], :]
    dst1 = cv2.bitwise_and(s_img, mask1)
    dst2 = cv2.bitwise_and(roi, mask2)
    dst3 = cv2.bitwise_or(dst1, dst2)
    dst[y_offset:y_offset + s_img.shape[1], x_offset:x_offset + s_img.shape[0], :] = dst3

    return dst 

def rotate_image(img, angle):
    """
    이미지를 지정된 각도만큼 회전시키는 함수.

    Args:
        image_path: 이미지 파일 경로.
        angle: 회전 각도 (양수: 반시계 방향, 음수: 시계 방향).

    Returns:
        회전된 이미지 (numpy array).
        이미지 회전에 실패한 경우 None 반환.
    """

    try:
        # 이미지 읽기
        image = img.copy()

        # 이미지 중심점 계산
        height, width, _ = image.shape
        center = (width // 2, height // 2)

        # 회전 행렬 생성
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        # 이미지 회전
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

        # 테두리 검은색 영역을 흰색으로 채우기
        mask = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)  # GrayScale로 변경
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)  # Threshold를 이용하여 배경과 객체 분리

        # 마스크 채우기 (중간 흰색 영역 포함)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

        mask_inv = cv2.bitwise_not(mask)  # Mask 반전
        rotated_image = cv2.bitwise_and(rotated_image, rotated_image, mask=mask)  # 객체만 남기고 배경 제거
        white_background = np.full_like(rotated_image, (255, 255, 255))  # 흰색 배경 생성
        white_background = cv2.bitwise_and(white_background, white_background, mask=mask_inv)  # 흰색 배경에 Mask 적용
        rotated_image = cv2.bitwise_or(rotated_image, white_background)  # 객체와 흰색 배경 합성

        return rotated_image

    except Exception as e:
        print(f"Error rotating image: {e}")
        return None


def quaternion_to_euler(z, w):

  # Yaw (z-axis rotation)
  siny_cosp = +2.0 * (w * z)
  cosy_cosp = +1.0 - 2.0 * (z * z)
  yaw = math.atan2(siny_cosp, cosy_cosp)

  # return [roll, pitch, yaw]
  return yaw

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
        self.timer_1sec.setInterval(100)
        self.timer_1sec.timeout.connect(self.updateSyncData)
        self.timer_1sec.start()

        self.pixmap = QPixmap()

        self.img = cv2.imread("/home/hdk/ws/final_project/data/pyri_map_final.png")
        # self.img = cv2.rotate(self.img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        self.pyri_icon = cv2.imread("/home/hdk/ws/final_project/data/pytokkih_test.png")
        # self.pyri_icon = cv2.resize(self.pyri_icon, (70,70))

        # pyri1 default
        x_offset = 30
        y_offset = 35
        icon = rotate_image(self.pyri_icon, 15)
        icon = icon[130:180, 130:180, :]
        temp_img = overlay_image(self.img, icon, x_offset, y_offset)


        # pyri2 default
        x_offset = 577
        y_offset = 225
        icon = rotate_image(self.pyri_icon, 345)
        icon = icon[130:180, 130:180, :]
        temp_img = overlay_image(temp_img, icon, x_offset, y_offset)

        cvt_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        h,w,c = cvt_img.shape
        qimage = QImage(cvt_img.data, w,h,w*c, QImage.Format_RGB888)

        self.pixmap = self.pixmap.fromImage(qimage)
        self.pixmap = self.pixmap.scaled(self.lbl_map.width(), self.lbl_map.height())
        self.lbl_map.setPixmap(self.pixmap)
        self.btn_test1.clicked.connect(self.btn_test1Clicked)
        self.btn_test2.clicked.connect(self.btn_test2Clicked)

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

                circle_ls = []
                circle_ls = ast.literal_eval(parts[-1])
                temp_img = self.img.copy()
                for vals in circle_ls:
                    translation_x = float(vals[0])
                    translation_y = float(vals[1])
                    x_offset = int(577 - (547 / 3.4) * translation_y)
                    if x_offset > 577:
                        x_offset = 577
                    elif x_offset < 0:
                        x_offset = 0
                    y_offset = int(225 - (190 / 1.5) * translation_x)
                    if y_offset > 225:
                        y_offset = 225
                    elif y_offset < 0:
                        y_offset = 0
                    x_offset += 25
                    y_offset += 25
                    temp_img = draw_transparent_circle(temp_img, (int(x_offset), int(y_offset)))

                # pyri1
                translation_x = float(parts[2])
                translation_y = float(parts[3])
                rotation_z = float(parts[4])
                rotation_w = float(parts[5])

                x_offset = int(577 - (547 / 3.4) * translation_y)
                if x_offset > 577:
                    x_offset = 577
                elif x_offset < 0:
                    x_offset = 0
                y_offset = int(225 - (190 / 1.5) * translation_x)
                if y_offset > 225:
                    y_offset = 225
                elif y_offset < 0:
                    y_offset = 0

                ang = int(round(math.degrees(quaternion_to_euler(rotation_z, rotation_w)), 0)) 
                icon = rotate_image(self.pyri_icon, ang)
                icon = icon[130:180, 130:180, :]
                temp_img = overlay_image(temp_img, icon, x_offset, y_offset)

                # pyri1
                translation_x = float(parts[6])
                translation_y = float(parts[7])
                rotation_z = float(parts[8])
                rotation_w = float(parts[9])

                x_offset = int(577 - (547 / 3.4) * translation_y)
                if x_offset > 577:
                    x_offset = 577
                elif x_offset < 0:
                    x_offset = 0
                y_offset = int(225 - (190 / 1.5) * translation_x)
                if y_offset > 225:
                    y_offset = 225
                elif y_offset < 0:
                    y_offset = 0

                ang = int(round(math.degrees(quaternion_to_euler(rotation_z, rotation_w)), 0)) 
                icon = rotate_image(self.pyri_icon, ang)
                icon = icon[130:180, 130:180, :]
                temp_img = overlay_image(temp_img, icon, x_offset, y_offset)

                cvt_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
                h,w,c = cvt_img.shape
                qimage = QImage(cvt_img.data, w,h,w*c, QImage.Format_RGB888)

                self.pixmap = self.pixmap.fromImage(qimage)
                self.pixmap = self.pixmap.scaled(self.lbl_map.width(), self.lbl_map.height())
                self.lbl_map.setPixmap(self.pixmap)


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

    def btn_test2Clicked(self):
        # 서버로부터 정보 동기화
        messages = ["Test2Clicked"]
        response = requestTCP(messages)
        print(response)



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