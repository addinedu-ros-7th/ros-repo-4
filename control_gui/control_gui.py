import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
import cv2

from_class = uic.loadUiType("./Control_GUI/Control_GUI.ui")[0]

class WindowClass(QDialog, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.pixmap = QPixmap()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindows = WindowClass()
    myWindows.show()
    sys.exit(app.exec_())