# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QProcess, QIODevice
from PyQt5.QtWidgets import QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QApplication
import subprocess
import cv2
import numpy as np
import collections
import time
from keras.models import load_model

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.chats = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.chats.setEnabled(False)
        self.chats.setGeometry(QtCore.QRect(30, 110, 715, 391))
        self.chats.setObjectName("chats")

        self.btn_send_2 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_send_2.setGeometry(QtCore.QRect(60, 10, 113, 32))
        self.btn_send_2.setObjectName("btn_send_2")

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 28))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.process_cam1 = QProcess()
        self.process_cam1.startDetached("python", ["cam1.py"])  # cam1.py 프로세스 시작

        self.process_coco_voice = QProcess()
        self.process_coco_voice.readyReadStandardOutput.connect(self.read_output)

        self.retranslateUi(MainWindow)
        self.btn_send_2.clicked.connect(self.run_coco_friend_voice)

    def run_coco_friend_voice(self):
        self.process_coco_voice.start("python", ["coco_friend_voice2.py"])

    def read_output(self):
        output = self.process_coco_voice.readAllStandardOutput().data().decode()
        lines = output.splitlines()

        if len(lines) >= 2:
            coco_line = lines[0].strip()
            user_line = lines[1].strip()

            self.chats.appendPlainText(coco_line)
            self.chats.appendPlainText(user_line)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_send_2.setText(_translate("MainWindow", "Open File"))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 얼굴 인식을 5초 동안만 수행하도록 수정
        start_time = time.time()
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        emotion_model = load_model('./models/emotion_model.hdf5')
        expression_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        while cap.isOpened() and time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (64, 64))
                face_roi = np.expand_dims(face_roi, axis=-1)
                face_roi = np.expand_dims(face_roi, axis=0)
                face_roi = face_roi / 255.0

                output = emotion_model.predict(face_roi)[0]
                expression_index = np.argmax(output)
                expression_label = expression_labels[expression_index]
                cv2.putText(frame, expression_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    subprocess.Popen(["python", "coco_friend_voice2.py"])
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
