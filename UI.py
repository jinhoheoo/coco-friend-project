# # -*- coding: utf-8 -*-

# from PyQt5 import QtCore, QtGui, QtWidgets
# from PyQt5.QtCore import QProcess
# import subprocess
# import coco_friend_voice2

# class Ui_MainWindow(object):
#     def setupUi(self, MainWindow):
#         MainWindow.setObjectName("MainWindow")
#         MainWindow.resize(800, 600)
#         self.centralwidget = QtWidgets.QWidget(MainWindow)
#         self.centralwidget.setObjectName("centralwidget")
        
#         self.textEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
#         self.textEdit.setGeometry(QtCore.QRect(30, 110, 715, 391))
#         self.textEdit.setObjectName("textEdit")
        
#         MainWindow.setCentralWidget(self.centralwidget)
        
#         self.process = QProcess()
#         self.process.setProcessChannelMode(QProcess.MergedChannels)
#         self.process.readyReadStandardOutput.connect(self.on_readyReadStandardOutput)
#         self.process.start("python", ["coco_friend_voice2.py"])
        
#     def on_readyReadStandardOutput(self):
#         data = self.process.readAllStandardOutput().data().decode()
#         self.textEdit.appendPlainText(data)

#     def run_coco_friend_voice(self):
#         self.process.start("python", ["coco_friend_voice2.py"])

#     def read_output(self):
#         output = self.process.readAllStandardOutput().data().decode()
#         # Split the output into "코코" and "사용자" lines
#         lines = output.splitlines()

#         # Check if there are at least two lines
#         if len(lines) >= 2:
#             coco_line = lines[0].strip()
#             user_line = lines[1].strip()

#             # Append the lines to the PlainTextEdit
#             self.textEdit.appendPlainText(coco_line)
#             self.textEdit.appendPlainText(user_line)
#             #prompt = f"코코야, 사용자는 {cam1.VAL1} 한 상태야. 3살에서 5살 사이의 아이가 이해할 수 있게 너무 기계적이지 않게, 사람처럼 따뜻하고 자연스럽고 간단하게 반말로 대답해줘."
#             #self.textEdit.appendPlainText(f"{coco_friend_voice2.prompt}")
#             #self.textEdit.appendPlainText(f"나 : {coco_friend_voice2.text}")  # 사용자의 입력된 음성을 출력합니다.    
    

# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     MainWindow = QtWidgets.QMainWindow()
#     ui = Ui_MainWindow()
#     ui.setupUi(MainWindow)
#     MainWindow.show()
#     sys.exit(app.exec_())


from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QProcess

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.textEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(30, 110, 715, 391))
        self.textEdit.setObjectName("textEdit")
        
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.process_coco_voice = QProcess()
        self.process_coco_voice.setProcessChannelMode(QProcess.MergedChannels)
        self.process_coco_voice.readyReadStandardOutput.connect(self.on_readyReadStandardOutput)
        self.process_coco_voice.start("python", ["coco_friend_voice2.py"])
        
    def on_readyReadStandardOutput(self):
        data = self.process_coco_voice.readAllStandardOutput().data().decode()
        self.textEdit.appendPlainText(data)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
