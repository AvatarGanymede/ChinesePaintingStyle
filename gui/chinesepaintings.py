# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'chinesepaintings.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from .notitle import TitleBar


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(958, 535)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        MainWindow.setWindowFlags(Qt.FramelessWindowHint)  # 隐藏边框
        # 标题栏
        self.titleBar = TitleBar(MainWindow)
        self.titleBar.setGeometry(QtCore.QRect(-1, -1, 960, 42))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        self.titleBar.setFont(font)
        self.titleBar.setStyleSheet("font:bold;background-image: url(:/pic/imgs/bg.png);color:#f9f1db;")
        self.titleBar.windowMinimumed.connect(MainWindow.showMinimized)
        self.titleBar.windowClosed.connect(MainWindow.close)
        self.titleBar.windowMoved.connect(self.move)
        MainWindow.windowTitleChanged.connect(self.titleBar.setTitle)
        MainWindow.windowIconChanged.connect(self.titleBar.setIcon)

        self.background = QtWidgets.QLabel(self.centralwidget)
        self.background.setGeometry(QtCore.QRect(0, 0, 960, 540))
        self.background.setStyleSheet("background-image: url(:/pic/imgs/bg.png);")
        self.background.setText("")
        self.background.setObjectName("background")
        self.org_image = QtWidgets.QLabel(self.centralwidget)
        self.org_image.setGeometry(QtCore.QRect(80, 110, 320, 320))
        self.org_image.setStyleSheet("border-width: 5px;\n"
"border-style: solid;\n"
"border-color: rgb(192, 72, 81);")
        self.org_image.setText("")
        self.org_image.setObjectName("org_image")
        self.after_image = QtWidgets.QLabel(self.centralwidget)
        self.after_image.setGeometry(QtCore.QRect(560, 110, 320, 320))
        self.after_image.setStyleSheet("border-width: 5px;\n"
"border-style: solid;\n"
"border-color: rgb(192, 72, 81);")
        self.after_image.setText("")
        self.after_image.setObjectName("after_image")
        self.transfer = QtWidgets.QPushButton(self.centralwidget)
        self.transfer.setGeometry(QtCore.QRect(430, 290, 101, 41))
        self.transfer.setStyleSheet("QPushButton{\n"
"font: 32pt \"方正字迹-周崇谦小篆繁体\";\n"
"color: rgb(192, 72, 81);\n"
"background-color:transparent;\n"
"}\n"
"QPushButton:hover{\n"
"color: #f9f1db;\n"
"}")
        self.transfer.setObjectName("transfer")
        self.OIMG = QtWidgets.QLabel(self.centralwidget)
        self.OIMG.setGeometry(QtCore.QRect(190, 70, 81, 41))
        self.OIMG.setStyleSheet("font: 32pt \"方正字迹-周崇谦小篆繁体\";\n"
"color: rgb(192, 72, 81);")
        self.OIMG.setObjectName("OIMG")
        self.AIMG = QtWidgets.QLabel(self.centralwidget)
        self.AIMG.setGeometry(QtCore.QRect(680, 70, 81, 41))
        self.AIMG.setStyleSheet("font: 32pt \"方正字迹-周崇谦小篆繁体\";\n"
"color: rgb(192, 72, 81);")
        self.AIMG.setObjectName("AIMG")
        self.open = QtWidgets.QPushButton(self.centralwidget)
        self.open.setGeometry(QtCore.QRect(430, 190, 101, 41))
        self.open.setStyleSheet("QPushButton{\n"
"font: 32pt \"方正字迹-周崇谦小篆繁体\";\n"
"color: rgb(192, 72, 81);\n"
"background-color:transparent;\n"
"}\n"
"QPushButton:hover{\n"
"color: #f9f1db;\n"
"}")
        self.open.setObjectName("open")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(80, 460, 321, 31))
        self.progressBar.setStyleSheet("QProgressBar {\n"
"    border: 2px solid grey;\n"
"    border-radius: 5px;\n"
"    border-color: #c04851;\n"
"    text-align: center;\n"
"    font: 75 12pt \"黑体-简\";\n"
"    color: #b78d12;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"    background-color: #c04851;\n"
"    width: 20px;\n"
"}")
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.pause = QtWidgets.QPushButton(self.centralwidget)
        self.pause.setGeometry(QtCore.QRect(560, 450, 101, 41))
        self.pause.setStyleSheet("QPushButton{\n"
"font: 32pt \"方正字迹-周崇谦小篆繁体\";\n"
"color: rgb(192, 72, 81);\n"
"background-color:transparent;\n"
"}\n"
"QPushButton:hover{\n"
"color: #f9f1db;\n"
"}")
        self.pause.setObjectName("pause")
        self.play = QtWidgets.QPushButton(self.centralwidget)
        self.play.setGeometry(QtCore.QRect(780, 450, 101, 41))
        self.play.setStyleSheet("QPushButton{\n"
"font: 32pt \"方正字迹-周崇谦小篆繁体\";\n"
"color: rgb(192, 72, 81);\n"
"background-color:transparent;\n"
"}\n"
"QPushButton:hover{\n"
"color: #f9f1db;\n"
"}")
        self.play.setObjectName("play")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.open.clicked.connect(MainWindow.Open)
        self.transfer.clicked.connect(MainWindow.Transfer)
        self.pause.clicked.connect(MainWindow.Pause)
        self.play.clicked.connect(MainWindow.Play)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("", ""))
        self.transfer.setText(_translate("MainWindow", "转化"))
        self.OIMG.setText(_translate("MainWindow", "原图"))
        self.AIMG.setText(_translate("MainWindow", "国画"))
        self.open.setText(_translate("MainWindow", "打开"))
        self.pause.setText(_translate("MainWindow", "暂停"))
        self.play.setText(_translate("MainWindow", "播放"))

from . import source_rc
