# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'seq2seq.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from train import TRAIN
import numpy as np
import jieba
import time

class Ui_MainWindow(object):
    def __init__(self):
        self.tr=TRAIN()
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 480)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(240, 50, 171, 31))
        font = QtGui.QFont()
        font.setFamily("Algerian")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(150, 230, 441, 71))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout_2.addWidget(self.lineEdit_2)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Algerian")
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(20, 110, 591, 211))
        brush = QtGui.QBrush(QtGui.QColor(170, 255, 255))
        brush.setStyle(QtCore.Qt.Dense6Pattern)
        self.graphicsView.setBackgroundBrush(brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 255, 255))
        brush.setStyle(QtCore.Qt.NoBrush)
        self.graphicsView.setForegroundBrush(brush)
        self.graphicsView.setObjectName("graphicsView")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(30, 140, 441, 73))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setFamily("Algerian")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.horizontalLayout.addWidget(self.lineEdit_4)
        self.label_2.raise_()
        self.lineEdit_4.raise_()
        self.widget1 = QtWidgets.QWidget(self.centralwidget)
        self.widget1.setGeometry(QtCore.QRect(150, 340, 441, 25))
        self.widget1.setObjectName("widget1")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget1)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.widget1)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.horizontalLayout_3.addWidget(self.lineEdit_3)
        self.pushButton = QtWidgets.QPushButton(self.widget1)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_3.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.widget1)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_3.addWidget(self.pushButton_2)
        self.label.raise_()
        self.label_2.raise_()
        self.layoutWidget.raise_()
        self.lineEdit_3.raise_()
        self.pushButton.raise_()
        self.pushButton_2.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 640, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton_2.clicked.connect(self.lineEdit_3.clear)
        self.pushButton_2.clicked.connect(self.lineEdit_4.clear)
        self.pushButton_2.clicked.connect(self.lineEdit_2.clear)
        self.pushButton.clicked.connect(self.predict)
        #self.pushButton.clicked.connect(self.lineEdit_3.clear)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.lineEdit_3, self.pushButton)
        MainWindow.setTabOrder(self.pushButton, self.pushButton_2)
        MainWindow.setTabOrder(self.pushButton_2, self.lineEdit_2)
        MainWindow.setTabOrder(self.lineEdit_2, self.lineEdit_4)
        MainWindow.setTabOrder(self.lineEdit_4, self.graphicsView)

    def predict(self):

        line_input=self.lineEdit_3.text()
        if line_input=='':
            self.lineEdit_3.setText('输入不能为空')
            return
        try:
            print(line_input)
            test_list = jieba.lcut(line_input)
            test_list = [self.tr.word_index[i] for i in test_list]
            print('test', test_list)
        except BaseException:
            self.lineEdit_3.setText('Sorry,ROBOT暂未录入您输入的单词')
            return
        input_list = np.ones([1, 23])
        for i in range(len(test_list)):
            input_list[0][i] = test_list[i]
        print('input_list', input_list)
        result=self.tr.predict(input_list)
        print('result:',result,type(result))
        for i in range(len(line_input)):
            show_input=line_input[:i+1]
            self.lineEdit_2.setText(show_input)
            QApplication.processEvents()
            time.sleep(0.2)
        if result=='':
            self.lineEdit_4.setText('...')
            return
        for i in range(len(result)):
            show_result=result[:i+1]
            self.lineEdit_4.setText(show_result)
            QApplication.processEvents()
            time.sleep(0.2)
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "对话机器人程序"))
        self.label_3.setText(_translate("MainWindow", ":Me"))
        self.label_2.setText(_translate("MainWindow", "Robot："))
        self.pushButton.setText(_translate("MainWindow", "Submit"))
        self.pushButton_2.setText(_translate("MainWindow", "Clear"))
