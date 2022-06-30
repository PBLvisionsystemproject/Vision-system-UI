import argparse
import imutils
import math
import matplotlib.pyplot as plt
import time
import socket
import urllib
import platform, subprocess, os
import cpuinfo
import threading
from queue import Queue
import struct
import zlib
import pickle
import base64
import inspect

import warnings
warnings.filterwarnings('ignore') 
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile
import time
import json
import cv2 as cv2
import csv
from csv import DictWriter
import pandas as pd

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from datetime import date
from datetime import datetime

from PIL import Image
from _thread import *
from time import sleep
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from numpy import (arctan, arccos, arcsin, arctan2)
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from threading import Thread 
from socketserver import ThreadingMixIn

from PyQt5 import Qt
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QScrollBar,QSplitter,QTableWidgetItem,QTableWidget,QComboBox,QVBoxLayout,QGridLayout,QDialog, QWidget, QPushButton, QApplication, QMainWindow,QAction,QMessageBox,QLabel,QTextEdit,QProgressBar,QLineEdit
from PyQt5.QtCore import pyqtSlot, QSettings, QTimer, QUrl, QDir
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QApplication

from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime,
                          QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase,
                         QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PyQt5 import uic
import psutil
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from pathlib import Path
from collections import deque

import platform
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

class visionSystem(QDialog):
	def __init__(self):
		super(visionSystem, self).__init__()
		loadUi('UI_Form.ui', self)
		self.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowMaximizeButtonHint 
							| QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowTitleHint 
							| QtCore.Qt.WindowCloseButtonHint)

		self.setWindowIcon(QtGui.QIcon("logo/logo apps.png"))
		self.pixmap = QPixmap('logo/machine_learning.jpg')
		self.realtime_cam.setPixmap(self.pixmap)
		self.process_cam.setPixmap(self.pixmap)

		self.open_camera.clicked.connect(self.open_cam)


	def verify_by_user(self):
		answer = QtWidgets.QMessageBox.question(self, "Are you sure you want to quit ?", "Task is in progress !",QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
		return answer == QtWidgets.QMessageBox.Yes

	def keyPressEvent(self, event):
		if event.key() == QtCore.Qt.Key_Escape:
			self.close()
		else:
			super(ObjectInspection, self).keyPressEvent(event)

	def closeEvent(self, event):
		if self.verify_by_user():
			event.accept()
		else:
			event.ignore()


	def testDevice(source):
		cap = cv2.VideoCapture(source)
		if cap is None or not cap.isOpened():
			print('Warning: unable to open video source: ', source)

	def make_1080p(self):
		cap = cv2.VideoCapture(int(self.usb_com.currentText()))
		cap.set(3, 1920)
		cap.set(4, 1080)

	def make_720p(self):
		cap = cv2.VideoCapture(int(self.usb_com.currentText()))
		cap.set(3, 1280)
		cap.set(4, 720)

	def make_480p(self):
		cap = cv2.VideoCapture(int(self.usb_com.currentText()))
		cap.set(3, 920)
		cap.set(4, 840)
		

	def change_res(self, width, height):
		cap = cv2.VideoCapture(int(self.usb_com.currentText()))
		cap.set(3, width)
		cap.set(4, height)

	def open_cam(self):
		cap = cv2.VideoCapture(int(self.usb_com.currentText()))
		self.make_720p()
		w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
		h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
		number_multiply = 2
		w = w * number_multiply
		h = h * number_multiply
		print(w , h)
		width_img = int(w)
		height_img = int(h)
		print(width_img , height_img)

		while True:
			ret, captureCam_real = cap.read()

			#_________________________________________________________________________________________________________________Light Effect
			if self.light_eff.isChecked() == True:
				#captureCam_real[:,:,2] = np.clip(int(self.contrast_vision_system.value()) * captureCam_real[:,:,2] + int(self.brightness_vision_system.value()), 0, 255);
				captureCam_real = cv2.convertScaleAbs(captureCam_real, alpha=int(self.contrast_vision_system.value()), beta=int(self.brightness_vision_system.value()))
			

			width = int(captureCam_real.shape[1]*1)
			height = int(captureCam_real.shape[0]*1.255)
			dim = (width, height)

			captureCam_real_afterResize = cv2.resize(captureCam_real, dim, interpolation =cv2.INTER_AREA)

			
			#_________________________________________________________________________________________________________________Mirror
			if self.mirror_image.isChecked() == True:
				captureCam_real_afterResize = cv2.flip(captureCam_real_afterResize, 1)

			#_________________________________________________________________________________________________________________Rotate
			if self.rotate_image.isChecked() == True:
				if self.rotation_deg.currentText() == "90 CW":
					captureCam_real_afterResize = cv2.rotate(captureCam_real_afterResize, cv2.ROTATE_90_CLOCKWISE)
				if self.rotation_deg.currentText() == "90 CCW":
					captureCam_real_afterResize = cv2.rotate(captureCam_real_afterResize, cv2.ROTATE_90_COUNTERCLOCKWISE)
				if self.rotation_deg.currentText() == "180":
					captureCam_real_afterResize = cv2.rotate(captureCam_real_afterResize, cv2.ROTATE_180)

			#_________________________________________________________________________________________________________________Gray Image
			if self.rgb_gray.isChecked() == True:
				captureCam_real_afterResize = cv2.cvtColor(captureCam_real_afterResize, cv2.COLOR_BGR2GRAY)

			# cv2.imshow('object detection', captureCam_real_afterResize)
			
			self.openCamera(captureCam_real_afterResize,1)

			if cv2.waitKey(25) & 0xFF == ord('q') or self.close_cam.isDown():
				self.pixmap = QPixmap('logo/machine_learning.jpg')
				self.realtime_cam.setPixmap(self.pixmap)
				self.process_cam.setPixmap(self.pixmap)
				cv2.destroyAllWindows()
				break

		cap.release()
		cv2.destroyAllWindows()




	def openCamera(self,img,window=0):
		qformat = QImage.Format_Indexed8
		if len(img.shape) == 3:
			if img.shape[2] == 4:
				qformat = QImage.Format_RGBA8888
			else:
				qformat = QImage.Format_RGB888
		outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0],qformat)
		#BGR to RGB
		outImage = outImage.rgbSwapped()

		try:
			if window == 1:
				self.realtime_cam.setPixmap(QPixmap.fromImage(outImage))
				self.realtime_cam.setScaledContents(True)
			if window == 2:
				self.process_cam.setPixmap(QPixmap.fromImage(outImage))
				self.process_cam.setScaledContents(True)
		except Exception as e:
			pass
			


if __name__ == '__main__':
	app = QtWidgets.QApplication(sys.argv)
	window = visionSystem()
	window.setWindowTitle('Vision System')
	window.show()
	sys.exit(app.exec_())