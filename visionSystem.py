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
import tensorflow as tf
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
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util

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

import sys
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

		self.open_camera.clicked.connect(self.open_cam)



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
		ret, captureCam = cap.read()

		captureCam = cv2.flip(captureCam, 0)
		captureCam = cv2.flip(captureCam, 1)

		self.openCamera(self,captureCam,2)


		while True:
			ret, captureCam_real = cap.read()
			width = int(captureCam_real.shape[1]*1)
			height = int(captureCam_real.shape[0]*1.255)
			dim = (width, height)

			captureCam_real_afterResize = cv2.resize(captureCam_real, dim, interpolation =cv2.INTER_AREA)

			if self.mirror_image.isChecked() == True:
				captureCam_real_afterResize = cv2.flip(captureCam_real_afterResize, 1)

			if self.rotate_image.isChecked() == True:
				if self.rotation_deg.currentText() == "90 CW":
					captureCam_real_afterResize = cv2.rotate(captureCam_real_afterResize, cv2.ROTATE_90_CLOCKWISE)
				if self.rotation_deg.currentText() == "90 CCW":
					captureCam_real_afterResize = cv2.rotate(captureCam_real_afterResize, cv2.ROTATE_90_COUNTERCLOCKWISE)
				if self.rotation_deg.currentText() == "180":
					captureCam_real_afterResize = cv2.rotate(captureCam_real_afterResize, cv2.ROTATE_180)

			self.openCamera(self, captureCam_real_afterResize, 1)




	def openCamera(self,img,window=1, display_number=1):
		qformat = QImage.Format_Indexed8
		if len(img.shape) == 3:
			if img.shape[2] == 4:
				qformat = QImage.Format_RGBA8888
			else:
				qformat = QImage.Format_RGB888
		outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0],qformat)
		#BGR to RGB
		outImage = outImage.rgbSwapped()

		if window == 1:
			if display_number == 0:
				self.realtime_cam.setPixmap(QPixmap.fromImage(outImage))
				self.realtime_cam.setScaledContents(True)
			if display_number == 1:
				self.realtime_cam.setPixmap(QPixmap.fromImage(outImage))
				self.realtime_cam.setScaledContents(True)
		if window == 2:
			if display_number == 0:
				self.process_cam.setPixmap(QPixmap.fromImage(outImage))
				self.process_cam.setScaledContents(True)
			if display_number == 1:
				self.process_cam.setPixmap(QPixmap.fromImage(outImage))
				self.process_cam.setScaledContents(True)


if __name__ == '__main__':
	app = QtWidgets.QApplication(sys.argv)
	window = visionSystem()
	window.setWindowTitle('Vision System')
	window.show()
	sys.exit(app.exec_())