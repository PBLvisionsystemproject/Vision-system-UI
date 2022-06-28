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

if __name__ == '__main__':
	app = QtWidgets.QApplication(sys.argv)
	window = visionSystem()
	window.setWindowTitle('Vision System')
	window.show()
	sys.exit(app.exec_())