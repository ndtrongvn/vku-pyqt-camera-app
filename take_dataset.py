from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5 import QtCore
import constant
from pathlib import Path

import os
import sys
import time
import glob
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K
from mtcnn import MTCNN
import cv2

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))
model.load_weights('vgg_face_weights.h5')
vgg_face=Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)

detector = MTCNN()

class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def run(self):
        for i in range(10, 0, -1):
            time.sleep(1)
            self.progress.emit(i - 1)
        self.finished.emit()

class Worker_Capture(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    is_finished = False

    def run(self):
        for i in range(1000):
            if self.is_finished:
                break
            time.sleep(0.2)
            self.progress.emit(i + 1)
        self.finished.emit()
    def stop(self):
        self.is_finished = True

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setMinimumSize(QtCore.QSize(1000, 500))
        self.setMaximumSize(QtCore.QSize(1000, 500))
        self.cameraComboBox = QComboBox()
        self.mainWidget = QWidget()
        self.info_panel()

        self.available_cameras = QCameraInfo.availableCameras()
        if not self.available_cameras:
            self.showdialog(constant.CRITICAL, "Camera not found!!!")
            print("Camera not found!!")
            exit()
            pass
        else:
            cameras = [constant.camera_index[i] for i in range(0, len(self.available_cameras))]
            self.cameraComboBox.addItems(cameras)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.viewfinder = QCameraViewfinder()
        
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.viewfinder)
        self.hbox.addWidget(self.leftWidget)
        
        self.mainWidget.setLayout(self.hbox)
        self.setCentralWidget(self.mainWidget)
        self.viewfinder.show()
        self.cameraComboBox.setCurrentIndex(1)
        self.cameraComboBox.currentIndexChanged.connect(self.select_camera)
        self.cameraComboBox.setCurrentIndex(0)

        photo_action = QAction("Take photo...", self)
        photo_action.setStatusTip("Take a photo now")
        photo_action.triggered.connect(self.take_photo)

        self.setWindowTitle("VKU Camera")
        self.show()
        self.save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "photos")

    def info_panel(self):
        self.leftWidget = QWidget()
        self.vbox = QVBoxLayout()
        self.vbox.addStretch(0)
        self.textbox_name = QLineEdit()
        lbl_name = QLabel('Name', self)
        self.textbox_id = QLineEdit()
        lbl_id = QLabel('ID', self)
        self.btn_takephoto = QPushButton("Take photos now")
        self.vbox.addStretch()
        self.vbox.addWidget(self.cameraComboBox)
        self.vbox.addWidget(lbl_name)
        self.vbox.addWidget(self.textbox_name)
        self.vbox.addWidget(lbl_id)
        self.vbox.addWidget(self.textbox_id)
        self.vbox.addWidget(self.btn_takephoto)
        self.vbox.addStretch()
        self.leftWidget.setLayout(self.vbox)
        self.btn_takephoto.clicked.connect(self.runLongTask)

    def select_camera(self, i):
        print('Changing to camera ', i)
        self.camera = QCamera(self.available_cameras[i])
        self.camera.setViewfinder(self.viewfinder)
        self.camera.setCaptureMode(QCamera.CaptureStillImage)
        self.camera.error.connect(lambda: self.alert(self.camera.errorString()))
        self.camera.start()

        self.capture = QCameraImageCapture(self.camera)
        err = self.capture.errorString()
        if err:
            print("Err", err)
            self.showdialog(constant.CRITICAL, err[0])
        
        # self.capture.imageCaptured.connect(lambda d, i: self.status.showMessage("Image %04d captured" % self.save_seq))

        self.current_camera_name = self.available_cameras[i].description()
        self.save_seq = 0

    def take_photo(self):
        uid = self.textbox_id.text().strip()
        uname = self.textbox_name.text().strip()
        if uid == "" or uname == "":
            self.showdialog(constant.CRITICAL, "Please input ID and name")
            return
        self.viewfinder.setContrast(0)
        timestamp = time.strftime("%d-%b-%Y-%H_%M_%S")
        output_path = os.path.join(self.save_path, uid,"%s-%04d-%s.jpg" % (
            self.current_camera_name,
            self.save_seq,
            timestamp
        ))
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        self.capture.capture(output_path)
        self.save_seq += 1

    def showdialog(self, dialog_type, content):
        title = constant.dialog_type[dialog_type]
        icon = constant.icon_type[dialog_type]
        msg = QMessageBox()
        msg.setIcon(icon)
        msg.setText(content)
        msg.setWindowTitle(title)
        msg.setStandardButtons(QMessageBox.Ok)
        real = msg.exec_()

    def reportProgress(self, n):
        self.status.showMessage(f"Terminate taking photo at: {n}")
        
    def reportProgress_capture(self, n):
        print(f'Taking photo {n}')
        self.take_photo()

    def crop_raw_face(self, uid):
        Path(f'./traindata/{uid}').mkdir(parents=True, exist_ok=True)
        paths = glob.glob(f'./photos/{uid}/*.jpg') # './images/*.jpg'
        for index, path in enumerate(paths):
            try:
                img = cv2.imread(path)
                faces = detector.detect_faces(img)
                for face in faces:
                    print(face)
                    left, top, width, height = face['box']
                    img = img[top:top+height, left:left+width]
                    cv2.imwrite(f'./traindata/{uid}/frame_{index}_{random.randint(0, 1000)}.jpg', img)
            except:
                print("exept: " + path)
    
    def runLongTask(self):
        uid = self.textbox_id.text().strip()
        uname = self.textbox_name.text().strip()
        if uid == "" or uname == "":
            self.showdialog(constant.CRITICAL, "Please input ID and name")
            return
        # Step 2: Create a QThread object
        self.thread = QThread()
        self.thread_capture = QThread()
        # Step 3: Create a worker object
        self.worker = Worker()
        self.worker_capture = Worker_Capture()
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        self.worker_capture.moveToThread(self.thread_capture)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.reportProgress)

        self.thread_capture.started.connect(self.worker_capture.run)
        self.worker_capture.finished.connect(self.thread_capture.quit)
        self.worker_capture.finished.connect(self.worker_capture.deleteLater)
        self.thread_capture.finished.connect(self.thread_capture.deleteLater)
        self.worker_capture.progress.connect(self.reportProgress_capture)
        # Step 6: Start the thread
        self.thread.start()
        self.thread_capture.start()

        def on_finished():
            self.crop_raw_face(uid)
            self.btn_takephoto.setEnabled(True)
            self.cameraComboBox.setEnabled(True)
            self.textbox_id.setEnabled(True)
            self.textbox_name.setEnabled(True)
            self.status.showMessage("Taking photos terminated!!")
            self.worker_capture.stop()
            self.thread_capture.quit()
            self.showdialog(constant.INFO, "Photos saved at: %s" % os.path.join(self.save_path, uid) )
        # Final resets
        self.btn_takephoto.setEnabled(False)
        self.textbox_id.setEnabled(False)
        self.textbox_name.setEnabled(False)
        self.cameraComboBox.setEnabled(False)
        self.thread.finished.connect(
            lambda: on_finished()
        )
    
        

if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setApplicationName("VKU Camera")

    window = MainWindow()
    app.exec_()
