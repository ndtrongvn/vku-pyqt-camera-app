from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap
import sys
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

from mtcnn import MTCNN
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K
import cv2
import time

cap = cv2.VideoCapture(0)
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

person_rep = {
    0: {'id':'SV_0001', 'name': 'Nguyen Bao M. Hoang', 'class': '17IT1', 'dob': '04/04/1999'},
    1: {'id':'SV_0002', 'name': 'Dao Minh Quan', 'class': '18IT3', 'dob': '06/09/2000'}
}
detector = MTCNN()
classifier_model=tf.keras.models.load_model('face_classifier_model.h5')

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_info = pyqtSignal(dict)

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            faces = detector.detect_faces(frame)
            _dict = {'id':'0000', 'name': 'NoName', 'class': 'None', 'dob': '00/00/0000'}
            for face in faces:
                left, top, width, height = face['box']
                if left < 0:
                    left = 0
                if top < 0:
                    top = 0
                img_crop = frame[top:top+height, left:left+width]

                img_crop = cv2.resize(img_crop, (224, 224))
                img_crop = np.expand_dims(img_crop,axis=0)
                img_crop = preprocess_input(img_crop)
                img_encode = vgg_face(img_crop)

                embed=K.eval(img_encode)
                person=classifier_model.predict(embed)
                _dict=person_rep[np.argmax(person)]

                cv2.rectangle(frame,(left,top),(left+width,top+height),(0,255,0), 2)
                # frame=cv2.putText(frame,_dict['name'],(left,top+30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA) 
            self.change_pixmap_signal.emit(frame)
            self.change_info.emit(_dict)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vku Recognizing Demo App")
        self.display_width = 700
        self.display_height = 1200
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('Webcam')
        topWidget = QWidget()
        hbox = QHBoxLayout()
        lbl_name = QLabel('Name: ', self)
        self.lbl_name_val = QLabel('NoName', self)
        lbl_id = QLabel('ID: ', self)
        self.lbl_id_val = QLabel('None', self)
        lbl_class = QLabel('Class: ', self)
        self.lbl_class_val = QLabel('None', self)
        lbl_dob = QLabel('DOB: ', self)
        self.lbl_dob_val = QLabel('0/0/0000', self)
        hbox.addWidget(lbl_name)
        hbox.addWidget(self.lbl_name_val)
        hbox.addStretch(0)
        hbox.addWidget(lbl_id)
        hbox.addWidget(self.lbl_id_val)
        hbox.addStretch(0)
        hbox.addWidget(lbl_class)
        hbox.addWidget(self.lbl_class_val)
        hbox.addStretch(0)
        hbox.addWidget(lbl_dob)
        hbox.addWidget(self.lbl_dob_val)
        topWidget.setLayout(hbox)

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(topWidget)
        vbox.addWidget(self.image_label)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.change_info.connect(self.on_change_info)
        # start the thread
        self.thread.start()

    def on_change_info(self, _inf):
        # print(_inf)
        self.lbl_name_val.setText(_inf['name'])
        self.lbl_name_val.adjustSize()
        self.lbl_id_val.setText(_inf['id'])
        self.lbl_id_val.adjustSize()
        self.lbl_class_val.setText(_inf['class'])
        self.lbl_class_val.adjustSize()
        self.lbl_dob_val.setText(_inf['dob'])
        self.lbl_dob_val.adjustSize()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        # """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        # """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(700, 900, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())