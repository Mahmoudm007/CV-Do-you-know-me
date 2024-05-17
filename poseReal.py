import sys
import cv2
import mtcnn
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.uic import loadUi
from PIL import Image
import os
import numpy as np
import mediapipe as mp
import json
import joblib
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calc_joints(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    # get the angle between three points
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


class PoseDetectionThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        counter = 0
        stage = None

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                frame = cv2.resize(frame, (600, 400))
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    angle = calc_joints(shoulder, elbow, wrist)

                    # Draw angle on the frame
                    cv2.putText(image, str(angle),
                                tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

                    if angle > 160:
                        stage = 'down'
                    if angle < 50 and stage == 'down':
                        stage = 'up'
                        counter += 1
                except Exception as e:
                    print("Error:", e)

                # Emit the frame for display
                qt_img = self.convert_cv_qt(image)
                self.change_pixmap_signal.emit(qt_img)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)
