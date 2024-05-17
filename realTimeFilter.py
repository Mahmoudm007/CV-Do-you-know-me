from PyQt5 import QtWidgets, QtGui, QtCore, uic
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
import os
class AddingSnapFilters(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    stop_signal = False

    def __init__(self):
        super().__init__()
        self.shape = None
        self.current_frame = None
        self.face_cascade = cv2.CascadeClassifier('XMLs/face.xml')
        self.glasses = cv2.imread("filters/glass.png", -1)
        self.mustache = cv2.imread("filters/mustache.png", -1)
        self.add_glasses_flag = False
        self.add_mustache_flag = False

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, cv_img = cap.read()
            if ret and not self.stop_signal:
                processed_frame = self.process_frame(cv_img)
                self.change_pixmap_signal.emit(processed_frame)
            else:
                break
        cap.release()

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, 0, minSize=(120, 120), maxSize=(350, 350))
        for (x, y, w, h) in faces:
            if h > 0 and w > 0:
                if self.add_glasses_flag:
                    glass_symin = int(y + 1.5 * h / 5)
                    glass_symax = int(y + 2.5 * h / 5)
                    sh_glass = glass_symax - glass_symin

                    face_glass_ori = frame[glass_symin:glass_symax, x:x + w]

                    glass = cv2.resize(self.glasses, (w, sh_glass), interpolation=cv2.INTER_CUBIC)

                    self.transparentOverlay(face_glass_ori, glass)

                if self.add_mustache_flag:
                    mus_symin = int(y + 3.5 * h / 6)
                    mus_symax = int(y + 5 * h / 6)
                    sh_mus = mus_symax - mus_symin

                    mus_glass_ori = frame[mus_symin:mus_symax, x:x + w]

                    mus = cv2.resize(self.mustache, (w, sh_mus), interpolation=cv2.INTER_CUBIC)

                    self.transparentOverlay(mus_glass_ori, mus)

        return frame

    @staticmethod
    def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
        overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
        h, w, _ = overlay.shape  ## size of foreground image
        rows, cols, channels = src.shape  ## size of background image
        y, x = pos[0], pos[1]

        for i in range(h):
            for j in range(w):
                if x + i > rows or y + j >= cols:
                    continue
                alpha = float(overlay[i][j][3] / 255)  ##  read the alpha chanel
                src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
        return src

    def stop(self):
        self.stop_signal = True

    def toggle_add_glasses(self):
        self.add_glasses_flag = not self.add_glasses_flag

    def toggle_add_mustache(self):
        self.add_mustache_flag = not self.add_mustache_flag
