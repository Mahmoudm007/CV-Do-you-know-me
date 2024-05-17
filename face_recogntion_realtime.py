import sys
import cv2
from deepface import DeepFace
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot
import threading

class recognitionDisplay(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    stop_signal = False
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.counter = 0

        self.reference_img = cv2.imread("database/ahmed.jpg")
        self.reference_name = "Ahmed"

        self.reference_img2 = cv2.imread("database/amgad.png")
        self.reference_name2 = "Amgad"

        self.reference_img3 = cv2.imread("database/Mohanad.jpg")
        self.reference_name3 = "Mohanad"

        self.reference_img4 = cv2.imread("database/Degla.jpg")
        self.reference_name4 = "Mahmoud"

        self.face_match = False
        self.name_flag = 0

    def run(self):
        while True:
            ret, frame = self.cap.read()

            if ret and not self.stop_signal:
                if self.counter % 30 == 0:
                    try:
                        threading.Thread(target=self.check_face, args=(frame.copy(),)).start()
                    except ValueError:
                        pass
                self.counter += 1

                if self.face_match:
                    if self.name_flag == 1:
                        cv2.putText(frame, f"{self.reference_name}", (28, 458), cv2.FONT_HERSHEY_SIMPLEX, 2, (6, 255, 8), 3)
                    elif self.name_flag == 2:
                        cv2.putText(frame, f"{self.reference_name2}", (28, 458), cv2.FONT_HERSHEY_SIMPLEX, 2, (6, 255, 8), 3)
                    elif self.name_flag == 3:
                        cv2.putText(frame, f"{self.reference_name3}", (28, 458), cv2.FONT_HERSHEY_SIMPLEX, 2, (6, 255, 8), 3)
                    elif self.name_flag == 4:
                        cv2.putText(frame, f"{self.reference_name4}", (28, 458), cv2.FONT_HERSHEY_SIMPLEX, 2, (6, 255, 8), 3)
                else:
                    cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (8, 8, 255), 3)

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(convert_to_qt_format)
            else:
                break

    def check_face(self, frame):
        try:
            if DeepFace.verify(frame, self.reference_img.copy())['verified']:
                self.face_match = True
                self.name_flag = 1
            elif DeepFace.verify(frame, self.reference_img2.copy())['verified']:
                self.face_match = True
                self.name_flag = 2
            elif DeepFace.verify(frame, self.reference_img3.copy())['verified']:
                self.face_match = True
                self.name_flag = 3
            elif DeepFace.verify(frame, self.reference_img4.copy())['verified']:
                self.face_match = True
                self.name_flag = 4
            else:
                self.face_match = False
        except ValueError:
            self.face_match = False

    def stop(self):
        self.stop_signal = True

