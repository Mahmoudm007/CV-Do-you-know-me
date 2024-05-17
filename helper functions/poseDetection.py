import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class PoseDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Pose Detection App")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.add_image)
        self.layout.addWidget(self.load_button)

        self.pose_detection_button = QPushButton("Detect Pose")
        self.pose_detection_button.clicked.connect(self.detect_pose)
        self.layout.addWidget(self.pose_detection_button)

        self.image_path = None

    def add_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        file_dialog.setViewMode(QFileDialog.Detail)
        files = file_dialog.getOpenFileNames(self, "Select Image")[0]
        for file in files:
            self.process_image(file)

    def process_image(self, file):
        self.image_path = file
        pixmap = QPixmap(file)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), aspectRatioMode=True, transformMode=False)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setScaledContents(True)


    def detect_pose(self):
        if self.image_path:
            image = cv2.imread(self.image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                results = pose.process(image_rgb)
                annotated_image = image_rgb.copy()
                mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.imwrite("annotated_image.jpg", annotated_image)
                pixmap = QPixmap("annotated_image.jpg")
                scaled_pixmap = pixmap.scaled(self.image_label.size(), aspectRatioMode=True, transformMode=False)
                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.setScaledContents(True)
        else:
            print("Please load an image first.")


def main():
    app = QApplication(sys.argv)
    window = PoseDetectionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
