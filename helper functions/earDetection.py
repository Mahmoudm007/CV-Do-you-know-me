import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class EarDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ear Detection App")
        self.setGeometry(100, 100, 1000, 800)  # Increased size of the main window

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)  # Center-align the label
        self.layout.addWidget(self.label)

        self.add_image_button = QPushButton("Add Image")
        self.add_image_button.clicked.connect(self.add_image)
        self.layout.addWidget(self.add_image_button)

    def add_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        file_dialog.setViewMode(QFileDialog.Detail)
        files = file_dialog.getOpenFileNames(self, "Select Image")[0]
        for file in files:
            self.process_image(file)

    def detect_ears(self, image_path):
        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Preprocess the image: Apply histogram equalization to improve contrast
        equalized = cv2.equalizeHist(gray)

        # Load the pre-trained Haar Cascade for ear detection
        ear_cascade = cv2.CascadeClassifier("XMLs/haarcascade_mcs_rightear.xml")  # Specify the path to ear cascade XML file

        # Detect ears in the image
        ears = ear_cascade.detectMultiScale(equalized, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

        # Draw rectangles around the detected ears
        for (x, y, w, h) in ears:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0 , 255, 0), 2)

        # Convert the image to RGB format for displaying in PyQt5
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to QImage
        height, width, channel = image_rgb.shape
        bytesPerLine = 3 * width
        q_image = QImage(image_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)

        # Convert QImage to QPixmap and display in QLabel
        pixmap = QPixmap.fromImage(q_image)

        # Scale the image to fit the label
        scaled_pixmap = pixmap.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)

        self.label.setPixmap(scaled_pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EarDetectionApp()
    window.show()
    sys.exit(app.exec_())
