import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np

class SkinColorDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skin Color Detection App")
        self.setGeometry(100, 100, 1200, 800)

        # Create a central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Add a QLabel to display original image
        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.original_image_label)

        # Add a QLabel to display skin color labeled image
        self.skin_color_label = QLabel()
        self.skin_color_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.skin_color_label)

        # Add a button to load an image
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_image_button)

        # Initialize image variable
        self.image = None

    def load_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.process_image(file_path)

    def detect_skin_color(self, image):
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for skin color in HSV
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Threshold the HSV image to get only skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Bitwise-AND mask and original image
        result = cv2.bitwise_and(image, image, mask=mask)

        # Convert BGR to RGB
        skin_color_image_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # Convert the skin color labeled image to QImage and display in the skin color label
        height, width, channel = skin_color_image_rgb.shape
        bytesPerLine = 3 * width
        q_image_skin_color = QImage(
            skin_color_image_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888
        )
        pixmap_skin_color = QPixmap.fromImage(q_image_skin_color)
        self.skin_color_label.setPixmap(pixmap_skin_color)

    def process_image(self, file_path):
        # Load the image
        self.image = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # Convert the OpenCV image to QImage and display in the original image label
        height, width, channel = image_rgb.shape
        bytesPerLine = 3 * width
        q_image = QImage(image_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.original_image_label.setPixmap(pixmap)

        # Detect skin color
        self.detect_skin_color(self.image)

        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SkinColorDetectionApp()
    window.show()
    sys.exit(app.exec_())
