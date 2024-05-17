import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QScrollArea
from PyQt5.QtGui import QImage, QPixmap

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Processor')
        self.setGeometry(100, 100, 400, 300)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setGeometry(10, 10, 380, 200)
        self.scroll_area.setWidgetResizable(True)

        self.image_label = QLabel(self.scroll_area)
        self.image_label.setGeometry(0, 0, 380, 200)

        self.upload_button = QPushButton('Upload Image', self)
        self.upload_button.setGeometry(150, 220, 100, 30)
        self.upload_button.clicked.connect(self.uploadImage)

        self.show()

    def uploadImage(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.bmp *.gif)", options=options)
        if file_name:
            image = cv2.imread(file_name)
            if image is not None:
                self.displayImage(image)
                self.processImage(image)

    def displayImage(self, image):
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        self.image_label.setPixmap(pixmap)

    def processImage(self, image):
        matrix = image.tolist()
        print(matrix)

def main():
    app = QApplication(sys.argv)
    window = ImageProcessor()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
