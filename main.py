import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QImage, QPixmap

from PyQt5 import uic
from PyQt5.QtGui import QIcon
import cv2

# CLASSES
import functions as f


class FaceRecgonitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()


    def init_ui(self):
        self.ui = uic.loadUi('design.ui', self)
        self.setWindowTitle("Face Recgonition Application")
        self.setWindowIcon(QIcon("icons/face-recognize.png"))
        self.load_ui_elements()


    def load_ui_elements(self):
         self.upload_button = self.ui.uploadButton
         self.input_image_label = self.ui.inputImageLabel
         self.connect_signals()
         
    def connect_signals(self):
        self.upload_button.clicked.connect(self.uploadImage)
        
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
        self.input_image_label.setPixmap(pixmap)
    
    # here we will load the model parameters and use it
    def processImage(self, image):
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = FaceRecgonitionApp()
    mainWin.show()
    sys.exit(app.exec())
