import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap

class SmileDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Smile Detection App")
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

        self.detect_button = QPushButton("Detect Smile")
        self.detect_button.clicked.connect(self.detect_smile)
        self.layout.addWidget(self.detect_button)

        self.image_path = None

    def add_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        file_dialog.setViewMode(QFileDialog.Detail)
        files = file_dialog.getOpenFileNames(self, "Select Image")[0]
        if len(files) > 0:
            self.image_path = files[0]
            pixmap = QPixmap(self.image_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)

    def detect_smile(self):
        if self.image_path:
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

            image = cv2.imread(self.image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=20)

            for (x, y, w, h) in smiles:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imwrite("smile_detected_image.jpg", image)
            pixmap = QPixmap("smile_detected_image.jpg")
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)
        else:
            print("Please load an image first.")

def main():
    app = QApplication(sys.argv)
    window = SmileDetectionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
