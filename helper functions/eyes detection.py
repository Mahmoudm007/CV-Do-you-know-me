import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Detection App")
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
            self.detect_eyes(file)

    def gabor_filter_analysis(self, eye_region):
        # Define parameters for Gabor filter bank
        ksize = 31  # Kernel size
        sigma = 5   # Standard deviation of the Gaussian envelope
        lambd = 10  # Wavelength of the sinusoidal factor
        gamma = 0.5 # Spatial aspect ratio
        psi = 0     # Phase offset

        # Create a bank of Gabor filter kernels
        kernels = []
        for theta in np.arange(0, np.pi, np.pi / 4):
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
            kernels.append(kernel)

        # Convolve the image with each Gabor filter kernel
        filtered_images = [cv2.filter2D(eye_region, cv2.CV_32F, kernel) for kernel in kernels]

        # Compute mean and standard deviation of the filtered images
        means = [np.mean(img) for img in filtered_images]
        stds = [np.std(img) for img in filtered_images]

        print("Gabor Filter Means:", means)
        print("Gabor Filter Stds:", stds)

        return means, stds

    def detect_eyes(self, image_path):
        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Preprocess the image: Apply histogram equalization to improve contrast
        equalized = cv2.equalizeHist(gray)

        # Load the pre-trained Haar Cascade for eye detection
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Detect eyes in the image
        
        # set k-nn to 3 for the image 1eye closed
        # set k-nn to 5 for the image other images of eyes
        eyes = eye_cascade.detectMultiScale(equalized, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))

        # Draw rectangles around the detected eyes and determine if they are open or closed
        for (x, y, w, h) in eyes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extract the eye region
            eye_region = gray[y:y + h, x:x + w]

            # Analyze intensity distribution within the eye region
            mean_intensity = eye_region.mean()

            # Analyze texture of the eye region using Gabor filters
            means, stds = self.gabor_filter_analysis(eye_region)

            # Determine if the eye is open or closed based on intensity and texture
            if mean_intensity < 80 or all(std < 10 for std in stds):
                status = "Closed"
            else:
                status = "Open"

            # Display the status (open/closed) next to the eye rectangle
            cv2.putText(image, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
