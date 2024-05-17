import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
from PIL import Image
import os
import cv2
import numpy as np
import json
import joblib

from EigenFaces import Eigenfaces


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def complex_and_array_decoder(obj):
    if '_complex_' in obj:
        return complex(obj['real'], obj['imag'])
    elif '_numpy_array_' in obj:
        return np.array(obj['data'])
    return obj


class DetectionClass(QMainWindow):
    def __init__(self):
        super().__init__()
        # Load the UI file
        loadUi('mainWindow.ui', self)
        self.setWindowTitle("Do you know me!!")
        self.imagePath = None
        self.scaleFactor = 1.3
        self.minNeighbors = 5
        self.minSize = (30, 30)
        
        
        #Recognition
        self.data_dir = 'YaleFaces/' 
        self.images = []
        self.labels=[]
        self.rec_image = None
        self.read_images(self.data_dir)
        self.model_data = None
        self.read_model()
        
        # Connect the button click to the add_image method
        self.browseBtn.clicked.connect(self.add_image)
        self.apply_ears.clicked.connect(self.detect_ears)
        self.apply_eyes.clicked.connect(self.detect_eyes)
        self.apply_mouths.clicked.connect(self.detect_mouths)
        self.apply_nose.clicked.connect(self.detect_nose)
        self.apply_pose.clicked.connect(self.detect_pose)
        self.show_db.clicked.connect(self.show_database)
        self.recognize.clicked.connect(self.recognize_face)


    def add_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        file_dialog.setViewMode(QFileDialog.Detail)
        files, _ = file_dialog.getOpenFileNames(self, "Select Image")
        for file in files:
            self.imagePath = file
            self.display_image(self.imagePath)


    def display_image(self, image_path):
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(self.input_image.size(), aspectRatioMode=True, transformMode=False)
        self.input_image.setPixmap(scaled_pixmap)
        self.input_image.setScaledContents(True)
        self.load_rec_image()

    # TODO: connect scaleFactor, minNeighbors, minSize to the line edits



    # Function to read images with unusual extensions and convert to OpenCV format
    def read_images(self , directory):
        
        for filename in os.listdir(directory):
            if filename.startswith('subject'):  # Adjust the extension as per your dataset
                filepath = os.path.join(directory, filename)
                with Image.open(filepath) as img:
                    # Convert image to numpy array
                    img=img.resize((100,100))
                    np_image = np.array(img)
                    self.images.append(np_image)
                    self.labels.append(filename)
                    
                    
                    
                    
        ############################### ears detection ##################################
    def detect_ears(self):
        # Load the image using OpenCV
        image = cv2.imread(self.imagePath)

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
        scaled_pixmap = pixmap.scaled(self.output_image.width(), self.output_image.height(), Qt.KeepAspectRatio)

        self.output_image.setPixmap(scaled_pixmap)

    ############################### smiles detection ##################################
    def detect_mouths(self):
        if self.imagePath:
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

            image = cv2.imread(self.imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=20)

            for (x, y, w, h) in smiles:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imwrite("smile_detected_image.jpg", image)
            pixmap = QPixmap("smile_detected_image.jpg")
            self.output_image.setPixmap(pixmap)
            self.output_image.setScaledContents(True)
        else:
            print("Please load an image first.")

    ############################### eye detection ##################################
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
    def detect_eyes(self):
        # Load the image using OpenCV
        image = cv2.imread(self.imagePath)

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
        scaled_pixmap = pixmap.scaled(self.output_image.width(), self.output_image.height(), Qt.KeepAspectRatio)

        self.output_image.setPixmap(scaled_pixmap)

    ############################### POSE detection ##################################
    def detect_pose(self):
        if self.imagePath:
            image = cv2.imread(self.imagePath)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert the image to grayscale for processing
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Convert grayscale image to RGB
            image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
            
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                results = pose.process(image_rgb)  # Process the grayscale image for pose detection
                
                # Draw the landmarks (points) and connection lines on the original RGB image
                mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),
                                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3))
                
                # Convert the annotated image back to BGR format for saving
                annotated_image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                # Save the annotated image
                cv2.imwrite("annotated_image.jpg", annotated_image_bgr)
                
                # Load the saved image and display it in the output label
                pixmap = QPixmap("annotated_image.jpg")
                scaled_pixmap = pixmap.scaled(self.output_image.size(), aspectRatioMode=True, transformMode=False)
                self.output_image.setPixmap(scaled_pixmap)
                self.output_image.setScaledContents(True)
        else:
            print("Please load an image first.")


    ############################### nose detection ##################################
    def detect_nose(self):
        # Load the image using OpenCV
        image = cv2.imread(self.imagePath)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Preprocess the image: Apply histogram equalization to improve contrast
        equalized = cv2.equalizeHist(gray)

        # Load the pre-trained Haar Cascade for nose detection
        nose_cascade = cv2.CascadeClassifier('XMLs/haarcascade_mcs_nose.xml')

        # Detect noses in the image
        noses = nose_cascade.detectMultiScale(equalized, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected noses
        for (x, y, w, h) in noses:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert the image to RGB format for displaying in PyQt5
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to QImage
        height, width, channel = image_rgb.shape
        bytesPerLine = 3 * width
        q_image = QImage(image_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)

        # Convert QImage to QPixmap and display in QLabel
        pixmap = QPixmap.fromImage(q_image)

        # Scale the image to fit the label
        scaled_pixmap = pixmap.scaled(self.output_image.width(), self.output_image.height(), Qt.KeepAspectRatio)

        self.output_image.setPixmap(scaled_pixmap)
        
            ############################### Recognition ##################################
    def read_model(self):
        # Define the file path
        file_path = 'data.json'

        # Read the JSON file using the custom decoder
        with open(file_path, 'r') as json_file:
            self.model_data = json.load(json_file, object_hook=complex_and_array_decoder)    
            
            
            
    def show_database(self):
        # Display images from each subject in 3 rows
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))  # 3 rows, 5 columns

        # Dictionary to store the index of the first image for each subject
        subject_indices = {}

        # Counter for the current row and column
        row_index = 0
        col_index = 0

        # Iterate through the image labels
        for filename in self.labels:
            subject = filename.split('.')[0]  # Extract the subject number from the filename

            # If the subject is encountered for the first time, assign a new index and display the image
            if subject not in subject_indices:
                subject_indices[subject] = (row_index, col_index)
                axes[row_index, col_index].imshow(self.images[self.labels.index(filename)], cmap='gray')
                axes[row_index, col_index].axis('off')
                axes[row_index, col_index].set_title(subject)

                # Update row and column indices
                col_index += 1
                if col_index == 5:  # Move to the next row if the current row is filled
                    row_index += 1
                    col_index = 0

            # Break the loop if we have displayed one image for each subject
            if len(subject_indices) == 15:
                break

        plt.tight_layout()
        plt.show()
    
    def load_rec_image(self):
        if 'subject' in self.imagePath:  # Check if the image path contains 'subject'
            with Image.open(self.imagePath) as img:

                self.rec_image = np.array(img)

                # Convert numpy array to QPixmap
                q_image = QImage(self.rec_image.data, self.rec_image.shape[1], self.rec_image.shape[0], QImage.Format_Indexed8)
                pixmap = QPixmap.fromImage(q_image)

                # Scale the image to fit the QLabel
                scaled_pixmap = pixmap.scaled(self.rec_input.size(), aspectRatioMode=True, transformMode=False)
                self.rec_input.setPixmap(scaled_pixmap)
                self.rec_input.setScaledContents(True)
        else:
            print("Image path does not contain 'subject'. Please select an appropriate image.")

    def recognize_face(self):
        self.model_load()
        img = np.copy(self.rec_image)

        img = cv2.resize(img, (100, 100))
        prediction = self.model.predict([img])[0]

        # Search for the index of the first image whose label contains the prediction substring
        index = next((i for i, label in enumerate(self.labels) if prediction in label), None)

        # If an index is found, retrieve the corresponding image; otherwise, return None
        if index is not None:
            corresponding_image = self.images[index]
            self.rec_image = np.array(corresponding_image)

            # Convert numpy array to QPixmap
            q_image = QImage(self.rec_image.data, self.rec_image.shape[1], self.rec_image.shape[0],
                             QImage.Format_Indexed8)
            pixmap = QPixmap.fromImage(q_image)

            # Scale the image to fit the QLabel
            scaled_pixmap = pixmap.scaled(self.rec_result.size(), aspectRatioMode=True, transformMode=False)
            self.rec_result.setPixmap(scaled_pixmap)
            self.rec_result.setScaledContents(True)

            # return corresponding_image
        else:
            print("No matching image found for the prediction:", prediction)
            return None
    
    def model_load(self):
        self.model = joblib.load("model.pkl")


    


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DetectionClass()
    window.show()
    sys.exit(app.exec_())
