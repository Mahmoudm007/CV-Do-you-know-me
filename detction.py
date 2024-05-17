import sys
import cv2
import mtcnn
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.uic import loadUi
from PIL import Image
import os
import numpy as np
import mediapipe as mp
import json
import joblib
from matplotlib import pyplot as plt
from EigenFaces import Eigenfaces
from detectionFacesRealTime import RealTimeVideoThread
from realTimeFilter import AddingSnapFilters
from face_recogntion_realtime import recognitionDisplay


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

        self.videoFrame = self.findChild(QtWidgets.QFrame, 'videoFrame')
        self.image_label = self.findChild(QtWidgets.QLabel, 'imageLabel')
        self.screenshot_button = self.findChild(QtWidgets.QPushButton, 'screenShotButton')

        #Recognition
        self.data_dir = 'YaleFaces/'
        self.images = []
        self.labels= []
        self.rec_image = None
        self.read_images(self.data_dir)
        self.model_data = None
        self.read_model()

        self.detector = mtcnn.MTCNN()

        # Connect the button click to the add_image method
        self.browseBtn.clicked.connect(self.add_image)
        self.browseBtnDetection.clicked.connect(self.add_image_detection)
        self.apply_faces.clicked.connect(self.detect_faces)
        self.apply_ears.clicked.connect(self.detect_ears)
        self.apply_eyes.clicked.connect(self.detect_eyes)
        self.apply_mouths.clicked.connect(self.detect_mouths)
        self.apply_nose.clicked.connect(self.detect_nose)
        self.apply_pose.clicked.connect(self.detect_pose)
        self.applySkinSegmentation.clicked.connect(self.detect_skin_color)
        self.show_db.clicked.connect(self.show_database)
        self.recognize.clicked.connect(self.recognize_face)
        self.screenshot_button.clicked.connect(self.take_screenshot)
        self.addGlassess.clicked.connect(self.add_glasses)
        self.addMus.clicked.connect(self.add_mustache)
        self.startReal.clicked.connect(self.start_real_time_video)
        self.closeReal.clicked.connect(self.stop_real_time_video)
        self.startSnap.clicked.connect(self.start_snap_filters)
        self.closeSnap.clicked.connect(self.stop_snap_filters)
        self.closeRec.clicked.connect(self.stop_real_time_rec)
        self.startRec.clicked.connect(self.start_real_time_rec)

        # self.real_time_reco = recognitionDisplay()
        # self.real_time_reco.change_pixmap_signal.connect(self.update_frame)
        # Initialize threads
        self.real_time_video_thread = RealTimeVideoThread()
        self.real_time_video_thread.change_pixmap_signal.connect(self.update_image_detection_real)

        self.snap_filters_thread = AddingSnapFilters()
        self.snap_filters_thread.change_pixmap_signal.connect(self.update_filter_image)

    def add_glasses(self):
        self.snap_filters_thread.toggle_add_glasses()

    def add_mustache(self):
        self.snap_filters_thread.toggle_add_mustache()

    @pyqtSlot(QImage)
    def update_frame(self, frame):
        self.realTimeReco.setPixmap(QPixmap.fromImage(frame))

    def start_real_time_rec(self):
        self.real_time_reco.start()

    def stop_real_time_rec(self):
        self.real_time_reco.stop()


    @pyqtSlot(np.ndarray)
    def update_image_detection_real(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt_real_time_detection(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt_real_time_detection(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

        # Calculate scaling factors for width and height
        width_ratio = self.image_label.width() / w
        height_ratio = self.image_label.height() / h
        ratio = min(width_ratio, height_ratio)

        # Scale the image to fit the label, maintaining aspect ratio
        scaled_w = int(w * ratio)
        scaled_h = int(h * ratio)
        p = convert_to_Qt_format.scaled(scaled_w, scaled_h, Qt.KeepAspectRatio)

        return QPixmap.fromImage(p)

    def start_real_time_video(self):
        self.real_time_video_thread.start()

    def stop_real_time_video(self):
        self.real_time_video_thread.stop()

    def start_snap_filters(self):
        self.snap_filters_thread.start()

    def stop_snap_filters(self):
        self.snap_filters_thread.stop()

    def take_screenshot(self):
        try:
            # Ensure the screenshots folder exists
            if not os.path.exists("screenshots"):
                os.makedirs("screenshots")

            # Capture the current frame from the video
            frame = self.real_time_video_thread.current_frame

            # Save the frame as a screenshot
            if frame is not None:
                cv2.imwrite(f"screenshots/screenshot_{len(os.listdir('screenshots')) + 1}.png", frame)
        except Exception as e:
            print("Error occurred while taking screenshot:", e)

    @pyqtSlot(np.ndarray)
    def update_filter_image(self, cv_img):
        qt_img = self.convert_cv_qt_filter_video(cv_img)
        self.filterCam.setPixmap(qt_img)

    def convert_cv_qt_filter_video(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape

        # Calculate scaling factors for width and height
        width_ratio = self.filterCam.width() / w
        height_ratio = self.filterCam.height() / h
        ratio = min(width_ratio, height_ratio)

        # Scale the image to fit the label, maintaining aspect ratio
        scaled_w = int(w * ratio)
        scaled_h = int(h * ratio)
        q_image = QtGui.QImage(rgb_image.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_image).scaled(scaled_w, scaled_h, Qt.KeepAspectRatio)

        return pixmap

    def add_image_detection(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        file_dialog.setViewMode(QFileDialog.Detail)
        files, _ = file_dialog.getOpenFileNames(self, "Select Image")
        for file in files:
            self.imagePath = file
            self.display_image_detection(self.imagePath)

    def display_image_detection(self, image_path):
        image = cv2.imread(self.imagePath)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytesPerLine = 3 * width
        q_image = QImage(image_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.input_image.width(), self.input_image.height(), Qt.KeepAspectRatio)
        self.input_image.setPixmap(scaled_pixmap)


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
        self.load_rec_image()
    # Function to read images with unusual extensions and convert to OpenCV format
    def read_images(self, directory):
        for filename in os.listdir(directory):
            if filename.startswith('subject'):  # Adjust the extension as per your dataset
                filepath = os.path.join(directory, filename)
                with Image.open(filepath) as img:
                    # Convert image to numpy array
                    img = img.resize((100, 100))
                    np_image = np.array(img)
                    self.images.append(np_image)
                    self.labels.append(filename)
    ############################### Face detection ##################################
    def detect_faces(self):
        if self.imagePath:
            print("Detecting faces...")
            image = cv2.imread(self.imagePath)
            if image is None:
                print("Failed to read image. Path:", self.imagePath)
                return
            print("Image loaded successfully.")
            faces = self.get_faces([image])
            if faces:
                print("Face detected.")
                # Convert the detected face image to QImage
                q_image = self.convert_cv_qt(faces[0])
                # Display the QImage in the output_image label
                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = pixmap.scaled(self.output_image.width(), self.output_image.height(), Qt.KeepAspectRatio)
                self.output_image.setPixmap(QPixmap.fromImage(q_image))
            else:
                print("No face detected in the image.")

    def detect_face(self, image):
        result = self.detector.detect_faces(image)
        if len(result) == 0:
            return None
        x1, y1, width, height = result[0]['box']
        x1, y1, width, height = int(x1), int(y1), int(width), int(height)
        face = image[y1:y1 + height, x1:x1 + width]
        face = cv2.resize(face, (160, 160))
        return face

    def get_faces(self, images):
        return [self.detect_face(image) for image in images]

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return convert_to_Qt_format

        ############################### ears detection ##################################
    def detect_ears(self):
        # Load the image using OpenCV
        image = cv2.imread(self.imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        ear_cascade = cv2.CascadeClassifier("XMLs/haarcascade_mcs_rightear.xml")  # Specify the path to ear cascade XML file
        scaleFactor_ears = float(self.scaleEars.text()) # 1.1
        minNeighbors_ears = int(self.earsNeighbors.text())  # 3
        ears = ear_cascade.detectMultiScale(equalized, scaleFactor=scaleFactor_ears, minNeighbors=minNeighbors_ears, minSize=(30, 30))
        for (x, y, w, h) in ears:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0 , 255, 0), 2)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytesPerLine = 3 * width
        q_image = QImage(image_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.output_image.width(), self.output_image.height(), Qt.KeepAspectRatio)
        self.output_image.setPixmap(scaled_pixmap)

    ############################### smiles detection ##################################
    def detect_mouths(self):
        if self.imagePath:
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

            image = cv2.imread(self.imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            scaleFactor_mouths = float(self.scaleMouths.text())  # 1.8
            minNeighbors_mouths = int(self.mouthsNeighbors.text())  # 21
            smiles = smile_cascade.detectMultiScale(gray, scaleFactor=scaleFactor_mouths, minNeighbors= minNeighbors_mouths)

            for (x, y, w, h) in smiles:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image_rgb.shape
            bytesPerLine = 3 * width
            q_image = QImage(image_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)

            # Convert QImage to QPixmap and display in QLabel
            pixmap = QPixmap.fromImage(q_image)
            # Scale the image to fit the label
            scaled_pixmap = pixmap.scaled(self.output_image.width(), self.output_image.height(), Qt.KeepAspectRatio)
            self.output_image.setPixmap(scaled_pixmap)

    ############################### eye detection ##################################
    def gabor_filter_analysis(self, eye_region):
        ksize = 31  # Kernel size
        sigma = 5   # Standard deviation of the Gaussian envelope
        lambd = 10  # Wavelength of the sinusoidal factor
        gamma = 0.5  # Spatial aspect ratio
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
        return means, stds

    def detect_eyes(self):
        image = cv2.imread(self.imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        # set k-nn to 3 for the image 1eye closed
        # set k-nn to 5 for the image other images of eyes
        scaleFactor_eyes = float(self.scaleEyes.text()) # 1.3
        minNeighbors_ears = int(self.eyesNeighbors.text())  # 3
        eyes = eye_cascade.detectMultiScale(equalized, scaleFactor=scaleFactor_eyes, minNeighbors=minNeighbors_ears, minSize=(30, 30))
        for (x, y, w, h) in eyes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            eye_region = gray[y:y + h, x:x + w]
            mean_intensity = eye_region.mean()
            means, stds = self.gabor_filter_analysis(eye_region)
            if mean_intensity < 80 or all(std < 10 for std in stds):
                status = "Closed"
            else:
                status = "Open"
            cv2.putText(image, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytesPerLine = 3 * width
        q_image = QImage(image_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.output_image.width(), self.output_image.height(), Qt.KeepAspectRatio)
        self.output_image.setPixmap(scaled_pixmap)

    ############################### POSE detection ##################################
    def detect_pose(self):
        if self.imagePath:
            image = cv2.imread(self.imagePath)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
            min_detection_confidence = float(self.min_detection_confidence.text())  # 0.5
            min_tracking_confidence = float(self.min_tracking_confidence.text())  # 0.5

            # Create the Pose object with the obtained confidence values
            with mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                              min_tracking_confidence=min_tracking_confidence) as pose:
                results = pose.process(image_rgb)
                mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3,
                                                                                       circle_radius=3),
                                          connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0),
                                                                                         thickness=3))
                annotated_image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                height, width, channel = image_rgb.shape
                bytesPerLine = 3 * width
                q_image = QImage(annotated_image_bgr.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = pixmap.scaled(self.output_image.width(), self.output_image.height(), Qt.KeepAspectRatio)
                self.output_image.setPixmap(scaled_pixmap)
        else:
            print("Please load an image first.")

    ############################### nose detection ##################################
    def detect_nose(self):
        image = cv2.imread(self.imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        nose_cascade = cv2.CascadeClassifier('XMLs/haarcascade_mcs_nose.xml')

        scaleFactor_nose = float(self.scaleNose.text())
        minNeighbors_nose = int(self.noseNeighbors.text())
        noses = nose_cascade.detectMultiScale(equalized, scaleFactor= scaleFactor_nose, minNeighbors= minNeighbors_nose, minSize=(30, 30))

        for (x, y, w, h) in noses:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytesPerLine = 3 * width
        q_image = QImage(image_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.output_image.width(), self.output_image.height(), Qt.KeepAspectRatio)
        self.output_image.setPixmap(scaled_pixmap)

    ################################### Skin detection ##################################
    def detect_skin_color(self, image):
        image = cv2.imread(self.imagePath)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        result = cv2.bitwise_and(image, image, mask=mask)
        skin_color_image_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        height, width, channel = skin_color_image_rgb.shape
        bytesPerLine = 3 * width
        q_image_skin_color = QImage(
            skin_color_image_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888
        )
        pixmap_skin_color = QPixmap.fromImage(q_image_skin_color)
        scaled_pixmap = pixmap_skin_color.scaled(self.output_image.width(), self.output_image.height(), Qt.KeepAspectRatio)
        self.output_image.setPixmap(scaled_pixmap)

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
                q_image = QImage(self.rec_image.data, self.rec_image.shape[1], self.rec_image.shape[0],
                                 QImage.Format_Indexed8)
                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = pixmap.scaled(self.rec_input.width(), self.rec_input.height(), Qt.KeepAspectRatio)
                # Scale the image to fit the QLabel
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
