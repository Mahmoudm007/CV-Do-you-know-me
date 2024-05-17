from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import cv2
import numpy as np


class RealTimeVideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    stop_signal = False

    def __init__(self):
        super().__init__()
        self.current_frame = None
        # Load the pre-trained Haar Cascade classifiers for face and eye detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, cv_img = cap.read()
            if ret and not self.stop_signal:
                # Store the current frame
                self.current_frame = cv_img
                # Detect faces in the current frame
                faces = self.face_cascade.detectMultiScale(cv_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                # Draw rectangles around the detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # Extract the face region
                    face_region = cv_img[y:y+h, x:x+w]
                    # Detect eyes in the face region
                    eyes = self.eye_cascade.detectMultiScale(face_region, scaleFactor=1.3, minNeighbors=5)
                    # Draw rectangles around the detected eyes
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(cv_img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 2)
                        # Extract the eye region
                        eye_region = face_region[ey:ey+eh, ex:ex+ew]
                        # Analyze intensity distribution within the eye region
                        mean_intensity = eye_region.mean()
                        # Analyze texture of the eye region using Gabor filters
                        means, stds = self.gabor_filter_analysis(eye_region)
                        # Determine if the eye is open or closed based on intensity and texture
                        if mean_intensity < 80 and all(std < 10 for std in stds):
                            status = "Closed"
                        else:
                            status = "Open"
                        # Display the status (open/closed) above the eye rectangle
                        cv2.putText(cv_img, status, (x+ex, y+ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Emit the updated frame for display
                self.change_pixmap_signal.emit(cv_img)
            else:
                break
        cap.release()

    def gabor_filter_analysis(self, eye_region):
        ksize = 31  # Kernel size
        sigma = 5   # Standard deviation of the Gaussian envelope
        lambd = 10  # Wavelength of the sinusoidal factor
        gamma = 0.5  # Spatial aspect ratio
        psi = 0     # Phase offset

        # Create a bank of Gabor filter kernels
        kernels = []
        for theta in np.arange(0, np.pi, np.pi / 4):
            # Bandpass filter
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
            kernels.append(kernel)

        # Convolve the image with each Gabor filter kernel
        filtered_images = [cv2.filter2D(eye_region, cv2.CV_32F, kernel) for kernel in kernels]

        # Compute mean and standard deviation of the filtered images
        means = [np.mean(img) for img in filtered_images]
        stds = [np.std(img) for img in filtered_images]
        return means, stds

    def detect_eyes(self, face_region):
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        return eyes

    def detect_eye_status(self, eye_region):
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        # Thresholding to binarize the eye region
        ret, thresh_eye = cv2.threshold(gray_eye, 25, 255, cv2.THRESH_BINARY_INV)
        # Calculate the area of the white pixels in the eye region
        white_area = np.sum(thresh_eye == 255)
        # Calculate the total area of the eye region
        total_area = eye_region.shape[0] * eye_region.shape[1]
        # Calculate the ratio of white area to total area
        white_ratio = white_area / total_area
        # Determine if the eye is closed or open based on the white area ratio
        if white_ratio < 0.3:
            return "Closed"
        else:
            return "Open"

    def stop(self):
        self.stop_signal = True