# CV Do You know Me!!

cv_do_you_know_me is a robust computer vision project that integrates multiple advanced features for facial and body part detection, real-time video processing, and facial recognition. This project is designed to explore the capabilities of various computer vision libraries and models, making it a comprehensive tool for developers and researchers interested in face detection, landmark tracking, and more.

## Table of Contents
1. [Face Detection](#face-detection)
2. [Eye Detection](#eye-detection)
3. [Ear Detection](#ear-detection)
4. [Mouth Detection](#mouth-detection)
5. [Nose Detection](#nose-detection)
6. [Skin Segmentation](#skin-segmentation)
7. [Pose Estimation](#pose-estimation)
8. [Real-time Face and Eye Detection](#real-time-face-and-eye-detection)
9. [Real-time Snapchat Filters](#real-time-snapchat-filters)
10. [PCI Faces Recognition](#pci-faces-recognition)
11. [Real-time Face Recognition](#real-time-face-recognition)

## Features

### 1. Face Detection
The Face Detection feature utilizes the `MTCNN` deep learning model to accurately detect human faces within an image. This model is specifically optimized for high performance and accuracy. After detecting faces, the feature crops the image around each detected face, producing isolated facial images. This can be particularly useful for applications like automated photo organization, facial recognition preprocessing, or even anonymization tasks where faces need to be extracted or blurred out.

![Face Detection](gifs/face.gif)

### 2. Eye Detection
Eye Detection is implemented using [Haarcascade XMLs](XMLs), a machine learning-based approach for object detection. This feature not only locates the eyes within a given image but also analyzes the state of the eyes, determining whether they are open or closed. This capability is critical for applications such as driver drowsiness detection systems, human-computer interaction, and emotion recognition systems. The Haarcascade classifiers are lightweight and efficient, making them suitable for real-time applications.

![Eye Detection](gifs/eye.gif)

### 3. Ear Detection
The Ear Detection feature also employs [Haarcascade XMLs](XMLs) to identify and locate ears in images. Users can adjust the `scaleFactor` and `minNeighbors` parameters to refine detection accuracy. The `scaleFactor` parameter controls how much the image size is reduced at each image scale, while `minNeighbors` specifies how many neighbors each rectangle should have to retain it. This feature can be essential in biometric identification systems, hearing aid fitting, or virtual accessory applications, where accurate ear localization is necessary.

![Ear Detection](gifs/ear.gif)

### 4. Mouth Detection
Mouth Detection is designed to identify the mouth region within an image using [Haarcascade XMLs](XMLs). In addition to detecting the mouth, this feature analyzes facial expressions to determine if the person is smiling. This can be leveraged in various applications, such as automated photo enhancement (e.g., detecting smiles for auto-capture), mood analysis, or even in systems that trigger actions based on user expressions.

![Mouth Detection](gifs/mouth.gif)

### 5. Nose Detection
The Nose Detection feature utilizes the [Haarcascade XMLs](XMLs) technique to locate the nose in an image. Accurate nose detection is often a critical component of comprehensive facial recognition systems and can also be used in applications like virtual try-on for glasses or medical applications where nasal analysis is required.

![Nose Detection](gifs/nose.gif)

### 6. Skin Segmentation
Skin Segmentation is a powerful feature that segments skin regions from the rest of the image. This feature can be instrumental in a variety of applications, including augmented reality (e.g., applying virtual makeup), deepfake technologies, and medical imaging where skin regions need to be isolated for further analysis. The segmentation process can be adjusted to handle different skin tones and lighting conditions, making it adaptable to various real-world scenarios.

![Skin Segmentation](gifs/skin.gif)

### 7. Pose Estimation
Pose Estimation uses the Mediapipe library to detect and segment the landmarks of a human pose. This feature tracks key points of the body, such as joints and skeletal structures, providing detailed information about body posture and movement. Pose estimation is widely used in fitness applications for exercise tracking, sports analytics, interactive gaming, and virtual reality, where accurate tracking of body movements is essential.

![Pose Estimation](gifs/pose.gif)

### 8. Real-time Face and Eye Detection
This feature provides real-time detection of faces and eyes using a webcam. It continuously scans the video feed for faces and identifies the eyes, including their status (open or closed). The real-time capability is achieved through efficient algorithms optimized for live processing, making this feature ideal for security surveillance, user engagement systems, and interactive video applications where immediate feedback is required.

![Real-time Face and Eye Detection](gifs/detection.gif)

### 9. Real-time Snapchat Filters
Real-time Snapchat Filters offer a fun and interactive experience by allowing users to add virtual accessories like glasses or mustaches in real-time as they look into the camera. Users can switch filters on and off with ease. This feature is built for entertainment applications, social media platforms, and even virtual meeting tools where users can personalize their appearance with playful elements.

![Real-time Snapchat Filters](gifs/filters-min.gif)

### 10. PCI Faces Recognition
The PCI Faces Recognition feature implements face recognition from scratch using eigenfaces and Principal Component Analysis (PCA). This method reduces the dimensionality of facial images while retaining the most critical information for recognition. The model is trained on the [YaleFaces dataset](https://www.kaggle.com/datasets/olgabelitskaya/yale-face-database), a widely used dataset in facial recognition research. The feature includes performance evaluation through confusion matrices and ROC curves, offering insights into the model's accuracy and reliability. This is especially useful for academic research and development of face recognition systems.

![PCI Faces Recognition](gifs/recognition.gif)

### 11. Real-time Face Recognition
This feature leverages a deep learning model to perform face recognition in real-time. Unlike traditional methods, the deep learning model can learn and extract features from a limited number of samples, making it robust in real-world applications. It processes live video from a webcam and identifies known faces with high accuracy. This feature is essential for security systems, personalized experiences in retail or customer service, and any application where quick and reliable face recognition is needed.


## Getting Started
To get started with this project, clone the repository and follow the setup instructions provided in the `Installation` section.

## Installation
Follow these steps to set up the project on your local machine:

```bash
git clone https://github.com/your_username/cv_do_you_know_me.git
cd cv_do_you_know_me
pip install -r requirements.txt


### Team
------------------------------------------

| Team Members' Names | 
|---------------------|
| [Ahmed Kamal](https://github.com/AhmedKamalMohammedElSayed)|
| [Amgad Atef](https://github.com/amg-eng)| 
| [Mahmoud Magdy](https://github.com/MahmoudMagdy404)|       
| [Mahmoud Mohamed ](https://github.com/Mahmoudm007)|     