import glob
import cv2
import numpy as np
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import roc_curve, auc


class Eigenfaces:
    def __init__(self):
        self.pca_compunant_dictionary = {}

    def fit(self, X, y, Eigen_threshold=0.90, min_distance_threshould=10000000):
        flatten_images = [image.flatten() for image in X]
        flatten_images = np.vstack(flatten_images)
        self.mean_image = np.mean(flatten_images, axis=0)
        flatten_images_centroid = flatten_images - self.mean_image
        coverance_matrix = (flatten_images_centroid.T @ flatten_images_centroid) / (len(X) - 1)
        # print(coverance_matrix.shape)
        print(coverance_matrix[0])

        eigenValues, eigenVectors = np.linalg.eig(coverance_matrix)
        print(eigenValues.shape)
        eigenValues_sum = sum(eigenValues)
        print(eigenValues_sum)
        used_eigenValues = []
        self.used_eigenVectors = []
        reached_sum = 0
        for i, j in zip(eigenValues, eigenVectors):
            if (reached_sum < Eigen_threshold * eigenValues_sum):
                used_eigenValues.append(i)
                self.used_eigenVectors.append(j)
                reached_sum += i
            else:
                break
        for i, label in enumerate(y):
            self.pca_compunant_dictionary[label] = []
            for vector in self.used_eigenVectors:
                weight = (flatten_images[i] - self.mean_image) @ vector
                self.pca_compunant_dictionary[label].append(weight)
        print(self.pca_compunant_dictionary)

    def predict(self, X):
        predictions = []
        for x in X:
            x_flat = x.flatten()
            value_weights = []
            for vector in self.used_eigenVectors:
                weight = (x_flat - self.mean_image) @ vector
                value_weights.append(weight)
            label_distance_dic = {}
            for label, weights in self.pca_compunant_dictionary.items():
                distance = np.sqrt(np.sum((np.array(weights) - np.array(value_weights)) ** 2))
                label_distance_dic[label] = distance
            sorted_dict = sorted(label_distance_dic.items(), key=lambda item: item[1])
            predictions.append(sorted_dict[0][0][:10])  # Take the top prediction
        return predictions

    def predict_proba(self, X):
        proba = []
        for x in X:
            x_flat = x.flatten()
            value_weights = []
            for vector in self.used_eigenVectors:
                weight = (x_flat - self.mean_image) @ vector
                value_weights.append(weight)
            label_distance_dic = {}
            for label, weights in self.pca_compunant_dictionary.items():
                distance = np.sqrt(np.sum((np.array(weights) - np.array(value_weights)) ** 2))
                label_distance_dic[label] = distance
            # Compute softmax probabilities based on distances
            distances = np.array([label_distance_dic[label] for label in self.pca_compunant_dictionary.keys()])
            exp_distances = np.exp(-distances)
            softmax = exp_distances / np.sum(exp_distances)
            proba.append({label: p for label, p in zip(self.pca_compunant_dictionary.keys(), softmax)})
        return proba

    def score(self, X, y):
        predictions = self.predict(X)
        correct = sum(1 for pred, true in zip(predictions, y) if pred[:10] == true[:10])
        return correct / len(y)

    def roc_curve(self, X, y):
        y_scores = self.predict_proba(X)
        fpr, tpr, thresholds = roc_curve(y, y_scores, pos_label='positive class label')
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    def get_category_images_and_labels(self, X, y, category):
        # Initialize lists to hold images and corresponding labels
        images = []
        labels = []

        # Iterate over the dataset and collect images and labels for the specified category
        for image, label in zip(X, y):
            if category in label:
                images.append(image)
                labels.append(label)

        return images, labels

    def calculate_roc(self, true_labels, pred_labels):
        # Calculate the ROC curve
        fpr, tpr, thresholds = self.roc_curve(true_labels, pred_labels)
        # Calculate the Area Under Curve (AUC)
        roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc

    def roc_curve(self, X, y):

        unique_classes = np.unique(y)

        # y_scores = self.predict(X)
        # Initialize a 2D list to hold images and corresponding labels for each category
        category_data = []

        # Iterate over each category
        for category in unique_classes:
            # Get images and labels for the current category
            category_images, category_labels = self.get_category_images_and_labels(X, y, category)

            # Predict labels for the current category
            y_pred_category = self.predict(category_images)

            # Store the images, true labels, and predicted labels for the current category
            category_data.append({
                category: [category_labels, y_pred_category],
            })
            self.calculate_roc(category_labels, y_pred_category)


def create_dataset(dataset_path, fixed_window_size=(62, 47)):
    class_labels = []
    total_no_images, no_of_classes = 0, 0

    for images in glob.glob(dataset_path + '/**', recursive=True):
        if images[-3:] in ['jpg', 'png']:
            total_no_images += 1

    images_dataset = np.zeros((total_no_images, fixed_window_size[0], fixed_window_size[1]), dtype='float64')
    i = 0

    testing_set_path = "dataSets/"
    os.makedirs(testing_set_path, exist_ok=True)

    for folder in glob.glob(dataset_path + '/*'):
        no_of_classes += 1
        print(folder)
        images_in_folder = glob.glob(folder + '/*')
        num_images_to_use = int(len(images_in_folder) * 0.7)
        print(num_images_to_use)

        for index, image in enumerate(glob.glob(folder + '/*')):
            if index > num_images_to_use:
                image_name = os.path.basename(image)
                testing_image_path = os.path.join(testing_set_path, folder.split("/")[-1], image_name)
                os.makedirs(os.path.dirname(testing_image_path), exist_ok=True)
                shutil.copy(image, testing_image_path)
                continue
            
            image = detect_largest_face(cv2.imread(image, cv2.IMREAD_GRAYSCALE))
            
            if image is None:
                continue
            
            class_labels.append(folder.split("/")[-1])

            resized_image = cv2.resize(image, (fixed_window_size[1], fixed_window_size[0]))
            images_dataset[i] = np.array(resized_image)
            i += 1

    # print(len(class_labels))
    # print(class_labels)

    np.save('dataSets/images_dataset.npy', images_dataset)
    np.save('dataSets/class_labels.npy', class_labels)


def pca(images_dataset):
    flattened_data = np.reshape(images_dataset, (images_dataset.shape[0], -1))

    mean_image = np.mean(flattened_data, axis=0, dtype='float64')

    centered_data = flattened_data - np.tile(mean_image, (images_dataset.shape[0], 1))

    covariance_matrix = centered_data.dot(centered_data.T) / images_dataset.shape[0]

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and eigenvectors by descending order
    sorted_eigenvectors = eigenvectors[:, eigenvalues.argsort()[::-1]]

    # Projecting the dataset onto principal components
    principal_components = centered_data.T @ sorted_eigenvectors

    # Normalizing the principal components
    real_part = np.real(principal_components.T)
    imaginary_part = np.imag(principal_components.T)
    normalized_real = preprocessing.normalize(real_part)
    normalized_imaginary = preprocessing.normalize(imaginary_part)
    eigenfaces = normalized_real + 1j * normalized_imaginary

    print(eigenfaces.shape)


    return eigenfaces, centered_data, mean_image


def detect_largest_face(image):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    largest_face = None
    largest_area = 0
    for (x, y, w, h) in faces:
        area = w * h
        if area > largest_area:
            largest_area = area
            largest_face = (x, y, w, h)

    if largest_face is not None:
        x, y, w, h = largest_face
        cropped_face = image[y:y+h, x:x+w]
        return cv2.resize(cropped_face, (62, 47))
    else:
        return None

    
def recognize_face(test_image, eigenfaces, training_data, mean_image, class_labels, num_eigenfaces=100, fixed_size=(62, 47)):
    face_image = detect_largest_face(test_image)

    if face_image is None:
        return "Not A Face"

    resized_face = cv2.resize(face_image, fixed_size[::-1])
    
    face_vector = np.reshape(resized_face, (resized_face.shape[0] * resized_face.shape[1])) - mean_image

    # Project the face vector onto the subspace of the chosen eigenfaces
    projected_face = eigenfaces[:num_eigenfaces].dot(face_vector)

    # Variables to store the minimum distance and corresponding class index
    min_distance = None
    closest_class_index = None

    # Calculate distances between the projected test image and each training image
    for i in range(training_data.shape[0]):
        training_data.shape[0]

        projected_training_face = eigenfaces[:num_eigenfaces].dot(training_data[i])

        distance = np.sqrt(np.sum((projected_face - projected_training_face)**2))

        if min_distance is None or distance < min_distance:
            min_distance = distance
            closest_class_index = i

    if min_distance < 1300:
        print(min_distance)
        return class_labels[closest_class_index]
    else:
        print(f"Face not recognized. Minimum distance: {min_distance}")
        return "Unknown"
    
create_dataset("./Georgia tech dataset/")