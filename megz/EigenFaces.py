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
from sklearn.metrics import roc_curve, auc


class Eigenfaces:
    def __init__(self):
        self.pca_compunant_dictionary={}
        
    def fit(self,X,y,Eigen_threshold=0.90,min_distance_threshould=10000000):
        flatten_images=[image.flatten() for image in X]
        flatten_images = np.vstack(flatten_images)
        self.mean_image=np.mean(flatten_images, axis=0)
        flatten_images_centroid = flatten_images - self.mean_image
        coverance_matrix=(flatten_images_centroid.T@flatten_images_centroid)/(len(X)-1)
        # print(coverance_matrix.shape)
        print(coverance_matrix[0])
        
        eigenValues, eigenVectors = np.linalg.eig(coverance_matrix)
        print(eigenValues.shape)
        eigenValues_sum=sum(eigenValues)
        print(eigenValues_sum)
        used_eigenValues=[]
        self.used_eigenVectors=[]
        reached_sum=0
        for i,j in zip(eigenValues, eigenVectors):
            if(reached_sum<Eigen_threshold*eigenValues_sum):
                used_eigenValues.append(i)
                self.used_eigenVectors.append(j)
                reached_sum+=i
            else:
                break
        for i,label in enumerate(y):
            self.pca_compunant_dictionary[label]=[]
            for vector in self.used_eigenVectors:
                weight=(flatten_images[i]-self.mean_image)@vector
                self.pca_compunant_dictionary[label].append(weight)
        print(self.pca_compunant_dictionary) 

    def predict(self, X):
        predictions = []
        for x in X:
            print(x)
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
