# Image Classification Using Bag of Words (BoW) Model with SIFT Features

## Description

This project aims to develop an image recognition application utilizing the Scale-Invariant Feature Transform (SIFT) features and a Bag-of-Words (BoW) model on the Describable Textures Dataset (DTD). By employing various machine learning algorithms such as K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and AdaBoost, the project evaluates and determines the optimal approach for texture classification.

## Dataset

The Describable Textures Dataset (DTD) is a comprehensive collection of textured images designed for the analysis, modeling, and synthesis of textures. It comprises 47 texture classes, each with 120 instances, capturing various textures found in fabrics, surfaces, natural materials, and artistic patterns. For this project, four classes were selected:
- Wrinkled
- Perforated
- Waffled
- Studded

A total of 480 images were used, and they were split into training, validation, and test sets.

## Models and Methods

### Bag of Words Model
The BoW model was trained using SIFT features to create a dictionary of descriptors. This dictionary was used to create histograms representing the frequency of descriptors in each image.

### Optimal K Value
The optimal number of clusters (K) for the KMeans algorithm was determined using the elbow method and the silhouette coefficient score.

### Classifiers
Three classifiers were tested:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- AdaBoost

Each classifier was fine-tuned to determine the best hyperparameters for texture classification.

## Results

### KNN Classifier
- **Optimal K**: 6 (Elbow Method)
- **Accuracy**: 72.5% (Validation Set)
- **Accuracy**: 76.67% (Test Set)

### SVM Classifier
- **Optimal C**: 80
- **Accuracy**: 79.17% (Validation Set)
- **Accuracy**: 80.83% (Test Set)

### AdaBoost Classifier
- **Optimal Number of Estimators**: 250
- **Accuracy**: 69.17% (Validation Set)
- **Accuracy**: 79.17% (Test Set)

### Confusion Matrices
- **Confusion matrices for each classifier are provided in the respective CSV files.**

## Conclusion

The SVM classifier achieved the highest accuracy with a value of 80.83% on the test set, indicating that the data is linearly separable. The project demonstrates that a diverse codebook can capture more trends, leading to better performance in texture classification.

## Requirements

To replicate this project, ensure you have the following dependencies installed:

```sh
torch
torchvision
numpy
opencv-python
scikit-learn
matplotlib
yellowbrick
