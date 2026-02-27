# cat-dog-image-classifier
A machine learning &amp; computer vision based Cat vs Dog image classifier built using SVM and OpenCV.

📌 Project Overview
This project is a Machine Learning based Computer Vision system that classifies images into Cat or Dog categories using:
1.Support Vector Machine (SVM)
2.OpenCV for image preprocessing
3.Feature extraction techniques

The objective of this project is to build a traditional ML-based image classifier without using deep learning, demonstrating strong fundamentals in machine learning and image processing.

🎯 Problem Statement
To build a binary image classification model that can accurately distinguish between cat and dog images using classical machine learning techniques.

🧠 Model Used
1.Support Vector Machine (SVC) from Scikit-learn
2.Kernel-based classification approach
3.Supervised learning algorithm

Why SVM?
1.Effective in high-dimensional spaces
2.Works well for binary classification
3.Good performance with limited data

🛠️ Tech Stack
Python
OpenCV
NumPy
Scikit-learn
Matplotlib
Jupyter Notebook

🔄 Project Workflow
1️⃣ Data Collection
Cat and Dog image dataset
2️⃣ Image Preprocessing
Image loading using OpenCV
Resizing images to fixed dimensions
Converting to grayscale (if applied)
Flattening image arrays
3️⃣ Feature Extraction
Converting images into numerical feature vectors
4️⃣ Model Training
Splitting data into training and testing sets
Training SVM classifier
5️⃣ Model Evaluation
Accuracy score
Performance analysis
6️⃣ Prediction
Classifies unseen images as Cat or Dog

📊 Results
Model Accuracy: (Add your accuracy here, e.g., 85%)
Binary classification output:
0 → Cat
1 → Dog

▶️ How to Run This Project
1.Install dependencies:
pip install -r requirements.txt

2.Run the Jupyter Notebook:
jupyter notebook

🚀 Future Improvements
Implement CNN using TensorFlow/Keras
Improve feature extraction techniques
Add real-time webcam classification
Deploy as a web application using Flask or Streamlit

💡 Key Learnings
Image preprocessing using OpenCV
Feature vector transformation
Implementation of SVM for classification
Understanding model evaluation metrics
Practical application of supervised learning

Author:
Ankit Mishra
