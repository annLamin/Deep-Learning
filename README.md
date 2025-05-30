Tomato Disease Classification Using Deep Learning
1. Introduction
Plant diseases significantly impact agricultural productivity. Early detection of diseases in crops like potatoes can help farmers take preventive measures. This project focuses on classifying potato leaf diseases using a Convolutional Neural Network (CNN). The model was trained on a dataset of potato leaf images and deployed using FastAPI and Node.js for a user-friendly web interface.
2. Project Overview
The goal was to develop an end-to-end deep learning pipeline for potato disease classification:

Data Collection: Gather tomato leaf images (healthy and diseased) from kaggle https://www.kaggle.com/datasets/arjuntejaswi/plant-village.

Preprocessing: Resize, normalize, and augment images.

Model Training: Build and train a CNN using TensorFlow/Keras.

Deployment: Serve the model via FastAPI and integrate with a Node.js frontend.

User Interface: Allow users to upload images and get predictions.

