# Tomato Disease Classification Using Deep Learning (CNN)
## 1. Introduction
Plant diseases significantly impact agricultural productivity. Early detection of diseases in crops like potatoes can help farmers take preventive measures. This project focuses on classifying potato leaf diseases using a Convolutional Neural Network (CNN). The model was trained on a dataset of potato leaf images and deployed using FastAPI and Node.js for a user-friendly web interface.
## 2. Project Overview
The goal was to develop an end-to-end deep learning pipeline for tomato disease classification:

- Data Collection: Gather tomato leaf images (healthy and diseased) from  [kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village).
- Preprocessing: Resize, normalize, and augment images.
- Model Training: Build and train a CNN using TensorFlow/Keras.
- Container and Deployment: Use Docker and serve the model via FastAPI and Google Cloud Platform (GCP)
- Frontend: integrate with a Node.js frontend.
- User Interface: Allow users to upload images and get predictions.

## 3. Tech Stack
- Backend & Model Training:

- Python, TensorFlow, Keras

- FastAPI (for API deployment)

- Docker (containerization)

- GCP (Google Cloud Platform for deployment)

- Frontend: Node.js (UI development)

### Tools & Libraries:

- OpenCV, Pillow (image processing)
- Matplotlib, NumPy (visualization & computations)
- TensorFlow Serving (model serving)
- Postman (API testing)

## 4. Workflow
### Architecture Diagram
![Architecture Diagram](https://github.com/annLamin/Deep-Learning/tree/main/workflow_diagram.jpg)


## 5. Data Collection & Preprocessing
### Dataset
- 10 classes: ['Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy'].

- Total Images: ~16,011.

- Split: 80% training, 20% validation.

### Preprocessing Steps
- Resizing: Images resized to 256x256 pixels.
- Normalization: Pixel values scaled to [0, 1].

### Data Augmentation:

- Rotation, flipping, zooming.
- Helps prevent overfitting.

## 6. Model Development (CNN)
### CNN Architecture
![Architecture Diagram](https://github.com/annLamin/Deep-Learning/tree/main/model_arch.jpg)

### Model Summary



## 7. Training & Evaluation
### Training Parameters
- Epochs: 50
- Batch Size: 32
- Optimizer: Adam
- Loss Function: Categorical Crossentropy

### Accuracy & Loss Curves



### Performance Metrics
#### Metric	Value
- Training Accuracy	98.5%
- Validation Accuracy	96.2%
- Test Accuracy	95.8%



## 8. Model Deployment
- FastAPI Backend
- Docker & GCP Deployment
- GCP Deployment: Containerized model deployed on Google Cloud Run.
- API endpoint exposed for Node.js UI.


### Web Application (Node.js UI)
#### Features
- Upload potato leaf images.

- Display disease prediction with confidence score.



## 10. Results & Performance Metrics
- Achieved 96% validation accuracy.

- FastAPI ensures low-latency predictions.

- Node.js provides a user-friendly interface.  


## 11. Conclusion
- Successfully developed a CNN-based tomato disease classifier.
- Deployed model using FastAPI & Node.js.
- This will help farmers detect diseases early, improving crop yield.


## 12. Future Enhancements
- Mobile app integration and Real-time camera-based detection.
- Build a chatbot for pesticide suggestions to combat disease.
- Weather forecast integration.
- Expand to other crops.


