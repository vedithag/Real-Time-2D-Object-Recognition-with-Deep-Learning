# Real-Time-2D-Object-Recognition-with-Deep-Learning

## **Project Overview**

This project develops a **Real-time 2-D Object Recognition System** that classifies various objects based on their shape, color, and texture. It combines traditional image processing methods with deep neural network (DNN) embeddings to achieve reliable object recognition on a white background. The primary objective is to implement a classification system leveraging various computer vision techniques, enabling the system to detect and classify objects in real time.

### **Core Features:**
- **Thresholding and Dynamic Thresholding:** Converts color images to binary for better background-foreground separation.
- **Morphological Processing:** Applies erosion and dilation operations to enhance image segmentation.
- **Segmentation and Feature Extraction:** Uses connected component analysis for detecting object boundaries and calculates features such as Hu Moments, aspect ratios, and region fills.
- **Training Mode:** Supports training in real time, allowing new object labels and feature vectors to be stored.
- **Classification Techniques:** Implements both K-Nearest Neighbors (KNN) and a DNN-based model for object classification.
- **Confusion Matrix for Performance Evaluation:** Provides accuracy, precision, and recall metrics to assess classification performance.

---

### **Technologies Used:**
- **Programming Language:** C++
- **Libraries:** OpenCV for image processing, OpenCV's DNN module for deep learning inference
- **Model:** Pre-trained ONNX model for deep neural network-based classification

### **Key Functional Modules:**

1. **Thresholding and Segmentation**
   - **Thresholding:** The `thresholdImage` function converts color images to grayscale and then applies a binary threshold to separate objects from the background.
   - **Dynamic Thresholding:** Adjusts the threshold based on K-means clustering, enhancing segmentation accuracy in varying lighting conditions.
   - **Morphological Operations:** Applies erosion and dilation to refine segmented regions and remove noise.

2. **Feature Extraction**
   - **Hu Moments and Shape Analysis:** Calculates Hu Moments and bounding box properties (aspect ratio, orientation) for each detected contour, providing shape-based descriptors for object classification.
   - **Region Features:** Extracts additional metrics such as the area, centroid, and contour orientation to aid in object differentiation.

3. **Training and Classification Modes**
   - **KNN Classification:** Utilizes KNN with Euclidean distance as the primary distance metric, allowing real-time classification based on previously stored feature vectors.
   - **DNN-Based Classification:** Uses a pre-trained DNN (ONNX) model to extract feature embeddings and classify objects based on cosine similarity.
   - **Confusion Matrix and Performance Metrics:** Tracks classification performance with precision and recall values, and identifies misclassifications to improve model accuracy.

---

### **Advanced Techniques:**
- **Dynamic Thresholding:** Clusters color values to dynamically calculate an adaptive threshold, ensuring robust segmentation under diverse conditions.
- **Combined KNN and DNN Classification:** Integrates both traditional KNN and advanced DNN-based classification, leveraging DNN’s ability to recognize high-level semantic features.
- **Custom Evaluation Metrics:** Implements a confusion matrix, measuring the true positives, false positives, and overall accuracy to continually assess and improve system performance.

### **Skills Demonstrated**

- **Computer Vision & Image Processing:** Proficient in segmentation, morphological processing, and advanced feature extraction techniques like Hu Moments and contour analysis.
- **Machine Learning & Deep Learning:** Experienced in DNN embeddings for high-dimensional vector comparison, including cosine similarity calculations for DNN-based classification.
- **C++ Programming:** Developed efficient, modular code using C++ with STL containers for high-performance data handling, showcasing object-oriented programming practices.

---

### **Dependencies**
- **C++ Compiler** (GCC recommended)
- **OpenCV 4.x**
- **ONNX DNN Model File:** A pre-trained model for embedding extraction, required for DNN-based classification.


### **Key Technical Skills Demonstrated**
#### **Computer Vision & Image Processing**
- Proficiency in image segmentation and feature extraction, using techniques such as binary thresholding, dynamic thresholding, and morphological processing.
- Extensive use of OpenCV for contour analysis, including Hu Moments, aspect ratios, orientation calculations, and histogram-based methods for accurate object recognition.

#### **Machine Learning & Deep Learning**
- Applied K-Nearest Neighbors (KNN) for real-time object classification using Euclidean distance and feature vectors.
- Utilized a pre-trained DNN model in ONNX format for advanced feature embeddings and similarity-based classification.
- Implemented cosine similarity and Euclidean distance metrics to compare high-dimensional embeddings and traditional feature vectors.

#### **C++ Programming**
- Structured C++ code that follows object-oriented programming principles for modular and efficient software design.
- Used STL containers like `vector` and `map` to handle data effectively, ensuring performance optimization for real-time processing.
- Experience with handling data from CSV files for feature storage and retrieval, as well as training new labels in real-time.
