# ✍️ Handwriting Recognition with EMNIST

This project is a deep learning-based application that recognizes **handwritten cursive characters** using the **EMNIST ByClass** dataset. It supports preprocessing, segmentation, and classification of handwritten input via a Streamlit web app interface.

---

## 📦 Features

- ✅ Character segmentation for cursive handwriting
- ✅ CNN model trained on EMNIST dataset (uppercase, lowercase, digits)
- ✅ Preprocessing using OpenCV (thresholding, resizing)
- ✅ Real-time prediction with Streamlit interface
- ✅ Modular structure with separate utilities

---

## 🧠 Model Overview

- **Model type:** Convolutional Neural Network (CNN)
- **Input size:** 28x28 grayscale images
- **Output:** 62 classes (26 uppercase + 26 lowercase + 10 digits)
- **Frameworks:** TensorFlow / Keras

---

## 🗂️ Folder Structure
![image](https://github.com/user-attachments/assets/e5be41a2-6972-4f6c-96fd-4d1eb62bbe10)



## 🗂️ Data Set 
> ⚠️ `emnist_data/` is not included in the repo due to size. See below for dataset download.
> https://www.kaggle.com/datasets/crawford/emnist
![image](https://github.com/user-attachments/assets/3dce3eee-f081-4d50-88cb-2123a37aab0c)

## 🖥️ Installation & Setup
### 1. Clone the repository
```bash
git clone https://github.com/meilisady/handwritingRecognition.git
cd handwritingRecognition
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run handwriting_cursive_recognition.py
```
