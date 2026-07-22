# 👁️ Diabetic Retinopathy Detection using Deep Learning

> An AI-powered web application for early detection and classification of Diabetic Retinopathy from retinal fundus images using **CLAHE, EfficientNet-B0, and SVM**.


## 📌 Overview

Diabetic Retinopathy (DR) is one of the leading causes of preventable blindness among diabetic patients. Early diagnosis can significantly reduce the risk of vision loss.

This project presents an automated deep learning framework that classifies retinal fundus images into **five severity levels** of diabetic retinopathy.

The application is built using:

- 🧠 EfficientNet-B0 for feature extraction
- ✨ CLAHE for image enhancement
- 📊 Support Vector Machine (SVM) for classification
- 🌐 Streamlit for an interactive web interface

---

## 🎯 Features

- Upload retinal fundus images
- Automatic retinal image validation
- CLAHE image enhancement
- Five-class DR classification
- Confidence score prediction
- Clean and interactive Streamlit UI
- Fast inference
- Easy deployment

---

## 🏗️ Model Pipeline

```
Input Fundus Image
        │
        ▼
Retinal Image Validation
        │
        ▼
CLAHE Enhancement
        │
        ▼
Resize & Normalize
        │
        ▼
EfficientNet-B0 Feature Extraction
        │
        ▼
Feature Scaling
        │
        ▼
Support Vector Machine (RBF)
        │
        ▼
Prediction
```

---

## 📊 DR Classes

| Class | Severity |
|--------|----------|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

---

## 📂 Project Structure

```
dr-detection-app/
│
├── app.py
├── requirements.txt
├── svm_model.pkl
├── scaler.pkl
├── effb0_se_best.pth
├── assets/
├── utils/
├── models/
├── images/
└── README.md
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/snehachalla9/dr-detection-app.git

cd dr-detection-app
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open at

```
http://localhost:8501
```

---

## 📦 Tech Stack

- Python
- PyTorch
- EfficientNet-B0
- OpenCV
- Scikit-learn
- Streamlit
- Pillow
- NumPy
- Pandas

---

## 📈 Dataset

**APTOS 2019 Blindness Detection**

- 3,662 retinal fundus images
- Five DR severity classes

Dataset:

https://www.kaggle.com/competitions/aptos2019-blindness-detection

---

## 📊 Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **82.95%** |
| Weighted Precision | 82.50% |
| Weighted Recall | 82.95% |
| Weighted F1 Score | 82.01% |
| Cohen's Kappa | **0.8844** |

---


## 🔮 Future Improvements

- Grad-CAM Visualization
- Explainable AI (XAI)
- PDF Report Generation
- Doctor Recommendation
- Cloud Deployment
- Mobile Application
- Multi-language Support

---

## ⚠️ Disclaimer

This application is developed **for educational and research purposes only** and should **not be used as a substitute for professional medical diagnosis.**

---

## 👩‍💻 Author

**Sneha Challa**

GitHub:
https://github.com/snehachalla9

---

## ⭐ Support

If you found this project useful,

⭐ Star the repository

🍴 Fork it

🤝 Contribute to improve it

---

## 📄 License

This project is licensed under the MIT License.
