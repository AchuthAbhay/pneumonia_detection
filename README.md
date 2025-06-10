# Pneumonia & Lung Disease Detection from Chest X-Rays 🫁🩻

A deep learning project for detecting **Pneumonia and lung diseases** from chest X-ray images using **ResNet-50**. Built with a user-friendly **Streamlit** interface and enhanced with **Grad-CAM** for explainable AI.

---

## 🚀 Features

- 🔍 Detects Pneumonia from X-ray images
- 🧠 Deep learning with **ResNet-50**
- 🎯 High accuracy on test images
- 📊 Grad-CAM visualization to explain model decisions
- 🌐 Streamlit web app for real-time predictions

---

## 🗂️ Dataset

Dataset used: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

- Structure: `train/`, `val/`, and `test/`
- Classes: `NORMAL`, `PNEUMONIA`

---

## 🧑‍💻 Tech Stack

- Python
- PyTorch
- ResNet-50 (pretrained on ImageNet)
- OpenCV, PIL, Matplotlib
- Grad-CAM
- Streamlit

---

## 📺 Demo Video

[![Watch Demo Video](https://img.shields.io/badge/Watch-Demo%20Video-blue?logo=google-drive)](https://drive.google.com/file/d/1VE1U9ofFAWf0xnU0aw3PQuKUxVO16Y8J/view?usp=sharing)


## 🛠️ Setup Instructions

```bash
# Clone the repository
git clone https://github.com/AchuthAbhay/pneumonia_detection.git
cd pneumonia_detection

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
