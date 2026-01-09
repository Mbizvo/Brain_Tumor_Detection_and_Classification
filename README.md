# 🧠 Brain Tumor Detection and Classification

This project implements a **deep learning–based medical imaging system** for detecting and classifying brain tumors from MRI scans. 
It combines **computer vision, convolutional neural networks (CNNs), and a web-based interface** to support automated and accurate diagnosis.
The system was developed as part of an academic and applied research project in **Data Science and Informatics**.

---

## 📌 Project Objectives

- Automatically detect whether a brain MRI contains a tumor
- Classify the tumor type from MRI images
- Provide a simple web interface for uploading and analyzing scans
- Demonstrate real-world use of deep learning in medical imaging

---

## 🏗 System Architecture

The project follows a **machine learning + web application architecture**:

### Components
- **Deep Learning Model** – Trained CNN for tumor detection & classification  
- **Flask Web App** – Upload MRI images and display predictions  
- **Model Layer** – Loads trained models and performs inference  
- **Frontend** – HTML, CSS, and JavaScript for user interaction  

---

## 📂 Project Structure

| Folder/File | Purpose |
|-------------|--------|
| `model/` | Training code to build the model |
| `test_data/` | Sample MRI images for testing |
| `templates/` | HTML templates for the web app |
| `static/` | CSS, JavaScript, and UI assets |
| `migrations/` | Database migrations |
| `instance/` | Local application database |
| `app.py` | Main Flask application |
| `manage.py` | App management and database handling |
| `requirements.txt` | Python dependencies |
| `update_dates.py` | Utility script |
| `LICENSE` | MIT License |

---

## 🧪 Model & Techniques

The system uses **Convolutional Neural Networks (CNNs)** to extract visual features from MRI images and classify them.

### Key techniques:
- Image preprocessing and normalization  
- Deep CNN feature extraction  
- Supervised learning with labeled MRI data  
- Classification into tumor categories  

---

## ⚙️ Technologies Used

| Area | Tools |
|------|-------|
| Programming | Python |
| Deep Learning | TensorFlow / Keras |
| Web Framework | Flask |
| Frontend | HTML, CSS, JavaScript |
| Data Processing | NumPy, OpenCV |
| Version Control | Git & GitHub |

---

## 🚀 How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/Mbizvo/brain_tumor_detection_and_classification.git
cd brain_tumor_detection_and_classification
