# 🔭 Radio Galaxy Morphological Classification

A deep learning system for classifying extragalactic radio sources into:

* **FR0**
* **FRI**
* **FRII**

This project includes:

* 🧠 CNN-based classification model
* ☁️ Deployed inference API (GCP Cloud Run)
* 🎯 Interactive web UI (Streamlit)

---

## 🚀 Live Demo

👉 **Try the app here:**
https://morphologicalclassificationofradiogalaxiesh5-7dpxrx9n4wkenetga.streamlit.app/

Upload a radio galaxy image and get:

* Predicted class (FR0 / FRI / FRII)
* Confidence score
* Class probability distribution

---

## 🧠 Model Overview

* Input: grayscale radio galaxy images
* Architecture: Convolutional Neural Network (CNN)
* Output:

  * Class label
  * Probability distribution over classes

---

## 🏗️ System Architecture

```
Streamlit UI  →  Cloud Run API  →  CNN Model
```

* **Frontend**: Streamlit (interactive UI)
* **Backend**: REST API deployed on GCP Cloud Run
* **Inference**: Model served via HTTP

---

## 📦 Project Structure

```
.
├── app/               # API (model + inference)
├── UI/                # Streamlit UI
│   ├── app.py
│   └── requirements.txt
├── Dockerfile         # API container
├── requirements.txt   # API dependencies
```

---

## 🔌 API Usage

### Health Check

```
GET /health
```

### Prediction

```
POST /
Content-Type: image/jpeg or image/png
```

Example:

```
curl -X POST \
  -H "Content-Type: image/jpeg" \
  --data-binary "@image.jpg" \
  https://radio-galaxy-api-783206752653.europe-west1.run.app/
```

---

## 🖥️ Run UI Locally

```
cd UI
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧪 Features

* End-to-end ML pipeline (training → deployment → UI)
* Real-time inference via REST API
* Clean UI for interaction
* Cloud-native deployment (GCP)

---

## 📌 Future Improvements

* Improve model accuracy
* Add Grad-CAM visualization
* Batch inference support
* Improve UI/UX

---

## 👤 Author

**Seyed Ali Rashidi (Farid)**
Computer Vision & AI Engineer
Infrared Imaging & Optical Systems

---

## ⭐ Notes

This project demonstrates:

* Computer Vision (CNN)
* MLOps (deployment, API, containerization)
* System design (UI + backend separation)
