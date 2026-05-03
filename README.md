# 🧬 Malaria Detection Using Deep Learning

A full-stack machine learning web application that detects malaria infection from microscopic blood cell images using a deep learning model.

---

## 🚀 Live Demo

* 🌐 Frontend (Netlify):
https://elegant-sunburst-0de7d4.netlify.app/

* ⚙️ Backend API (Render):
  https://malaria-backend-ussf.onrender.com/

---

## 📌 Overview

This project uses a deep learning model to classify cell images into:

* **Parasitized (Infected)**
* **Uninfected (Healthy)**

Users can upload an image through a web interface and get predictions in real time.

---

## 🧠 Tech Stack

* **Frontend:** HTML, CSS, JavaScript
* **Backend:** Flask (Python)
* **Model:** PyTorch (MobileNet / ResNet)
* **Deployment:**

  * Backend → Render
  * Frontend → Netlify
* **Containerization:** Docker

---

## ⚙️ System Architecture

Frontend (Netlify)
↓
Backend API (Render, Flask + Gunicorn)
↓
PyTorch Model (.pth)
↓
Prediction + Confidence

---

## 🖼️ Features

* Upload cell images
* Real-time prediction
* Confidence score output
* Lightweight deployment (MobileNet optimized)
* Cloud-hosted backend and frontend

---

## 📂 Project Structure

```
malaria-detector/
│
├── backend/
│   ├── main.py
│   ├── malaria_model.pth
│   ├── requirements.txt
│   ├── Dockerfile
│
├── frontend/
│   ├── index.html
│   ├── style.css
│
└── README.md
```

---

## 🧪 API Usage

### Endpoint:

```
POST /predict
```

### Example (curl):

```
curl -X POST -F "file=@test.png" https://malaria-backend-ussf.onrender.com/predict
```

### Response:

```json
{
  "prediction": "Parasitized",
  "confidence": 92.45
}
```

---

## ⚠️ Notes

* First request may take **20–30 seconds** (Render free tier sleep)
* Model runs on CPU (free-tier constraint)
* Ensure correct model architecture matches `.pth` file

---

## 🧑‍💻 Setup Locally

### Backend

```
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend

Open `index.html` in browser

---

## 🐳 Docker Usage

```
docker build -t malaria-app .
docker run -p 5000:5000 malaria-app
```

---

## 🎓 Learning Outcomes

* Built end-to-end ML system
* Deployed model using Docker
* Integrated frontend with backend API
* Handled real-world constraints (memory, latency)

---

## 📌 Future Improvements

* Use GPU-based inference
* Improve model accuracy
* Add explainable AI (prediction reasoning)
* Enhanced UI/UX (medical dashboard)

---

## 🙌 Acknowledgement

Dataset: Malaria Cell Images (NIH dataset via Kaggle)

---

## 📎 License

This project is for academic and educational purposes.
