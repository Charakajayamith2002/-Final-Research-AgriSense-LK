
---

# AgriSense-LK

### AI-Based Agricultural Business Prediction System for Sri Lanka

AgriSense-LK is a **machine learning–based decision support system** designed to identify the most profitable agricultural opportunities in Sri Lanka. The system analyzes crop prices, transportation costs, and market distances to predict profitable agricultural business options for farmers and traders.

The project integrates **Artificial Intelligence, Geospatial Analysis, and Market Analytics** to support data-driven agricultural decisions.

---

# Features

• Crop price prediction using **Machine Learning models**
• Economic center proximity analysis
• Profitability prediction for farmers and sellers
• Integration with **MongoDB database**
• Image-based crop recognition using **CNN models**
• AI-driven decision support system

---

# Technology Stack

| Technology     | Purpose                     |
| -------------- | --------------------------- |
| Python         | Core programming language   |
| TensorFlow     | Deep learning model         |
| Scikit-Learn   | Machine learning algorithms |
| Flask          | Web application framework   |
| MongoDB        | Database                    |
| Pandas / NumPy | Data processing             |
| LightGBM       | Prediction model            |
| Geopy          | Distance calculation        |
| Pillow         | Image processing            |

---

# System Requirements

• Python **3.10 or higher**
• MongoDB installed
• pip package manager

---

# Installation Guide

## 1. Clone the Repository

```bash
git clone https://github.com/your-repository/agrisense-lk.git
cd agrisense-lk
```

---

# 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

# 3. Install Required Packages

```bash
pip install numpy==1.26.4
pip install scipy==1.15.3
pip install pandas==2.2.3
pip install joblib==1.5.2
pip install tensorflow==2.19.0
pip install pymongo
pip install scikit-learn
pip install flask
pip install flask-pymongo
pip install lightgbm
pip install geopy
pip install pillow
pip install openpyxl
```

---

# Database Setup

Install and start **MongoDB**.

Create a database named:

```
DB
```

Then run the setup script:

```bash
python setup.py
```

This will initialize the database structure required for the system.

---

# Running the Application

Start the Flask application:

```bash
python app.py
```

Open your browser and navigate to:

```
http://localhost:5000
```

---

# Project Structure

```
AgriSense-LK
│
├── app.py
├── setup.py
├── model_loader.py
├── models/
├── dataset/
├── static/
├── templates/
└── README.md
```

---

# Research Contribution

This research introduces the **Economic-Center Analytics (Ranking & Proximity)** framework, combining:

• AI anomaly detection
• Haversine-based geospatial analysis
• Learning-to-rank algorithms

to optimize agricultural market decision-making.

---

# Authors

Vinuja
BSc (Hons) in Information Technology
Sri Lanka Institute of Information Technology (SLIIT)

---

# License

This project is developed for **academic and research purposes**.

---

If you want, I can also help you create **3 more important files for your research GitHub project**:

1️⃣ `requirements.txt` (auto install all packages with one command)
2️⃣ `setup.py` professional database initializer
3️⃣ **Perfect GitHub project description + tags** (very important for visibility)

Just tell me 👍
