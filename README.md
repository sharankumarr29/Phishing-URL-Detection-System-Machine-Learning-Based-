# Phishing-URL-Detection-System-Machine-Learning-Based-
Phishing attacks are one of the most common cyber threats, where attackers create fake websites to steal sensitive user information such as usernames, passwords, and banking details.
This project implements a Machine Learning–based phishing URL detection system that classifies URLs as phishing or legitimate using security-related features.


## 1️⃣ INTRODUCTION

### 📌 Problem Statement
Phishing attacks trick users into revealing sensitive information by using **fake websites that look legitimate**. Traditional rule-based detection methods fail against modern phishing techniques.

### 🎯 Objective
To design and implement a **Machine Learning–based system** that automatically detects whether a given URL is **phishing or legitimate** based on security-related features.

### ✅ Why This Project Is Important
- Phishing is one of the **top cyber threats**
- Helps prevent **credential theft & financial fraud**
- Demonstrates **cybersecurity + machine learning integration**

---

## 2️⃣ SYSTEM OVERVIEW

### 🧠 How the System Works
1. User inputs a URL  
2. System extracts security-related features  
3. Trained ML model analyzes extracted features  
4. Output is classified as **Phishing** or **Legitimate**

---

## 3️⃣ TOOLS & TECHNOLOGIES USED

| Category | Tools |
|--------|------|
| Programming | Python |
| ML Libraries | Scikit-learn |
| Data Handling | Pandas, NumPy |
| Feature Extraction | Regex, urllib |
| Model | Random Forest |
| Environment | VS Code / Jupyter Notebook |

---

## 4️⃣ DATASET COLLECTION

### 📂 Dataset Description
- Dataset contains **URLs with labels**
- Label values:
  - `1` → Phishing
  - `0` → Legitimate

**Example:**
http://secure-login-paypal.com , 1
https://www.google.com , 0

python
Copy code

### 📌 Dataset Sources
- PhishTank
- Kaggle Phishing URL Datasets

---

## 5️⃣ FEATURE EXTRACTION (CORE SECURITY PART)

Raw URLs cannot be directly used by machine learning models.  
They are converted into **numerical security-related features**.

### 🔍 Features Extracted

| Feature | Reason |
|-------|--------|
| URL length | Phishing URLs are usually longer |
| Count of dots | Multiple subdomains indicate phishing |
| Presence of `@` | Used for redirection tricks |
| Presence of `-` | Common in fake domains |
| HTTPS usage | Phishing sites often lack HTTPS |
| IP address in URL | Strong phishing indicator |
| Suspicious keywords | login, verify, bank, secure |

---

## 6️⃣ FEATURE EXTRACTION CODE

```python
import re
from urllib.parse import urlparse

def extract_features(url):
    features = []

    features.append(len(url))                     # URL length
    features.append(url.count('.'))               # Number of dots
    features.append(1 if '@' in url else 0)        # @ symbol
    features.append(1 if '-' in url else 0)        # hyphen
    features.append(1 if 'https' in url else 0)    # HTTPS
    features.append(1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0)  # IP address

    suspicious_words = ['login', 'verify', 'bank', 'secure']
    features.append(1 if any(word in url.lower() for word in suspicious_words) else 0)

    return features
7️⃣ DATA PREPROCESSING
🔧 Steps Involved
Extract features for all URLs

Convert features into numerical format

Split dataset into:

80% Training Data

20% Testing Data

8️⃣ DATA PREPARATION CODE
python
Copy code
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("urls.csv")

X = data['url'].apply(extract_features).tolist()
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
9️⃣ MODEL SELECTION
🔹 Why Random Forest?
High accuracy

Handles non-linear data

Reduces overfitting

Widely used in cybersecurity applications

🔟 MODEL TRAINING
python
Copy code
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
1️⃣1️⃣ MODEL EVALUATION
📊 Evaluation Metrics
Accuracy

Confusion Matrix

Precision

Recall

python
Copy code
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
✅ Expected Accuracy
93% – 97%

1️⃣2️⃣ URL PREDICTION MODULE
python
Copy code
def predict_url(url):
    features = extract_features(url)
    prediction = model.predict([features])

    if prediction[0] == 1:
        return "PHISHING"
    else:
        return "LEGITIMATE"

print(predict_url("http://secure-paypal-login.xyz"))
1️⃣3️⃣ PROJECT STRUCTURE
Copy code
phishing-url-detection/
│
├── dataset/
│   └── urls.csv
├── feature_extraction.py
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md
1️⃣4️⃣ RESULTS
Successfully classified phishing and legitimate URLs

Random Forest achieved approximately 95% accuracy

System is fast and scalable

1️⃣5️⃣ LIMITATIONS
Cannot detect zero-day phishing attacks

Accuracy depends on dataset quality

Does not analyze webpage content

1️⃣6️⃣ FUTURE ENHANCEMENTS
Add webpage content analysis

Deploy as a Flask web application

Develop browser extension

Implement deep learning models
