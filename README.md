# Phishing-URL-Detection-System-Machine-Learning-Based-
Phishing attacks are one of the most common cyber threats, where attackers create fake websites to steal sensitive user information such as usernames, passwords, and banking details.
This project implements a Machine Learning–based phishing URL detection system that classifies URLs as phishing or legitimate using security-related features.

🎯 Objectives

Detect phishing websites automatically

Reduce dependency on manual rule-based detection

Apply machine learning techniques to real-world cybersecurity problems

🛠️ Technologies Used

Programming Language: Python

Machine Learning: Scikit-learn

Data Processing: Pandas, NumPy

Feature Extraction: Regex, urllib

Model Used: Random Forest Classifier

🧠 System Architecture

Input URL

Feature extraction from URL

Machine Learning model analysis

Output classification (Phishing / Legitimate)

📂 Project Structure
phishing-url-detection/
│
├── dataset/
│   └── urls.csv
├── feature_extraction.py
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md

📊 Dataset

The dataset contains URLs labeled as:

1 → Phishing

0 → Legitimate

Example:
http://secure-login-paypal.com , 1
https://www.google.com        , 0


📌 Dataset sources:

PhishTank

Kaggle phishing datasets

🔍 Feature Extraction

Each URL is converted into numerical features before training.

Features Used:

URL length

Number of dots (.)

Presence of @ symbol

Presence of hyphen (-)

Use of HTTPS

IP address in URL

Presence of suspicious keywords (login, verify, bank, secure)

These features help identify patterns commonly used in phishing URLs.

🤖 Machine Learning Model

Algorithm: Random Forest Classifier

Reason for selection:

High accuracy

Handles non-linear data well

Reduces overfitting

Widely used in cybersecurity applications

📈 Model Performance

Achieved ~95% accuracy on test data

Evaluated using:

Accuracy score

Precision & Recall

Confusion matrix

▶️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/phishing-url-detection.git
cd phishing-url-detection

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model
python train_model.py

4️⃣ Predict a URL
python predict.py

🧪 Sample Output
Input URL: http://secure-paypal-login.xyz
Prediction: PHISHING

⚠️ Limitations

Cannot detect zero-day phishing attacks

Depends on dataset quality

Does not analyze webpage content

🚀 Future Enhancements

Flask-based web application

Browser extension integration

Deep learning models (LSTM, CNN)

Real-time URL scanning
