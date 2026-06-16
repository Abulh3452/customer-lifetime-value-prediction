# 📈 Customer Lifetime Value (CLV) Prediction Using RFM Analysis

## Overview

This project predicts Customer Lifetime Value (CLV) using RFM (Recency, Frequency, Monetary) Analysis and Machine Learning.

The system segments customers into:

* ⭐ High Value Customers
* 👍 Medium Value Customers
* ⚠ Low Value Customers

based on their purchasing behavior and predicts their future Customer Lifetime Value (CLV).

A Streamlit web application is developed to allow users to enter customer information and obtain CLV predictions instantly.

---

## Features

* Customer Segmentation using RFM Analysis
* Machine Learning-based CLV Prediction
* Streamlit Interactive Web Application
* Automatic Customer Classification
* User-Friendly Interface
* Real-Time Predictions

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* Joblib
* Streamlit

### Machine Learning

* Ridge
* K-Means Clustering (for customer segmentation)

---

## Dataset Features

The model uses the following customer attributes:

| Feature   | Description                                       |
| --------- | ------------------------------------------------- |
| Recency   | Number of days since the customer's last purchase |
| Frequency | Number of purchases made by the customer          |
| Monetary  | Total amount spent by the customer                |

---

## Project Workflow

1. Data Collection
2. Data Cleaning and Preprocessing
3. RFM Feature Engineering
4. Customer Segmentation
5. Model Training
6. CLV Prediction
7. Streamlit Deployment

---

## RFM Analysis

RFM Analysis evaluates customers based on:

### Recency (R)

How recently a customer made a purchase.

### Frequency (F)

How often a customer makes purchases.

### Monetary (M)

How much money a customer spends.

Customers are categorized into:

* High Value
* Medium Value
* Low Value

---

## Model Training

The dataset is divided according to customer segments.

Separate machine learning models are trained to predict future CLV for each customer segment.

The trained models are saved using Joblib.

Example:

```python
joblib.dump(models, "clv_regression_models.joblib")
```

---

## Running the Application

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start Streamlit

```bash
streamlit run app.py
```

---

## Application Interface

Users provide:

* Recency
* Frequency
* Monetary Value

The system:

1. Determines the customer segment.
2. Loads the appropriate model.
3. Predicts Customer Lifetime Value.
4. Displays the result.

---

## Example

Input:

* Recency = 30
* Frequency = 10
* Monetary = ₹1000

Output:

```text
Customer Segment: High Value
Predicted 90-Day CLV: ₹6,500
```

---

## Future Enhancements

* Advanced Customer Segmentation
* Deep Learning Models
* Customer Churn Prediction
* Interactive Dashboards
* Cloud Deployment
* Real-Time Business Analytics

---

## Author

Abul Hasan

### Academic Project

Customer Lifetime Value Prediction Using RFM Analysis and Machine Learning

---

## License

This project is developed for educational and research purposes.
