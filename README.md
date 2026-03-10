# Heart Disease Prediction API

This project is an **End-to-End Machine Learning API** built using **FastAPI** and **Logistic Regression**.
The API predicts the **10-year risk of heart disease** using the **Framingham dataset**.

## 🚀 Features

* Logistic Regression model for heart disease prediction
* FastAPI backend for serving the model
* Structured project architecture
* Interactive API documentation using Swagger UI

## 🧠 Machine Learning

The model is trained using **Logistic Regression** from Scikit-Learn.
It predicts whether a person is likely to develop heart disease in the next 10 years.

Target variable:

* `TenYearCHD`

  * **1 → Risk of heart disease**
  * **0 → No heart disease**

## 📂 Project Structure

```
heart_disease_api
│
├── app
│   ├── main.py
│   ├── schema.py
│   ├── prediction.py
│   └── model_loader.py
│
├── models
│   └── heart_disease.pkl
│
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/devsharmacodes/heart-disease-fastapi-api.git
```

Install dependencies:

```
pip install -r requirements.txt
```

## ▶️ Run the API

Start the FastAPI server:

```
uvicorn app.main:app --reload
```

Open the API documentation:

```
http://127.0.0.1:8000/docs
```

## 📊 Example Request

POST `/predict`

```
{
 "male": 1,
 "age": 52,
 "education": 2,
 "currentSmoker": 1,
 "cigsPerDay": 20,
 "BPMeds": 0,
 "prevalentStroke": 0,
 "prevalentHyp": 1,
 "diabetes": 0,
 "totChol": 250,
 "sysBP": 130,
 "diaBP": 85,
 "BMI": 27.5,
 "heartRate": 75,
 "glucose": 85
}
```

## 📌 Technologies Used

* Python
* FastAPI
* Scikit-Learn
* Pandas
* NumPy
* Uvicorn

## 👨‍💻 Author

Aryan Dev
