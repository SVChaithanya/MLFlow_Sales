# Sales Profit Prediction (MLflow + Scikit-Learn Pipeline)

This project predicts **Total Profit** using a full Machine Learning pipeline with:

- Data Cleaning  
- Outlier Removal  
- EDA (Heatmap, Histogram, Scatter plots)  
- Preprocessing (Imputation, Scaling, Encoding)  
- Model Training (RF, Linear Regression, KNN)  
- Hyperparameter Tuning (RandomizedSearchCV)  
- Stacking Regressor  
- MLflow Tracking & Model Logging


## 🔥 Features

### ✔ Full ML Pipeline  
- Numeric + Categorical preprocessing using `ColumnTransformer`  
- KNN Imputation  
- Standard Scaling  
- OneHotEncoding  

### ✔ Multiple Models Compared  
- RandomForestRegressor  
- LinearRegression  
- KNeighborsRegressor  

### ✔ Hyperparameter Tuning  
- RandomizedSearchCV for RF & KNN  

### ✔ Stacking Ensemble  
Final model is a Stacking Regressor with:
- RandomForest  
- Linear Regression  
- KNN  
- Final estimator = RandomForest  

### ✔ MLflow Tracking  
Everything is tracked:
- params  
- metrics  
- models  
- artifacts  
- pipeline models  

To view MLflow UI:  
python -m mlflow ui


Then open:  
**http://127.0.0.1:5000**



## 🚀 Project Structure

ml_project/
│
├── app.py # Main ML pipeline & training code
├── p1.csv # Sales dataset
└── README.md # Project documentation



## 📦 Installation

### 1️⃣ Create virtual environment
python -m venv .venv

markdown
Copy code

### 2️⃣ Activate
**Windows**
.venv\Scripts\activate

shell
Copy code

### 3️⃣ Install dependencies
pip install pandas numpy scikit-learn mlflow matplotlib seaborn



## ▶ Run the project

Start MLflow:
python -m mlflow ui

css
Copy code

Run your ML code:
python app.py

## 📊 MLflow Output

You will see:
- Metrics (R², RMSE)
- Parameters from tuned models
- Logged model artifacts
- Visualization of training runs


## 🧠 Model Logging

The final stacking model is logged using:

mlflow.sklearn.log_model(pipeline, artifact_path="model", signature=signature)

You can load the model later for prediction or deployment.


## 📈 Results

The project prints:
- R² score  
- RMSE  
- Best params for RF  
- Best params for KNN  
- Comparison of all models  


## 💡 Future Improvements

- Integrate **Streamlit UI**  
- Deploy using **FastAPI**  
- Convert to **PySpark + Spark MLlib** for big data  
- Add Docker support  
- Add unit tests  

