📊 Credit Card Default Prediction — ML Project Documentation

🧾 Project Title:

Prediction of Credit Card Default Risk using Random Forest Classifier

---

📁 Project Structure:

```
ML Project/
│
├── Data/
├   |── cleanddata/
│   │   └── cleaned.csv   
│   ├── preprocessdata/
│   │   ├── x.csv
│   │   └── y.csv
│   ├── rawdata/
│   │   └── rawdata.csv
│
├── models/
│   └── models.pkl
│
├── notebooks/
│   ├── datacleaning.ipynb
│   ├── datacollection.ipynb
│   ├── eda.ipynb
│   ├── modelbuilding.ipynb
│   └── preprocess.ipynb
│
├── src/
│   ├── datacleaning.py
│   ├── datacollection.py
│   ├── modelbuilding.py
│   └── preprocess.py
│
├── app.py
├── requirements.txt
└── venv/
```

---

 🧠 Project Overview

This project predicts whether a credit card client will default on their next payment based on demographic, payment, and billing data collected from April to September 2005 in Taiwan.

---

 🎯 Objective

To build an accurate ML model that can:

* Predict default risk.
* Be deployed as a Streamlit web app.
* Assist financial institutions in evaluating credit risk.

---

 📊 Dataset Information

**Source**: UCI Machine Learning Repository
**Features**:

* Demographics: `AGE`, `SEX`, `EDUCATION`, `MARRIAGE`
* Payment history: `PAY_0` to `PAY_6`
* Bill statements: `BILL_AMT1` to `BILL_AMT6`
* Previous payments: `PAY_AMT1` to `PAY_AMT6`
* Target: `default.payment.next.month`

---

 🛠️ Steps Involved

 1. 📥 Data Collection

* Raw dataset imported as `rawdata.csv`.

  2. 🧹 Data Cleaning (`datacleaning.ipynb` / `datacleaning.py`)

* Handled missing values, inconsistent labels (e.g. merging `EDUCATION=0,5,6` into `others`)
* Encoded categorical features.

 3. 📊 Exploratory Data Analysis (`eda.ipynb`)

* Distribution plots for target classes.
* Correlation heatmap.
* Boxplots for numeric columns.

 4. ⚙️ Feature Engineering (`preprocess.py`)

* Normalization/scaling where required.
* Removed highly correlated or redundant features.

 5. 🧪 Model Building (`modelbuilding.ipynb` / `modelbuilding.py`)

* Multiple models trained: Logistic Regression, Decision Tree, Random Forest, etc.
* Random Forest gave best results.

**Metrics for Best Model (Random Forest)**:

```
Precision: 0.83 (class 0), 0.86 (class 1)
Recall:    0.86 (class 0), 0.82 (class 1)
F1-Score:  0.84 (both classes)
Accuracy:  84%
```

 6. 💾 Model Saving

* Best model serialized using `joblib` or `pickle` → `models/models.pkl`.

 7. 🌐 Deployment using Streamlit (`app.py`)

* Inputs for all 23 features via `st.number_input()` and `st.selectbox()`.
* Model inference and prediction display.
* UI organized by sections: Demographics, Repayment Status, Bill Amounts, etc.

---

 🚀 How to Run the App

### 🔧 Prerequisites:

Install dependencies using:

```bash
pip install -r requirements.txt
```

### ▶️ Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📌 Notable Code Snippets (from `app.py`):

```python
PAY_0 = st.number_input("Repayment in September")
BILL_AMT1 = st.number_input("Bill Amount in September")
prediction = model.predict([final_input])
st.success("Prediction: Will Default" if prediction[0]==1 else "Prediction: Will Not Default")
```

---

## 📈 Improvements and Next Steps

* Add SHAP/feature importance for explainability.
* Enable batch prediction for multiple users.
* Collect live feedback to retrain model.

---

## 📃 Requirements (requirements.txt)

```
pandas
numpy
scikit-learn
streamlit
matplotlib
joblib
```

---

## 🧑‍💻 Author

**Ashwini Sawant**
Role: Data Scientist | ML Developer
Tools Used: Python, Streamlit, Scikit-learn, VS Code, Pandas, Matplotlib

