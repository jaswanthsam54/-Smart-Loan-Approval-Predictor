# 🏦 Smart Loan Approval Predictor

A machine learning project that predicts whether a loan application should be
approved or rejected based on applicant financial and demographic data.

---

## 📌 Problem Statement

Banks and financial institutions receive thousands of loan applications daily.
Manual review is slow and inconsistent. This project builds an automated ML
pipeline that predicts loan approval with high accuracy — helping reduce
default risk and speed up decisions.

---

## 📊 Dataset

- **Type:** Synthetically generated loan applicant data
- **Size:** 1,000 rows × 11 features
- **Target:** `Loan_Status` — 1 (Approved) / 0 (Rejected)

**Features used:**

| Feature | Description |
|---|---|
| Age | Applicant age (21–65) |
| Income | Annual income |
| Loan_Amount | Requested loan amount |
| Loan_Term | Repayment period (months) |
| Credit_Score | Credit score (300–850) |
| Employment_Years | Years of employment |
| Num_Dependents | Number of dependents |
| Education | High School / Graduate / Post-Graduate |
| Self_Employed | Yes / No |
| Existing_Loans | Number of existing loans |
| Property_Area | Urban / Rural / Semi-Urban |

**Models trained:**
- Logistic Regression (baseline)
- Random Forest
- XGBoost ✅ (best performer)

---

## 📈 Results

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~0.84 | ~0.83 | ~0.89 |
| Random Forest | ~0.89 | ~0.88 | ~0.92 |
| **XGBoost** | **~0.91** | **~0.89** | **~0.93** |

> Replace with your actual numbers from notebook output.

---

## 📉 Visualizations

- Approval rate by Education, Employment, Property Area
- Feature distributions by Loan Status
- Correlation Heatmap
- ROC Curve (all 3 models)
- Confusion Matrix
- Feature Importance (XGBoost)

---

## 🗂️ Project Structure

loan-approval-predictor/

│

├── loan_approval_predictor.ipynb   # Main notebook (all 13 cells)

├── approval_by_category.png        # EDA plot

├── distributions.png               # Feature distribution plot

├── correlation_heatmap.png         # Heatmap

├── model_comparison.png            # Bar chart comparison

├── confusion_matrix.png            # Best model CM

├── roc_curve.png                   # ROC curves

├── feature_importance.png          # XGBoost importances

└── README.md                       # This file

---

## 🛠️ Tech Stack

- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/loan-approval-predictor.git
cd loan-approval-predictor
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyterlab
```

**3. Launch Jupyter Lab**
```bash
jupyter lab
```

**4. Open and run**

Open `loan_approval_predictor.ipynb` and run all cells top to bottom.

---

## 🔍 Sample Prediction

```python
new_applicant = {
    'Age': 35,
    'Income': 75000,
    'Loan_Amount': 15000,
    'Credit_Score': 720,
    'Employment_Years': 8,
    ...
}
# Output: ✅ APPROVED | Approval Probability: 87.4%
```

---

## 👤 Author

**Jaswanth**
B.Tech CSE — ACE Engineering College, Hyderabad

---

## 📄 License

MIT License — free to use and modify.

---

## ⚙️ ML Pipeline
