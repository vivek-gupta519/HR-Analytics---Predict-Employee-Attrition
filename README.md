# HR Analytics – Predict Employee Attrition  

## 📌 Objective  
Use analytics to understand the main causes of employee resignation and predict future attrition. The project combines **Python (EDA & modeling)**, **Power BI (visualization)**, and **Explainable AI (SHAP analysis)** to deliver actionable insights for HR decision-making.  

---

## 🛠 Tools & Libraries  
- **Python**: Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, Imbalanced-learn, SHAP  
- **Visualization**: Power BI  
- **Reports**: PDF (recommendations), Confusion matrix & performance metrics  

---

## 📂 Project Structure  
```bash
├── data/                  # Raw and cleaned HR datasets
├── notebooks/             # Jupyter notebooks for EDA, modeling, SHAP
├── reports/               # Model reports, SHAP plots, PDF recommendations
├── powerbi/               # Power BI .pbix dashboard
├── images/                # Confusion matrix, feature importance charts
├── README.md              # Project documentation
```

---

## 📊 Workflow  

### 1. Data Preparation  
- Import and clean HR dataset  
- Handle missing values & outliers  
- Encode categorical features, scale numerical features  
- Create engineered features (e.g., Salary Band, Tenure Group)  

### 2. Exploratory Data Analysis (EDA)  
- Attrition rates by **Department, Job Role, Salary Band, Gender, Age Group**  
- Relationship between **Promotions, Overtime, Tenure** and attrition  
- Correlation heatmaps & distribution plots  

### 3. Modeling  
- Models used: **Logistic Regression** and **Decision Tree**  
- Preprocessing pipeline: Scaling + OneHotEncoding + SMOTE (for class imbalance)  
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC  
- Confusion matrix for error analysis  

### 4. Explainability (SHAP Analysis)  
- **Global interpretation**: Identify key drivers of attrition (salary, promotions, overtime, tenure)  
- **Local interpretation**: Understand why specific employees are predicted at risk  

### 5. Visualization in Power BI  
- **Dashboard KPIs**:  
  - Total Employees  
  - Attrition Rate  
  - Avg Tenure  
  - Avg Salary  
- **Reports**:  
  - Attrition by Department & Job Role  
  - Salary Band vs Attrition Rate  
  - SHAP-based Feature Importance (imported from Python)  
  - Employee-level risk scores & explanations  
- **Filters**: Department, Job Role, Gender, Salary Band  

### 6. Deliverables  
✔ Power BI dashboard (.pbix)  
✔ Model accuracy report + confusion matrix images  
✔ PDF with attrition prevention suggestions  

---

## 📈 Sample Results (Illustrative)  
- **Best model**: Decision Tree (F1-score = 0.78, Accuracy = 0.82)  
- **Top drivers of attrition**:  
  - Low Salary Band  
  - No promotion in last 3 years  
  - High overtime hours  
  - Short tenure (<2 years)  

---

## 📑 Attrition Prevention Recommendations  
1. **Competitive compensation** – review salary bands with high attrition  
2. **Career progression** – promote internal mobility, structured promotion cycles  
3. **Early retention** – mentorship & check-ins for employees within first year  
4. **Workload balance** – monitor & reduce excessive overtime  
5. **Manager training** – improve leadership & recognition culture  
6. **Flexible benefits** – introduce remote/hybrid options where possible  
7. **Targeted interventions** – apply retention offers to high-risk employees (based on model predictions)  

---

## 🚀 How to Run  

### 1. Clone the repository  
```bash
git clone https://github.com/your-username/hr-analytics-attrition.git
cd hr-analytics-attrition
```

### 2. Install dependencies  
```bash
pip install -r requirements.txt
```

### 3. Run Jupyter notebooks  
- `EDA.ipynb` → Perform data exploration  
- `Modeling.ipynb` → Train Logistic Regression & Decision Tree  
- `SHAP_Analysis.ipynb` → Generate explainability plots  

### 4. Power BI Dashboard  
- Open `powerbi/HR_Attrition.pbix`  
- Connect to cleaned dataset (`data/clean_hr.csv`)  
- Refresh visuals to see updated metrics  

---

## 📘 Future Enhancements  
- Deploy model as **Streamlit/Dash web app** for HR teams  
- Add **real-time attrition risk scoring** with new employee data  
- Explore **advanced models**: Random Forest, XGBoost, LightGBM  
- Build **automated retraining pipeline** (e.g., ZenML, Airflow)  

---

