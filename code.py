*** 1) Project plan & milestones ***

#Data ingestion & cleaning (missing values, datatypes, encode categorical, feature engineering).

#EDA (overview, department-wise attrition, salary bands, promotions, tenure).

#Baseline models: Logistic Regression + Decision Tree.

#Improve: preprocessing pipeline, class imbalance handling, hyperparameter tuning.

#Explainability: SHAP values (global + local).

#Visualizations + Power BI dashboard.

#Deliverables: Power BI .pbix, Model accuracy report + confusion matrix images, PDF with attrition prevention suggestions.

*** 2) EDA & preprocessing (Python) ***

#Copy this into a notebook. Replace hr.csv with your filename.

# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load
df = pd.read_csv('hr.csv')
df.info()
df.head()

# Basic cleaning
# Example columns: 'EmployeeID','Age','Department','JobRole','MonthlyIncome','Attrition','YearsAtCompany','NumPromotions',...
# Convert Attrition to binary
df['Attrition_flag'] = df['Attrition'].map({'Yes':1,'No':0})

# Missing values
print(df.isna().sum())

# Example feature engineering
df['SalaryBand'] = pd.qcut(df['MonthlyIncome'], q=4, labels=['Low','Med','High','VeryHigh'])
df['TenureYears'] = df['YearsAtCompany']  # rename for clarity

# EDA plots (examples)
plt.figure(figsize=(8,5))
sns.countplot(x='Department', hue='Attrition', data=df)
plt.title('Department-wise Attrition Counts')
plt.xticks(rotation=30)
plt.tight_layout()

# Attrition rate by salary band
attr_by_salary = df.groupby('SalaryBand')['Attrition_flag'].mean().reset_index()
print(attr_by_salary)

plt.figure(figsize=(6,4))
sns.barplot(x='SalaryBand', y='Attrition_flag', data=attr_by_salary)
plt.ylabel('Attrition Rate')
plt.title('Attrition Rate by Salary Band')
plt.tight_layout()


#EDA checklist to run:

#Attrition % overall and by Department/JobRole/SalaryBand/Gender/Age group.

#Distribution of tenure, monthly income, performance rating, number of promotions.

#Correlations (numeric features).

#Crosstabs: Promotions vs Attrition, Recent Hires vs Attrition (YearsAtCompany<2), Overtime vs Attrition.

*** 3) Modeling pipeline (preprocessing + training + evaluation) ***

#This pipeline handles categorical encoding, scaling, and offers both logistic and tree models. It also shows how to produce a confusion matrix and classification report.

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE  # pip install imbalanced-learn

# Features and target - adapt columns
target = 'Attrition_flag'
exclude_cols = ['EmployeeID','Attrition','Attrition_flag']
X = df.drop(columns=exclude_cols)
y = df[target]

# Detect numeric & categorical
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
])

# Create pipeline using SMOTE + model. Use imblearn pipeline for SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Example: Logistic Regression
pipe_lr = ImbPipeline(steps=[
    ('pre', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# Train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No','Yes'], yticklabels=['No','Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.tight_layout()


Decision Tree example & hyperparameter tuning:

pipe_dt = ImbPipeline(steps=[
    ('pre', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', DecisionTreeClassifier(random_state=42))
])

param_grid = {
    'clf__max_depth':[3,5,8,12,None],
    'clf__min_samples_leaf':[1,5,10,20]
}

gs = GridSearchCV(pipe_dt, param_grid, cv=5, scoring='f1', n_jobs=-1)
gs.fit(X_train, y_train)
print('Best params:', gs.best_params_)
best = gs.best_estimator_
y_pred_dt = best.predict(X_test)
print('Accuracy (DT):', accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

** 4) SHAP explainability **

Use TreeExplainer for tree model, KernelExplainer for logistic (or TreeExplainer on a tree-based model). Example for decision tree:

import shap

# Precompute transformed data for explainer
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# If using decision tree:
model = best.named_steps['clf']  # decision tree model
explainer = shap.TreeExplainer(model)
# shap expects 2D array of features in same transformed order
shap_values = explainer.shap_values(X_test_trans)

# Global importance
shap.summary_plot(shap_values, X_test_trans, plot_type='bar')

# Local explanation for one sample
i = 0
shap.force_plot(explainer.expected_value[1], shap_values[1][i,:], matplotlib=True)


## Notes:

##With OneHotEncoder you may want to capture feature names: ohe = preprocessor.named_transformers_['cat']; feature_names = num_cols + list(ohe.get_feature_names_out(cat_cols)) — then pass a DataFrame to SHAP with those column names.

##For LogisticRegression, KernelExplainer can be slow; consider using a tree-based model (RandomForest/LightGBM) for faster TreeExplainer and often better performance.

** 5) Model accuracy report + confusion matrix (deliverable) **

#Create a small report (markdown or PDF) that includes:

#Dataset used (rows, columns, date).

#Preprocessing steps.

#Model(s) tried and hyperparameters.

#Metrics: Accuracy, Precision, Recall, F1, ROC-AUC.

##Confusion matrix chart (as PNG).

#Feature importance (SHAP summary plots).
#Use sklearn.metrics.roc_auc_score and sklearn.metrics.plot_roc_curve for ROC.

** 6) Power BI dashboard design (suggested layout & measures) **

Dashboard pages:

Overview — KPIs: Total employees, Attrition rate (overall %), Avg tenure, Avg salary, % Recently promoted (last 12 months).

Attrition by Department/JobRole — stacked bar with attrition counts & rates; drill-down by JobRole.

Drivers — visuals for top features from SHAP (e.g., low salary, lack of promotion, long hours).

Employee profiles — table with employee-level risk score (model prediction), key features, link to recommended action.

Suggested DAX measures:

TotalEmployees = COUNTROWS('HR')
TotalAttrition = CALCULATE(COUNTROWS('HR'), 'HR'[Attrition] = "Yes")
AttritionRate = DIVIDE([TotalAttrition], [TotalEmployees], 0)

AvgTenure = AVERAGE('HR'[YearsAtCompany])
AvgSalary = AVERAGE('HR'[MonthlyIncome])

RecentPromotionPct = DIVIDE(CALCULATE(COUNTROWS('HR'), 'HR'[YearsSinceLastPromotion] <= 1), [TotalEmployees], 0)


Tips:

Import model predictions (CSV) with columns: EmployeeID, PredictedRisk (0-1), PredLabel. Join in Power BI.

Use conditional formatting to highlight high-risk employees.

Add slicers: Department, JobRole, Gender, SalaryBand.

Use bookmarks to show “Top risk” and “What-if” scenarios.

7) SHAP -> Power BI integration

Export SHAP summary (feature and mean abs SHAP) as CSV and import to Power BI to visualize global feature importance.

For local explanations: generate per-employee top 5 contributing features and their SHAP values and import into Power BI to show “why this employee is at risk.”

8) Attrition prevention suggestions (content for PDF)

Use this as sections in the PDF: Evidence-based recommendations with short action items.

Competitive compensation

Salary bands with high attrition should be reviewed.

Action: Market salary review for affected roles within 30 days.

Promotion & career path

Lack of promotions strongly correlates with resignations.

Action: Formalize 12-month development plans; increase internal mobility.

Onboarding & early retention

High attrition in first 6-12 months — bolster onboarding, mentorship.

Action: Assign mentors for new hires; 30/60/90-day check-ins.

Workload & overtime

Excess overtime links to attrition — consider resource planning and hiring.

Action: Monitor overtime per team and redistribute tasks.

Manager training & engagement

Managers drive retention — train managers on feedback and recognition.

Action: Quarterly manager effectiveness survey + coaching.

Flexible benefits & wellbeing

Introduce flexible hours or remote options for at-risk groups.

Action: Pilot 6-month flexible hours in two departments.

Targeted interventions using model

Use model risk scores to run targeted retention offers (training, pay review, career discussions).

Action: Define intervention A/B tests and measure effectiveness over 6 months.

9) Templates & ready-to-use text

Executive summary (short):

Using historical HR data (N rows, M columns) we built classification models to predict employee attrition. Best model: Decision Tree/Logistic Regression with F1 = X. Key drivers: low salary band, no promotions in last 3 years, high overtime hours, and short tenure. Recommended interventions: market-based salary adjustments, promotion & career roadmaps, manager training, and targeted retention offers to high-risk employees.

PDF section: Implementation roadmap

Month 0-1: Data & model productionization; HR dashboard creation.

Month 1-3: Pilot retention initiatives for top-2 risk reasons.

Month 3-6: Evaluate pilot, scale successful interventions.

10) Quick checklist before production

Validate model on holdout set (time-split if possible).

Ensure PII policies for employee-level data; restrict access to risk lists.

Retrain model regularly (quarterly).

Monitor drift and fairness by protected attributes.
