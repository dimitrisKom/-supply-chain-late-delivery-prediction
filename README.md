🚚 Supply Chain Late Delivery Risk Prediction
A machine learning pipeline built in KNIME Analytics Platform with Python (scikit-learn) to predict late delivery risk in supply chain shipment data.

📌 Project Overview
Late deliveries are one of the most costly problems in logistics operations. This project builds an end-to-end predictive pipeline that ingests raw shipment data, preprocesses it, and trains multiple classification models to predict whether a shipment is at risk of being delivered late — enabling proactive decision-making in supply chain management.

🔧 Tech Stack
Tool	Purpose
KNIME Analytics Platform 5.x	Workflow orchestration & data pipeline
Python (scikit-learn)	Machine learning model training & evaluation
SQL / SQLite	Data storage and querying
pandas	Data manipulation
🗂️ Pipeline Architecture
CSV Reader
    │
    ▼
Missing Value Handler
    │
    ▼
Column Filter
    │
    ├──────────────────────────────────────────┐
    ▼                                          ▼
Python Script                          Python Script (#10)
Data Exploration                       (Exploratory Analysis)
(nunique, dtypes)
    │
    ├─────────────────┬──────────────────┐
    ▼                 ▼                  ▼
Random Forest   Decision Tree    Logistic Regression
Classifier      Classifier       Classifier
    │                 │                  │
    └─────────────────┴──────────────────┘
                      │
                      ▼
               SQLite Connector
                      │
                ┌─────┴──────┐
                ▼            ▼
           DB Writer    DB Query Readers
                        (Results retrieval)
🤖 Models Trained
All models use GridSearchCV for hyperparameter tuning and are evaluated on the following metrics:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Models
Random Forest Classifier — ensemble method, robust to overfitting
Decision Tree Classifier — interpretable baseline model
Logistic Regression — linear baseline with StandardScaler normalization
🎯 Target Variable
Late_delivery_risk — Binary classification:

1 = Shipment is at risk of late delivery
0 = Shipment is expected to arrive on time
Features used (after preprocessing)
All relevant shipment features excluding data-leaking columns:

Days for shipping (real)
Days for shipment (scheduled)
Delivery Status
Order Id / Order Customer Id
📊 Data
The dataset contains supply chain and e-commerce shipment records with features including shipping mode, customer location, product category, order details, and delivery outcomes.

Data is loaded via CSV Reader node in KNIME. Place your dataset in the /data folder or update the CSV Reader path accordingly.

📊 Results
Three classification models were trained with GridSearchCV hyperparameter tuning. Results on the held-out test set:

Algorithm	Accuracy	Precision	Recall	F1-Score
Random Forest	69.23%	81.19%	57.10%	67.05%
Logistic Regression	69.15%	84.22%	53.83%	65.68%
Decision Tree	69.48%	83.30%	55.46%	66.59%
Key findings:

All three models show high precision (>81%) but comparatively lower recall (~55%) — they're conservative, flagging late-delivery risk only when fairly confident, at the cost of missing some real delays.
Feature importance analysis identified Shipping Mode (Standard Class) as the strongest predictor of late-delivery risk.
Planned improvements:

Address class imbalance with class_weight='balanced' to improve recall (catch more true late deliveries).
Extract Month from the order date to capture seasonality effects.
Extend the pipeline with a dedicated fraud-detection model and K-Means-based automated customer segmentation.
🗃️ SQL Analysis
Two exploratory analyses were run directly against the SQLite database via KNIME's DB Query Reader nodes.

Customer Segmentation (VIP / Standard / Loss-Making)
SELECT "Order Customer Id",
       COUNT(DISTINCT "Order Id") AS "Total_Unique_Orders",
       COUNT("Order Item Id") AS Total_Items_Purchased,
       SUM("Order Profit Per Order") AS Total_Profit,
       SUM(CASE WHEN "Late_delivery_risk" = 1 THEN 1 ELSE 0 END) AS Delayed_Orders,
       SUM(CASE WHEN "Late_delivery_risk" = 1 THEN 1.0 ELSE 0.0 END) / COUNT("Order Item Id") * 100 AS Delay_Rate_Percentage,
       CASE
           WHEN SUM("Order Profit Per Order") > 500 AND COUNT(DISTINCT "Order Id") >= 5 THEN '1. VIP Customer'
           WHEN SUM("Order Profit Per Order") < 0 THEN '3. Loss-Making Customer'
           ELSE '2. Standard Customer'
       END AS Customer_Segment
FROM orders
GROUP BY "Order Customer Id"
HAVING COUNT(DISTINCT "Order Id") > 1
ORDER BY Total_Profit DESC;
Sample output — top VIP customers by total profit:

Order Customer Id	Total Orders	Total Profit ($)	Segment
2,641	11	2,441.97	1. VIP Customer
1,657	11	2,196.92	1. VIP Customer
9,833	7	1,938.39	1. VIP Customer
Fraud Risk by Region
SELECT "Order Region",
       COUNT("Order Item Id") AS Total_transactions,
       SUM(CASE WHEN "Order Status" = 'SUSPECTED_FRAUD' THEN 1 ELSE 0 END) AS Fraud_Cases,
       SUM(CASE WHEN "Order Status" = 'SUSPECTED_FRAUD' THEN 1.0 ELSE 0.0 END) / COUNT("Order Item Id") * 100 AS Fraud_Rate_Percentage,
       SUM(CASE WHEN "Order Status" = 'SUSPECTED_FRAUD' THEN "Sales" ELSE 0 END) AS Value_At_Risk
FROM orders
GROUP BY "Order Region"
HAVING SUM(CASE WHEN "Order Status" = 'SUSPECTED_FRAUD' THEN 1 ELSE 0 END) > 0
ORDER BY Value_At_Risk DESC;
Top 3 highest-risk regions:

Order Region	Total Transactions	Fraud Rate (%)	Value At Risk ($)
Western Europe	27,109	2.601	148,507.55
Central America	28,341	2.226	126,740.04
South America	14,935	2.417	71,856.88
🐍 Python Code (from KNIME Python Script nodes)
Data Exploration (Python Script #10)
import knime.scripting.io as knio
import pandas as pd

df = knio.input_tables[0].to_pandas()
dtypes = df.dtypes
unique = df.nunique()
explore = pd.DataFrame({'Column': df.columns, 'Data Type': dtypes.astype(str), 'nunique': unique})
explore = explore.sort_values(by='nunique', ascending=False)
knio.output_tables[0] = knio.Table.from_pandas(explore)
Random Forest Classifier (Python Script #4)
import knime.scripting.io as knio
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

df = knio.input_tables[0].to_pandas()
y = df["Late_delivery_risk"]

cols_to_drop = [
    "Late_delivery_risk", "Days for shipping (real)", "Days for shipment (scheduled)",
    "Delivery Status", "Order Id", "Order Customer Id", "Order Item Id", "Customer Id"
]
X_raw = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

num_cols = X_raw.select_dtypes(include=["number"]).columns.tolist()
cat_cols = ["Shipping Mode", "Type", "Category Name", "Customer Segment"]
X_filtered = X_raw[num_cols + cat_cols]
X = pd.get_dummies(X_filtered, columns=cat_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20], 'min_samples_split': [2, 10]}
base_rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=base_rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("--- Random Forest (Tuned) Results ---")
print("Best Params:", grid_search.best_params_)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("f1-score:", f1)
print('--confusion matrix--')
print(cm)

importances = best_model.feature_importances_
importance_df = pd.DataFrame({'column': X.columns, 'importance': importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)
knio.output_tables[0] = knio.Table.from_pandas(importance_df)
Decision Tree Classifier (Python Script #5)
import knime.scripting.io as knio
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

df = knio.input_tables[0].to_pandas()
y = df["Late_delivery_risk"]

cols_to_drop = [
    "Late_delivery_risk", "Days for shipping (real)", "Days for shipment (scheduled)",
    "Delivery Status", "Order Id", "Order Customer Id", "Order Item Id", "Customer Id"
]
X_raw = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

num_cols = X_raw.select_dtypes(include=["number"]).columns.tolist()
cat_cols = ["Shipping Mode", "Type", "Category Name", "Customer Segment"]
X_filtered = X_raw[num_cols + cat_cols]
X = pd.get_dummies(X_filtered, columns=cat_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}
model = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Decision tree results")
print("Accuracy", accuracy)
print("Precision", precision)
print("f1-score", f1)
print('--confusion matrix--')
print(cm)

importances = best_model.feature_importances_
importance_df = pd.DataFrame({'column': X.columns, 'importance': importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)
knio.output_tables[0] = knio.Table.from_pandas(importance_df)
Logistic Regression (Python Script #6)
import pandas as pd
import knime.scripting.io as knio
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

df = knio.input_tables[0].to_pandas()
y = df['Late_delivery_risk']

cols_to_drop = [
    "Late_delivery_risk", "Days for shipping (real)", "Days for shipment (scheduled)",
    "Delivery Status", "Order Id", "Order Customer Id", "Order Item Id", "Customer Id"
]
X_raw = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

num_cols = X_raw.select_dtypes(include=['number']).columns.tolist()
cat_cols = ['Shipping Mode', 'Category Name', 'Customer Segment', 'Type']
cat_cols = [c for c in cat_cols if c in X_raw.columns]
X_filtered = X_raw[num_cols + cat_cols]
X = pd.get_dummies(X_filtered, columns=cat_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

param_grid = {'C': [0.1, 1, 10]}
model = LogisticRegression(max_iter=2000, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_sc, y_train)

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test_sc)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print("---lr results---")
print("Best Params:", grid_search.best_params_)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("f1-score:", f1)
print("Confusion Matrix:\n", cm)

importances = best_model.coef_[0]
importance_df = pd.DataFrame({'column': X.columns, 'coefficient': importances})
importance_df['abs_coefficient'] = importance_df['coefficient'].abs()
importance_df = importance_df.sort_values(by='abs_coefficient', ascending=False)
knio.output_tables[0] = knio.Table.from_pandas(importance_df)
🚀 How to Run
Install KNIME Analytics Platform (v5.x recommended)
Install the KNIME Python Integration extension
Make sure Python is configured in KNIME with the required libraries:
pip install pandas scikit-learn
Open KNIME_project11.knwf in KNIME
Update the CSV Reader node to point to your dataset
Execute the workflow (Shift + F7)
📁 Repository Structure
📦 supply-chain-late-delivery-prediction
 ┣ 📄 KNIME_project11.knwf   ← Main KNIME workflow file
 ┣ 📁 data/                  ← Place your dataset here
 ┗ 📄 README.md
👤 Author
Dimitrios Komkoudis MSc Candidate in Supply Chain Management & Logistics | BSc Mathematics Aristotle University of Thessaloniki (AUTH) 📧 dkomkoud@gmail.com 🔗 LinkedIn

📄 License
This project is open source and available under the MIT License.
