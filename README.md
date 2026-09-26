# 🚚 Supply Chain Late Delivery Risk Prediction

A machine learning pipeline built in **KNIME Analytics Platform** with **Python (scikit-learn)** to predict late delivery risk in supply chain shipment data.

---

## 📌 Project Overview

Late deliveries are one of the most costly problems in logistics operations. This project builds an end-to-end predictive pipeline that ingests raw shipment data, preprocesses it, and trains multiple classification models to predict whether a shipment is at risk of being delivered late — enabling proactive decision-making in supply chain management.

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| KNIME Analytics Platform 5.x | Workflow orchestration & data pipeline |
| Python (scikit-learn) | Machine learning model training & evaluation |
| SQL / SQLite | Data storage and querying |
| pandas | Data manipulation |

---

## 🗂️ Pipeline Architecture

```
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
```

---

## 🤖 Models Trained

All models use **GridSearchCV** for hyperparameter tuning and are evaluated on the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Models:
1. **Random Forest Classifier** — ensemble method, robust to overfitting
2. **Decision Tree Classifier** — interpretable baseline model
3. **Logistic Regression** — linear baseline with StandardScaler normalization

---

## 🎯 Target Variable

**`Late_delivery_risk`** — Binary classification:
- `1` = Shipment is at risk of late delivery
- `0` = Shipment is expected to arrive on time

### Features used (after preprocessing):
All relevant shipment features excluding data-leaking columns:
- `Days for shipping (real)`
- `Days for shipment (scheduled)`
- `Delivery Status`
- `Order Id` / `Order Customer Id`

---

## 📊 Data

The dataset contains supply chain and e-commerce shipment records with features including shipping mode, customer location, product category, order details, and delivery outcomes.

> Data is loaded via CSV Reader node in KNIME. Place your dataset in the `/data` folder or update the CSV Reader path accordingly.

---

## 🚀 How to Run

1. Install [KNIME Analytics Platform](https://www.knime.com/downloads) (v5.x recommended)
2. Install the **KNIME Python Integration** extension
3. Make sure Python is configured in KNIME with the required libraries:
   ```
   pip install pandas scikit-learn
   ```
4. Open `KNIME_project11.knwf` in KNIME
5. Update the CSV Reader node to point to your dataset
6. Execute the workflow (Shift + F7)

---

## 📁 Repository Structure

```
📦 supply-chain-late-delivery-prediction
 ┣ 📄 KNIME_project11.knwf   ← Main KNIME workflow file
 ┣ 📁 data/                  ← Place your dataset here
 ┗ 📄 README.md
```

---

## 👤 Author

**Dimitrios Komkoudis**  
MSc Candidate in Supply Chain Management & Logistics | BSc Mathematics  
Aristotle University of Thessaloniki (AUTH)  
📧 dkomkoud@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/dimitrios-komkoudis-40488631a)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
