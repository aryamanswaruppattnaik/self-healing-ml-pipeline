# Dataset Documentation — Self-Healing ML Pipeline

This folder contains all datasets used in the Self-Healing Machine Learning Pipeline project, which focuses on financial fraud detection and adaptive retraining using PSI (Population Stability Index) for drift detection.

Each dataset here represents a unique  domain view of financial and customer behavior data — collectively forming a multidisciplinary dataset for robust model adaptation.

---

# 1. Credit Card Fraud Detection Dataset
 Source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
File:`creditcard.csv`  
 Size: ~150 MB  
Samples: 284,807 transactions  
Features: 31 anonymized numerical columns (V1–V28, Time, Amount)  
Target: `Class` (0 = Non-Fraud, 1 = Fraud)

Description: 
This dataset contains real credit card transactions made by European cardholders in September 2013. All input features are the result of a PCA transformation due to confidentiality constraints. It is used here as the **baseline training dataset** for fraud classification.

 Project Role: 
Serves as the base domain for model training — our “reference population” for PSI drift calculations.

---

## 2. PaySim Synthetic Financial Transactions
 Source: [Kaggle – PaySim Mobile Money Simulator](https://www.kaggle.com/datasets/ealaxi/paysim1)  
 File:`PS_20174392719_1491204439457_log.csv`  
 Size:~450 MB  
 Samples: 6,362,620 transactions  
 Features: 11 columns including transaction type, amount, old/new balances, fraud indicators  
 Target: `isFraud` (1 = Fraud, 0 = Genuine)

 Description: 
PaySim is a synthetic simulator based on real financial logs from a mobile money service. It mirrors the transactional flow, user balance variations, and fraud patterns.  
Used as the target domain to test domain drift and adaptive retraining.

 Project Role: 
Forms the drifted domain to measure how the trained model on Credit Card data performs when exposed to PaySim data, enabling PSI-based self-healing retraining.

---

# 3. Telco Customer Churn Dataset
Source: [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
 File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`  
Size:~1 MB  
 Samples: 7,043 customers  
Features: 21 attributes including demographics, service usage, and churn behavior  
 Target:`Churn` (Yes / No)

Description:
This dataset provides telecom customer behavioral patterns — including tenure, contract type, billing, and payment preferences. Although not directly financial, it is used to introduce *behavioral drift signals* related to user retention and stability.

Project Role:
Used as an auxiliary behavioral dataset to simulate cross-domain variability and test how well the self-healing system adapts across non-identical data distributions.

---

##  Summary Table

| Dataset | Samples | Features | Target | Domain Type | Role in Project |
|----------|----------|-----------|----------|---------------|----------------|
| Credit Card Fraud | 284,807 | 31 | `Class` | Real Financial | Baseline Model Training |
| PaySim | 6,362,620 | 11 | `isFraud` | Synthetic Financial | Drift Testing + Retraining |
| Telco Churn | 7,043 | 21 | `Churn` | Behavioral | Auxiliary Drift Simulation |

---

##  Usage Notes

- Place downloaded datasets under `./data/`
- Use the Kaggle API or manual download to fetch files:
  ```bash
  kaggle datasets download -d mlg-ulb/creditcardfraud -p ./data
  kaggle datasets download -d ealaxi/paysim1 -p ./data
  kaggle datasets download -d blastchar/telco-customer-churn -p ./data
  unzip './data/*.zip' -d './data/'
