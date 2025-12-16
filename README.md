# Credit Risk Probability Model for Alternative Data

## Credit Scoring Business Understanding

### 1\) Basel II and why interpretability + documentation matter

Basel II emphasizes that banks must measure, manage, and hold capital against risk in a structured and auditable way. This makes our credit scoring model more than a prediction tool: it becomes part of the bank’s risk management framework, where assumptions, data, methodology, and performance must be clearly documented for internal governance and supervisory review. An interpretable model (or an explainable modeling approach) helps stakeholders understand what drives risk estimates, supports validation and monitoring, and enables consistent decision-making. In practice, this means we must prioritize traceability (data lineage + feature definitions), reproducibility (versioned code and experiments), and clear reporting of model limitations and performance metrics.



### 2\) Why we need a proxy default label (and risks)

The dataset does not include a direct loan performance outcome (e.g., “defaulted” vs “repaid”), which is normally required to train a supervised credit risk model. To proceed, we create a proxy target variable based on observable customer behavior—using RFM (Recency, Frequency, Monetary) patterns to identify disengaged customers and label them as higher-risk. This proxy enables model training and experimentation, but it introduces business risk because the proxy may not perfectly reflect true default behavior. The key risks include rejecting good customers (false positives), approving risky customers (false negatives), and building a model that optimizes for “engagement” rather than actual repayment. Therefore, proxy-based models must be treated as a first iteration, monitored carefully, and re-trained once true repayment labels become available.



### 3\) Trade-offs: Logistic Regression + WoE vs Gradient Boosting

A Logistic Regression scorecard with WoE transformation is typically easier to interpret, audit, and govern: each variable’s contribution is transparent, the model is stable, and it is straightforward to generate reason codes for accept/reject decisions. This is valuable in a regulated environment where stakeholders need clarity on why a customer was scored as risky. In contrast, Gradient Boosting models often deliver stronger predictive performance by capturing nonlinearities and interactions, especially with rich behavioral and categorical data. However, they are harder to explain end-to-end, require more advanced monitoring, and may increase model risk if their behavior changes under data drift. In a regulated financial context, the best choice balances predictive power with explainability, stability, monitoring complexity, and compliance requirements; a common approach is to benchmark both, then select the simplest model that meets performance and governance standards.


Why a proxy “default” label is necessary (and its risks)

This dataset does not contain a direct loan default label. Without an observed outcome, supervised learning cannot be applied directly. Therefore, we create a proxy target that approximates risk based on observed customer behavior (RFM engagement patterns).

Key business risks of proxy labels:

Label noise / misclassification: “low engagement” may not always mean “will default.”

Bias risk: the proxy can systematically label certain customer segments as risky due to behavioral patterns unrelated to ability to repay.

Model overconfidence: high performance against a proxy target may not translate to real-world default prediction.

Governance risk: decisions based on a proxy require careful documentation and conservative use (e.g., as one input to underwriting, not the only one).

Interpretable vs complex models in regulated settings

Trade-offs between models:

Logistic Regression + WoE (more interpretable)

Pros: explainable coefficients, easier to justify decisions, stable scorecard-style deployment.

Cons: may underfit nonlinear patterns and interactions.

Gradient Boosting / Tree Ensembles (higher performance potential)

Pros: strong predictive power, captures nonlinearities/interactions.

Cons: harder to explain; requires extra work for interpretability (feature importance, SHAP), monitoring, and governance.

In a regulated financial context, the best choice balances performance, transparency, stability, and operational simplicity.

Why RFM behavioral data can work

Transaction behavior can be converted into a predictive risk signal using:

Recency (how recently the customer transacted),

Frequency (how often they transact),

Monetary value (how much they transact).

These behavioral patterns can segment customers into engagement groups, and the least engaged segment can be used as a high-risk proxy for modeling.

Exploratory Data Analysis Summary (Task 2)
Dataset overview

Shape: 95,662 rows × 16 columns

Unique transactions: TransactionId is unique

Unique customers: 3,742 (CustomerId)

Unique accounts: 3,633 (AccountId)

Currency: UGX only

Country code: 256 only (constant)

Missing values

No missing values were detected in the raw dataset (missing ratio output was empty), indicating the dataset is complete at the column level.

Numerical distributions and outliers

Key numeric fields: Amount, Value, PricingStrategy, FraudResult

Amount is highly skewed with large outliers:

min: -1,000,000

max: 9,880,000

median: 1,000

Value is also highly skewed:

min: 2

max: 9,880,000

median: 1,000

FraudResult is very rare:

mean ≈ 0.002 → strong class imbalance for fraud

Boxplots and histograms indicate heavy-tailed distributions (extreme values), so robust aggregation and scaling are important during feature engineering.

Categorical distributions (most frequent)

ProductCategory: dominated by financial_services (45,405) and airtime (45,027); others are much smaller.

ChannelId: mostly ChannelId_3 (56,935) and ChannelId_2 (37,141).

ProviderId: mainly ProviderId_4 (38,189) and ProviderId_6 (34,186).

PricingStrategy: mostly 2 (79,848) and 4 (13,562).

Time-based patterns

Extracted features showed:

Most transactions occur during daytime hours (peaks around business hours).

Fridays have the highest transaction count.

Transaction months are concentrated in 2018-11, 2018-12, 2019-01, 2019-02, with 2018-12 being the highest.

Correlation analysis (numerical)

Amount and Value are strongly correlated (~0.99).

FraudResult shows moderate positive correlation with Amount/Value (note: correlation does not imply causation).

CountryCode is constant, so correlations with it are not meaningful.

Top insights (3–5)

Customer behavior is highly skewed (large outliers in Amount/Value), requiring aggregation and robust scaling.

Two product categories dominate (financial_services and airtime), suggesting concentrated purchasing behavior.

Country and currency are constant, likely providing no predictive value and can be dropped.

Amount and Value are almost redundant due to ~0.99 correlation; careful feature selection is needed.

FraudResult is extremely rare, so it should be handled carefully if used (imbalance).

Feature Engineering (Task 3)

We created a reproducible feature pipeline that produces customer-level features suitable for modeling:

Aggregate features per customer:

total_transaction_amount

avg_transaction_amount

transaction_count

std_transaction_amount

Temporal features (from the customer’s most recent transaction):

last_tx_hour, last_tx_day, last_tx_month, last_tx_year

Categorical summarization (mode per customer):

mode_product_category, mode_channel_id, mode_provider_id, mode_pricing_strategy

We validate that:

raw data is (95,662 × 16)

engineered customer feature frame is (3,742 × 13)

final model matrix (after preprocessing/encoding) is (3,742 × 31)

Proxy Target Engineering (Task 4)

Because there is no default label, we construct a proxy target using RFM clustering:

Compute RFM metrics per customer using a fixed snapshot date.

Scale RFM and cluster customers into 3 groups using K-Means with a fixed random_state.

Select the least engaged cluster (typically low frequency and low monetary) as the high-risk proxy.

Create is_high_risk and merge it with customer features.

Model Training & MLflow Tracking (Task 5)
Training approach

Split customer dataset into train/test with a fixed random_state.

Train and compare at least two models (e.g., Logistic Regression, Gradient Boosting).

Tune hyperparameters using GridSearchCV or RandomizedSearchCV.

Track all runs with MLflow:

parameters

metrics (Accuracy, Precision, Recall, F1, ROC-AUC)

artifacts (trained model)

Model selection

Best run selected based on ROC-AUC, then registered in the MLflow Model Registry as:

Registered model name: credit-risk-model

Alias: Production → Version 1

To reproduce:

python -m src.train --data_path data/raw/data.csv --experiment_name credit-risk-task5

Deployment, Docker, and CI/CD (Task 6)
FastAPI service

A FastAPI app loads the best model from the MLflow registry and exposes:

GET /health

POST /predict → returns risk_probability and credit_score

Run locally:

$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
$env:MODEL_URI="models:/credit-risk-model@Production"
python -m uvicorn src.api.main:app --reload


Swagger UI:

http://127.0.0.1:8000/docs

Example request/response (tested)

Request

{
  "features": {
    "total_transaction_amount": -10000.0,
    "avg_transaction_amount": -10000.0,
    "transaction_count": 1,
    "std_transaction_amount": null,
    "last_tx_hour": 16,
    "last_tx_day": 21,
    "last_tx_month": 11,
    "last_tx_year": 2018,
    "mode_product_category": "airtime",
    "mode_channel_id": "ChannelId_2",
    "mode_provider_id": "ProviderId_4",
    "mode_pricing_strategy": 4
  }
}


Response

{
  "risk_probability": 1.0,
  "credit_score": 300,
  "model_uri": "models:/credit-risk-model@Production"
}

Docker

Build and run the API using Docker:

docker compose up --build

CI/CD (GitHub Actions)

On each push/PR to main, GitHub Actions runs:

flake8 (lint)

pytest (unit tests)
The workflow fails if linting or tests fail.
