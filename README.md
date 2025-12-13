# Credit Risk Probability Model for Alternative Data

## Credit Scoring Business Understanding

### 1\) Basel II and why interpretability + documentation matter

Basel II emphasizes that banks must measure, manage, and hold capital against risk in a structured and auditable way. This makes our credit scoring model more than a prediction tool: it becomes part of the bank’s risk management framework, where assumptions, data, methodology, and performance must be clearly documented for internal governance and supervisory review. An interpretable model (or an explainable modeling approach) helps stakeholders understand what drives risk estimates, supports validation and monitoring, and enables consistent decision-making. In practice, this means we must prioritize traceability (data lineage + feature definitions), reproducibility (versioned code and experiments), and clear reporting of model limitations and performance metrics.



### 2\) Why we need a proxy default label (and risks)

The dataset does not include a direct loan performance outcome (e.g., “defaulted” vs “repaid”), which is normally required to train a supervised credit risk model. To proceed, we create a proxy target variable based on observable customer behavior—using RFM (Recency, Frequency, Monetary) patterns to identify disengaged customers and label them as higher-risk. This proxy enables model training and experimentation, but it introduces business risk because the proxy may not perfectly reflect true default behavior. The key risks include rejecting good customers (false positives), approving risky customers (false negatives), and building a model that optimizes for “engagement” rather than actual repayment. Therefore, proxy-based models must be treated as a first iteration, monitored carefully, and re-trained once true repayment labels become available.



### 3\) Trade-offs: Logistic Regression + WoE vs Gradient Boosting

A Logistic Regression scorecard with WoE transformation is typically easier to interpret, audit, and govern: each variable’s contribution is transparent, the model is stable, and it is straightforward to generate reason codes for accept/reject decisions. This is valuable in a regulated environment where stakeholders need clarity on why a customer was scored as risky. In contrast, Gradient Boosting models often deliver stronger predictive performance by capturing nonlinearities and interactions, especially with rich behavioral and categorical data. However, they are harder to explain end-to-end, require more advanced monitoring, and may increase model risk if their behavior changes under data drift. In a regulated financial context, the best choice balances predictive power with explainability, stability, monitoring complexity, and compliance requirements; a common approach is to benchmark both, then select the simplest model that meets performance and governance standards.



