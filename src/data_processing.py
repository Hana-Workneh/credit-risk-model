from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED_COLUMNS = [
    "CustomerId",
    "Amount",
    "TransactionStartTime",
    "ProductCategory",
    "ChannelId",
    "ProviderId",
    "PricingStrategy",
]


def _mode(series: pd.Series) -> Optional[object]:
    """Safe mode: returns the most frequent value, or None if empty."""
    if series.empty:
        return None
    vc = series.value_counts(dropna=True)
    if vc.empty:
        return None
    return vc.index[0]


class CustomerFeatureAggregator(BaseEstimator, TransformerMixin):
    """
    Converts transaction-level data into customer-level features.

    Output columns include:
      - total_transaction_amount
      - avg_transaction_amount
      - transaction_count
      - std_transaction_amount
      - last_tx_hour, last_tx_day, last_tx_month, last_tx_year
      - mode_product_category, mode_channel_id, mode_provider_id, mode_pricing_strategy
    """

    def __init__(self, snapshot_date: Optional[str] = None):
        self.snapshot_date = snapshot_date  # kept for future (RFM task), not used here

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Parse datetime
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce", utc=True)

        # Sort so "last" transaction is last row per customer
        df = df.sort_values(["CustomerId", "TransactionStartTime"])

        # Aggregate numeric features (required)
        agg_numeric = (
            df.groupby("CustomerId")["Amount"]
            .agg(
                total_transaction_amount="sum",
                avg_transaction_amount="mean",
                transaction_count="count",
                std_transaction_amount="std",
            )
            .reset_index()
        )

        # Last transaction time components per customer
        last_tx = df.groupby("CustomerId").tail(1)[
            ["CustomerId", "TransactionStartTime"]
        ].copy()

        last_tx["last_tx_hour"] = last_tx["TransactionStartTime"].dt.hour
        last_tx["last_tx_day"] = last_tx["TransactionStartTime"].dt.day
        last_tx["last_tx_month"] = last_tx["TransactionStartTime"].dt.month
        last_tx["last_tx_year"] = last_tx["TransactionStartTime"].dt.year

        last_tx = last_tx.drop(columns=["TransactionStartTime"])

        # Categorical "mode" per customer (most frequent category used by that customer)
        cat_agg = df.groupby("CustomerId").agg(
            mode_product_category=("ProductCategory", _mode),
            mode_channel_id=("ChannelId", _mode),
            mode_provider_id=("ProviderId", _mode),
            mode_pricing_strategy=("PricingStrategy", _mode),
        ).reset_index()

        # Merge everything
        out = agg_numeric.merge(last_tx, on="CustomerId", how="left").merge(cat_agg, on="CustomerId", how="left")

        return out


@dataclass
class FeatureConfig:
    numeric_features: List[str]
    categorical_features: List[str]


def build_feature_config() -> FeatureConfig:
    numeric = [
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
        "last_tx_hour",
        "last_tx_day",
        "last_tx_month",
        "last_tx_year",
    ]
    categorical = [
        "mode_product_category",
        "mode_channel_id",
        "mode_provider_id",
        "mode_pricing_strategy",
    ]
    return FeatureConfig(numeric_features=numeric, categorical_features=categorical)


def build_preprocessor(config: FeatureConfig) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, config.numeric_features),
            ("cat", categorical_pipe, config.categorical_features),
        ]
    )
    return preprocessor


def build_processing_pipeline() -> Pipeline:
    """
    Full Task 3 pipeline:
      raw transactions -> customer-level aggregation -> preprocessing (impute/encode/scale)
    """
    config = build_feature_config()
    pipeline = Pipeline(
        steps=[
            ("aggregate", CustomerFeatureAggregator()),
            ("preprocess", build_preprocessor(config)),
        ]
    )
    return pipeline


def make_customer_feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convenience helper: returns the engineered customer-level dataframe (before encoding)."""
    return CustomerFeatureAggregator().transform(raw_df)


from src.target_engineering import build_customer_target

def build_customer_training_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a customer-level dataset ready for model training:
      - engineered customer features (Task 3)
      - is_high_risk target (Task 4)
    """
    features = make_customer_feature_frame(raw_df)
    target, high_cluster, summary, snap = build_customer_target(raw_df)

    out = features.merge(target, on="CustomerId", how="left")
    if out["is_high_risk"].isna().any():
        raise ValueError("Some customers are missing is_high_risk after merge.")

    return out