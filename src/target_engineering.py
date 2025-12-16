
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class RFMConfig:
    customer_col: str = "CustomerId"
    datetime_col: str = "TransactionStartTime"
    amount_col: str = "Value"  # use Value (absolute), aligns with your dataset
    n_clusters: int = 3
    random_state: int = 42


def compute_rfm(
    df: pd.DataFrame,
    config: RFMConfig = RFMConfig(),
    snapshot_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Compute customer-level RFM:
      - Recency: days since last transaction (lower is better engagement)
      - Frequency: number of transactions
      - Monetary: sum of Value

    Returns:
      rfm_df: columns [CustomerId, recency, frequency, monetary]
      snapshot_date: timestamp used for recency calculation
    """
    data = df.copy()

    for col in [config.customer_col, config.datetime_col, config.amount_col]:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    data[config.datetime_col] = pd.to_datetime(data[config.datetime_col], errors="coerce", utc=True)

    # Snapshot date default: 1 day after the latest observed transaction
    if snapshot_date is None:
        max_dt = data[config.datetime_col].max()
        if pd.isna(max_dt):
            raise ValueError("All TransactionStartTime values are NaT after parsing.")
        snapshot_date = max_dt + pd.Timedelta(days=1)

    grouped = data.groupby(config.customer_col)

    last_tx = grouped[config.datetime_col].max()
    frequency = grouped[config.datetime_col].count()
    monetary = grouped[config.amount_col].sum()

    rfm = pd.DataFrame(
        {
            config.customer_col: last_tx.index,
            "recency": (snapshot_date - last_tx).dt.days.astype(int),
            "frequency": frequency.values.astype(int),
            "monetary": monetary.values.astype(float),
        }
    ).reset_index(drop=True)

    return rfm, snapshot_date


def assign_high_risk_label(
    rfm_df: pd.DataFrame,
    config: RFMConfig = RFMConfig(),
) -> Tuple[pd.DataFrame, int, pd.DataFrame]:
    """
    Cluster customers into 3 groups using RFM and label the least engaged cluster as high-risk.

    Least engaged cluster definition (typical):
      - high recency (worse)
      - low frequency (worse)
      - low monetary (worse)

    Returns:
      labeled_rfm: rfm_df + columns [cluster, is_high_risk]
      high_risk_cluster: int
      cluster_summary: mean recency/frequency/monetary per cluster
    """
    rfm = rfm_df.copy()

    needed = [config.customer_col, "recency", "frequency", "monetary"]
    missing = [c for c in needed if c not in rfm.columns]
    if missing:
        raise ValueError(f"rfm_df is missing columns: {missing}")

    X = rfm[["recency", "frequency", "monetary"]].to_numpy(dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=config.n_clusters,
        random_state=config.random_state,
        n_init=10,
    )
    rfm["cluster"] = kmeans.fit_predict(Xs)

    # Summarize clusters in original units for interpretation
    cluster_summary = (
        rfm.groupby("cluster")[["recency", "frequency", "monetary"]]
        .mean()
        .sort_index()
    )

    # Determine high-risk cluster using ranks:
    # - recency: higher is worse => rank descending (highest gets rank 1)
    # - frequency: lower is worse => rank ascending (lowest gets rank 1)
    # - monetary: lower is worse => rank ascending (lowest gets rank 1)
    ranks = pd.DataFrame(index=cluster_summary.index)
    ranks["recency_rank"] = cluster_summary["recency"].rank(ascending=False)
    ranks["frequency_rank"] = cluster_summary["frequency"].rank(ascending=True)
    ranks["monetary_rank"] = cluster_summary["monetary"].rank(ascending=True)

    # Higher total rank score => more "high-risk-like"
    ranks["risk_score"] = ranks["recency_rank"] + ranks["frequency_rank"] + ranks["monetary_rank"]

    high_risk_cluster = int(ranks["risk_score"].idxmin())

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    return rfm, high_risk_cluster, cluster_summary


def build_customer_target(
    df: pd.DataFrame,
    config: RFMConfig = RFMConfig(),
    snapshot_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, int, pd.DataFrame, pd.Timestamp]:
    """
    Convenience wrapper: compute RFM, cluster, and return customer-level target table:
      columns [CustomerId, is_high_risk]
    """
    rfm, snap = compute_rfm(df, config=config, snapshot_date=snapshot_date)
    labeled_rfm, high_cluster, summary = assign_high_risk_label(rfm, config=config)
    target = labeled_rfm[[config.customer_col, "is_high_risk"]].copy()
    return target, high_cluster, summary, snap
