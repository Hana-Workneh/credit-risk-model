import pandas as pd

from src.target_engineering import compute_rfm, assign_high_risk_label, build_customer_target


def _sample_df():
    # 6 customers with clearly different engagement patterns
    return pd.DataFrame(
        {
            "CustomerId": ["A", "A", "B", "C", "D", "E", "F"],
            "TransactionStartTime": [
                "2018-11-01T00:00:00Z",
                "2018-11-02T00:00:00Z",
                "2019-02-10T10:00:00Z",  # recent
                "2019-02-10T11:00:00Z",  # recent
                "2018-11-03T00:00:00Z",  # old
                "2018-11-03T00:00:00Z",  # old
                "2019-02-10T12:00:00Z",  # recent
            ],
            "Value": [20, 30, 5000, 4000, 10, 15, 6000],
            "Amount": [20, 30, 5000, 4000, 10, 15, 6000],
            "ProductCategory": ["airtime"] * 7,
            "ChannelId": ["ChannelId_3"] * 7,
            "ProviderId": ["ProviderId_6"] * 7,
            "PricingStrategy": [2] * 7,
        }
    )


def test_compute_rfm_outputs_columns():
    df = _sample_df()
    rfm, snap = compute_rfm(df)
    assert {"CustomerId", "recency", "frequency", "monetary"}.issubset(set(rfm.columns))
    assert rfm.shape[0] == df["CustomerId"].nunique()


def test_assign_high_risk_label_is_binary_and_nontrivial():
    df = _sample_df()
    rfm, _ = compute_rfm(df)
    labeled, high_cluster, summary = assign_high_risk_label(rfm)

    assert "is_high_risk" in labeled.columns
    assert set(labeled["is_high_risk"].unique()).issubset({0, 1})
    assert labeled["is_high_risk"].sum() >= 1
    assert labeled["is_high_risk"].sum() < labeled.shape[0]
    assert isinstance(high_cluster, int)
    assert summary.shape[0] == 3  # k=3


def test_build_customer_target_returns_customer_table():
    df = _sample_df()
    target, high_cluster, summary, snap = build_customer_target(df)
    assert target.shape[0] == df["CustomerId"].nunique()
    assert set(target["is_high_risk"].unique()).issubset({0, 1})
