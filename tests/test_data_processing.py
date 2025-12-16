import pandas as pd

from src.data_processing import make_customer_feature_frame, build_processing_pipeline


def _sample_df():
    return pd.DataFrame(
        {
            "CustomerId": ["C1", "C1", "C2"],
            "Amount": [1000.0, -20.0, 500.0],
            "TransactionStartTime": ["2018-11-15T02:18:49Z", "2018-11-15T03:18:49Z", "2018-11-16T10:00:00Z"],
            "ProductCategory": ["airtime", "financial_services", "airtime"],
            "ChannelId": ["ChannelId_3", "ChannelId_2", "ChannelId_3"],
            "ProviderId": ["ProviderId_6", "ProviderId_4", "ProviderId_6"],
            "PricingStrategy": [2, 2, 2],
            # extra columns allowed
            "TransactionId": ["T1", "T2", "T3"],
        }
    )


def test_customer_feature_frame_columns():
    df = _sample_df()
    feat = make_customer_feature_frame(df)

    expected_cols = {
        "CustomerId",
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
        "last_tx_hour",
        "last_tx_day",
        "last_tx_month",
        "last_tx_year",
        "mode_product_category",
        "mode_channel_id",
        "mode_provider_id",
        "mode_pricing_strategy",
    }
    assert expected_cols.issubset(set(feat.columns))


def test_pipeline_fit_transform_runs():
    df = _sample_df()
    pipe = build_processing_pipeline()
    X = pipe.fit_transform(df)
    # Should produce 2 customers (C1, C2)
    assert X.shape[0] == 2
