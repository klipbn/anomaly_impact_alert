from __future__ import annotations

import pandas as pd

from anomaly_impact_alert.anomaly_detector import (
    AnomalyParams,
    analyze_latest_point,
    month_position_category,
)


def test_month_position_category_has_four_special_values():
    assert month_position_category(pd.Timestamp("2026-02-01")) == 1
    assert month_position_category(pd.Timestamp("2026-02-02")) == 2
    assert month_position_category(pd.Timestamp("2026-02-27")) == -2
    assert month_position_category(pd.Timestamp("2026-02-28")) == -1
    assert month_position_category(pd.Timestamp("2026-02-15")) == 0


def test_daily_ci_uses_same_month_position_when_available():
    dates = pd.date_range("2021-10-01", "2022-01-01", freq="D")
    values = pd.Series(10.0, index=dates)
    special_dates = [
        date
        for date in dates
        if month_position_category(date) == 1
    ]
    values.loc[special_dates] = 100.0
    values.loc[pd.Timestamp("2022-01-01")] = 100.0
    df = pd.DataFrame({"time_at": dates, "metric_value": values.to_numpy()})

    result = analyze_latest_point(
        df,
        metric_name="amount",
        granularity="daily",
        params=AnomalyParams(
            granularity="daily",
            ci_min_points_same_bin_daily=4,
            enable_iforest=False,
            enable_lof=False,
            enable_stl=False,
            enable_sesd=False,
            enable_cusum=False,
        ),
    )

    assert float(result.iloc[0]["ci_mean"]) == 100.0
