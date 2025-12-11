from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd


@dataclass
class ImpactConfig:
    time_col: str = "time_at"
    group_col: str = "group_col"
    metric_col: str = "metric_value"
    window_days: int = 5
    top_k: int = 3
    exclude_groups: Tuple[str, ...] = ("total", "-")  # кого выкидывать из групп


def _fmt_num(x: float) -> str:
    try:
        return f"{x:,.0f}"
    except Exception:
        return str(int(round(x)))


def _fmt_pct(x: float) -> str:
    if np.isnan(x) or np.isinf(x):
        return "0.0%"
    return f"{x:.1f}%"


def _compute_impact_for_date(
    df_impact: pd.DataFrame,
    now: pd.Timestamp,
    cfg: ImpactConfig,
) -> pd.DataFrame:
    """
    Возвращает таблицу вкладов по группам на дату now
    Колонки: section, metric_now, metric_last, metric_Δ_abs, metric_Δ_pct, metric_Δ_impact_pct
    """
    t, g, v = cfg.time_col, cfg.group_col, cfg.metric_col

    work = df_impact[[t, g, v]].copy()
    work[t] = pd.to_datetime(work[t])
    work[g] = work[g].astype(str)

    # исключаем Total, '-' (регистронезависимо)
    mask_excl = work[g].str.lower().isin([x.lower() for x in cfg.exclude_groups])
    work = work.loc[~mask_excl]

    df_now = (
        work[work[t] == now]
        .groupby(g, dropna=False, as_index=False)[v]
        .sum()
        .rename(columns={g: "section", v: "metric_now"})
    )

    if df_now.empty:
        return pd.DataFrame(
            columns=[
                "section",
                "metric_now",
                "metric_last",
                "metric_Δ_abs",
                "metric_Δ_pct",
                "metric_Δ_impact_pct",
            ]
        )

    # предыдущие дни
    last_days = [now - timedelta(days=i) for i in range(1, cfg.window_days + 1)]
    df_last = (
        work[work[t].isin(last_days)]
        .groupby(g, dropna=False, as_index=False)[v]
        .mean()
        .rename(columns={g: "section", v: "metric_last"})
    )

    # объединяем
    out = pd.merge(df_now, df_last, on="section", how="outer").fillna(0.0)

    # дельты
    out["metric_Δ_abs"] = out["metric_now"] - out["metric_last"]

    def _safe_pct(now_val, last_val):
        if last_val == 0:
            return 0.0
        return (now_val / last_val - 1.0) * 100.0

    out["metric_Δ_pct"] = out.apply(
        lambda r: _safe_pct(r["metric_now"], r["metric_last"]),
        axis=1,
    )

    total_delta = float(out["metric_Δ_abs"].sum())
    denom = abs(total_delta) if abs(total_delta) > 0 else 1.0
    out["metric_Δ_impact_pct"] = (out["metric_Δ_abs"] / denom) * 100.0

    return out


def _build_impact_text(impact_df: pd.DataFrame, top_k: int) -> str:
    """
    Формирует строки в формате:
    1. <section>: -76,044 (-23.5%), вклад: -45.6%
    """
    if impact_df.empty:
        return ""

    total_delta = float(impact_df["metric_Δ_abs"].sum())

    if total_delta < 0:
        ordered = impact_df.sort_values("metric_Δ_impact_pct").head(top_k)
    else:
        ordered = impact_df.sort_values(
            "metric_Δ_impact_pct", ascending=False
        ).head(top_k)

    lines: List[str] = []
    for i, row in enumerate(ordered.itertuples(index=False), 1):
        sect = getattr(row, "section")
        d_abs = getattr(row, "metric_Δ_abs")
        d_pct = getattr(row, "metric_Δ_pct")
        imp = getattr(row, "metric_Δ_impact_pct")
        line = (
            f"{i}. {sect}: {_fmt_num(d_abs)} ({_fmt_pct(d_pct)}), "
            f"вклад: {_fmt_pct(imp)}"
        )
        lines.append(line)

    return "\n".join(lines)


def _one_date_pipeline(
    now: pd.Timestamp,
    df_impact: pd.DataFrame,
    cfg: ImpactConfig,
) -> Tuple[pd.Timestamp, str]:
    """Вычисляет impact_text для одной даты"""
    imp = _compute_impact_for_date(df_impact, now, cfg)
    text = _build_impact_text(imp, cfg.top_k)
    return now, text


def attach_impact_text(
    df_anomaly: pd.DataFrame,
    df_impact: pd.DataFrame,
    *,
    config: Optional[ImpactConfig] = None,
    time_col_anom: str = "time_at",
    anomaly_flag_col: str = "anomaly_final",
    output_col: str = "impact_text",
) -> pd.DataFrame:
    """
    Базовая функция: добавляет одну колонку импакта.
    """
    cfg = config or ImpactConfig()

    a = df_anomaly.copy()
    a[time_col_anom] = pd.to_datetime(a[time_col_anom])

    # Даты с аномалиями
    anomaly_dates = (
        a.loc[a[anomaly_flag_col] == 1, time_col_anom]
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    if not anomaly_dates:
        a[output_col] = ""
        return a

    results: dict[pd.Timestamp, str] = {}
    for d in anomaly_dates:
        dt, text = _one_date_pipeline(pd.Timestamp(d), df_impact, cfg)
        results[dt] = text

    a[output_col] = a[time_col_anom].map(results).fillna("")

    return a


def attach_multi_impact(
    df_anomaly: pd.DataFrame,
    df: pd.DataFrame,
    *,
    dims: Sequence[str],
    time_col: str = "time_at",
    metric_col: str = "metric_value",
    total_value: str = "total",
    time_col_anom: str = "time_at",
    anomaly_flag_col: str = "anomaly_final",
    window_days: int = 5,
    top_k: int = 3,
    exclude_groups: Tuple[str, ...] = ("total", "-"),
    prefix: str = "impact_text_",
) -> pd.DataFrame:
    """
    Принимает:
      - df_anomaly  — датафрейм с аномалиями (time_col_anom, anomaly_flag_col)
      - df          — полный датафрейм с измерениями
      - dims        — список измерений, по которым считать вклад (["country", "platform", ...])

    Логика по каждому измерению dim:
      - берём строки, где:
          dim != total_value
          все остальные измерения из dims == total_value
      - group_col = dim
      - считаем импакты
      - добавляем колонку prefix + dim  (например impact_text_country)
    """
    result = df_anomaly.copy()

    for dim in dims:
        other_dims = [d for d in dims if d != dim]

        mask = df[dim] != total_value
        for od in other_dims:
            mask &= df[od] == total_value

        df_impact_dim = (
            df.loc[mask, [time_col, dim, metric_col]]
            .rename(columns={dim: "group_col"})
        )

        cfg = ImpactConfig(
            time_col=time_col,
            group_col="group_col",
            metric_col=metric_col,
            window_days=window_days,
            top_k=top_k,
            exclude_groups=exclude_groups,
        )

        col_name = f"{prefix}{dim}"
        result = attach_impact_text(
            df_anomaly=result,
            df_impact=df_impact_dim,
            config=cfg,
            time_col_anom=time_col_anom,
            anomaly_flag_col=anomaly_flag_col,
            output_col=col_name,
        )

    return result
