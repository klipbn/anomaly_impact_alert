from __future__ import annotations

from dataclasses import dataclass, asdict, field, is_dataclass
from typing import Optional, List, Literal, Dict, Any
import numpy as np
import pandas as pd
import bottleneck as bn
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import STL
import warnings


columns_true: List[str] = [
    "time_at", "metric_value", "ci_mean", "ci_std", "ci_upper", "ci_lower",
    "ci_alert", "z_score", "z_alert", "iforest_alert", "lof_alert",
    "stl_resid", "stl_alert", "sesd_alert", "cusum_alert", "anomaly_final",
    "metric_name", "granularity"
]


# =========================
# Параметры тюнинга (Config)
# =========================
@dataclass
class AnomalyParams:
    """
    Конфиг для управления чувствительностью и окнами.
    По умолчанию значения соответствуют текущей логике.
    Любой параметр можно переопределить при вызове функций.

    NB: Формулы внутри расчётов не менялись.
    """

    # Общие
    granularity: Literal["hourly", "daily"] = "hourly"

    # CI/Z (MAD по rolling history)
    ci_k: float = 1.44
    z_threshold: float = 1.44
    rolling_window_hourly: int = 24
    rolling_window_daily: int = 7

    # STL (сезонность/период)
    stl_period_hourly: int = 24 * 7
    stl_period_daily: int = 7
    stl_std_multiplier: float = 2.0

    # SESD / Seasonal ESD
    sesd_alpha: float = 0.1
    seasonality_hourly: int = 24 * 7
    seasonality_daily: int = 7
    sesd_window_hourly: int = 24
    sesd_window_daily: int = 7
    sesd_ppd_hourly: int = 24    # points per day (для trend_window)
    sesd_ppd_daily: int = 1
    sesd_hybrid: bool = True     # hybrid в seasonal_esd_full

    # LOF / IForest
    contamination_threshold: float = 0.15  # IsolationForest contamination
    lof_contamination: float = 0.15
    lof_neighbors_hourly: int = 10
    lof_neighbors_daily: int = 10

    # CUSUM
    cusum_k: float = 0.5
    cusum_h: float = 5
    cusum_reference_window: int = 50

    # Вспомогательные переключатели
    enable_sesd: bool = True
    enable_stl: bool = True
    enable_iforest: bool = True
    enable_lof: bool = True
    enable_cusum: bool = True


def seasonal_esd_full(
    ts: np.ndarray,
    window: int = 50,
    seasonality: Optional[int] = 24 * 7,
    trend_window: int = 2,
    alpha: float = 0.2,
    hybrid: bool = True
) -> bool:
    """
    Проверка последней точки временного ряда на аномалию с учётом сезонности и тренда

    :param ts: (np.ndarray) массив с dtype=[('f0', 'float32'), ('f1', 'bool')], где f1 — маска выброшенных точек
    :param window: окно для reference_filter
    :param seasonality: длина сезонности (например, 24*7 для почасового ряда)
    :param trend_window: окно для удаления тренда
    :param alpha: уровень значимости
    :param hybrid: использовать ли MAD вместо стандартного отклонения
    :return: True/False — является ли последняя точка аномалией
    """
    values, mask = ts['f0'], ts['f1']
    trend_window = max(trend_window, 2)

    def calc_zscore(arr):
        if hybrid:
            median = np.median(arr)
            mad = np.median(np.abs(arr - median)) or 1e-9
            return (arr - median) / mad
        return stats.zscore(arr, ddof=1, nan_policy='omit')

    def get_seasonal_residual(data):
        detrended = data[trend_window - 1:] - bn.move_mean(data, window=trend_window, min_count=trend_window)[trend_window - 1:]
        avg = np.array([bn.nanmean(detrended[i::seasonality]) for i in range(seasonality)])
        avg -= bn.nanmean(avg)
        seasonal = np.tile(avg, len(detrended) // seasonality + 1)[:len(detrended)]
        return detrended - seasonal

    import warnings
    
    def grubbs_statistic(x, m):
        masked = np.ma.array(x, mask=m)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*'partition' will ignore the mask of the MaskedArray.*",
                category=UserWarning,
                module="numpy.core.fromnumeric",
            )
            return np.abs(calc_zscore(masked))[-1]

    def grubbs_critical(n):
        t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        return ((n - 1) * t) / np.sqrt(n * (1 + t**2 / (n - 2)))

    def esd_test(x, m):
        n = len(x) - np.sum(m)
        if n < 3:
            return False
        return grubbs_statistic(x, m) > grubbs_critical(n)

    def reference_check(x):
        diff_now = np.diff(x[-window:])
        diff_prev = np.diff(x[-window - seasonality:-seasonality])
        delta = np.abs(diff_now - diff_prev)
        q_now = np.quantile(diff_now, 1 - alpha)
        q_diff = np.quantile(delta, 1 - alpha)
        return delta[-1] >= q_diff and abs(diff_now[-1]) >= q_now

    # фильтры
    if mask[-2] and not np.array_equal(mask[-5:-1], [True] * 4):
        outlier_diff = True
    else:
        outlier_diff = window == 1 or reference_check(values)

    if outlier_diff:
        residual = get_seasonal_residual(values)
        adjusted_mask = mask[trend_window - 1:]
        outlier_esd = esd_test(residual, adjusted_mask)
    else:
        outlier_esd = False

    return bool(outlier_diff and outlier_esd)


def _resolve_params(
    granularity: Literal["hourly", "daily"],
    params: Optional[AnomalyParams],
    overrides: Optional[Dict[str, Any]] = None
) -> AnomalyParams:
    base = AnomalyParams(granularity=granularity)
    if params is not None:
        if not is_dataclass(params):
            raise TypeError("params must be an AnomalyParams dataclass")
        base = AnomalyParams(**{**asdict(base), **asdict(params)})
    if overrides:
        base = AnomalyParams(**{**asdict(base), **overrides})
    return base


def calculate_anomalies(
    df: pd.DataFrame,
    time_col: str = "time_at",
    value_col: str = "metric_value",
    freq: Literal["hourly", "daily"] = "hourly",
    # --- legacy args  ---
    ci_k: float = 1.44,
    z_threshold: float = 1.44,
    contamination_threshold: float = 0.15,
    lof_contamination: float = 0.15,
    lof_neighbors_hourly: int = 10,
    lof_neighbors_daily: int = 10,
    stl_std_multiplier: float = 2.0,
    sesd_alpha: float = 0.1,

    params: Optional[AnomalyParams] = None,
    **overrides
) -> Optional[pd.DataFrame]:

    resolved = _resolve_params(freq, params, overrides)

    if params is None:
        legacy_over = dict(
            ci_k=ci_k,
            z_threshold=z_threshold,
            contamination_threshold=contamination_threshold,
            lof_contamination=lof_contamination,
            lof_neighbors_hourly=lof_neighbors_hourly,
            lof_neighbors_daily=lof_neighbors_daily,
            stl_std_multiplier=stl_std_multiplier,
            sesd_alpha=sesd_alpha,
        )
        legacy_over = {k: v for k, v in legacy_over.items() if k not in overrides}
        resolved = _resolve_params(freq, resolved, legacy_over)

    df = df.copy()
    df[value_col] = df[value_col].astype(float)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    if df[value_col].iloc[-1] <= 0.000001:
        return None

    # Выбор окон/параметров по гранулярности
    if resolved.granularity == "hourly":
        rolling_window = resolved.rolling_window_hourly
        stl_period = resolved.stl_period_hourly
        sesd_ppd = resolved.sesd_ppd_hourly
        seasonality = resolved.seasonality_hourly
        sesd_window = resolved.sesd_window_hourly
        lof_neighbors = resolved.lof_neighbors_hourly
    else:  # daily
        rolling_window = resolved.rolling_window_daily
        stl_period = resolved.stl_period_daily
        sesd_ppd = resolved.sesd_ppd_daily
        seasonality = resolved.seasonality_daily
        sesd_window = resolved.sesd_window_daily
        lof_neighbors = resolved.lof_neighbors_daily

    # перед расчётами
    df["hour"] = df[time_col].dt.hour
    df["dow"]  = df[time_col].dt.dayofweek  # 0..6
    
    def compute_ci_and_z_mad(row, df_hist, value_col, ci_k, z_threshold,
                             rolling_window_days, freq,
                             min_points_same_bin=4,  # сколько точек достаточно для того же часа/дня недели
                             eps=1e-9):
        # только раньше текущего времени
        base_time_mask = df_hist[time_col] < row[time_col]
    
        if freq == "hourly":
            bin_mask = (df_hist["hour"] == row["hour"]) & base_time_mask
        elif freq == "daily":
            bin_mask = (df_hist["dow"] == row["dow"]) & base_time_mask
        else:
            raise ValueError(f"Unsupported freq: {freq}")
    
        hist_bin = (
            df_hist.loc[bin_mask, [time_col, value_col]]
                  .sort_values(time_col)
                  .tail(min_points_same_bin * 8) 
        )
    
        if len(hist_bin) >= min_points_same_bin:
            history = hist_bin[value_col]
        else:
            history = (
                df_hist.loc[base_time_mask, [time_col, value_col]]
                       .sort_values(time_col)
                       .tail(rolling_window_days)[value_col]
            )
    
        if history.empty:
            return pd.Series([np.nan]*7)
    
        median = history.median()
        mad = np.median(np.abs(history - median))
        sigma_mad = max(mad * 1.4826, eps)
    
        ci_upper = median + ci_k * sigma_mad
        ci_lower = median - ci_k * sigma_mad
        ci_alert = int((row[value_col] < ci_lower) or (row[value_col] > ci_upper))
    
        z_score = (row[value_col] - median) / sigma_mad
        z_alert = int(abs(z_score) > z_threshold)
    
        return pd.Series([median, sigma_mad, ci_upper, ci_lower, ci_alert, z_score, z_alert])
    
    df[["ci_mean","ci_std","ci_upper","ci_lower","ci_alert","z_score","z_alert"]] = df.apply(
        lambda row: compute_ci_and_z_mad(
            row,
            df,
            value_col,
            resolved.ci_k,
            resolved.z_threshold,
            rolling_window,
            resolved.granularity,
            min_points_same_bin=4,
            eps=1e-6
        ),
        axis=1
    )

    # Isolation Forest
    if resolved.enable_iforest:
        iso_forest = IsolationForest(
            contamination=resolved.contamination_threshold,
            max_samples="auto",
            random_state=42
        )
        df["iforest_alert"] = iso_forest.fit_predict(df[[value_col]])
        df["iforest_alert"] = df["iforest_alert"].apply(lambda x: 1 if x == -1 else 0)
    else:
        df["iforest_alert"] = 0

    # LOF
    if resolved.enable_lof:
        lof = LocalOutlierFactor(
            n_neighbors=lof_neighbors,
            contamination=resolved.lof_contamination,
            metric="euclidean"
        )
        lof_result = lof.fit_predict(df[[value_col]])
        df["lof_alert"] = pd.Series(lof_result, index=df.index).apply(lambda x: 1 if x == -1 else 0)
    else:
        df["lof_alert"] = 0

    # STL
    if resolved.enable_stl:
        stl = STL(df[value_col], period=stl_period, robust=False)
        res = stl.fit()
        df["stl_resid"] = res.resid
        df["stl_alert"] = (abs(res.resid) > (resolved.stl_std_multiplier * np.std(res.resid))).astype(int)
    else:
        df["stl_resid"] = np.nan
        df["stl_alert"] = 0

    # SESD
    df["sesd_alert"] = 0
    if resolved.enable_sesd:
        buffer_size = seasonality + sesd_window
        if len(df) > buffer_size:
            array = np.array([(x, False) for x in df[value_col].iloc[:buffer_size].values], dtype="float32,bool")
            calc_range = df.index[buffer_size:]
            for point in calc_range:
                try:
                    is_outlier = seasonal_esd_full(
                        array,
                        seasonality=seasonality,
                        alpha=resolved.sesd_alpha,
                        trend_window=sesd_ppd,
                        window=sesd_window,
                        hybrid=resolved.sesd_hybrid
                    )
                    df.loc[point, "sesd_alert"] = int(is_outlier)
                except Exception:
                    df.loc[point, "sesd_alert"] = 0
                array[:-1] = array[1:]
                array[-1] = (df.loc[point, value_col], False)

    # CUSUM
    def calculate_cusum(series, k=0.5, h=5, reference_window=50):
        """
        Возвращает Series с флагами CUSUM (1=аномалия) для сдвига среднего
        :param series: pd.Series — данные
        :param k: целевое смещение
        :param h: порог срабатывания
        :param reference_window: на сколько точек назад брать среднее и std
        """
        mean = np.mean(series.iloc[:reference_window])
        std = np.std(series.iloc[:reference_window])
        s_pos, s_neg = 0, 0
        flags = np.zeros(len(series))
        for i, x in enumerate(series):
            s_pos = max(0, s_pos + (x - mean - k) / std)
            s_neg = min(0, s_neg + (x - mean + k) / std)
            if s_pos > h or s_neg < -h:
                flags[i] = 1
                s_pos, s_neg = 0, 0
        return pd.Series(flags, index=series.index, dtype=int)

    if resolved.enable_cusum:
        df["cusum_alert"] = calculate_cusum(
            df[value_col],
            k=resolved.cusum_k,
            h=resolved.cusum_h,
            reference_window=resolved.cusum_reference_window
        )
    else:
        df["cusum_alert"] = 0

    # Итоговая метка
    def mark_anomalies(df_: pd.DataFrame) -> pd.DataFrame:
        df_ = df_.copy()
        conditions = (df_["ci_alert"] == 1) | (df_["z_alert"] == 1)
        summed = df_[["iforest_alert", "lof_alert", "stl_alert", "sesd_alert", "cusum_alert"]].sum(axis=1)
        df_["anomaly_final"] = 0
        df_.loc[conditions & (summed >= 2), "anomaly_final"] = 1
        return df_

    df = mark_anomalies(df)

    # df = df.drop("hour", axis=1)/
    df = df.drop(columns=[c for c in ["hour", "dow"] if c in df.columns])
    return df.reset_index()


def analyze_latest_point(
    df_two_cols: pd.DataFrame,
    metric_name: str,
    granularity: Literal["hourly", "daily"] = "hourly",
    params: Optional[AnomalyParams] = None,
    **overrides
) -> pd.DataFrame:
    """
    Принимает DF ТОЛЬКО со столбцами: time_at, metric_value
    Возвращает одну строку по последней дате со всеми признаками и флагами

    Можно передавать:
      - params=AnomalyParams(...)
      - любые overrides (например, stl_period_daily=14, ci_k=1.2, ...)
    """
    if set(df_two_cols.columns) != {"time_at", "metric_value"}:
        cols = list(df_two_cols.columns)
        if len(cols) == 2:
            df_two_cols = df_two_cols.rename(columns={cols[0]: "time_at", cols[1]: "metric_value"})
        else:
            raise ValueError("Input dataframe must contain exactly two columns: time_at, metric_value")

    df = df_two_cols.copy()
    df["time_at"] = pd.to_datetime(df["time_at"])
    df = df.sort_values("time_at")

    latest_time = df["time_at"].max()
    latest_value = float(df.loc[df["time_at"] == latest_time, "metric_value"].iloc[0])

    res = calculate_anomalies(
        df[["time_at", "metric_value"]],
        time_col="time_at",
        value_col="metric_value",
        freq=granularity,
        params=params,
        **overrides
    )

    if res is not None and not res.empty:
        last_row = res.loc[res["time_at"] == latest_time].copy()
    else:
        # Фолбэк, чтобы всегда вернуть строку
        last_row = pd.DataFrame([{
            "time_at": latest_time,
            "metric_value": latest_value,
            "ci_mean": np.nan, "ci_std": np.nan, "ci_upper": np.nan, "ci_lower": np.nan,
            "ci_alert": 0, "z_score": np.nan, "z_alert": 0,
            "iforest_alert": 0, "lof_alert": 0,
            "stl_resid": np.nan, "stl_alert": 0,
            "sesd_alert": 0, "cusum_alert": 0,
            "anomaly_final": 0
        }])

    last_row["metric_name"] = metric_name
    last_row["granularity"] = granularity

    for col in columns_true:
        if col not in last_row.columns:
            last_row[col] = np.nan
    last_row = last_row[columns_true]

    if len(last_row) > 1:
        last_row = last_row.iloc[[-1]]

    return last_row.reset_index(drop=True)
    