from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass
from typing import Optional, List, Literal, Dict, Any, Union
import numpy as np
import pandas as pd
import bottleneck as bn
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import STL


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
    Любой параметр можно переопределить при вызове функций.

    ВАЖНО:
    - Если передаёшь готовый AnomalyParams в params=..., пресет sensitivity игнорируется.
    - Если params=None, можно использовать sensitivity (1..6) как преднастройку.
    """

    # Общие
    granularity: Literal["hourly", "daily"] = "hourly"

    # CI/Z (MAD по rolling history)
    ci_k: float = 1.44
    z_threshold: float = 1.44
    rolling_window_hourly: int = 24
    rolling_window_daily: int = 7

    # --- Advanced CI ---
    # использовать ли сначала bin по часу / дню недели, а потом fall-back к rolling
    ci_use_same_bin: bool = True
    # минимальное число точек в bin'е, чтобы использовать его (по умолчанию 4)
    ci_min_points_same_bin_hourly: int = 4
    ci_min_points_same_bin_daily: int = 4
    # насколько длинным делать хвост bin'а (min_points_same_bin * этот фактор)
    ci_bin_tail_factor: int = 8
    # сглаживание оценок σ (MAD) по времени; 0 = без сглаживания
    ci_smooth_mad_hourly: int = 0
    ci_smooth_mad_daily: int = 0
    # опциональный клип σ, чтобы не было слишком узких/широких интервалов
    ci_std_clip_min: Optional[float] = None
    ci_std_clip_max: Optional[float] = None

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


# ---------- SESD (Seasonal ESD ядро) ----------
def seasonal_esd_full(
    ts: np.ndarray,
    window: int = 50,
    seasonality: Optional[int] = 24 * 7,
    trend_window: int = 2,
    alpha: float = 0.2,
    hybrid: bool = True
) -> bool:
    """
    Проверка последней точки временного ряда на аномалию с учётом сезонности и тренда.

    :param ts: (np.ndarray) массив с dtype=[('f0', 'float32'), ('f1', 'bool')],
               где f1 — маска выброшенных точек.
    """
    values, mask = ts["f0"], ts["f1"]
    trend_window = max(trend_window, 2)

    def calc_zscore(arr):
        if hybrid:
            median = np.median(arr)
            mad = np.median(np.abs(arr - median)) or 1e-9
            sigma = mad * 1.4826  # MAD → sigma
            return (arr - median) / sigma
        return stats.zscore(arr, ddof=1, nan_policy="omit")

    def get_seasonal_residual(data):
        detrended = (
            data[trend_window - 1:]
            - bn.move_mean(data, window=trend_window, min_count=trend_window)[
                trend_window - 1:
            ]
        )
        avg = np.array(
            [bn.nanmean(detrended[i::seasonality]) for i in range(seasonality)]
        )
        avg -= bn.nanmean(avg)
        seasonal = np.tile(avg, len(detrended) // seasonality + 1)[: len(detrended)]
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
        diff_prev = np.diff(x[-window - seasonality : -seasonality])
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


# ---------- Presets для чувствительности ----------
def _normalize_sensitivity(
    sensitivity: Optional[Union[int, str]]
) -> Optional[int]:
    """
    Приводим sensitivity к int 1..6 или None.

    1 — очень чувствительно (узкие интервалы, много алертов)
    6 — мега слабо (широкие интервалы, минимум алертов)
    """
    if sensitivity is None:
        return None

    if isinstance(sensitivity, int):
        if 1 <= sensitivity <= 6:
            return sensitivity
        else:
            raise ValueError("sensitivity int must be in range 1..6")

    if isinstance(sensitivity, str):
        s = sensitivity.lower().strip()
        mapping_str = {
            "1": 1,
            "very_sensitive": 1,
            "очень_чувствительно": 1,
            "2": 2,
            "sensitive": 2,
            "чувствительно": 2,
            "3": 3,
            "medium": 3,
            "средне": 3,
            "4": 4,
            "low": 4,
            "слабо": 4,
            "5": 5,
            "very_low": 5,
            "очень_слабо": 5,
            "6": 6,
            "ultra_low": 6,
            "mega_low": 6,
            "мега_слабо": 6,
        }
        if s in mapping_str:
            return mapping_str[s]
        else:
            raise ValueError(f"Unknown sensitivity string: {s}")

    raise TypeError("sensitivity must be int 1..6 or str")


def _apply_sensitivity_profile(
    base: AnomalyParams,
    level_int: int,
) -> AnomalyParams:
    """
    Настраивает базовый AnomalyParams под желаемую чувствительность.

    level_int:
      1 — Очень чувствительно
      2 — Чувствительно
      3 — Средне
      4 — Слабо чувствительно
      5 — Очень слабо
      6 — Мега слабо
    """

    # стартуем с базового словаря
    d = asdict(base)

    if level_int == 1:
        # ОЧЕНЬ ЧУВСТВИТЕЛЬНО
        d.update(
            dict(
                ci_k=1.5,
                z_threshold=1.5,
                stl_std_multiplier=1.0,
                sesd_alpha=0.22,
                contamination_threshold=0.15,
                lof_contamination=0.15,
            )
        )
    elif level_int == 2:
        # ЧУВСТВИТЕЛЬНО
        d.update(
            dict(
                ci_k=1.8,
                z_threshold=1.8,
                stl_std_multiplier=1.2,
                sesd_alpha=0.18,
                contamination_threshold=0.12,
                lof_contamination=0.12,
            )
        )
    elif level_int == 3:
        # СРЕДНЯЯ ЧУВСТВИТЕЛЬНОСТЬ
        d.update(
            dict(
                ci_k=2.0,
                z_threshold=2.0,
                stl_std_multiplier=1.5,
                sesd_alpha=0.12,
                contamination_threshold=0.10,
                lof_contamination=0.10,
            )
        )
    elif level_int == 4:
        # СЛАБО ЧУВСТВИТЕЛЬНО
        d.update(
            dict(
                ci_k=2.5,
                z_threshold=2.5,
                stl_std_multiplier=2.0,
                sesd_alpha=0.09,
                contamination_threshold=0.08,
                lof_contamination=0.08,
            )
        )
    elif level_int == 5:
        # ОЧЕНЬ СЛАБО
        d.update(
            dict(
                ci_k=3.0,
                z_threshold=3.0,
                stl_std_multiplier=2.5,
                sesd_alpha=0.07,
                contamination_threshold=0.06,
                lof_contamination=0.06,
            )
        )
    elif level_int == 6:
        # МЕГА СЛАБО (почти только экстремали)
        d.update(
            dict(
                ci_k=4.0,
                z_threshold=4.0,
                stl_std_multiplier=3.0,
                sesd_alpha=0.05,
                contamination_threshold=0.05,
                lof_contamination=0.05,
            )
        )
    else:
        # technically не должны сюда попасть
        return base

    return AnomalyParams(**d)


def _resolve_params(
    granularity: Literal["hourly", "daily"],
    params: Optional[AnomalyParams],
    overrides: Optional[Dict[str, Any]] = None,
    sensitivity: Optional[Union[int, str]] = None,
) -> AnomalyParams:
    """
    Логика:
      1) Стартуем с базового AnomalyParams(granularity=...)
      2) Если params is None и sensitivity задан → применяем пресет к базе
      3) Если params не None → поверх базы накатываем params (ручная конфигурация)
      4) Поверх всего применяем overrides (явные аргументы функций)
    """
    base = AnomalyParams(granularity=granularity)

    # если пользователь НЕ передал params, можно использовать пресет по чувствительности
    if params is None and sensitivity is not None:
        level_int = _normalize_sensitivity(sensitivity)
        base = _apply_sensitivity_profile(base, level_int)

    if params is not None:
        if not is_dataclass(params):
            raise TypeError("params must be an AnomalyParams dataclass")
        base = AnomalyParams(**{**asdict(base), **asdict(params)})

    if overrides:
        base = AnomalyParams(**{**asdict(base), **overrides})

    return base


# ---------- Основной расчёт ----------
def calculate_anomalies(
    df: pd.DataFrame,
    time_col: str = "time_at",
    value_col: str = "metric_value",
    freq: Literal["hourly", "daily"] = "hourly",
    # --- legacy args (оставлены для обратной совместимости) ---
    ci_k: float = 1.44,
    z_threshold: float = 1.44,
    contamination_threshold: float = 0.15,
    lof_contamination: float = 0.15,
    lof_neighbors_hourly: int = 10,
    lof_neighbors_daily: int = 10,
    stl_std_multiplier: float = 2.0,
    sesd_alpha: float = 0.1,
    params: Optional[AnomalyParams] = None,
    sensitivity: Optional[Union[int, str]] = None,  # <--- НОВЫЙ АРГУМЕНТ
    **overrides,
) -> Optional[pd.DataFrame]:

    # 1) нормальный путь: sensitivity / params / overrides
    resolved = _resolve_params(
        freq,
        params=params,
        overrides=overrides,
        sensitivity=sensitivity,
    )

    # 2) legacy-режим — ТОЛЬКО если не заданы ни params, ни sensitivity.
    #    То есть старый код без изменений продолжит работать.
    if params is None and sensitivity is None:
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
        # не трогаем то, что уже есть в overrides
        for k, v in legacy_over.items():
            if k in overrides:
                continue
            if hasattr(resolved, k):
                setattr(resolved, k, v)

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
        ci_min_points_same_bin = resolved.ci_min_points_same_bin_hourly
        ci_smooth_mad_window = resolved.ci_smooth_mad_hourly
    else:  # daily
        rolling_window = resolved.rolling_window_daily
        stl_period = resolved.stl_period_daily
        sesd_ppd = resolved.sesd_ppd_daily
        seasonality = resolved.seasonality_daily
        sesd_window = resolved.sesd_window_daily
        lof_neighbors = resolved.lof_neighbors_daily
        ci_min_points_same_bin = resolved.ci_min_points_same_bin_daily
        ci_smooth_mad_window = resolved.ci_smooth_mad_daily

    # перед расчётами
    df["hour"] = df[time_col].dt.hour
    df["dow"] = df[time_col].dt.dayofweek  # 0..6

    def compute_ci_and_z_mad(
        row,
        df_hist,
        value_col,
        ci_k,
        z_threshold,
        rolling_window_days,
        freq,
        min_points_same_bin,
        bin_tail_factor,
        use_same_bin: bool,
        eps=1e-9,
    ):
        # только раньше текущего времени
        base_time_mask = df_hist[time_col] < row[time_col]

        history = None

        if use_same_bin:
            if freq == "hourly":
                bin_mask = (df_hist["hour"] == row["hour"]) & base_time_mask
            elif freq == "daily":
                bin_mask = (df_hist["dow"] == row["dow"]) & base_time_mask
            else:
                raise ValueError(f"Unsupported freq: {freq}")

            hist_bin = (
                df_hist.loc[bin_mask, [time_col, value_col]]
                .sort_values(time_col)
                .tail(min_points_same_bin * bin_tail_factor)
            )

            if len(hist_bin) >= min_points_same_bin:
                history = hist_bin[value_col]

        # fall-back: обычный rolling хвост
        if history is None:
            history = (
                df_hist.loc[base_time_mask, [time_col, value_col]]
                .sort_values(time_col)
                .tail(rolling_window_days)[value_col]
            )

        if history.empty:
            return pd.Series([np.nan] * 7)

        median = history.median()
        mad = np.median(np.abs(history - median))
        sigma_mad = max(mad * 1.4826, eps)

        ci_upper = median + ci_k * sigma_mad
        ci_lower = median - ci_k * sigma_mad
        ci_alert = int((row[value_col] < ci_lower) or (row[value_col] > ci_upper))

        z_score = (row[value_col] - median) / sigma_mad
        z_alert = int(abs(z_score) > z_threshold)

        return pd.Series(
            [median, sigma_mad, ci_upper, ci_lower, ci_alert, z_score, z_alert]
        )

    df[
        [
            "ci_mean",
            "ci_std",
            "ci_upper",
            "ci_lower",
            "ci_alert",
            "z_score",
            "z_alert",
        ]
    ] = df.apply(
        lambda row: compute_ci_and_z_mad(
            row,
            df,
            value_col,
            resolved.ci_k,
            resolved.z_threshold,
            rolling_window,
            resolved.granularity,
            min_points_same_bin=ci_min_points_same_bin,
            bin_tail_factor=resolved.ci_bin_tail_factor,
            use_same_bin=resolved.ci_use_same_bin,
            eps=1e-6,
        ),
        axis=1,
    )

    # --- Пост-обработка CI: клип и сглаживание σ ---
    if resolved.ci_std_clip_min is not None or resolved.ci_std_clip_max is not None:
        df["ci_std"] = df["ci_std"].clip(
            lower=resolved.ci_std_clip_min, upper=resolved.ci_std_clip_max
        )

    if ci_smooth_mad_window and ci_smooth_mad_window > 1:
        df["ci_std"] = df["ci_std"].rolling(
            ci_smooth_mad_window, min_periods=1
        ).mean()

    # пересчёт границ CI и z-score после сглаживания/клипа σ
    nonzero_std = df["ci_std"].replace(0, np.nan)
    df["ci_upper"] = df["ci_mean"] + resolved.ci_k * df["ci_std"]
    df["ci_lower"] = df["ci_mean"] - resolved.ci_k * df["ci_std"]
    df["ci_alert"] = (
        (df[value_col] < df["ci_lower"]) | (df[value_col] > df["ci_upper"])
    ).astype(int)
    df["z_score"] = (df[value_col] - df["ci_mean"]) / nonzero_std
    df["z_score"] = df["z_score"].replace([np.inf, -np.inf], np.nan)
    df["z_alert"] = (df["z_score"].abs() > resolved.z_threshold).astype(int)

    # Isolation Forest
    if resolved.enable_iforest:
        iso_forest = IsolationForest(
            contamination=resolved.contamination_threshold,
            max_samples="auto",
            random_state=42,
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
            metric="euclidean",
        )
        lof_result = lof.fit_predict(df[[value_col]])
        df["lof_alert"] = pd.Series(lof_result, index=df.index).apply(
            lambda x: 1 if x == -1 else 0
        )
    else:
        df["lof_alert"] = 0

    # STL
    if resolved.enable_stl:
        stl = STL(df[value_col], period=stl_period, robust=False)
        res = stl.fit()
        df["stl_resid"] = res.resid
        df["stl_alert"] = (
            abs(res.resid) > (resolved.stl_std_multiplier * np.std(res.resid))
        ).astype(int)
    else:
        df["stl_resid"] = np.nan
        df["stl_alert"] = 0

    # SESD
    df["sesd_alert"] = 0
    if resolved.enable_sesd:
        buffer_size = seasonality + sesd_window
        if len(df) >= buffer_size:
            values = df[value_col].to_numpy()
            idx = df.index.to_numpy()

            array = np.array(
                [(x, False) for x in values[:buffer_size]], dtype="float32,bool"
            )

            for i in range(buffer_size, len(df)):
                array[:-1] = array[1:]
                array[-1] = (values[i], False)

                try:
                    is_outlier = seasonal_esd_full(
                        array,
                        seasonality=seasonality,
                        alpha=resolved.sesd_alpha,
                        trend_window=sesd_ppd,
                        window=sesd_window,
                        hybrid=resolved.sesd_hybrid,
                    )
                    df.loc[idx[i], "sesd_alert"] = int(is_outlier)
                    if is_outlier:
                        array[-1]["f1"] = True
                except Exception:
                    df.loc[idx[i], "sesd_alert"] = 0

    # CUSUM
    def calculate_cusum(series, k=0.5, h=5, reference_window=50):
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
            reference_window=resolved.cusum_reference_window,
        )
    else:
        df["cusum_alert"] = 0

    # Итоговая метка
    def mark_anomalies(df_: pd.DataFrame) -> pd.DataFrame:
        df_ = df_.copy()
        conditions = (df_["ci_alert"] == 1) | (df_["z_alert"] == 1)
        summed = df_[
            ["iforest_alert", "lof_alert", "stl_alert", "sesd_alert", "cusum_alert"]
        ].sum(axis=1)
        df_["anomaly_final"] = 0
        df_.loc[conditions & (summed >= 2), "anomaly_final"] = 1
        return df_

    df = mark_anomalies(df)

    df = df.drop(columns=[c for c in ["hour", "dow"] if c in df.columns])
    return df.reset_index()


def analyze_latest_point(
    df_two_cols: pd.DataFrame,
    metric_name: str,
    granularity: Literal["hourly", "daily"] = "hourly",
    params: Optional[AnomalyParams] = None,
    sensitivity: Optional[Union[int, str]] = None,  # <--- НОВЫЙ АРГУМЕНТ
    **overrides,
) -> pd.DataFrame:
    """
    Принимает DF ТОЛЬКО со столбцами: time_at, metric_value.
    Возвращает одну строку по последней дате со всеми признаками и флагами.

    Варианты использования:
      1) Явный конфиг:
         p = AnomalyParams(...)
         analyze_latest_point(..., params=p)

      2) Пресет чувствительности:
         analyze_latest_point(..., sensitivity=1)   # очень чувствительно
         analyze_latest_point(..., sensitivity=6)   # мега слабо

         также можно строкой:
         sensitivity="very_sensitive", "medium", "ultra_low", "очень_чувствительно", ...
    """
    if set(df_two_cols.columns) != {"time_at", "metric_value"}:
        cols = list(df_two_cols.columns)
        if len(cols) == 2:
            df_two_cols = df_two_cols.rename(
                columns={cols[0]: "time_at", cols[1]: "metric_value"}
            )
        else:
            raise ValueError(
                "Input dataframe must contain exactly two columns: time_at, metric_value"
            )

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
        sensitivity=sensitivity,
        **overrides,
    )

    if res is not None and not res.empty:
        last_row = res.loc[res["time_at"] == latest_time].copy()
    else:
        last_row = pd.DataFrame(
            [
                {
                    "time_at": latest_time,
                    "metric_value": latest_value,
                    "ci_mean": np.nan,
                    "ci_std": np.nan,
                    "ci_upper": np.nan,
                    "ci_lower": np.nan,
                    "ci_alert": 0,
                    "z_score": np.nan,
                    "z_alert": 0,
                    "iforest_alert": 0,
                    "lof_alert": 0,
                    "stl_resid": np.nan,
                    "stl_alert": 0,
                    "sesd_alert": 0,
                    "cusum_alert": 0,
                    "anomaly_final": 0,
                }
            ]
        )

    last_row["metric_name"] = metric_name
    last_row["granularity"] = granularity

    for col in columns_true:
        if col not in last_row.columns:
            last_row[col] = np.nan
    last_row = last_row[columns_true]

    if len(last_row) > 1:
        last_row = last_row.iloc[[-1]]

    return last_row.reset_index(drop=True)
