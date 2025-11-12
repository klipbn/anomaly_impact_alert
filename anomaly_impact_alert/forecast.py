from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Literal, Dict
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from prophet import Prophet
import holidays
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL


# =============== Конфиг ===============

@dataclass
class BFConfig:
    time_col: str = "time_at"
    value_col: str = "metric_value"

    # 'daily' или 'hourly' (если None — пытаемся угадать по шагу)
    granularity: Optional[Literal["daily", "hourly"]] = None

    # Даты-цели (ровно для них делаем прогноз); если None — последняя дата в DF
    targets: Optional[List[pd.Timestamp]] = None

    # Минимум точек для моделей
    min_points_prophet: int = 28
    min_points_ets: int = 28

    # Бэктест для весов ансамбля
    backtest_window_daily: int = 28
    backtest_window_hourly: int = 24 * 7

    # Робастная очистка истории
    winsorize: bool = True
    winsor_k: float = 4.0               # медиана ± k*MAD по сезонному индексу

    # Prophet
    holidays_country: str = "RU"
    seasonality_mode: Literal["additive", "multiplicative"] = "multiplicative"
    interval_width: float = 0.8
    uncertainty_samples: int = 0 
    changepoint_prior_scale: float = 0.1
    seasonality_prior_scale: float = 10.0
    yearly: bool = True                 # для daily
    weekly: bool = True
    daily_seasonality_for_hourly: bool = True
    add_monthly_for_daily: bool = True 

    # ETS (Holt-Winters)
    ets_trend: Optional[str] = "add"
    ets_seasonal_daily: Optional[str] = "add"
    ets_seasonal_hourly: Optional[str] = "add"

    # Log-трансформация (auto/True/False)
    log_y: Literal["auto", True, False] = "auto"
    log_cv_threshold: float = 0.3

    seed: int = 42


# Утилиты

def _infer_granularity(ds: pd.Series) -> str:
    ds = pd.to_datetime(ds.sort_values().unique())
    if len(ds) < 2:
        return "daily"
    deltas = np.diff(ds.values.astype("datetime64[ns]")).astype("timedelta64[h]").astype(int)
    step_h = int(pd.Series(deltas).mode().iloc[0])
    return "hourly" if step_h in (1, 2, 3, 4, 6, 12, 24) and step_h <= 24 else "daily"

def _seasonal_index(df: pd.DataFrame, granularity: str) -> np.ndarray:
    if granularity == "hourly":
        return df["ds"].dt.hour.values
    # daily
    return df["ds"].dt.weekday.values

def _winsorize_by_seasonal_median(df: pd.DataFrame, granularity: str, k: float) -> pd.DataFrame:
    """
    Робастно ограничиваем выбросы: для каждой сезонной корзины (час или день недели)
    y := median ± k*MAD.
    """
    work = df.copy()
    idx = _seasonal_index(work, granularity)
    work["__bin"] = idx
    y = work["y"].values

    for b in np.unique(idx):
        mask = work["__bin"] == b
        vals = y[mask]
        if vals.size == 0:
            continue
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med)) or 1e-9
        lo, hi = med - k * mad, med + k * mad
        y[mask] = np.clip(vals, lo, hi)

    work["y"] = y
    return work.drop(columns="__bin")

def _need_log(y: pd.Series, cfg: BFConfig) -> Tuple[bool, float]:
    if cfg.log_y is True:
        eps = max(1e-6, 1e-3 * np.nanmedian(y))
        return True, eps
    if cfg.log_y is False:
        return False, 0.0
    # auto
    if (y <= 0).any():
        return False, 0.0
    cv = float(np.nanstd(y) / (np.nanmean(y) + 1e-9))
    use_log = cv > cfg.log_cv_threshold
    eps = max(1e-6, 1e-3 * np.nanmedian(y))
    return use_log, eps

def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-9
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

def _seasonal_naive(hist: pd.DataFrame, tgt: pd.Timestamp, granularity: str) -> float:
    if hist.empty:
        return np.nan
    if granularity == "hourly":
        mask = hist["ds"].dt.hour == tgt.hour
        tail = hist.loc[mask].tail(24*7)
        return float(tail["y"].mean()) if not tail.empty else float(hist["y"].iloc[-1])
    else:
        mask = hist["ds"].dt.weekday == tgt.weekday()
        tail = hist.loc[mask].tail(4)
        return float(tail["y"].mean()) if not tail.empty else float(hist["y"].iloc[-1])

def _make_holidays(df_hist: pd.DataFrame, country: str) -> Optional[pd.DataFrame]:
    try:
        years = list(range(int(df_hist["ds"].dt.year.min()), int(df_hist["ds"].dt.year.max()) + 2))
        if country.upper() == "RU":
            hol = holidays.Russia(years=years)
        else:
            hol = holidays.country_holidays(country.upper(), years=years)
        return pd.DataFrame({"ds": list(hol.keys()), "holiday": "holiday"})
    except Exception:
        return None


# Модели: Prophet / STL+ETS

def _prophet_one(hist: pd.DataFrame, tgt: pd.Timestamp, cfg: BFConfig) -> float:
    """Prophet 1-step, учится на ds<tgt."""
    df = hist.copy()
    use_log, eps = _need_log(df["y"], cfg)
    if use_log:
        df["y"] = np.log(df["y"] + eps)

    hdf = _make_holidays(df, cfg.holidays_country)

    base = dict(seasonality_mode=cfg.seasonality_mode,
                changepoint_prior_scale=cfg.changepoint_prior_scale,
                seasonality_prior_scale=cfg.seasonality_prior_scale,
                interval_width=cfg.interval_width,
                uncertainty_samples=cfg.uncertainty_samples)

    if cfg.granularity == "hourly":
        m = Prophet(holidays=hdf, daily_seasonality=cfg.daily_seasonality_for_hourly,
                    weekly_seasonality=cfg.weekly, yearly_seasonality=False, **base)
        m.add_seasonality(name="weekly_hourly", period=24*7, fourier_order=15)
        m.add_seasonality(name="in_day", period=24, fourier_order=6)
    else:
        m = Prophet(holidays=hdf, daily_seasonality=False,
                    weekly_seasonality=cfg.weekly, yearly_seasonality=cfg.yearly, **base)
        if cfg.add_monthly_for_daily:
            m.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    m.fit(df)
    yhat = float(m.predict(pd.DataFrame({"ds": [tgt]}))["yhat"].iloc[0])
    if use_log:
        yhat = np.exp(yhat) - eps
    return yhat

def _stl_ets_one(hist: pd.DataFrame, tgt: pd.Timestamp, cfg: BFConfig) -> float:
    """STL + ETS 1-step, на ds<tgt."""
    y = hist["y"].astype(float).values
    seasonal_periods = 24 if cfg.granularity == "hourly" else 7

    # если не хватает для STL прямая ETS
    if len(y) < max(cfg.min_points_ets, seasonal_periods*2):
        seasonal = cfg.ets_seasonal_hourly if cfg.granularity == "hourly" else cfg.ets_seasonal_daily
        try:
            model = ExponentialSmoothing(
                y,
                trend=cfg.ets_trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated"
            ).fit(optimized=True, use_brute=True)
            return float(model.forecast(1)[0])
        except Exception:
            return _seasonal_naive(hist, tgt, cfg.granularity)

    # STL для удаления сезонки, ETS по тренду/остатку
    stl = STL(hist["y"].astype(float), period=seasonal_periods, robust=True)
    res = stl.fit()
    resid = res.trend + res.resid  # без сезонной компоненты (она экстраполируется как средняя)
    try:
        model = ExponentialSmoothing(
            resid.values,
            trend=cfg.ets_trend,
            seasonal=None,
            initialization_method="estimated"
        ).fit(optimized=True, use_brute=True)
        fc_resid = float(model.forecast(1)[0])

        if cfg.granularity == "hourly":
            bin_val = hist["ds"].iloc[-1]  # приблизительно ниже точнее
            tgt_bin = tgt.hour
            bins = hist["ds"].dt.hour
        else:
            tgt_bin = tgt.weekday()
            bins = hist["ds"].dt.weekday
        seas_vals = res.seasonal[bins == tgt_bin]
        seas_next = float(seas_vals.mean()) if len(seas_vals) else 0.0
        return fc_resid + seas_next
    except Exception:
        return _seasonal_naive(hist, tgt, cfg.granularity)


# Ансамбль с автовесами

def _backtest_weights(hist: pd.DataFrame, cfg: BFConfig) -> Dict[str, float]:
    """
    Возвращает веса моделей {'prophet': w1, 'ets': w2, 'naive': w3} на основе sMAPE
    на последних W точках (одношаговый бэктест)
    """
    W = cfg.backtest_window_hourly if cfg.granularity == "hourly" else cfg.backtest_window_daily
    ds_sorted = hist["ds"].sort_values().values
    if len(ds_sorted) < max(cfg.min_points_prophet, cfg.min_points_ets) + 5:
        # мало данных значт фиксированные разумные веса
        return {"prophet": 0.5, "ets": 0.3, "naive": 0.2}

    cut_start = max(0, len(ds_sorted) - W - 1)
    test_points = [pd.Timestamp(d) for d in ds_sorted[cut_start+1:]]

    y_true_p, y_pred_p = [], []
    y_true_e, y_pred_e = [], []
    y_true_n, y_pred_n = [], []

    for tgt in test_points:
        train = hist[hist["ds"] < tgt]
        true = float(hist.loc[hist["ds"] == tgt, "y"].iloc[0])

        # Prophet
        try:
            yp = _prophet_one(train, tgt, cfg)
        except Exception:
            yp = np.nan
        # ETS
        try:
            ye = _stl_ets_one(train, tgt, cfg)
        except Exception:
            ye = np.nan
        # Naive
        yn = _seasonal_naive(train, tgt, cfg.granularity)

        if not np.isnan(yp):
            y_true_p.append(true); y_pred_p.append(yp)
        if not np.isnan(ye):
            y_true_e.append(true); y_pred_e.append(ye)
        y_true_n.append(true); y_pred_n.append(yn)

    scores = {}
    if y_true_p:
        scores["prophet"] = _smape(np.array(y_true_p), np.array(y_pred_p))
    if y_true_e:
        scores["ets"] = _smape(np.array(y_true_e), np.array(y_pred_e))
    scores["naive"] = _smape(np.array(y_true_n), np.array(y_pred_n))

    # ошибки в веса: w ~ 1/error
    inv = {k: (1.0 / max(v, 1e-6)) for k, v in scores.items()}
    s = sum(inv.values())
    weights = {k: v / s for k, v in inv.items()}

    # наличие всех ключей
    for k in ("prophet", "ets", "naive"):
        weights.setdefault(k, 0.0)
    # нормировка
    s2 = sum(weights.values()) or 1.0
    weights = {k: v / s2 for k, v in weights.items()}
    return weights


# Публичная функция

def forecast_values_for_targets_better(
    df: pd.DataFrame,
    *,
    cfg: Optional[BFConfig] = None
) -> pd.DataFrame:
    """
    Возвращает DF с колонками: time_at, forecast, forecast_prophet, forecast_ets, forecast_naive, w_prophet, w_ets, w_naive
    Прогноз делается ТОЛЬКО для cfg.targets (или для последней даты, если targets=None)
    Для каждой цели модель учится на истории < target. Единственные обязательные колоноки во входе: time_at, metric_value
    """
    cfg = cfg or BFConfig()
    s = df[[cfg.time_col, cfg.value_col]].rename(columns={cfg.time_col: "ds", cfg.value_col: "y"}).copy()
    s["ds"] = pd.to_datetime(s["ds"])
    s["y"] = s["y"].astype(float)
    s = s.sort_values("ds").dropna()

    # определяем гранулярность
    if cfg.granularity is None:
        cfg.granularity = _infer_granularity(s["ds"])

    # winsorize по сезонным корзинам
    if cfg.winsorize:
        s_clean = _winsorize_by_seasonal_median(s, cfg.granularity, cfg.winsor_k)
    else:
        s_clean = s.copy()

    # список целей
    if cfg.targets is None or len(cfg.targets) == 0:
        targets = [pd.Timestamp(s["ds"].max())]
    else:
        targets = sorted(pd.to_datetime(pd.Series(cfg.targets)).unique().tolist())

    out_rows = []

    for tgt in targets:
        # История только до цели
        hist = s_clean[s_clean["ds"] < tgt].copy()

        # Веса ансамбля на основе короткого бэктеста на этой истории
        try:
            w = _backtest_weights(hist, cfg)
        except Exception:
            w = {"prophet": 0.5, "ets": 0.3, "naive": 0.2}

        # Прогнозы отдельных моделей
        try:
            yp = _prophet_one(hist, tgt, cfg)
        except Exception:
            yp = np.nan

        try:
            ye = _stl_ets_one(hist, tgt, cfg)
        except Exception:
            ye = np.nan

        yn = _seasonal_naive(hist, tgt, cfg.granularity)

        # Если что-то из моделей NaN перенормируем веса на существующие
        parts = {"prophet": yp, "ets": ye, "naive": yn}
        valid = {k: v for k, v in parts.items() if not np.isnan(v)}
        if not valid:  # совсем беда
            yhat = np.nan
            w_use = {"prophet": 0.0, "ets": 0.0, "naive": 1.0}
        else:
            w_use = {k: w.get(k, 0.0) for k in valid.keys()}
            ssum = sum(w_use.values()) or 1.0
            w_use = {k: v / ssum for k, v in w_use.items()}
            yhat = sum(valid[k] * w_use[k] for k in valid.keys())

        out_rows.append({
            "time_at": pd.Timestamp(tgt),
            "forecast": float(yhat) if yhat is not None else np.nan,
            "forecast_prophet": float(yp) if not np.isnan(yp) else np.nan,
            "forecast_ets": float(ye) if not np.isnan(ye) else np.nan,
            "forecast_naive": float(yn),
            "w_prophet": w_use.get("prophet", 0.0),
            "w_ets": w_use.get("ets", 0.0),
            "w_naive": w_use.get("naive", 0.0),
        })

    return pd.DataFrame(out_rows).sort_values("time_at").reset_index(drop=True)
