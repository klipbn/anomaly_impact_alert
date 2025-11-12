from __future__ import annotations

import math
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Callable, Tuple, List

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

@dataclass
class AlertConfig:
    # имена колонок во входном df
    time_col: str = "time_at"
    value_col: str = "metric_value"
    anomaly_col: str = "anomaly_final"
    metric_name_col: str = "metric_name"
    granularity_col: str = "granularity"

    impact_blocks: Optional[List[Tuple[str, str]]] = None

    # прогноз
    forecast_col: Optional[str] = "forecast"
    forecast_alt_cols: Tuple[str, str, str] = ("forecast_prophet", "forecast_ets", "forecast_naive")
    forecast_weight_cols: Tuple[str, str, str] = ("w_prophet", "w_ets", "w_naive")

    # график
    plot_window_points: int = 36
    figure_size: Tuple[int, int] = (15, 6)

    # подписи срезов
    slice1_name: Optional[str] = "Срез1"
    slice1_value: Optional[str] = "Total"
    slice2_name: Optional[str] = "Срез2"
    slice2_value: Optional[str] = "Total"

    # ссылки
    links: Optional[List[Tuple[str, str]]] = None

    # только если аномалия
    anomaly_only: bool = True


def _fmt_compact(n: float) -> str:
    if n is None or (isinstance(n, float) and (math.isnan(n) or math.isinf(n))):
        return "н/д"
    sgn = "-" if n < 0 else ""
    n = abs(float(n))
    if n >= 1_000_000_000:
        return f"{sgn}{n/1_000_000_000:.1f}".replace(".", ",") + "B"
    if n >= 1_000_000:
        return f"{sgn}{n/1_000_000:.1f}".replace(".", ",") + "M"
    if n >= 1_000:
        return f"{sgn}{n/1_000:.1f}".replace(".", ",") + "K"
    return f"{sgn}{n:,.0f}".replace(",", " ")


def _fmt_pct(x: Optional[float]) -> str:
    return "н/д" if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else f"{x:.1f}%"


def _calc_vs(prev_val: Optional[float], now_val: float) -> Optional[float]:
    if prev_val is None or prev_val == 0:
        return None
    return (now_val / prev_val - 1.0) * 100.0

def _y_scale_and_unit(mx: float) -> Tuple[float, str]:
    if mx >= 1e9:
        return 1e9, "млрд"
    if mx >= 1e6:
        return 1e6, "млн"
    if mx >= 1e3:
        return 1e3, "тыс"
    return 1.0, ""


def _pick_row_for_now(df: pd.DataFrame, now: datetime, tcol: str) -> pd.DataFrame:
    ts = pd.Timestamp(now)
    df = df.copy()
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    if getattr(df[tcol].dt, "tz", None) is not None:
        df[tcol] = df[tcol].dt.tz_convert(None)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None) if hasattr(ts, "tz_convert") else ts.replace(tzinfo=None)

    hit = df.loc[df[tcol] == ts]
    if not hit.empty:
        return hit.sort_values(tcol).tail(1)

    hit = df.loc[df[tcol].dt.normalize() == ts.normalize()]
    if not hit.empty:
        return hit.sort_values(tcol).tail(1)
    return pd.DataFrame()


def make_plot_image(df: pd.DataFrame, now: pd.Timestamp, metric_name: str, cfg: AlertConfig) -> str:
    t, v = cfg.time_col, cfg.value_col
    cols = [c for c in ["ci_upper", "ci_lower", "ci_mean", cfg.anomaly_col] if c in df.columns]

    df_fig = df[[t, v, *cols]].sort_values(t).copy()
    if len(df_fig) > cfg.plot_window_points:
        df_fig = df_fig.tail(cfg.plot_window_points)

    fig, ax = plt.subplots(figsize=cfg.figure_size)

    ax.plot(df_fig[t], df_fig[v], linewidth=1.6, label="Метрика")
    if "ci_upper" in df_fig.columns and "ci_lower" in df_fig.columns:
        ax.plot(df_fig[t], df_fig["ci_upper"], linestyle="--", linewidth=1.2, label="CI верх")
        ax.plot(df_fig[t], df_fig["ci_lower"], linestyle="--", linewidth=1.2, label="CI низ")
    if "ci_mean" in df_fig.columns:
        ax.plot(df_fig[t], df_fig["ci_mean"], linestyle=":", linewidth=1.2, label="CI mean")

    if cfg.anomaly_col in df_fig.columns:
        ano = df_fig[df_fig[cfg.anomaly_col] == 1]
        if not ano.empty:
            drops = ano[ano[v] < ano.get("ci_mean", ano[v])]
            rises = ano[ano[v] >= ano.get("ci_mean", ano[v])]
            if not drops.empty:
                ax.scatter(drops[t], drops[v], label="Аномальное падение", zorder=5, s=80)
            if not rises.empty:
                ax.scatter(rises[t], rises[v], label="Аномальный рост", zorder=5, s=80)

    ax.set_title(f"Аномалии · {metric_name}", fontsize=14)
    ax.set_xlabel("Дата", fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=90, ha="right")

    y_max = float(df_fig[v].max()) if not df_fig.empty else 1.0
    scale, unit = _y_scale_and_unit(y_max)
    ax.set_ylabel(f"Значение {unit}".strip(), fontsize=12)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/scale:,.2f}".replace(",", " ")))
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

def _find_prev_values(df: pd.DataFrame, now: pd.Timestamp, cfg: AlertConfig) -> Tuple[Optional[float], Optional[float]]:
    t, v, gcol = cfg.time_col, cfg.value_col, cfg.granularity_col
    gran = df.loc[df[t] == now, gcol].iloc[0] if gcol in df.columns and (df[t] == now).any() else "daily"
    if gran == "hourly":
        prev_1 = now - timedelta(hours=24)
        prev_7 = now - timedelta(hours=24*7)
    else:
        prev_1 = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        prev_7 = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
    v1 = df.loc[df[t] == prev_1, v]
    v7 = df.loc[df[t] == prev_7, v]
    return (float(v1.iloc[0]) if not v1.empty else None,
            float(v7.iloc[0]) if not v7.empty else None)


def _resolve_forecast_from_row(row: pd.Series, cfg: AlertConfig) -> Optional[float]:
    fcol = cfg.forecast_col
    if fcol and fcol in row.index and pd.notna(row[fcol]):
        try:
            return float(str(row[fcol]).replace(" ", "").replace(",", ""))
        except Exception:
            try:
                return float(row[fcol])
            except Exception:
                pass

    pcol, ecol, ncol = cfg.forecast_alt_cols
    wp, we, wn = cfg.forecast_weight_cols
    if all(c in row.index for c in (pcol, ecol, ncol, wp, we, wn)):
        parts, weights = [], []
        for c, w in ((pcol, wp), (ecol, we), (ncol, wn)):
            try:
                val = float(str(row[c]).replace(" ", "").replace(",", ""))
                wt = float(row[w])
                if pd.notna(val) and pd.notna(wt):
                    parts.append(val); weights.append(wt)
            except Exception:
                continue
        if parts and sum(weights) != 0:
            wsum = sum(weights)
            return float(sum(p * (w / wsum) for p, w in zip(parts, weights)))
    return None


def build_caption(alert_row: pd.Series, cfg: AlertConfig) -> str:
    now = pd.to_datetime(alert_row[cfg.time_col])
    metric_name = str(alert_row.get(cfg.metric_name_col, "metric"))
    val_now = float(str(alert_row[cfg.value_col]).replace(" ", "").replace(",", "")) \
        if isinstance(alert_row[cfg.value_col], str) else float(alert_row[cfg.value_col])

    ci_mean = alert_row.get("ci_mean", np.nan)
    try:
        ci_mean = float(str(ci_mean).replace(" ", "").replace(",", "")) if isinstance(ci_mean, str) else float(ci_mean)
    except Exception:
        ci_mean = np.nan
    sign = "🔴 Падение" if (not np.isnan(ci_mean) and val_now < ci_mean) else "🟢 Рост"

    vs_last_day = alert_row.get("vs_last_day", None)
    vs_last_week = alert_row.get("vs_last_week", None)

    forecast_val = _resolve_forecast_from_row(alert_row, cfg)
    diff_val = None if (forecast_val is None) else (val_now - forecast_val)

    gran = alert_row.get(cfg.granularity_col, "daily")
    dt_fmt = "%Y-%m-%d %H:%M" if gran == "hourly" else "%Y-%m-%d"
    header = f"{sign} | {now:{dt_fmt}} | <b>{metric_name}</b>\n\n"

    slice_line = ""
    if cfg.slice1_name and cfg.slice1_value:
        slice_line += f"{cfg.slice1_name} = {cfg.slice1_value}"
    if cfg.slice2_name and cfg.slice2_value:
        slice_line += (", " if slice_line else "") + f"{cfg.slice2_name} = {cfg.slice2_value}"
    if slice_line:
        slice_line = "Срез: " + slice_line + "\n\n"

    body_main = f"Значение: <b>{_fmt_compact(val_now)}</b> (DoD: {_fmt_pct(vs_last_day)}, WoW: {_fmt_pct(vs_last_week)})\n"
    if forecast_val is not None:
        body_main += f"Прогноз: {_fmt_compact(forecast_val)} (diff: {_fmt_compact(diff_val)})\n\n"
    else:
        body_main += "Прогноз: н/д\n\n"

    impact_text = ""
    if cfg.impact_blocks:
        for heading, col in cfg.impact_blocks:
            if col in alert_row.index:
                txt = str(alert_row.get(col) or "").strip()
                if txt:
                    if impact_text:
                        impact_text += "\n"
                    impact_text += f"{heading}\n{txt}\n"

    links_block = ""
    if cfg.links:
        for title, url in cfg.links:
            links_block += f'\n🔎 <a href="{url}">{title}</a>'

    caption = header + slice_line + body_main + impact_text + links_block
    caption = re.sub(r"(<)\s*(\d+)", r"&lt; \2", caption) #TODO: убрать костыль в разметк
    return caption

def send_telegram_message(
    token: str,
    chat_id: str,
    image_path: Optional[str],
    caption_html: str,
) -> dict:
    """
    Отправляет сообщение в Telegram Bot API
    Если image_path=None — sendMessage, иначе sendPhoto
    """
    base = f"https://api.telegram.org/bot{token}"
    params = {"chat_id": chat_id, "parse_mode": "HTML", "disable_web_page_preview": True}

    if image_path:
        url = f"{base}/sendPhoto"
        with open(image_path, "rb") as f:
            files = {"photo": f}
            data = {**params, "caption": caption_html}
            resp = requests.post(url, data=data, files=files, timeout=30)
    else:
        url = f"{base}/sendMessage"
        data = {**params, "text": caption_html}
        resp = requests.post(url, data=data, timeout=30)

    try:
        return resp.json()
    except Exception:
        return {"ok": False, "status_code": getattr(resp, "status_code", None), "text": getattr(resp, "text", "")}

def send_alert_for_date(
    df_final: pd.DataFrame,
    now: datetime,
    *,
    metric_name: Optional[str] = None,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
    cfg: Optional[AlertConfig] = None,
    plot_func: Optional[Callable[[pd.DataFrame, pd.Timestamp, str, AlertConfig], str]] = None,
    also_return: bool = False,
) -> Optional[dict]:
    """
    Берёт df_final (уже с прогнозами/импактами), находит строку на дату `now`,
    проверяет anomaly_final == 1 (если cfg.anomaly_only=True), строит график, собирает текст и отправляет

    Возвращает payload Telegram (или dict с caption+image_path при also_return=True),
    либо None, если на дату нет строки/аномалии
    """
    cfg = cfg or AlertConfig()
    t, v, a, mcol, gcol = cfg.time_col, cfg.value_col, cfg.anomaly_col, cfg.metric_name_col, cfg.granularity_col

    df = df_final.copy()
    df[t] = pd.to_datetime(df[t], errors="coerce")

    row = _pick_row_for_now(df, now, t)
    if row.empty:
        if also_return:
            return {"skipped": True, "reason": "no_row_for_now",
                    "now": str(now), "min_ts": str(df[t].min()), "max_ts": str(df[t].max())}
        return None

    if cfg.anomaly_only and (pd.isna(row.iloc[0].get(a, np.nan)) or int(row.iloc[0][a]) != 1):
        if also_return:
            return {"skipped": True, "reason": "no_anomaly_flag", "now": str(now)}
        return None

    alert_row = row.iloc[0]

    if ("vs_last_day" not in df.columns) or ("vs_last_week" not in df.columns) or \
       (pd.isna(alert_row.get("vs_last_day", np.nan)) and pd.isna(alert_row.get("vs_last_week", np.nan))):
        prev1, prev7 = _find_prev_values(df, pd.Timestamp(now), cfg)
        df.loc[df[t] == pd.Timestamp(now), "vs_last_day"] = _calc_vs(prev1, float(alert_row[v]))
        df.loc[df[t] == pd.Timestamp(now), "vs_last_week"] = _calc_vs(prev7, float(alert_row[v]))
        alert_row = df.loc[df[t] == pd.Timestamp(now)].iloc[0]

    metric_name_effective = metric_name or str(alert_row.get(mcol, "metric"))

    plt_func = plot_func or make_plot_image
    maybe_cols = [t, v, "ci_upper", "ci_lower", "ci_mean", a, gcol]
    cols_exist = [c for c in maybe_cols if c in df.columns]
    plot_df = df[cols_exist].copy() if cols_exist else df.copy()
    img_path = plt_func(plot_df, pd.Timestamp(now), metric_name_effective, cfg)

    caption = build_caption(alert_row, cfg)

    result = None
    if token and chat_id:
        result = send_telegram_message(token=token, chat_id=chat_id, image_path=img_path, caption_html=caption)

    if also_return:
        return {"caption": caption, "image_path": img_path, "send_result": result}
    return result
