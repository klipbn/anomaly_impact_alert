from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Callable, Tuple, List, Literal

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import re


# =========================
# -------- Конфиг ---------
# =========================

@dataclass
class AlertConfig:
    # имена колонок во входном df
    time_col: str = "time_at"
    value_col: str = "metric_value"
    anomaly_col: str = "anomaly_final"
    metric_name_col: str = "metric_name"
    granularity_col: str = "granularity"

    # [DEPRECATED] — оставлено для обратной совместимости
    impact_bu_col: Optional[str] = "impact_text_bu"
    impact_platform_col: Optional[str] = "impact_text_platform"
    impact_bu_heading: str = "Изменение за счет продуктов:"
    impact_platform_heading: str = "Изменение за счет площадок:"

    # NEW: импакт-блоки
    impact_blocks: Optional[List[Tuple[str, str]]] = None

    # колонка финального прогноза
    forecast_col: Optional[str] = "forecast"

    # альтернативные столбцы (если forecast пуст): forecast = w_p*p + w_e*e + w_n*n
    forecast_alt_cols: Tuple[str, str, str] = (
        "forecast_prophet",
        "forecast_ets",
        "forecast_naive",
    )
    forecast_weight_cols: Tuple[str, str, str] = (
        "w_prophet",
        "w_ets",
        "w_naive",
    )

    # формат и окно графика
    plot_window_points: int = 36
    figure_size: Tuple[int, int] = (15, 6)

    # форматирование чисел
    value_decimals: int = 2
    pct_decimals: int = 0
    forecast_decimals: Optional[int] = None
    diff_decimals: Optional[int] = None

    # ось X
    x_major_freq: Optional[int] = 1
    x_major_unit: str = "day"      # day / hour / week / month
    x_minor_freq: Optional[int] = None
    x_minor_unit: Optional[str] = None
    x_date_format_daily: str = "%Y-%m-%d"
    x_date_format_hourly: str = "%Y-%m-%d %H:%M"
    x_tick_rotation: int = 90

    # ось Y
    y_min: Optional[float] = 0
    y_max: Optional[float] = None
    y_tick_step: Optional[float] = None
    y_label_decimals: int = 2
    y_use_compact_unit: bool = True

    # подписи осей
    x_label: str = "Дата"
    y_label: str = "Значение"

    # заголовок
    plot_title_prefix: str = "Аномалии"

    # сетка
    grid_enabled: bool = True
    grid_linestyle: str = "--"
    grid_linewidth: float = 0.5
    grid_alpha: float = 0.7

    # легенда
    legend_loc: str = "upper left"

    # подписи и срезы
    slice1_name: Optional[str] = "Продукт"
    slice1_value: Optional[str] = "Total"
    slice2_name: Optional[str] = "Проект"
    slice2_value: Optional[str] = "Total"

    # ссылки (HTML)
    links: Optional[List[Tuple[str, str]]] = (
        ("Дашборд по аномалиям", "https://superset.vk.team/superset/dashboard/6134"),
        ("Дашборд по факторному анализу", "https://superset.vk.team/superset/dashboard/5109/"),
    )

    # API VK Teams
    vkteams_api_url: str = "https://api.internal.myteam.mail.ru/bot/v1"

    # отправлять только если аномалия
    anomaly_only: bool = True

    # -------- NEW: шаблоны сообщения --------
    message_style: Literal["standard", "extended_inline", "analytic_blocks"] = "standard"

    # сравнение со средним за N дней
    show_avg_compare: bool = False
    avg_compare_days: int = 7
    avg_compare_label: str = "vs avg"

    # разделитель между импакт-блоками
    impact_join_separator: str = "\n\n"


# =========================
# ----- Утилиты текста ----
# =========================

def _is_bad_number(x) -> bool:
    try:
        return x is None or pd.isna(x) or np.isinf(x)
    except Exception:
        return True


def _fmt_num(x: Optional[float], decimals: int = 2, thousands_sep: str = " ") -> str:
    if _is_bad_number(x):
        return "н/д"
    try:
        s = f"{float(x):,.{decimals}f}"
        return s.replace(",", " ").replace(".", ",")
    except Exception:
        return "н/д"


def _fmt_compact(n: float, decimals: int = 2) -> str:
    if _is_bad_number(n):
        return "н/д"

    sgn = "-" if float(n) < 0 else ""
    n = abs(float(n))

    if n >= 1_000_000_000:
        val = f"{n / 1_000_000_000:.{decimals}f}".replace(".", ",")
        return f"{sgn}{val}B"
    if n >= 1_000_000:
        val = f"{n / 1_000_000:.{decimals}f}".replace(".", ",")
        return f"{sgn}{val}M"
    if n >= 1_000:
        val = f"{n / 1_000:.{decimals}f}".replace(".", ",")
        return f"{sgn}{val}K"

    return _fmt_num(float(f"{sgn}{n}"), decimals=decimals)


def _fmt_pct(x: Optional[float], decimals: int = 0) -> str:
    if _is_bad_number(x):
        return "н/д"
    return f"{float(x):.{decimals}f}%".replace(".", ",")


def _calc_vs(prev_val: Optional[float], now_val: float) -> Optional[float]:
    if prev_val is None or prev_val == 0:
        return None
    return (now_val / prev_val - 1.0) * 100.0


# =========================
# -------- График ---------
# =========================

def _y_scale_and_unit(max_value: float) -> Tuple[float, str]:
    if max_value >= 1e9:
        return 1e9, "млрд"
    elif max_value >= 1e6:
        return 1e6, "млн"
    elif max_value >= 1e3:
        return 1e3, "тыс"
    return 1.0, ""


def _make_date_locator(unit: Optional[str], freq: Optional[int]):
    if unit is None or freq is None:
        return None

    unit = unit.lower()
    if unit == "hour":
        return mdates.HourLocator(interval=freq)
    if unit == "day":
        return mdates.DayLocator(interval=freq)
    if unit == "week":
        return mdates.WeekdayLocator(interval=freq)
    if unit == "month":
        return mdates.MonthLocator(interval=freq)

    return None


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
    df_fig[t] = pd.to_datetime(df_fig[t], errors="coerce")
    df_fig = df_fig.dropna(subset=[t])

    if len(df_fig) > cfg.plot_window_points:
        df_fig = df_fig.tail(cfg.plot_window_points)

    fig, ax = plt.subplots(figsize=cfg.figure_size)

    ax.plot(df_fig[t], df_fig[v], label="Метрика", linewidth=1.6)
    if "ci_upper" in df_fig.columns and "ci_lower" in df_fig.columns:
        ax.plot(df_fig[t], df_fig["ci_upper"], linestyle="--", linewidth=1.2, label="CI верх")
        ax.plot(df_fig[t], df_fig["ci_lower"], linestyle="--", linewidth=1.2, label="CI низ")
    if "ci_mean" in df_fig.columns:
        ax.plot(df_fig[t], df_fig["ci_mean"], linestyle=":", linewidth=1.2, label="CI mean")

    if cfg.anomaly_col in df_fig.columns:
        ano = df_fig[df_fig[cfg.anomaly_col] == 1]
        if not ano.empty:
            base_line = ano["ci_mean"] if "ci_mean" in ano.columns else ano[v]
            drops = ano[ano[v] < base_line]
            rises = ano[ano[v] >= base_line]

            if not drops.empty:
                ax.scatter(drops[t], drops[v], color="red", label="Аномальное падение", zorder=5, s=80)
            if not rises.empty:
                ax.scatter(rises[t], rises[v], color="green", label="Аномальный рост", zorder=5, s=80)

    ax.set_title(f"{cfg.plot_title_prefix} {metric_name}", fontsize=14)
    ax.set_xlabel(cfg.x_label, fontsize=12)

    y_max_data = float(df_fig[v].max()) if not df_fig.empty else 1.0
    y_min_data = float(df_fig[v].min()) if not df_fig.empty else 0.0

    if cfg.y_use_compact_unit:
        scale, unit = _y_scale_and_unit(max(abs(y_max_data), abs(y_min_data)))
    else:
        scale, unit = 1.0, ""

    y_label_full = f"{cfg.y_label} {unit}".strip()
    ax.set_ylabel(y_label_full, fontsize=12)

    if cfg.y_min is not None or cfg.y_max is not None:
        ax.set_ylim(
            bottom=cfg.y_min if cfg.y_min is not None else None,
            top=cfg.y_max if cfg.y_max is not None else None,
        )

    if cfg.y_tick_step is not None:
        ax.yaxis.set_major_locator(mticker.MultipleLocator(base=cfg.y_tick_step))

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, _: f"{x / scale:,.{cfg.y_label_decimals}f}".replace(",", " ")
        )
    )

    gran = "daily"
    if cfg.granularity_col in df.columns and not df[cfg.granularity_col].dropna().empty:
        gran = str(df[cfg.granularity_col].dropna().iloc[-1]).lower()

    major_locator = _make_date_locator(cfg.x_major_unit, cfg.x_major_freq)
    if major_locator is not None:
        ax.xaxis.set_major_locator(major_locator)

    date_fmt = cfg.x_date_format_hourly if gran == "hourly" else cfg.x_date_format_daily
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))

    minor_locator = _make_date_locator(cfg.x_minor_unit, cfg.x_minor_freq)
    if minor_locator is not None:
        ax.xaxis.set_minor_locator(minor_locator)

    fig.autofmt_xdate(rotation=cfg.x_tick_rotation, ha="right")

    if cfg.grid_enabled:
        ax.grid(
            True,
            linestyle=cfg.grid_linestyle,
            linewidth=cfg.grid_linewidth,
            alpha=cfg.grid_alpha,
        )

    ax.legend(loc=cfg.legend_loc, framealpha=0.9)
    plt.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


# =========================
# ----- Генерация текста ---
# =========================

def _find_prev_values(df: pd.DataFrame, now: pd.Timestamp, cfg: AlertConfig) -> Tuple[Optional[float], Optional[float]]:
    t, v, gcol = cfg.time_col, cfg.value_col, cfg.granularity_col
    gran = df.loc[df[t] == now, gcol].iloc[0] if gcol in df.columns and (df[t] == now).any() else "daily"

    if gran == "hourly":
        prev_1 = now - timedelta(hours=24)
        prev_7 = now - timedelta(hours=24 * 7)
    else:
        prev_1 = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        prev_7 = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)

    v1 = df.loc[df[t] == prev_1, v]
    v7 = df.loc[df[t] == prev_7, v]
    return (
        float(v1.iloc[0]) if not v1.empty else None,
        float(v7.iloc[0]) if not v7.empty else None,
    )


def _find_mean_for_prev_n_days(
    df: pd.DataFrame,
    now: pd.Timestamp,
    cfg: AlertConfig,
) -> Optional[float]:
    t, v, gcol = cfg.time_col, cfg.value_col, cfg.granularity_col

    work = df[[t, v] + ([gcol] if gcol in df.columns else [])].copy()
    work[t] = pd.to_datetime(work[t], errors="coerce")
    work = work.dropna(subset=[t])

    gran = "daily"
    hit_now = work.loc[work[t] == now]
    if gcol in work.columns and not hit_now.empty:
        gran = str(hit_now[gcol].iloc[0]).lower()

    if gran == "hourly":
        start_ts = now - timedelta(hours=cfg.avg_compare_days * 24)
        mask = (work[t] < now) & (work[t] >= start_ts)
    else:
        start_ts = now.normalize() - timedelta(days=cfg.avg_compare_days)
        mask = (work[t] < now.normalize()) & (work[t] >= start_ts)

    vals = pd.to_numeric(work.loc[mask, v], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


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
        parts = []
        weights = []
        for c, w in ((pcol, wp), (ecol, we), (ncol, wn)):
            try:
                val = float(str(row[c]).replace(" ", "").replace(",", ""))
                wt = float(row[w])
                if pd.notna(val) and pd.notna(wt):
                    parts.append(val)
                    weights.append(wt)
            except Exception:
                continue
        if parts and sum(weights) != 0:
            wsum = sum(weights)
            return float(sum(p * (w / wsum) for p, w in zip(parts, weights)))

    return None


def _render_impact_blocks(alert_row: pd.Series, cfg: AlertConfig) -> str:
    impact_text = ""
    blocks = []

    if cfg.impact_blocks and isinstance(cfg.impact_blocks, list):
        blocks = [(str(h), str(c)) for (h, c) in cfg.impact_blocks]

    if not blocks:
        if cfg.impact_bu_col:
            blocks.append((cfg.impact_bu_heading, cfg.impact_bu_col))
        if cfg.impact_platform_col:
            blocks.append((cfg.impact_platform_heading, cfg.impact_platform_col))

    rendered = []
    for heading, col in blocks:
        if col in alert_row.index:
            txt = str(alert_row.get(col) or "").strip()
            if txt:
                rendered.append(f"{heading}\n{txt}")

    return cfg.impact_join_separator.join(rendered)


def _build_main_metrics_line(
    val_now: float,
    vs_last_day: Optional[float],
    vs_last_week: Optional[float],
    vs_avg_n_days: Optional[float],
    forecast_val: Optional[float],
    diff_val: Optional[float],
    cfg: AlertConfig,
) -> str:
    forecast_decimals = cfg.forecast_decimals if cfg.forecast_decimals is not None else cfg.value_decimals
    diff_decimals = cfg.diff_decimals if cfg.diff_decimals is not None else cfg.value_decimals

    comps = [
        f"DoD: {_fmt_pct(vs_last_day, cfg.pct_decimals)}",
        f"WoW: {_fmt_pct(vs_last_week, cfg.pct_decimals)}",
    ]

    if cfg.show_avg_compare:
        comps.append(f"{cfg.avg_compare_label}: {_fmt_pct(vs_avg_n_days, cfg.pct_decimals)}")

    line = f"Значение: <b>{_fmt_compact(val_now, cfg.value_decimals)}</b> ({', '.join(comps)})\n"

    if forecast_val is not None:
        line += (
            f"Прогноз: {_fmt_compact(forecast_val, forecast_decimals)} "
            f"(diff: {_fmt_compact(diff_val, diff_decimals)})"
        )
    else:
        line += "Прогноз: н/д"

    return line


def build_caption(alert_row: pd.Series, cfg: AlertConfig) -> str:
    now = pd.to_datetime(alert_row[cfg.time_col])
    metric_name = str(alert_row.get(cfg.metric_name_col, "metric"))

    raw_val_now = alert_row[cfg.value_col]
    val_now = float(str(raw_val_now).replace(" ", "").replace(",", "")) if isinstance(raw_val_now, str) else float(raw_val_now)

    ci_mean = alert_row.get("ci_mean", np.nan)
    try:
        ci_mean = float(str(ci_mean).replace(" ", "").replace(",", "")) if isinstance(ci_mean, str) else float(ci_mean)
    except Exception:
        ci_mean = np.nan

    if np.isnan(ci_mean):
        sign = "⚪ Аномалия"
    elif val_now < ci_mean:
        sign = "🔴 Падение"
    elif val_now > ci_mean:
        sign = "🟢 Рост"
    else:
        sign = "⚪ Аномалия"

    vs_last_day = alert_row.get("vs_last_day", None)
    vs_last_week = alert_row.get("vs_last_week", None)
    vs_avg_n_days = alert_row.get("vs_avg_n_days", None)

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

    main_line = _build_main_metrics_line(
        val_now=val_now,
        vs_last_day=vs_last_day,
        vs_last_week=vs_last_week,
        vs_avg_n_days=vs_avg_n_days,
        forecast_val=forecast_val,
        diff_val=diff_val,
        cfg=cfg,
    )

    impact_text = _render_impact_blocks(alert_row, cfg)

    links_block = ""
    if cfg.links:
        for title, url in cfg.links:
            links_block += f'\n🔎 <a href="{url}">{title}</a>'

    # ---------- style 1 ----------
    if cfg.message_style == "standard":
        body = main_line + "\n\n"
        if impact_text:
            body += impact_text + "\n"
        return header + slice_line + body + links_block

    # ---------- style 2 ----------
    if cfg.message_style == "extended_inline":
        body = main_line + "\n\n"
        if impact_text:
            body += impact_text + "\n"
        return header + slice_line + body + links_block

    # ---------- style 3 ----------
    if cfg.message_style == "analytic_blocks":
        status_block = (
            f"<b>Статус:</b> {sign}\n"
            f"<b>Дата:</b> {now:{dt_fmt}}\n"
            f"<b>Метрика:</b> {metric_name}\n\n"
        )

        fact_block = (
            f"<b>Факт:</b> {_fmt_compact(val_now, cfg.value_decimals)}\n"
            f"<b>DoD:</b> {_fmt_pct(vs_last_day, cfg.pct_decimals)}\n"
            f"<b>WoW:</b> {_fmt_pct(vs_last_week, cfg.pct_decimals)}\n"
        )

        if cfg.show_avg_compare:
            fact_block += f"<b>{cfg.avg_compare_label}:</b> {_fmt_pct(vs_avg_n_days, cfg.pct_decimals)}\n"

        if forecast_val is not None:
            fact_block += (
                f"<b>Прогноз:</b> {_fmt_compact(forecast_val, cfg.value_decimals)}\n"
                f"<b>Отклонение от прогноза:</b> {_fmt_compact(diff_val, cfg.value_decimals)}\n\n"
            )
        else:
            fact_block += "<b>Прогноз:</b> н/д\n\n"

        factors_block = ""
        if impact_text:
            factors_block = f"<b>Ключевые факторы:</b>\n{impact_text}\n"

        return status_block + slice_line + fact_block + factors_block + links_block

    return header + slice_line + main_line + "\n\n" + impact_text + links_block


# =========================
# -------- Отправква -------
# =========================

def send_vkteams_message(
    token: str,
    chat_id: str,
    image_path: Optional[str],
    caption_html: str,
    cfg: AlertConfig,
) -> dict:
    url = f"{cfg.vkteams_api_url}/messages/sendFile" if image_path else f"{cfg.vkteams_api_url}/messages/sendText"
    data = {"token": token, "chatId": chat_id, "parseMode": "HTML"}

    if image_path:
        with open(image_path, "rb") as f:
            files = {"file": f}
            data["caption"] = caption_html
            resp = requests.post(url, data=data, files=files, timeout=30)
    else:
        data["text"] = caption_html
        resp = requests.post(url, data=data, timeout=30)

    try:
        return resp.json()
    except Exception:
        return {"ok": False, "status_code": resp.status_code, "text": resp.text}


# =========================
# ------- Оркестратор -----
# =========================

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
    cfg = cfg or AlertConfig()
    t, v, a, mcol, gcol = (
        cfg.time_col,
        cfg.value_col,
        cfg.anomaly_col,
        cfg.metric_name_col,
        cfg.granularity_col,
    )

    df = df_final.copy()
    df[t] = pd.to_datetime(df[t], errors="coerce")

    row = _pick_row_for_now(df, now, t)
    if row.empty:
        if also_return:
            return {
                "skipped": True,
                "reason": "no_row_for_now",
                "now": str(now),
                "min_ts": str(df[t].min()),
                "max_ts": str(df[t].max()),
            }
        return None

    if cfg.anomaly_only and (pd.isna(row.iloc[0][a]) or int(row.iloc[0][a]) != 1):
        if also_return:
            return {"skipped": True, "reason": "no_anomaly_flag", "now": str(now)}
        return None

    alert_row = row.iloc[0]

    if ("vs_last_day" not in df.columns) or ("vs_last_week" not in df.columns) or (
        pd.isna(alert_row.get("vs_last_day", np.nan)) and pd.isna(alert_row.get("vs_last_week", np.nan))
    ):
        prev1, prev7 = _find_prev_values(df, pd.Timestamp(now), cfg)
        df.loc[df[t] == pd.Timestamp(now), "vs_last_day"] = _calc_vs(prev1, float(alert_row[v]))
        df.loc[df[t] == pd.Timestamp(now), "vs_last_week"] = _calc_vs(prev7, float(alert_row[v]))
        alert_row = df.loc[df[t] == pd.Timestamp(now)].iloc[0]

    # сравнение со средним за N дней
    if cfg.show_avg_compare and (
        ("vs_avg_n_days" not in df.columns) or pd.isna(alert_row.get("vs_avg_n_days", np.nan))
    ):
        avg_val = _find_mean_for_prev_n_days(df, pd.Timestamp(now), cfg)
        df.loc[df[t] == pd.Timestamp(now), "vs_avg_n_days"] = _calc_vs(avg_val, float(alert_row[v]))
        alert_row = df.loc[df[t] == pd.Timestamp(now)].iloc[0]

    metric_name_effective = metric_name or str(alert_row.get(mcol, "metric"))

    plt_func = plot_func or make_plot_image
    maybe_cols = [t, v, "ci_upper", "ci_lower", "ci_mean", a, gcol]
    cols_exist = [c for c in maybe_cols if c in df.columns]
    plot_df = df[cols_exist].copy() if cols_exist else df.copy()
    img_path = plt_func(plot_df, pd.Timestamp(now), metric_name_effective, cfg)

    caption = build_caption(alert_row, cfg)
    caption = re.sub(r"(<)\s*(\d+)", r"&lt; \2", caption)

    result = None
    if token and chat_id:
        result = send_vkteams_message(
            token=token,
            chat_id=chat_id,
            image_path=img_path,
            caption_html=caption,
            cfg=cfg,
        )

    if also_return:
        return {"caption": caption, "image_path": img_path, "send_result": result}
    return result
