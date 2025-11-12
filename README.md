# anomaly_impact_alert
<img src="docs/logo.png" alt="logo" width="200">

**Anomaly Impact Alert** — a lightweight Python toolkit for **detecting anomalies**, **forecasting metrics**, **explaining deviations**, and **sending alerts** (e.g. via Telegram)

---

## Purpose
This library helps analytics and monitoring teams automatically:
- Detect abnormal metric values in time-series data
- Forecast expected ranges (with confidence intervals)
- Explain the drivers behind changes (countries, platforms, etc.)
- Send structured alerts to Telegram

---

## 📊 Example Output

### Detection visualization
Shows detected anomalies (red dots), confidence intervals, and model outputs:

<img src="docs/example_detection.png" alt="Anomaly Detection]" width="900">


### Telegram alert
Example of an automatically formatted alert with top contributing segments:

<img src="docs/example_telegram.png" alt="Telegram Alert" width="400">

---

## Quick Start

```bash
pip install anomaly_impact_alert
```

```python
from anomaly_impact_alert import (
    AnomalyParams, analyze_latest_point,
    ImpactConfig, attach_impact_text,
    BFConfig, forecast_values_for_targets_better,
    AlertConfig, send_alert_for_date,
)

# Detect anomalies
params = AnomalyParams(threshold=3.0)
anomalies = analyze_latest_point(df, params)

# Forecast expected values
forecast_cfg = BFConfig(periods=7)
forecast = forecast_values_for_targets_better(df, forecast_cfg)

# Explain the impact
impact_cfg = ImpactConfig(metric="amount")
text = attach_impact_text(anomalies, impact_cfg)

# Send alert to Telegram
alert_cfg = AlertConfig(bot_token="YOUR_BOT_TOKEN", chat_id="YOUR_CHAT_ID")
send_alert_for_date(text, alert_cfg)
```

---

## 📦 Modules Overview

| Module | Description |
|--------|--------------|
| `anomaly_detector` | Detects anomalies using statistical and ML methods |
| `forecast` | Forecasts values and confidence intervals |
| `impact_explainer` | Explains which dimensions cause deviations |
| `alert_bot_telegram` | Sends rich alerts to Telegram |

---

## ⚙️ Dependencies
`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `prophet`, `tqdm`, `requests`, `python-telegram-bot`

---

## 🧾 License
MIT © 2025 Aleksey Voronko, Petr Devitsin
