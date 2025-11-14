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

## Demo Notebook
**Anomaly Impact Alert** is a lightweight Python toolkit for:  
- **Anomaly detection** (Z-Score, CI, STL, SESD, LOF, Isolation Forest, and more)  
- **Forecasting metrics** (Prophet, ETS, STL ensemble)  
- **Explaining deviations** (impact decomposition by segments)  
- **Sending alerts** (e.g., via Telegram)

This notebook demonstrates how to use **anomaly_impact_alert** in practice:  
- Data preparation  
- Anomaly detection workflow  
- Time-series forecasting  
- Contribution analysis across segments  
- Automated alert generation and explanations  

 **Try it in Google Colab:**  👉 [Open in Colab](https://colab.research.google.com/drive/117E7v4KhX6wqLylcK-1g47Ee_pnNCHYd)


---

## Example Output

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

**Try it in Google Colab:**  👉 [Open in Colab](https://colab.research.google.com/drive/117E7v4KhX6wqLylcK-1g47Ee_pnNCHYd)


---

## Modules Overview

| Module | Description |
|--------|--------------|
| `anomaly_detector` | Detects anomalies using statistical and ML methods |
| `forecast` | Forecasts values and confidence intervals |
| `impact_explainer` | Explains which dimensions cause deviations |
| `alert_bot_telegram` | Sends rich alerts to Telegram |

---

## Dependencies
`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `prophet`, `tqdm`, `requests`, `python-telegram-bot`

---

## License
MIT © 2025 Aleksey Voronko
