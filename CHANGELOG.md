# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).


## [0.4.11] — 2026-09-02
### Added
- Calendar-aware daily baseline: the 1st, 2nd, second-to-last and last days of
  a month are now grouped into separate calendar bins
  (`AnomalyParams.ci_use_month_position`, default `True`).
- For special calendar categories a fallback to the weekday baseline is used
  when there is not enough history.
- Daily special categories allow three historical points, matching the
  availability of such dates in a standard 90-day window.
- New public helper `month_position_category(date)`.
- Unit tests for the calendar-aware baseline (`tests/`).
- This changelog.

### Fixed
- Base alerts are no longer mistakenly labeled as growth when `ci_mean` is
  missing (now shown as "⚪ Аномалия").
- Robust history winsorizing in `forecast` now operates on a copy of the
  numpy array and no longer depends on pandas view/copy behavior.
- Declared dependencies now match actual imports: added `bottleneck`,
  `scipy`, `statsmodels`, `holidays`; removed unused `tqdm` and
  `python-telegram-bot`.

### Changed
- Example notebook extended with full VK Teams and Telegram configuration
  walkthroughs (message templates, axis/number formatting, N-day average
  comparison).


## [0.4.10] — 2026-03-25
### Added
- Message templates for alerts: `standard`, `extended_inline`, `analytic_blocks`.
- Comparison of the current value with the average over the previous N days:
  `show_avg_compare`, `avg_compare_days`, `avg_compare_label`.
- Number formatting options in `impact_explainer`: separate precision for
  absolute values (`decimals_value`) and percentages (`decimals_pct`).
- Extensive plot customization in alerts:
  - X axis major/minor tick step and type (`x_major_unit`, `x_major_freq`,
    `x_minor_unit`, `x_minor_freq`), date formats for daily/hourly data,
    label rotation;
  - Y axis limits (`y_min`, `y_max`), tick step (`y_tick_step`), label
    precision (`y_label_decimals`), compact scaling (`тыс`/`млн`/`млрд`);
  - axis labels, title, grid and legend toggles.
- Custom impact texts across multiple dimensions with backward compatibility.
- Unified message and plot settings for VK Teams and Telegram.

### Changed
- `attach_multi_impact` and related formatting support separate precision for
  values and percentages.
- Alert messages use a single, more flexible template.
- Impact text generation moved to universal rendering via the `impact_blocks`
  list.
- Telegram alerting brought to full feature parity with VK Teams.
- An empty line is now inserted between impact blocks by default
  (`impact_join_separator="\n\n"`).

### Fixed
- Impact blocks no longer stick together without a blank line.
- Improved alert readability with many factor blocks.
- Removed configuration discrepancies between the VK Teams and Telegram
  implementations.


## [0.4.8] — 2025-12-11
### Added
- Sensitivity profiles (`sensitivity=1…6`) for quick detector tuning.
- Per-call method toggles: `enable_sesd`, `enable_lof`, `enable_iforest`,
  `enable_stl`, `enable_cusum`.
- Advanced CI settings: bin history requirements, bin tail
  (`ci_bin_tail_factor`), MAD smoothing (`ci_smooth_mad_*`), optional σ clip
  (`ci_std_clip_min`/`ci_std_clip_max`).

### Changed
- Improved CI logic and fallback chain: bin → rolling.
- Seasonal ESD strengthened and stabilized (hybrid MAD-z score, trend check).
- Updated detector merging logic in `anomaly_final`.

### Fixed
- Narrow confidence intervals at small MAD values; division-by-zero edge
  cases.
- STL stability across sensitivity presets.


## [0.4.7] — 2025-11-20
### Fixed
- Robust Z-Score calculation logic in `seasonal_esd_full`.

### Changed
- Unified MAD normalization: `seasonal_esd_full` now uses the same 1.4826
  coefficient as the CI/MAD block.


## [0.4.6] — 2025-11-20
### Added
- Impact calculation across multiple dimensions in a single
  `attach_multi_impact` call.
- Generation of `impact_text_<dimension>` columns in `df_anomaly` (country,
  platform and any other dimension).
- Unified interface with a `dims` list instead of per-column calls.
- Automatic validation and data preparation: `total` level filtering,
  `group_col` transformation, date normalization.

### Changed
- Parallel execution removed (simplified calculation model, no
  `ThreadPoolExecutor`).

### Fixed
- Potential threading issues with a large number of anomaly dates.
- Impact text mapping risks across consecutive calls.


## [0.4.2] — 2025-11-14
### Changed
- Packaging and metadata improvements for the public release.

## [0.4.0] — 2025-11-14
### Added
- Telegram alerts: chart + compact text message (`alert_bot_telegram`).

### Fixed
- Minor bugs and inaccuracies.

## [0.3.0] — 2025-11-13
### Changed
- Reworked confidence interval calculation.
- Usage example notebook added.

### Fixed
- Minor bugs and inaccuracies.

## [0.2.0] — 2025-11-12
### Added
- `impact_explainer.py`: factor decomposition (impact) and TOP-3 contribution
  generation (by products/platforms).
- `forecast.py`: standalone "normal level" forecast module
  (Prophet + ETS + seasonal naive, automatic weights by sMAPE, bias
  calibration).
- `alert_bot.py`: support for external `forecast`, `forecast_*`, `w_*`,
  `impact_text_*` columns; the bot no longer computes the forecast itself.
- Configurable labels: slice names, impact block titles, dashboard links.
- Improved X axis and CI rendering on the alert chart.

### Changed
- Fully modular pipeline: anomalies → (opt.) impacts → (opt.) forecast →
  (opt.) bot.
- Simplified input contract: only `time_at` and `metric_value` are required.

### Fixed
- More robust CI/Z (median + MAD), careful SESD/STL/LOF fallbacks.
- Transparent logic of the final `anomaly_final` flag.

## 0.1.0 — 2025-10-07
### Added
- Initial version: anomaly detector and a built-in forecast inside the bot.

### Notes
- The embedded forecast proved hard to tune, and in 0.2.0 it was moved to a
  separate module and passed to the bot as a `forecast` column.
