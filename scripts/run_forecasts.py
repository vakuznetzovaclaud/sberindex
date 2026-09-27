"""Прогнозный стенд. Запуск: python scripts/run_forecasts.py <категория 0–5> <модель> [...]
prophet_tuned собирается из посчитанных prophet_g0…g7: python scripts/run_forecasts.py <категория> prophet_tuned"""
import sys

from ews import backtest, baselines, foundation, models, neural

MODELS = {
    "snaive": models.snaive,
    "snaive_growth": models.snaive_growth,
    "panel_factor_panel": lambda c: models.panel(c, use_national=False),
    "panel": models.panel,
    "lgbm": models.lgbm_correction,
    "lgbm_global": models.lgbm_global,
    "naive_sa": models.stat_on_seasonally_adjusted("naive"),
    "ses_sa": models.stat_on_seasonally_adjusted("ses"),
    "theta_sa": models.stat_on_seasonally_adjusted("theta"),
    "lgbm_text": models.lgbm_correction,
    "lgbm_text_regex": models.lgbm_correction,
    "lgbm_nowcast_text": models.lgbm_correction,
    "timesfm25": foundation.fm_on_deviation("timesfm25"),
    "tirex2": foundation.fm_on_deviation("tirex2"),
    "tirex2_sa": foundation.fm_on_seasonally_adjusted("tirex2"),
    "tirex2_raw": foundation.fm_on_raw("tirex2"),
    "timesfm25_sa": foundation.fm_on_seasonally_adjusted("timesfm25"),
    "chronos2_sa": foundation.fm_on_seasonally_adjusted("chronos2"),
    "chronos2_sa_cross": foundation.fm_on_seasonally_adjusted("chronos2", cross=True),
    "lstm_sa": neural.timecast_on_seasonally_adjusted("lstm"),
    "tcn_sa": neural.timecast_on_seasonally_adjusted("tcn"),
    "patchtst_sa": neural.timecast_on_seasonally_adjusted("patchtst"),
    "chronos2_sa_cov": foundation.chronos2_sa_covariates(),
    "chronos2_sa_cov_cross": foundation.chronos2_sa_covariates(cross=True),
    "chronos2_sa_ft": foundation.chronos2_finetuned_sa(),
    "chronos2_sa_ft_lr1e5": foundation.chronos2_finetuned_sa(steps=150, lr=1e-5),
    "prophet_default": baselines.prophet_model(baselines.CONFIGS["prophet_default"]),
    "prophet_timecast": baselines.prophet_model(baselines.CONFIGS["prophet_timecast"]),
    **{f"prophet_g{k}": baselines.prophet_model(kw) for k, kw in enumerate(baselines.GRID)},
}
# по отдельному ряду и медленно — на стратифицированной выборке 400 МО
SLOW = {"prophet_timecast", *[f"prophet_g{k}" for k in range(len(baselines.GRID))]}
# с текстовыми признаками МО-месяца (акты ЧС + новости МЧС, размеченные моделью или словарём)
TEXT = {"lgbm_text": "llm", "lgbm_text_regex": "regex", "lgbm_nowcast_text": "llm"}
NOWCAST_DAY = 15                              # наукаст: прогноз в середине месяца, тексты — по 15-е число

if __name__ == "__main__":
    cat = int(sys.argv[1])
    if sys.argv[2:] == ["prophet_tuned"]:          # сборка из уже посчитанных конфигураций сетки
        print(baselines.select_online([f"prophet_g{k}" for k in range(len(baselines.GRID))], cat))
        sys.exit()
    for name in sys.argv[2:]:
        out = backtest.run(name, MODELS[name], cat, subset=backtest.stratified_subset() if name in SLOW else None,
                           text=TEXT.get(name), nowcast_day=NOWCAST_DAY if "nowcast" in name else None)
        print(name, "категория", cat, "—", len(out), "прогнозов", flush=True)
