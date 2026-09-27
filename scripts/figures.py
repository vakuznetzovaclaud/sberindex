"""Рисунки отчёта report/fig/*.png из прогнозов и итоговых таблиц outputs/ — теми же функциями, что и таблицы отчёта,
поэтому числа на рисунках и в таблицах одни и те же. Запуск: python scripts/figures.py"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from ews import evaluate as ev, panel as pnl
from ews.paths import OUTPUTS

matplotlib.use("Agg")

FIG = Path(__file__).resolve().parents[1] / "report" / "fig"
INK, MUTED, ACCENT, WARN, GRID = "#16181d", "#8a909c", "#0b6e4f", "#b4442f", "#e6e8ec"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.dpi": 200, "savefig.bbox": "tight"})


def ru(x, _=None):
    """Число по-русски: неразрывный пробел между разрядами, десятичная запятая."""
    return f"{x:,.10g}".replace(",", "\u00a0").replace(".", ",")


RU = FuncFormatter(ru)
# подписи моделей — как в таблицах отчёта
NAMES = {"snaive": "сезонный наивный", "snaive_growth": "наивный с приростом г/г", "prophet_default": "Prophet по умолчанию",
         "prophet_tuned": "Prophet с подбором", "panel": "панель: фактор + отклонение", "lgbm": "LightGBM",
         "ses_sa": "сглаживание (скорр.)", "theta_sa": "Theta (скорр.)", "timesfm25": "TimesFM-2.5 (сырое отклонение)",
         "timesfm25_sa": "TimesFM-2.5 (скорр.)", "tirex2": "TiRex-2 (сырое отклонение)", "tirex2_sa": "TiRex-2 (скорр.)",
         "chronos2_sa": "Chronos-2 (скорр.)", "ens_main": "ансамбль",
         "lstm_sa": "LSTM (скорр.)", "tcn_sa": "TCN (скорр.)", "patchtst_sa": "PatchTST (скорр.)"}


def forecast_mae(cat=0):
    """MAE моделей по «Все категории», все точки 2024 г., общие строки."""
    names = [n for n in NAMES if n in ev.available(cat) and n != "prophet_tuned"]
    t = ev.table(names, cat).set_index("модель").MAE.sort_values()
    snaive = t.pop("snaive")                          # на порядок хуже остальных — в подпись, а не на шкалу
    fig, ax = plt.subplots(figsize=(6.2, 0.26 * len(t) + 0.6))
    colors = [ACCENT if n == "ens_main" else (WARN if n.startswith("prophet") else "#4a5568") for n in t.index]
    ax.barh([NAMES[n] for n in t.index][::-1], t.values[::-1], color=colors[::-1], height=0.62)
    for y, v in enumerate(t.values[::-1]):
        ax.text(v + t.max() * 0.01, y, ru(round(v)), va="center", fontsize=8)
    ax.set_xlabel(f"MAE, руб. на жителя в месяц; сезонный наивный — {ru(round(snaive))}")
    ax.xaxis.set_major_formatter(RU); ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "forecast_mae.png"); plt.close(fig)


def month_error(cat=0, models=("panel", "lgbm", "tirex2", "tirex2_sa", "ens_main")):
    Y, ids, months, meta = pnl.load()
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for n in models:
        if n not in ev.available(cat):
            continue
        p = ev.load(n, cat)
        e = pd.Series(np.abs(p.yhat.to_numpy() - Y[cat][p.i, p.target])).groupby(p.target.map(lambda t: months[t][2:])).mean()
        ax.plot(e.index, e.values, marker="o", ms=3, lw=1.6 if n == "ens_main" else 1.0,
                color=ACCENT if n == "ens_main" else None, label=NAMES.get(n, n))
    ax.set_ylabel("MAE, руб."); ax.yaxis.set_major_formatter(RU); ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.savefig(FIG / "month_error.png"); plt.close(fig)


def detectors():
    r = pd.read_csv(OUTPUTS / "detectors_synthetic.csv")
    s = r.groupby("детектор")[["VUS-PR", "полнота@3%", "NAB-подобная"]].mean().sort_values("VUS-PR")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 0.24 * len(s) + 0.8), sharey=True)
    for ax, col in zip(axes, s.columns):
        colors = [ACCENT if "текст" in d else INK if d == "6 категорий: Стауффер" else "#2f6fb4" if "ансамбль" in d else "#a3a9b5"
                  for d in s.index]
        ax.barh(s.index, s[col], color=colors, height=0.62)
        ax.set_title(col, fontsize=9); ax.xaxis.set_major_formatter(RU); ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "detectors.png"); plt.close(fig)


def warning(source="llm"):
    r = pd.read_csv(OUTPUTS / "tables" / f"warning_{source}.csv").set_index("вариант")
    col = "Маркетплейсы: подъём"
    s = r[col].dropna()
    fig, ax = plt.subplots(figsize=(6.2, 0.28 * len(s) + 0.6))
    ax.barh(s.index[::-1], s.values[::-1], color=[WARN if "плацебо" in i else ACCENT for i in s.index[::-1]], height=0.6)
    ax.axvline(1, color=INK, lw=0.8)
    ax.set_xlabel("во сколько раз чаще, чем обычно в тех же МО, — провал трат на маркетплейсах\nв месяце эпизода или следующем")
    ax.xaxis.set_major_formatter(RU); ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "warning_lift.png"); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (forecast_mae, month_error, detectors, warning):
        f()
        print("готово:", f.__name__)
