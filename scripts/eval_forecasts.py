"""Таблицы качества прогноза: модель × категория × окно оценки. Запуск: python scripts/eval_forecasts.py
Окна: все точки 2024 года; «разработка» (точки прогноза 01–06.2024, на них принимались решения) и «контроль»
(07–11.2024, при выборе конструкции не использовались). Модели на выборке 400 МО (Prophet в медленных конфигурациях) сравниваются в отдельной
таблице на тех же 400 МО. Дополнительные таблицы: проверки заголовочных чисел, ошибка по месяцам цели, попарный тест
Диболда–Мариано ансамбля с соседними моделями, варианты ансамбля."""
import numpy as np
import pandas as pd

from ews import baselines, evaluate as ev, panel as pnl
from ews.backtest import origins, stratified_subset
from ews.paths import OUTPUTS

ORDER = ["snaive", "snaive_growth", "prophet_default", "prophet_timecast", "prophet_tuned", "panel_factor_panel", "panel",
         "lgbm", "lgbm_global", "tirex2_raw", "timesfm25", "tirex2", "naive_sa", "ses_sa", "theta_sa", "chronos2_sa",
         "chronos2_sa_cross", "chronos2_sa_cov", "chronos2_sa_cov_cross", "timesfm25_sa", "tirex2_sa", "chronos2_sa_ft",
         "chronos2_sa_ft_lr1e5", "lstm_sa", "tcn_sa", "patchtst_sa"]
# состав ансамблей задан до оценки: панель, бустинг и три FM с равными весами
ENSEMBLES = {"ens_fm_sa": ["timesfm25_sa", "tirex2_sa", "chronos2_sa"],
             "ens_main": ["panel", "lgbm", "timesfm25_sa", "tirex2_sa", "chronos2_sa"]}
INTERVALS = ["panel", "lgbm", "ens_main"]
SUBSET_ONLY = {"prophet_timecast", "prophet_tuned"}


def windows(months):
    o = origins(months)
    return {"все точки": o, "разработка 01–06": [t for t in o if months[t] <= "2024-06"],
            "контроль 07–11": [t for t in o if months[t] >= "2024-07"]}


def main():
    _, _, months, _ = pnl.load()
    out, lines = [], ["# Качество прогноза трат на жителя МО", ""]
    for cat, cname in enumerate(pnl.CATS):
        have = ev.available(cat)
        for e, members in ENSEMBLES.items():
            if all(m in have for m in members):
                ev.ensemble(members, cat, e)
        have = ev.available(cat)
        full = [m for m in ORDER if m in have and m not in SUBSET_ONLY] + [e for e in ENSEMBLES if e in have]
        sub = [m for m in ORDER if m in have and m in SUBSET_ONLY]
        for wname, ws in windows(months).items():
            groups = [("все МО", full)] + ([("400 МО", full + sub)] if sub else [])
            for scope, names in groups:
                t = ev.table(names, cat, base="snaive", ref="prophet_default" if "prophet_default" in names else None, origins=ws)
                t.insert(0, "окно", wname); t.insert(0, "МО", scope); t.insert(0, "категория", cname)
                out.append(t)
                if wname == "все точки":
                    lines += [f"## {cname} — {scope}, все точки 2024", "", t.drop(columns=["категория", "МО", "окно"]).round(3).to_markdown(index=False), ""]
    res = pd.concat(out)
    res.to_csv(OUTPUTS / "tables" / "forecast_metrics.csv", index=False)
    piv = res[res["МО"] == "все МО"].pivot_table(index="модель", columns=["окно", "категория"], values="MAE")
    lines += ["## MAE по окнам (все МО)", "", piv.round(1).to_markdown(), ""]
    iv = []
    for cat, cname in enumerate(pnl.CATS):
        for n in INTERVALS:
            if n in ev.available(cat):
                iv.append(ev.conformal(n, cat).reset_index().assign(категория=cname, модель=n))
    if iv:
        iv = pd.concat(iv)
        iv.to_csv(OUTPUTS / "tables" / "forecast_intervals.csv", index=False)
        lines += ["## Интервалы (скользящий конформный метод)", "", iv.round(3).to_markdown(index=False), ""]
    (OUTPUTS / "tables" / "forecast_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(piv.round(1).to_string())
    checks(months)


def mae(p, Y):
    return np.abs(p.yhat.to_numpy() - Y[p.i, p.target])


def checks(months, cat=0):
    """Проверки заголовочных чисел по «Все категории»: без декабрьских целей, ансамбль без FM (FM заменены
    сглаживанием на том же входе), разброс фиксированных настроек Prophet, R² прироста внутри месяца цели, R²_oos к
    наивному прогнозу с приростом; ошибка по месяцам цели; попарный тест DM ансамбля; варианты ансамбля."""
    Y = pnl.load()[0][cat]
    L = {n: ev.load(n, cat) for n in ["ens_main", "prophet_default", "snaive_growth", "panel", "lgbm", "ses_sa", "tirex2", "tirex2_sa", "tirex2_raw"]}
    dec = months.index("2024-12")
    nodec = {n: mae(L[n][L[n].target != dec], Y).mean() for n in ("ens_main", "prophet_default")}
    key = ["i", "origin", "h", "target"]
    nofm = pd.concat([L[m].set_index(key).yhat for m in ("panel", "lgbm", "ses_sa")], axis=1).mean(1).rename("yhat").reset_index()
    sub = stratified_subset()
    grid = [mae(g[g.i.isin(sub)], Y).mean() for g in (ev.load(f"prophet_g{k}", cat) for k in range(len(baselines.GRID)))]
    p = L["ens_main"]
    a, prev = Y[p.i, p.target], Y[p.i, p.target - 12]
    g = p.assign(ga=a / prev - 1, gf=p.yhat.to_numpy() / prev - 1)                 # прирост, как R² г/г в evaluate.table
    within = np.mean([1 - ((x.ga - x.gf) ** 2).sum() / ((x.ga - x.ga.mean()) ** 2).sum() for _, x in g.groupby("target")])
    e, es = mae(p, Y), mae(L["snaive_growth"], Y)
    rows = [("MAE ансамбля без декабрьских целей", nodec["ens_main"]), ("MAE Prophet по умолчанию без декабрьских целей", nodec["prophet_default"]),
            ("разрыв с Prophet без декабря, %", 100 * (1 - nodec["ens_main"] / nodec["prophet_default"])),
            ("MAE ансамбля без FM (панель, LightGBM, сглаживание)", mae(nofm, Y).mean()),
            ("MAE Prophet: лучшая из 8 настроек сетки, 400 МО", min(grid)), ("MAE Prophet: худшая из 8 настроек сетки, 400 МО", max(grid)),
            ("R² г/г внутри месяца цели, среднее по месяцам", within), ("R²_oos к наивному с приростом г/г", 1 - (e ** 2).sum() / (es ** 2).sum())]
    pd.DataFrame(rows, columns=["показатель", "значение"]).to_csv(OUTPUTS / "tables" / "forecast_checks.csv", index=False)
    by_m = pd.DataFrame({n: pd.Series(mae(L[n], Y)).groupby(L[n].target.map(lambda t: months[t]).to_numpy()).mean()
                         for n in ["ens_main", "prophet_default", "panel", "lgbm", "tirex2", "tirex2_sa", "tirex2_raw"]})
    by_m.rename_axis("месяц цели").to_csv(OUTPUTS / "tables" / "forecast_by_month.csv")
    pairs = []
    for other in ["timesfm25_sa", "tirex2_sa", "chronos2_sa_cov_cross", "lgbm", "snaive_growth"]:
        q = ev.load(other, cat)
        row = {"против": other}
        for h in (1, 2, 3):
            l1 = pd.Series(mae(p[p.h == h], Y)).groupby(p[p.h == h].target.to_numpy()).mean()
            l2 = pd.Series(mae(q[q.h == h], Y)).groupby(q[q.h == h].target.to_numpy()).mean()
            row[f"DM p, h={h}"] = ev.dm_test(l1.values, l2.values, h)
        pairs.append(row)
    pd.DataFrame(pairs).to_csv(OUTPUTS / "tables" / "forecast_dm_pairs.csv", index=False)
    members = ENSEMBLES["ens_main"]
    variants = {"среднее (ens_main)": "ens_main", "медиана": ev.ensemble(members, cat, "ens_main_median", how="median"),
                "веса по обратной прошлой ошибке": ev.ensemble_online(members, cat, "ens_main_online")}
    pd.DataFrame([(k, mae(ev.load(v, cat), Y).mean()) for k, v in variants.items()], columns=["ансамбль", "MAE"]) \
        .to_csv(OUTPUTS / "tables" / "forecast_ensembles.csv", index=False)


if __name__ == "__main__":
    main()
