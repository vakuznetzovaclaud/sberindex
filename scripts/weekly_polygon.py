"""Национальный недельный полигон 2025–2026: недельные приросты г/г СберИндекса по 46 категориям. Сравнение тревог
детектора без календаря и с календарём переходящих дат (модель учится только на прошлых неделях).
Запуск: python scripts/weekly_polygon.py"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ews import holidays, national
from ews.paths import OUTPUTS, config

THR = 4.0


def main():
    cfg = config()["period"]
    w = national.download("ver-izmenenie-trat-po-kategoriyam")
    w["period"] = pd.to_datetime(w["period"])
    X = w.pivot_table(index="period", columns="category", values="value").sort_index()
    X.columns = [c.strip() for c in X.columns]
    cal = holidays.weekly_diff(X.index)
    cal.index = X.index
    rows = []
    for c in X.columns:
        x = X[c]
        base = x.shift(1).rolling(6, min_periods=4).median()
        r = (x - base).dropna()
        for t in r.index[(r.index >= cfg["weekly_from"]) & (r.index <= cfg["weekly_to"])]:
            past = r[r.index < t]
            if len(past) < 30:
                continue
            mad = np.median(np.abs(past - past.median())) * 1.4826 or 1.0
            z_raw = (r[t] - past.median()) / mad
            m = Ridge(alpha=1.0).fit(cal.loc[past.index].to_numpy(), past.to_numpy())
            res_past = past.to_numpy() - m.predict(cal.loc[past.index].to_numpy())
            mad2 = np.median(np.abs(res_past - np.median(res_past))) * 1.4826 or 1.0
            z_cal = (r[t] - m.predict(cal.loc[[t]].to_numpy())[0] - np.median(res_past)) / mad2
            rows.append({"категория": c, "неделя": t.date(), "г/г": x[t], "z без календаря": z_raw, "z с календарём": z_cal,
                         "календарная неделя": bool((cal.loc[t].abs() > 0).any())})
    d = pd.DataFrame(rows)
    # «эхо» в рядах г/г: сдвиг уровня уходит из базы сравнения через 52 недели и даёт тревогу противоположного знака
    d["неделя"] = pd.to_datetime(d["неделя"])
    d["эхо"] = False
    for k, r in d[d["z без календаря"].abs() > THR].iterrows():
        prev = d[(d["категория"] == r["категория"]) & (d["неделя"] - r["неделя"] + pd.Timedelta(weeks=52)).abs().le(pd.Timedelta(weeks=1))]
        d.loc[k, "эхо"] = bool((prev["z без календаря"] * np.sign(r["z без календаря"]) < -THR).any())
    d["неделя"] = d["неделя"].dt.date
    d.to_csv(OUTPUTS / "weekly_polygon.csv", index=False)
    raw, adj = d[d["z без календаря"].abs() > THR], d[d["z с календарём"].abs() > THR]
    print(f"недель × категорий: {len(d)}; тревог без календаря: {len(raw)} (из них на календарных неделях {raw['календарная неделя'].mean():.0%}); "
          f"с календарём: {len(adj)} (на календарных неделях {adj['календарная неделя'].mean():.0%})")
    out = d[(d["z без календаря"].abs() > THR) & ~d["календарная неделя"]]
    print(f"вне недель праздников: {len(out)}, из них эхо прошлогодних сдвигов: {int(out['эхо'].sum())}")
    print("\nтревоги с календарём (|z| > 4):")
    print(adj.sort_values("неделя")[["неделя", "категория", "г/г", "z без календаря", "z с календарём"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
