"""Детекторы по тратам на реальном событии: паводок весны 2024 г. в Оренбургской, Курганской и Тюменской областях.
Что увидели бы детекторы, которые знают только траты, и когда: доля МО с тревогой в апреле (или в апреле–мае — GLR
смотрит на окно) при одинаковой для всех доле тревог 3% в остальных МО-месяцах 2024 г. Группы МО — по числу сообщений
о подтоплении с названием МО до конца апреля (outputs/case_flood2024.csv): 5 и больше, 1–4, ни одного; контроль —
все МО вне трёх регионов. Кроме детекторов из сравнения — порог z и GLR по одним маркетплейсам: категория выбрана по
разделу «что режимы ЧС делают с тратами», а не по этому случаю.
Вторая таблица — проверка по группе, которую задаёт текст: −z МО с 5+ сообщениями в апреле против всех МО вне трёх
регионов (односторонний тест Манна–Уитни) по каждой категории. Так текст указывает, где искать, а траты подтверждают.
Запуск: python scripts/eval_detectors_flood.py"""
import numpy as np
import pandas as pd
from scipy import stats

from ews import detect, panel as pnl, synth
from ews.paths import OUTPUTS


FA = 0.03
MP = 4                                                    # индекс категории «Маркетплейсы» в панели


def main():
    Y, ids, months, meta = pnl.load()
    Z = synth.h1_residuals(Y, months, synth.nat_series())
    D = detect.all_detectors(Z, synth.knn(meta))
    D["порог z, маркетплейсы"] = detect.threshold(Z, cat=MP)
    D["GLR, маркетплейсы"] = detect.glr(Z, cat=MP)
    case = pd.read_csv(OUTPUTS / "case_flood2024.csv")
    pos = {t: k for k, t in enumerate(ids)}
    case["i"] = case.territory_id.map(pos)
    groups = {"5+ сообщений": case[case["сообщений"] >= 5].i, "1–4 сообщения": case[case["сообщений"].between(1, 4)].i,
              "без сообщений": case[case["сообщений"] == 0].i, "на сайте МЧС": case[case["сайт МЧС"] >= 1].i}
    flood = np.zeros(len(ids), bool); flood[case.i] = True
    apr, may = months.index("2024-04"), months.index("2024-05")
    lo, hi = months.index("2024-02"), months.index("2024-12")
    ctrl = np.zeros(Z[0].shape, bool); ctrl[:, lo:hi + 1] = True
    ctrl[flood, apr:may + 1] = False                      # порог — по МО-месяцам вне паводка
    ctrl &= ~np.isnan(Z[0])
    rows = []
    for name, S in D.items():
        thr = np.quantile(S[ctrl & ~np.isnan(S)], 1 - FA)
        a = S > thr
        row = {"детектор": name}
        for g, idx in groups.items():
            idx = idx.to_numpy()
            row[f"{g}: апрель"] = float(a[idx, apr].mean())
            row[f"{g}: апрель–май"] = float((a[idx, apr] | a[idx, may]).mean())
        other = ~flood
        row["вне трёх регионов: апрель"] = float(a[other, apr].mean())
        rows.append(row)
    r = pd.DataFrame(rows).sort_values("5+ сообщений: апрель", ascending=False)
    n = {g: len(v) for g, v in groups.items()}
    pd.set_option("display.width", 250)
    print(f"МО в группах: {n}; доля тревог в контроле {FA:.0%}")
    print(r.round(3).to_string(index=False))
    r.to_csv(OUTPUTS / "tables" / "detectors_flood.csv", index=False)
    grp = []
    for c, cname in enumerate(pnl.CATS):
        x = -Z[c][:, apr]
        v, ref = x[groups["5+ сообщений"].to_numpy()], x[~flood]
        v, ref = v[~np.isnan(v)], ref[~np.isnan(ref)]
        grp.append({"категория": cname, "МО": len(v), "медиана −z": float(np.median(v)), "медиана −z вне регионов": float(np.median(ref)),
                    "p (Манн–Уитни)": float(stats.mannwhitneyu(v, ref, alternative="greater").pvalue)})
    g = pd.DataFrame(grp)
    print("\nгруппа «5+ сообщений», апрель, против МО вне трёх регионов:")
    print(g.round(3).to_string(index=False))
    g.to_csv(OUTPUTS / "tables" / "detectors_flood_group.csv", index=False)
    # устойчивость к источнику: основная группа считается по всем источникам, включая новостные Telegram-ленты,
    # собранные для этого разбора; здесь — группы только по сообщениям МЧС (сайты и Telegram-каналы МЧС)
    mchs = case["сайт МЧС"] + case["Telegram МЧС"]
    alt = {"5+ сообщений, все источники": case["сообщений"] >= 5, "5+ сообщений МЧС": mchs >= 5,
           "1+ сообщение МЧС": mchs >= 1, "1+ сообщение на сайте МЧС": case["сайт МЧС"] >= 1}
    src = []
    for gname, mask in alt.items():
        idx = case[mask].i.to_numpy()
        row = {"группа": gname, "МО": len(idx)}
        for c, cname in enumerate(pnl.CATS):
            x = -Z[c][:, apr]
            v, ref = x[idx], x[~flood]
            v, ref = v[~np.isnan(v)], ref[~np.isnan(ref)]
            row[f"{cname}: p"] = float(stats.mannwhitneyu(v, ref, alternative="greater").pvalue)
        src.append(row)
    src = pd.DataFrame(src)
    n_tests = int(src.filter(regex=": p$").notna().sum().sum())      # поправка Бонферрони на все p таблицы (4 группы × 6 категорий)
    src.insert(src.columns.get_loc("Маркетплейсы: p") + 1, "Маркетплейсы: p с поправкой", (n_tests * src["Маркетплейсы: p"]).clip(upper=1))
    print("\nустойчивость к источнику (Манн–Уитни, апрель, против МО вне трёх регионов):")
    print(src.round(4).to_string(index=False))
    src.to_csv(OUTPUTS / "tables" / "detectors_flood_group_sources.csv", index=False)


if __name__ == "__main__":
    main()
