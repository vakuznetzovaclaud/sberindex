"""Ручная проверка разметки на случайных выборках: 40 актов о введении и изменении режимов (зерно 2026) сверяются
с распознанным текстом акта, 50 новостей МЧС, признанных событием (зерно 777), — с заголовком и текстом новости;
выборки, вердикты и итоги — report/checks/acts_sample.csv, news_sample.csv, summary.csv. Вердикты хранятся в тех же
файлах и при повторной выгрузке сохраняются. Запуск: python scripts/audit_samples.py"""
import json

import pandas as pd

from ews import acts, news_classify as nc, text_features as tf
from ews.paths import PROCESSED, ROOT

OUT = ROOT / "report" / "checks"
# акт: событие — «да» (режим о текущем бедствии), «нет» (правила поведения, выплаты, субсидии, COVID-19, приём
# переселенцев и т.п.), «старое» (правка зоны ЧС прошлых лет); режим, причина и территория оцениваются у «да»:
# причина — категория совпадает с главной причиной в акте; территория — зона события, а не только формальная зона режима
ACT_VERDICT = ["событие", "режим_верен", "причина_верна", "территория_верна", "комментарий"]
# новость: «событие» — названы текущие последствия для жителей (подтоплены дома, дворы или дороги, отрезаны сёла,
# эвакуация, отключения, пожар у сёл или дым над ними, откачка воды из домов); «слабое» — бедствие настоящее, но
# последствия не названы или единичны (крупные пожары вдали от сёл, одиночный пожар, спад воды); «не событие» —
# превентивные работы и режимы, учения, разъяснения, итоги сезона, бытовые происшествия
NEWS_VERDICT = ["оценка", "комментарий"]


def keep(df, name, key, cols):
    """Вердикты из прежней выгрузки того же файла (по ключу)."""
    f = OUT / name
    if not f.exists():
        return df.assign(**{c: "" for c in cols})
    return df.join(pd.read_csv(f, dtype={key: str}, keep_default_na=False).set_index(key)[cols], on=key)


def acts_sample():
    ex = pd.DataFrame([json.loads(l) for l in open(acts.extracted_path())])
    lst = pd.read_parquet(acts.DIR / "acts_2023_2024.parquet").set_index("eo")
    ex = ex.join(lst[["action", "pub_date", "title"]], on="eo")
    smp = ex[ex.action.isin(["введение", "изменение"])].sample(40, random_state=2026)
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    rules = ev.groupby("eo").agg(действие=("action", "first"), старый_акт=("old", "first"), категория=("cause", "first"),
                                 охват=("scope", "first"), число_мо=("territory_id", "count"))
    # в событиях — тот же отбор, что у признаков и системы тревог (text_features, warning)
    rules["в_событиях"] = rules.действие.isin(["введение", "изменение"]) & ~rules.старый_акт & rules.категория.isin(tf.RELEVANT)
    d = pd.DataFrame({"eo": smp.eo, "дата_публикации": smp.pub_date, "заголовок": smp.title, "режим": smp["режим"],
                      "весь_субъект": smp["вся_территория_субъекта"],
                      "мо": smp["муниципальные_образования"].map("; ".join),
                      "пункты": smp["населённые_пункты"].map(lambda x: "; ".join(x or [])),
                      "причина_дословно": smp["причина_дословно"]}).join(rules, on="eo")
    return keep(d, "acts_sample.csv", "eo", ACT_VERDICT)


def news_sample():
    lab = nc.labels()
    smp = lab[lab.apply(nc.is_event, axis=1)].sample(50, random_state=777)
    d = pd.DataFrame({"id": smp.id.astype(str), "регион": smp.region_code, "дата": smp.dt.str[:10], "заголовок": smp.title,
                      "причина": smp["причина"], "места": smp["места"].map("; ".join)})
    return keep(d, "news_sample.csv", "id", NEWS_VERDICT)


def summary(a, n):
    cnt = lambda m: int(m.sum())
    no, old, cur = a.событие.eq("нет"), a.событие.eq("старое"), a[a.событие.eq("да")]
    rows = [("акты", "в выборке", len(a)),
            ("акты", "не события", cnt(no)),
            ("акты", "не события в реестре событий", cnt(no & a.в_событиях)),
            ("акты", "правки старых зон ЧС", cnt(old)),
            ("акты", "правки старых зон ЧС в реестре событий", cnt(old & a.в_событиях)),
            ("акты", "текущие режимы", len(cur)),
            ("акты", "текущие режимы в реестре событий", cnt(cur.в_событиях)),
            ("акты", "тип режима верен", cnt(cur.режим_верен.eq("да"))),
            ("акты", "причина верна", cnt(cur.причина_верна.eq("да"))),
            ("акты", "территория верна", cnt(cur.территория_верна.eq("да"))),
            ("новости", "в выборке", len(n)),
            ("новости", "событие", cnt(n.оценка.eq("событие"))),
            ("новости", "слабое", cnt(n.оценка.eq("слабое"))),
            ("новости", "не событие", cnt(n.оценка.eq("не событие"))),
            ("новости", "точность строго, %", round(100 * n.оценка.eq("событие").mean())),
            ("новости", "точность с оговорками, %", round(100 * n.оценка.isin(["событие", "слабое"]).mean()))]
    return pd.DataFrame(rows, columns=["выборка", "показатель", "значение"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a, n = acts_sample(), news_sample()
    a.to_csv(OUT / "acts_sample.csv", index=False)
    n.to_csv(OUT / "news_sample.csv", index=False)
    s = summary(a, n)
    s.to_csv(OUT / "summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
