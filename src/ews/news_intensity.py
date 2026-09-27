"""Интенсивность новостей о событии по МО: заголовки новостей ГУ МЧС региона, где названо МО (или его населённый пункт)
и есть слова об ущербе нужного типа, в окне вокруг даты знания события."""

import pandas as pd

from .news_mchs import DIR as MCHS_DIR

IMPACT = {
    "паводок/наводнение": r"подтоп|затоп|паводк|половод|уровень воды|вода зашла|вышла из берег|дамб|эвакуир",
    "лесные/природные пожары": r"лесн\w* пожар|природн\w* пожар|ландшафтн|(?<![а-яё])палы|пал\w* (?:сух|трав)|задымлен|эвакуир",
    "авария ЖКХ/тепло-, энерго-, водоснабжение": r"без тепла|отоплен|теплоснабж|электроснабж|без света|водоснабж|котельн|авари",
    "техногенная авария/разлив/обрушение/взрыв": r"авари|разлив|обрушен|взрыв|утечк|эвакуир",
    "метеоявления (ветер, снег, мороз)": r"ветер|ураган|снег|метел|гололед|мороз|шквал|обесточ|без света",
    "атаки БПЛА/обстрелы": r"беспилот|бпла|атак|обстрел|эвакуир",
}


def titles(region_code):
    f = MCHS_DIR / "titles" / f"titles_{int(region_code):02d}.parquet"
    return pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["region_code", "id", "dt", "title"])


def intensity(matcher, region_code, territory_id, cause, date_from, date_to):
    t = titles(region_code)
    if t.empty:
        return 0, None
    t = t[(t.dt >= date_from) & (t.dt <= date_to + "T23:59")]
    pat = IMPACT.get(cause)
    if pat:
        t = t[t.title.str.contains(pat, case=False, regex=True)]
    hits = t[t.title.apply(lambda s: territory_id in matcher.match(s, region_code))]
    return len(hits), (hits.dt.min() if len(hits) else None)
