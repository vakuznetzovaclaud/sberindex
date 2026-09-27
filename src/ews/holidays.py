"""Календарь РФ: нерабочие и сокращённые дни по производственному календарю (xmlcalendar.ru), православная Пасха,
дни подарков (14 февраля, 23 февраля, 8 марта, День матери). Для недельных рядов г/г — разность календарных признаков
«эта неделя минус сопоставимая неделя год назад» (сдвиг на 364 дня): переходящие даты дают ложные скачки."""
import datetime as dt
import json

import pandas as pd

from .paths import RAW, config
from .web import get

DIR = RAW / "calendar"


def production(years=range(2022, 2027)):
    """{дата: 'выходной' | 'сокращённый'} по производственному календарю."""
    out = {}
    DIR.mkdir(parents=True, exist_ok=True)
    for y in years:
        f = DIR / f"cal{y}.json"
        if not f.exists():
            f.write_text(get(config()["sources"]["production_calendar"].format(year=y)))
        for m in json.loads(f.read_text())["months"]:
            for d in m["days"].split(","):
                day = int(d.rstrip("*+"))
                out[dt.date(y, m["month"], day)] = "сокращённый" if d.endswith("*") else "выходной"
    return out


def orthodox_easter(year):
    """Православная Пасха (юлианский алгоритм Меёса) в григорианском календаре."""
    a, b, c = year % 4, year % 7, year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    month, day = divmod(d + e + 114, 31)
    return dt.date(year, month, day + 1) + dt.timedelta(days=13)


def mothers_day(year):
    d = dt.date(year, 11, 30)
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)


def day_features(d, prod):
    weekend = d.weekday() >= 5
    kind = prod.get(d)
    return {"праздник в будний день": int(kind == "выходной" and not weekend), "рабочий в выходной": int(weekend and d not in prod),
            "сокращённый": int(kind == "сокращённый"), "пасха": int(d == orthodox_easter(d.year)),
            "день подарков": int((d.month, d.day) in {(2, 14), (2, 23), (3, 8)} or d == mothers_day(d.year)),
            "канун нового года": int((d.month, d.day) in {(12, 29), (12, 30), (12, 31)})}


def weekly_diff(week_ends):
    """Календарные признаки недели [D-6, D] минус недели [D-370, D-364] для списка дат конца недели."""
    prod = production()
    rows = []
    for D in pd.to_datetime(week_ends):
        D = D.date()
        cur = pd.DataFrame([day_features(D - dt.timedelta(days=k), prod) for k in range(7)]).sum()
        prev = pd.DataFrame([day_features(D - dt.timedelta(days=364 + k), prod) for k in range(7)]).sum()
        rows.append((cur - prev).rename(pd.Timestamp(D)))
    return pd.DataFrame(rows)
