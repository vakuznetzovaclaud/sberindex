"""Новости сайтов главных управлений МЧС России по субъектам (<код>.mchs.gov.ru): архив с фильтром по датам.
Единая структура для всех регионов; у каждой новости — заголовок, дата и время публикации; текст — по запросу."""
import html
import re
import time

import pandas as pd

from .paths import RAW, config
from .web import get

DIR = RAW / "mchs"
MON = {m: i + 1 for i, m in enumerate("января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split())}
CARD = re.compile(r'class="articles-item__title" href="/deyatelnost/press-centr/novosti/(\d+)">(.*?)</a>.*?'
                  r'class="articles-item__date">\s*(\d{1,2})\s+(\S+)\s+(\d{4})(?:,\s*(\d{1,2}):(\d\d))?', re.S)


def news_url(region_code):
    """Лента новостей ГУ МЧС региона (адрес — configs/base.yaml, sources.mchs_news)."""
    return config()["sources"]["mchs_news"].format(region=f"{int(region_code):02d}")


def listing(region_code, date_from, date_to):
    """Все новости региона за период: [(id, дата-время, заголовок)]; даты — ДД.ММ.ГГГГ."""
    base = news_url(region_code)
    pause = config()["crawl"]["pause_s"]
    rows, page, last = {}, 1, 1
    while page <= last:
        s = ""
        for _ in range(3):
            s = get(base, {"news_date_from": date_from, "news_date_to": date_to, "page": page}, timeout=60)
            if "articles-item" in s or "novosti?" not in s:
                break
            time.sleep(5)
        for m in CARD.finditer(s):
            nid, title, d, mon, y, hh, mm = m.groups()
            if mon not in MON:
                continue
            rows[int(nid)] = (f"{y}-{MON[mon]:02d}-{int(d):02d}T{int(hh or 0):02d}:{mm or '00'}",
                              re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", title))).strip())
        last = max([last] + [int(x) for x in re.findall(r"novosti\?[^\"]*page=(\d+)", s)])
        page += 1
        time.sleep(pause)
    return pd.DataFrame([(region_code, k, *v) for k, v in rows.items()], columns=["region_code", "id", "dt", "title"])


def text(region_code, nid):
    s = get(f"{news_url(region_code)}/{nid}", timeout=60)
    i, j = s.find('class="public__text"'), s.find('class="share-block"')
    seg = s[i:j] if i >= 0 and j > i else ""
    return re.sub(r"\s+", " ", " ".join(html.unescape(re.sub(r"<[^>]+>", " ", p)) for p in re.findall(r"<p[^>]*>(.*?)</p>", seg, re.S))).strip()
