"""Публичные Telegram-каналы региональных управлений МЧС и региональных новостей — через веб-превью t.me/s (без
аккаунта и API). Поиск в превью отдаёт лишь ~20 последних совпадений, поэтому лента листается целиком по номерам постов
(?before=N — 20 постов до N), тема отбирается локально. Только посты каналов, без комментариев; пауза между запросами;
тексты хранятся локально и в репозиторий не входят. Используется для разбора паводка 2024 г. (второй источник)."""
import datetime as dt
import html
import json
import re
import time

from .paths import RAW, config
from .web import get

DIR = RAW / "telegram"
POST = re.compile(r'<div class="tgme_widget_message_wrap.*?data-post="([^"]+)".*?(?=<div class="tgme_widget_message_wrap|\Z)', re.S)
# каналы разбора паводка весны 2024 г.: управления МЧС и крупнейшие региональные новостные ленты трёх областей
FLOOD_CHANNELS = {56: ["orenburg_online56", "pojarnoe_depo"], 45: ["kurgan45mchs", "kurganskayaobl"], 72: ["MCHSTYUMEN72"]}


def _parse(page):
    out = []
    for m in POST.finditer(page):
        block = m.group(0)
        t = re.search(r'<time datetime="([^"]+)"', block)
        txt = re.search(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', block, re.S)
        if t:
            text = html.unescape(re.sub(r"<br\s*/?>", "\n", re.sub(r"<(?!br)[^>]+>", "", txt.group(1)))).strip() if txt else ""
            out.append({"post": m.group(1), "id": int(m.group(1).split("/")[-1]), "dt": t.group(1), "text": text})
    return out


def _page(channel, before=None):
    time.sleep(config()["crawl"]["pause_s"])
    return _parse(get(config()["sources"]["telegram_preview"].format(channel=channel), {"before": before} if before else None, timeout=30))


def _first_id_after(channel, day):
    """Номер поста, с которого лента идёт не раньше даты day (двоичный поиск по номерам)."""
    top = _page(channel)
    if not top:
        return None
    lo, hi = 1, max(p["id"] for p in top) + 1
    while hi - lo > 20:
        mid = (lo + hi) // 2
        posts = _page(channel, mid)
        if not posts or max(p["dt"] for p in posts)[:10] < day:
            lo = mid
        else:
            hi = mid
    return hi


def collect(channel, since, until):
    """Все посты канала за [since, until] (ГГГГ-ММ-ДД); кэш на диске."""
    f = DIR / f"{channel}_{since}_{until}.jsonl"
    if f.exists():
        return [json.loads(l) for l in open(f, encoding="utf-8")]
    after = (dt.date.fromisoformat(until) + dt.timedelta(days=1)).isoformat()
    before, got = _first_id_after(channel, after), {}
    while before and before > 1:
        posts = _page(channel, before)
        if not posts:
            before -= 20                                   # удалённые посты
            continue
        for p in posts:
            if since <= p["dt"][:10] <= until:
                got[p["id"]] = p | {"channel": channel}
        if min(p["dt"] for p in posts)[:10] < since:
            break
        before = min(p["id"] for p in posts)
    DIR.mkdir(parents=True, exist_ok=True)
    rows = sorted(got.values(), key=lambda p: p["dt"])
    with open(f, "w", encoding="utf-8") as fh:
        for p in rows:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    return rows


def flood_posts(since="2024-02-15", until="2024-05-31"):
    """Посты каналов разбора паводка с регионом канала."""
    out = []
    for rc, chans in FLOOD_CHANNELS.items():
        for ch in chans:
            out += [p | {"region_code": rc} for p in collect(ch, since, until)]
    return out
