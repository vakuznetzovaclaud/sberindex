"""Национальные ряды портала СберИндекса (выгрузки parquet по названию набора)."""
import hashlib
import shutil

import pandas as pd

from .paths import RAW, ROOT, config
from .web import get

DIR = RAW / "portal"
SNAPSHOT = ROOT / "data" / "snapshots" / "portal"      # выгрузки, на которых посчитана работа (23.09.2026)


def download(slug):
    """Выгрузка набора портала. Портал дополняет ряды новыми периодами и пересчитывает недавние недели, поэтому
    используемые выгрузки лежат в репозитории (data/snapshots/portal) и закреплены контрольной суммой в
    configs/base.yaml (sources.portal_sha256); с портала набор скачивается, только если снимка нет."""
    src = config()["sources"]
    DIR.mkdir(parents=True, exist_ok=True)
    f = DIR / f"{slug}.parquet"
    if not f.exists() and (SNAPSHOT / f.name).exists():
        shutil.copy(SNAPSHOT / f.name, f)
    if not f.exists():
        # curl -k: сервер портала не передаёт промежуточный сертификат, и без -k загрузка падает везде, где его нет
        # в хранилище системы; подлинность файла проверяет контрольная сумма
        f.write_bytes(get(src["portal_api"] + f"/dataset/v1/download/{slug}/parquet", binary=True, insecure=True))
    if hashlib.sha256(f.read_bytes()).hexdigest() != src["portal_sha256"][slug]:
        raise RuntimeError(f"{f}: контрольная сумма не совпадает с configs/base.yaml — это другая выгрузка портала "
                           f"(ряд дополнен или пересчитан), числа работы на ней не воспроизведутся")
    return pd.read_parquet(f)


def monthly(kind="Всего"):
    """Национальные безналичные расходы по типу («Всего», «Продовольственные товары», «Общественное питание», …), млрд руб."""
    cs = download("consumer-spending")
    s = cs[cs["type"] == kind].assign(period=lambda d: pd.to_datetime(d["period"])).set_index("period")["value"].sort_index()
    s.index = s.index.to_period("M").astype(str)
    return s


# Национальный ряд, соответствующий категории панели по смыслу (выбор сделан до оценки, не по тесту).
# Для здоровья, маркетплейсов и транспорта национального аналога на портале нет — фактор только по панели.
CATEGORY_SERIES = {"Все категории": "Всего", "Продовольствие": "Продовольственные товары", "Общественное питание": "Общественное питание"}


def for_category(cat_name):
    kind = CATEGORY_SERIES.get(cat_name)
    return monthly(kind) if kind else None
