"""Панель «МО × месяц × категория» безналичных расходов СберИндекса (архив хакатона Data -> Sense с territory_id)
и справочник МО СберИндекса (ОКТМО, регион, центр, полигоны)."""
import hashlib
import sqlite3
import struct
import subprocess

import numpy as np
import pandas as pd

from .paths import RAW, PROCESSED, config
from .rosstat import population
from .web import get

SRC = RAW / "sberindex"
CATS = ["Все категории", "Продовольствие", "Здоровье", "Общественное питание", "Маркетплейсы", "Транспорт"]
# короткие подписи категорий для разборов событий (без «Здоровья»)
CATS_SHORT = {0: "Все", 1: "Продовольствие", 3: "Общепит", 4: "Маркетплейсы", 5: "Транспорт"}


def download():
    """Архив хакатона и справочник границ МО. Версии закреплены контрольными суммами в configs/base.yaml."""
    c = config()["sources"]
    SRC.mkdir(parents=True, exist_ok=True)
    for key, name in [("hackathon_zip", "hackathonlicence.zip"), ("borders_rar", "t_dict_municipal.rar")]:
        f = SRC / name
        if not f.exists():
            # curl -k: сертификат сайта выдан российским удостоверяющим центром, которого нет в стандартных хранилищах;
            # подлинность архива проверяет контрольная сумма
            f.write_bytes(get(c[key], timeout=600, binary=True, insecure=True))
        if hashlib.sha256(f.read_bytes()).hexdigest() != c[f"{key}_sha256"]:
            raise RuntimeError(f"{f}: контрольная сумма не совпадает с configs/base.yaml")
        subprocess.run(["bsdtar", "-xf", str(f), "-C", str(SRC)], check=True)


def directory():
    f = next(SRC.rglob("t_dict_municipal_districts.xlsx"))
    d = pd.read_excel(f)
    return d.sort_values("year_to").drop_duplicates("territory_id", keep="last").set_index("territory_id")


def _polygon_centers():
    """Центр охватывающего прямоугольника полигона МО из справочника границ (GeoPackage: в заголовке геометрии —
    minx, maxx, miny, maxy). Нужен для внутригородских территорий Москвы и Петербурга: у них в справочнике нет
    координат центра."""
    con = sqlite3.connect(next(SRC.rglob("t_dict_municipal_districts_poly.gpkg")))
    d = pd.read_sql("select territory_id, year_to, geom from t_dict_municipal_districts_poly", con)
    d = d.sort_values("year_to").drop_duplicates("territory_id", keep="last")
    out = {}
    for tid, g in zip(d.territory_id, d.geom):
        g = bytes(g)
        if (g[3] >> 1) & 7:                                       # в заголовке есть охватывающий прямоугольник
            x0, x1, y0, y1 = struct.unpack("<4d" if g[3] & 1 else ">4d", g[8:40])
            out[int(tid)] = ((y0 + y1) / 2, (x0 + x1) / 2)
    return out


def build():
    """Полная панель: МО с полной историей 24 месяца по всем шести категориям. Возвращает (Y[кат, МО, мес], ids, months, meta)."""
    cons = pd.read_parquet(next(SRC.rglob("consumption.parquet")))
    cfg = config()["period"]
    months = pd.period_range(cfg["months_from"], cfg["months_to"], freq="M").astype(str)
    wide = cons.pivot_table(index=["territory_id", "date"], columns="category", values="value")
    wide = wide.reindex(columns=CATS)
    full = wide.dropna().groupby(level=0).size()
    ids = np.array(sorted(full[full == len(months)].index))
    Y = np.stack([wide[c].unstack(level=1).reindex(index=ids, columns=months).to_numpy(float) for c in CATS])
    meta = directory().reindex(ids)[["municipal_district_name", "municipal_district_name_short", "municipal_district_type",
                                     "region_code", "region_name", "oktmo", "municipal_district_center",
                                     "municipal_district_center_lat", "municipal_district_center_lon"]]
    meta.columns = ["name", "name_short", "mo_type", "region_code", "region_name", "oktmo", "center", "lat", "lon"]
    poly = _polygon_centers()
    no_xy = meta.lat.isna()
    meta.loc[no_xy, "lat"] = [poly[t][0] for t in meta.index[no_xy]]
    meta.loc[no_xy, "lon"] = [poly[t][1] for t in meta.index[no_xy]]
    meta["population"] = _population(meta)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED / "panel.npz", Y=Y, ids=ids, months=np.array(months), cats=np.array(CATS))
    meta.to_parquet(PROCESSED / "mo_meta.parquet")
    return Y, ids, months, meta


def _population(meta):
    """Население на 1 января 2024 (2023, если нет) по ОКТМО из БД ПМО Росстата."""
    p = population()
    p = p[p.mest == "Все население"].copy()
    p["v"] = pd.to_numeric(p.indicator_value, errors="coerce")
    p = p.dropna(subset=["v"]).sort_values("year").drop_duplicates("oktmo", keep="last").set_index("oktmo")["v"]
    return meta["oktmo"].map(lambda o: p.get(str(o).replace("-", "")[:8]) if isinstance(o, str) else None)


def load():
    z = np.load(PROCESSED / "panel.npz", allow_pickle=True)
    return z["Y"], list(z["ids"]), list(z["months"]), pd.read_parquet(PROCESSED / "mo_meta.parquet")
