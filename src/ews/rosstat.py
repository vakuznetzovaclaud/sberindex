"""Численность населения МО из базы данных показателей муниципальных образований Росстата (каталог «Если быть точным»,
CC BY 4.0). Архивы разделов весят гигабайты, поэтому zip читается прямо по сети: объект-файл с HTTP Range запросами
позволяет стандартному zipfile прочитать оглавление и нужный член архива, не скачивая остальное."""
import hashlib
import io
import urllib.request
import zipfile

import pandas as pd

from .paths import RAW, config
from .web import get

POPULATION = (31, "Y48112027")          # оценка численности постоянного населения на 1 января


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, block=1 << 20):
        self.url, self.pos, self.block, self.cache = url, 0, block, {}
        req = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _block(self, k):
        if k not in self.cache:
            a, b = k * self.block, min(self.size, (k + 1) * self.block) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"})
            self.cache = {k: urllib.request.urlopen(req, timeout=180).read()} | {x: v for x, v in list(self.cache.items())[-3:]}
        return self.cache[k]

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            k, off = divmod(self.pos, self.block)
            chunk = self._block(k)[off:off + n]
            out += chunk; self.pos += len(chunk); n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b)); b[:len(data)] = data
        return len(data)


def population(years=("2023", "2024")):
    out = RAW / "rosstat" / "population.parquet"
    if out.exists():
        return pd.read_parquet(out)
    section, code = POPULATION
    z = zipfile.ZipFile(HTTPRangeFile(config()["sources"]["bdmo"].format(section=section)))
    member = next(n for n in z.namelist() if code in n)
    rows = []
    with z.open(member) as fh:
        header = None
        for line in io.TextIOWrapper(fh, encoding="utf-8"):
            f = line.rstrip("\n").split(";")
            if header is None:
                header = f; continue
            r = dict(zip(header, f))
            if r.get("year") in years:
                rows.append(r)
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return df


def oktmo():
    """Классификатор ОКТМО (портал открытых данных Росстата), раздел 2 — населённые пункты муниципальных образований.
    Версия закреплена адресом и контрольной суммой в configs/base.yaml: классификатор обновляется каждый месяц."""
    out = RAW / "rosstat" / "oktmo.csv"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(get(config()["sources"]["oktmo"], timeout=300, binary=True))
    if hashlib.sha256(out.read_bytes()).hexdigest() != config()["sources"]["oktmo_sha256"]:
        raise RuntimeError(f"{out}: контрольная сумма не совпадает с configs/base.yaml")
    return out
