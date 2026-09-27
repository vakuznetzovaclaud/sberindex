"""Загрузка по HTTP через curl с повторами: сайты госорганов часто отвечают медленно или обрывают соединение."""
import os
import shutil
import subprocess
import time
from pathlib import Path


def get(url, params=None, timeout=60, tries=5, binary=False, insecure=False):
    cmd = ["curl", "-s", "-L", "-m", str(timeout), "-A", "Mozilla/5.0"] + (["-k"] if insecure else [])
    if params:
        cmd += ["-G", url] + sum([["--data-urlencode", f"{k}={v}"] for k, v in params.items()], [])
    else:
        cmd += [url]
    for k in range(tries):
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0 and r.stdout:
            return r.stdout if binary else r.stdout.decode("utf-8", "ignore")
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"не удалось загрузить {url}")


def chrome():
    """Chrome или Chromium для печати HTML в PDF: переменная CHROME, стандартные пути macOS, затем команды в PATH."""
    for c in [os.environ.get("CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "/Applications/Chromium.app/Contents/MacOS/Chromium", "google-chrome", "google-chrome-stable", "chromium",
              "chromium-browser"]:
        if c and (Path(c).exists() or shutil.which(c)):
            return c
    raise RuntimeError("Chrome или Chromium не найден: укажите путь в переменной CHROME")
