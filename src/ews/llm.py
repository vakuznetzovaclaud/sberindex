"""Локальная открытая языковая модель через Ollama. Используется только для извлечения фактов из текста в заданную
JSON-схему; модель ничего не прогнозирует и не дополняет по памяти (защита от заглядывания вперёд)."""
import functools
import json
import urllib.request

from .paths import config


@functools.cache
def check_model():
    """Файл модели в Ollama должен совпадать с закреплённым в configs/base.yaml (llm.digest): тег в реестре может
    указывать на пересобранный файл, и разметка тогда тихо изменится."""
    c = config()["llm"]
    tags = json.loads(urllib.request.urlopen(c["endpoint"].replace("/api/chat", "/api/tags"), timeout=30).read())
    digest = {m["name"]: m["digest"] for m in tags["models"]}.get(c["model"])
    if digest != c["digest"]:
        raise RuntimeError(f"модель {c['model']}: файл {digest} не совпадает с закреплённым {c['digest']}")


def ask(prompt, schema, num_ctx=8192, timeout=600):
    check_model()
    c = config()["llm"]
    body = {"model": c["model"], "stream": False, "format": schema, "options": {"temperature": c["temperature"], "num_ctx": num_ctx},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(c["endpoint"], data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(json.loads(urllib.request.urlopen(req, timeout=timeout).read())["message"]["content"])
