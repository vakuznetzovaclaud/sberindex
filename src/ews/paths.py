"""Пути и конфигурация проекта."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
(OUTPUTS / "tables").mkdir(parents=True, exist_ok=True)   # таблицы скриптов оценки


def config(name="base"):
    return yaml.safe_load(open(ROOT / "configs" / f"{name}.yaml", encoding="utf-8"))
