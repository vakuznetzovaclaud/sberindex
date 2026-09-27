"""Нейросетевые модели библиотеки TimeCast (LSTM, TCN, PatchTST) на том же входе, что у фундаментальных моделей:
отклонение МО от общего фактора без сезонного профиля 2023 г. На 13–23 точках одного ряда такие модели не обучить,
поэтому модель одна на все МО: окна длиной WINDOW из истории до T0 включительно, цель — значение через h месяцев,
по модели на горизонт. Окно центрируется своим средним (уровень МО модель не запоминает). Архитектуры, функция потерь
(MSE) и оптимизатор со своим шагом — из TimeCast; меняется только длина патча PatchTST под короткое окно.
Код библиотеки не копируется: он скачивается с GitHub на закреплённом коммите (configs/base.yaml, sources.timecast)."""
import functools
import subprocess
import sys

import numpy as np

from .models import Ctx, _decompose, factor_forecast, sa_deviation
from .paths import RAW, config

WINDOW, EPOCHS, BATCH = 6, 30, 256
DIR = RAW / "timecast"


@functools.cache
def _timecast():
    """Каталог моделей TimeCast на закреплённом коммите (скачивается один раз) — в пути импорта."""
    src = config()["sources"]["timecast"]
    if not (DIR / ".git").exists():
        subprocess.run(["git", "clone", "-q", src["url"], str(DIR)], check=True)
    head = subprocess.run(["git", "-C", str(DIR), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != src["commit"]:
        subprocess.run(["git", "-C", str(DIR), "checkout", "-q", src["commit"]], check=True)
    sys.path.insert(0, str(DIR))


def _model(name):
    _timecast()
    if name == "lstm":
        from models.lstm import LSTM
        return LSTM(input_size=1)
    if name == "tcn":
        from models.tcn import TCN
        return TCN(input_size=1)
    if name == "patchtst":
        from models.patch_tst import PatchTST
        return PatchTST(input_size=1, patch_len=3)
    raise ValueError(name)


def windows(usa, h):
    """Обучающие окна по всем МО: вход — WINDOW точек до t, цель — точка t + h; всё в пределах известной истории."""
    X, y = [], []
    for t in range(WINDOW, usa.shape[1] - h + 1):
        x = usa[:, t - WINDOW:t]
        m = x.mean(1, keepdims=True)
        X.append(x - m)
        y.append(usa[:, t + h - 1] - m[:, 0])
    return np.concatenate(X), np.concatenate(y)


def fit_predict(name, usa, h, seed=42):
    """Обучает модель на окнах истории и прогнозирует отклонение каждого МО через h месяцев от последнего окна."""
    import torch
    torch.manual_seed(seed)
    X, y = windows(usa, h)
    net = _model(name)
    opt = net.configure_optimizers()
    loss = torch.nn.MSELoss()
    Xt, yt = torch.tensor(X[..., None], dtype=torch.float32), torch.tensor(y[:, None], dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(Xt), generator=g).split(BATCH):
            opt.zero_grad()
            loss(net(Xt[idx]), yt[idx]).backward()
            opt.step()
    net.eval()
    last = usa[:, -WINDOW:]
    m = last.mean(1, keepdims=True)
    with torch.no_grad():
        return net(torch.tensor((last - m)[..., None], dtype=torch.float32)).numpy()[:, 0] + m[:, 0]


def timecast_on_seasonally_adjusted(name, lam=0.7):
    """Модель TimeCast name на отклонении МО без сезонности; сборка прогноза — как у FM (fm_on_seasonally_adjusted)."""
    def f(c: Ctx):
        F, U = _decompose(c)
        usa, s = sa_deviation(U, c.T0, lam)
        fh = factor_forecast(c)
        return np.stack([np.exp(fh[j] + fit_predict(name, usa, h) + s[:, (c.T0 + h) % 12]) for j, h in enumerate(c.H)], 1)
    return f
