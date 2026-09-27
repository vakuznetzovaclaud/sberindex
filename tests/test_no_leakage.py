"""Прогноз на момент T0 не должен меняться, если испортить всё, что стало известно после T0: траты МО всех категорий,
национальный ряд и текстовые признаки. Prophet (отдельная модель на каждый ряд МО, обучается только на ряде до T0) здесь
не проверяется: его прогон занимает минуты."""
import numpy as np
import pandas as pd
import pytest

from ews import models

N, T = 60, 24
MONTHS = [str(p) for p in pd.period_range("2023-01", "2024-12", freq="M")]


def _panel(seed=0):
    """Искусственная панель: траты трёх категорий с годовой сезонностью, национальный ряд, справочник МО, тексты."""
    g = np.random.default_rng(seed)
    ya = np.exp(10 + g.normal(0, 0.1, (3, N, T)) + np.sin(np.arange(T) / 12 * 2 * np.pi)[None, None, :] * 0.1)
    nat = pd.Series(np.exp(8 + g.normal(0, 0.05, 93)), index=[str(p) for p in pd.period_range("2018-12", periods=93, freq="M")])
    meta = pd.DataFrame({"population": g.integers(5_000, 500_000, N), "region_code": g.integers(1, 10, N)})
    tx = {"act_new": g.poisson(0.1, (N, T)).astype(float), "news_mo": g.poisson(0.5, (N, T)).astype(float)}
    return ya, nat, meta, tx


YA, NAT, META, TX = _panel()


@pytest.fixture
def rng():
    """Свой генератор в каждом тесте: результат не зависит от порядка и состава запущенных тестов."""
    return np.random.default_rng(0)


@pytest.mark.parametrize("fn", [models.snaive, models.snaive_growth, models.panel, lambda c: models.panel(c, use_national=False),
                                models.lgbm_correction, models.lgbm_global,
                                *[models.stat_on_seasonally_adjusted(k) for k in ("naive", "ses", "theta")]])
@pytest.mark.parametrize("T0", [12, 17, 22])
def test_future_does_not_leak(fn, T0, rng):
    c = models.Ctx(y=YA[1], T0=T0, H=[1, 2, 3], months=MONTHS, nat=NAT, meta=META, ya=YA, cat=1, tx=TX)
    before = fn(c)
    ya2, nat2 = YA.copy(), NAT.copy()
    ya2[:, :, T0 + 1:] *= rng.uniform(0.5, 2.0, ya2[:, :, T0 + 1:].shape)
    nat2[nat2.index > MONTHS[T0]] *= 3.0
    tx2 = {k: v.copy() for k, v in TX.items()}
    for v in tx2.values():
        v[:, T0 + 1:] = rng.poisson(3.0, v[:, T0 + 1:].shape)
    after = fn(models.Ctx(y=ya2[1], T0=T0, H=[1, 2, 3], months=MONTHS, nat=nat2, meta=META, ya=ya2, cat=1, tx=tx2))
    assert np.allclose(before, after)


@pytest.mark.parametrize("T0", [12, 17, 22])
def test_nowcast_text_uses_only_next_month_start(T0, rng):
    """Наукаст: признаки первых дней месяца T0 + 1 разрешены, всё, что позже, — нет."""
    txp = {"act_new": rng.poisson(0.1, (N, T)).astype(float)}
    c = models.Ctx(y=YA[1], T0=T0, H=[1, 2, 3], months=MONTHS, nat=NAT, meta=META, ya=YA, cat=1, tx=TX, txp=txp)
    before = models.lgbm_correction(c)
    txp2 = {k: v.copy() for k, v in txp.items()}
    for v in txp2.values():
        v[:, T0 + 2:] = rng.poisson(3.0, v[:, T0 + 2:].shape)
    after = models.lgbm_correction(models.Ctx(y=YA[1], T0=T0, H=[1, 2, 3], months=MONTHS, nat=NAT, meta=META, ya=YA, cat=1,
                                              tx=TX, txp=txp2))
    assert np.allclose(before, after)


class _FakeFM:
    """Подмена фундаментальной модели: запоминает всё, что ей подали на вход, и возвращает последнее значение ряда.
    Утечка в FM возможна только через вход (сами веса данных панели не видели), поэтому тест сравнивает входы."""
    def __init__(self):
        self.seen = []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        """Как у Chronos-2: список тензоров [варианты × горизонт × квантили], по одному на ряд."""
        import torch
        x = inputs.numpy()[:, 0, :]
        self.seen.append(x.copy())
        return [torch.tensor(np.repeat(r[None, -1:], prediction_length, 1))[:, :, None] for r in x], None

    def predict_df(self, df, prediction_length, quantile_levels, cross_learning=False, batch_size=100, future_df=None):
        df = df.sort_values(["item_id", "timestamp"])
        self.seen.append(df.drop(columns="timestamp").to_numpy().copy())
        rows = []
        for item, g in df.groupby("item_id"):
            ts = pd.date_range(g.timestamp.iloc[-1], periods=prediction_length + 1, freq="MS")[1:]
            rows.append(pd.DataFrame({"item_id": item, "timestamp": ts, "predictions": g.target.iloc[-1]}))
        return pd.concat(rows)

    def fit(self, inputs, prediction_length, **kw):
        self.seen.append(np.stack(inputs).copy())
        return self


class _FakeTimesFM:
    """Подмена TimesFM-2.5 с тем же вызовом forecast(horizon, inputs): запоминает вход, прогноз — медиана ряда."""
    def __init__(self):
        self.seen = []

    def forecast(self, horizon, inputs):
        x = np.stack(inputs)
        self.seen.append(x.copy())
        return np.repeat(np.median(x, 1, keepdims=True), horizon, 1), None


class _FakeTiRex:
    """Подмена TiRex-2 с тем же вызовом forecast(tss, prediction_length, output_type): запоминает вход, все девять
    квантилей — медиана ряда."""
    def __init__(self):
        self.seen = []

    def forecast(self, tss, prediction_length, output_type="numpy"):
        x = np.stack([ts.target.numpy()[0] for ts in tss])
        self.seen.append(x.copy())
        return [np.full((1, 9, prediction_length), np.median(r)) for r in x]


FAKES = {"_chronos2": _FakeFM, "_timesfm25": _FakeTimesFM, "_tirex2": _FakeTiRex}


def _fm_models():
    """Имя варианта -> (загрузчик весов, который подменяется, функция прогноза)."""
    from ews import foundation as fd
    out = {"chronos2_deviation": ("_chronos2", fd.fm_on_deviation("chronos2")), "chronos2_raw": ("_chronos2", fd.fm_on_raw("chronos2")),
           "chronos2_sa": ("_chronos2", fd.fm_on_seasonally_adjusted("chronos2")),
           "chronos2_sa_cross": ("_chronos2", fd.fm_on_seasonally_adjusted("chronos2", cross=True)),
           "chronos2_sa_covariates": ("_chronos2", fd.chronos2_sa_covariates()),
           "chronos2_sa_covariates_cross": ("_chronos2", fd.chronos2_sa_covariates(cross=True)),
           "chronos2_sa_finetuned": ("_chronos2", fd.chronos2_finetuned_sa(steps=1))}
    for name, loader in (("timesfm25", "_timesfm25"), ("tirex2", "_tirex2")):
        out |= {f"{name}_deviation": (loader, fd.fm_on_deviation(name)), f"{name}_raw": (loader, fd.fm_on_raw(name)),
                f"{name}_sa": (loader, fd.fm_on_seasonally_adjusted(name))}
    return out


@pytest.mark.parametrize("name", list(_fm_models()))
@pytest.mark.parametrize("T0", [12, 17, 22])
def test_foundation_inputs_do_not_leak(name, T0, monkeypatch, rng):
    from ews import foundation as fd
    loader, fn = _fm_models()[name]
    runs = []
    for spoil in (False, True):
        fake = FAKES[loader]()
        monkeypatch.setattr(fd, loader, lambda: fake)
        ya, nat = YA.copy(), NAT.copy()
        if spoil:
            ya[:, :, T0 + 1:] *= rng.uniform(0.5, 2.0, ya[:, :, T0 + 1:].shape)
            nat[nat.index > MONTHS[T0]] *= 3.0
        out = fn(models.Ctx(y=ya[1], T0=T0, H=[1, 2, 3], months=MONTHS, nat=nat, meta=META, ya=ya, cat=1))
        runs.append((fake.seen, out))
    (seen_a, out_a), (seen_b, out_b) = runs
    assert len(seen_a) == len(seen_b) > 0
    assert all(np.allclose(a, b) for a, b in zip(seen_a, seen_b))
    assert np.allclose(out_a, out_b)


@pytest.mark.parametrize("T0", [15, 19, 22])
def test_detector_input_does_not_leak(T0, rng):
    """Вход детекторов: z месяца t (остаток одношагового прогноза, нормированный по прошлому ряда МО и очищенный от
    общей ошибки месяца) не зависит от трат и национального ряда после t. Контроль: при нормировке по всему ряду
    (online=False) z зависит от будущего, и последняя проверка это ловит."""
    from ews import synth
    ya, nat = YA.copy(), [NAT.copy() for _ in range(3)]
    ya2, nat2 = ya.copy(), [s.copy() for s in nat]
    ya2[:, :, T0 + 1:] *= rng.uniform(0.5, 2.0, ya2[:, :, T0 + 1:].shape)
    for s in nat2:
        s[s.index > MONTHS[T0]] *= 3.0
    a, b = synth.h1_residuals(ya, MONTHS, nat), synth.h1_residuals(ya2, MONTHS, nat2)
    assert np.isfinite(a[:, :, 13:T0 + 1]).all()
    assert np.allclose(a[:, :, :T0 + 1], b[:, :, :T0 + 1], equal_nan=True)
    full_a, full_b = synth.h1_residuals(ya, MONTHS, nat, online=False), synth.h1_residuals(ya2, MONTHS, nat2, online=False)
    assert not np.allclose(full_a[:, :, :T0 + 1], full_b[:, :, :T0 + 1], equal_nan=True)


@pytest.mark.parametrize("T0", [15, 19, 22])
def test_detectors_do_not_leak(T0):
    """Выход детекторов: оценка каждого детектора и обоих ансамблей в месяцы до T0 включительно не меняется, если
    испортить z после T0."""
    from ews import detect
    g = np.random.default_rng(1)
    z = g.normal(0, 1, (6, N, T))
    z[:, :, :13] = np.nan
    knn = np.array([np.r_[i, g.permutation(np.delete(np.arange(N), i))[:8]] for i in range(N)])
    z2 = z.copy()
    z2[:, :, T0 + 1:] = g.normal(-3, 2, z2[:, :, T0 + 1:].shape)
    a, b = detect.all_detectors(z, knn), detect.all_detectors(z2, knn)
    for name in a:
        assert np.allclose(a[name][:, :T0 + 1], b[name][:, :T0 + 1], equal_nan=True), name


def test_text_features_date_cutoff(monkeypatch):
    """Текстовые признаки месяца собираются только из текстов, опубликованных в этом месяце, а в режиме наукаста
    (day=15) — только до 15-го числа: акт от 25 марта и новость от 20 марта в признаки «по 15-е» не попадают,
    а тексты марта не попадают в признаки февраля."""
    from ews import text_features as tf
    ids = np.arange(100, 100 + N)
    meta = pd.DataFrame({"region_code": np.where(np.arange(N) < N // 2, 1, 2), "population": 1.0}, index=ids)
    ev = pd.DataFrame({"territory_id": ids[:3], "pub_date": ["2024-03-05", "2024-03-25", "2024-04-02"], "cause": tf.RELEVANT[0],
                       "action": "введение", "old": False, "scope": "перечень", "people": 0})
    links = pd.DataFrame({"region_code": [1, 1], "id": [1, 2], "dt": ["2024-03-10T10:00", "2024-03-20T10:00"], "territory_id": ids[3:5]})
    monkeypatch.setattr(tf.pnl, "load", lambda: (YA, ids, MONTHS, meta))
    monkeypatch.setattr(tf.pd, "read_parquet", lambda *a, **k: ev)
    monkeypatch.setattr(tf, "news_links", lambda rebuild=False, source="llm": links)
    monkeypatch.setattr(tf, "covered_regions", lambda: {1, 2})
    full, mid = tf.monthly(), tf.monthly(day=15)
    mar = MONTHS.index("2024-03")
    assert full["act_new"][:, mar].sum() == 2 and mid["act_new"][:, mar].sum() == 1
    assert full["news_mo"][:, mar].sum() == 2 and mid["news_mo"][:, mar].sum() == 1
    assert full["act_new"][:, :mar].sum() == 0 and full["news_mo"][:, :mar].sum() == 0


ORIGINS = list(range(12, 23))                     # точки прогноза: последний наблюдённый месяц — январь…ноябрь 2024 г.


def _member_forecasts(g):
    """Прогнозы трёх участников (разная точность) по всем точкам и горизонтам 1–3 с целью не позже декабря 2024 г."""
    base = pd.DataFrame([(i, o, h, o + h) for o in ORIGINS for h in (1, 2, 3) if o + h < T for i in range(N)],
                        columns=["i", "origin", "h", "target"])
    y = YA[0][base.i, base.target]
    return {m: base.assign(yhat=y * g.uniform(1 - s, 1 + s, len(base))) for m, s in (("m1", 0.05), ("m2", 0.15), ("m3", 0.10))}


@pytest.mark.parametrize("how", ["ensemble_online", "select_online"])
@pytest.mark.parametrize("T0", [14, 18, 21])
def test_online_weights_do_not_leak(how, T0, monkeypatch, tmp_path, rng):
    """Ансамбль с весами по прошлой ошибке и онлайн-подбор настроек Prophet: прогнозы точек до T0 включительно не
    меняются, если испортить траты после T0 и прогнозы участников из более поздних точек."""
    from ews import backtest, baselines, evaluate as ev, panel as pnl
    monkeypatch.setattr(ev, "DIR", tmp_path)                     # прогнозы пишутся во временную папку, не в outputs/
    monkeypatch.setattr(backtest, "DIR", tmp_path)
    runs = []
    for spoil in (False, True):
        ya, P = YA.copy(), _member_forecasts(np.random.default_rng(2))
        if spoil:
            ya[:, :, T0 + 1:] *= rng.uniform(0.5, 2.0, ya[:, :, T0 + 1:].shape)
            for p in P.values():
                late = p.origin > T0
                p.loc[late, "yhat"] *= rng.uniform(0.5, 2.0, late.sum())
        monkeypatch.setattr(pnl, "load", lambda ya=ya: (ya, np.arange(N), MONTHS, META))
        for m, p in P.items():
            p.to_parquet(tmp_path / f"{m}__c0.parquet")
        (ev.ensemble_online if how == "ensemble_online" else baselines.select_online)(list(P), 0, "out")
        runs.append(pd.read_parquet(tmp_path / "out__c0.parquet").sort_values(["origin", "h", "i"]).reset_index(drop=True))
    a, b = runs
    early = a.origin <= T0
    pd.testing.assert_frame_equal(a[early], b[early])
    assert not np.allclose(a.loc[~early, "yhat"], b.loc[~early, "yhat"])          # порча после T0 действительно была


@pytest.mark.parametrize("name", ["lstm", "tcn", "patchtst"])
@pytest.mark.parametrize("T0", [12, 22])
def test_timecast_models_do_not_leak(name, T0, rng, monkeypatch):
    """Нейросети TimeCast обучаются на окнах истории до T0: порча данных после T0 прогноз не меняет (2 эпохи — для скорости)."""
    from ews import neural
    try:
        neural._timecast()
    except Exception:
        pytest.skip("код TimeCast недоступен (нужна сеть для первой загрузки)")
    monkeypatch.setattr(neural, "EPOCHS", 2)
    fn = neural.timecast_on_seasonally_adjusted(name)
    c = models.Ctx(y=YA[1], T0=T0, H=[1, 2, 3], months=MONTHS, nat=NAT, meta=META, ya=YA, cat=1)
    before = fn(c)
    ya2 = YA.copy()
    ya2[:, :, T0 + 1:] *= rng.uniform(0.5, 2.0, ya2[:, :, T0 + 1:].shape)
    after = fn(models.Ctx(y=ya2[1], T0=T0, H=[1, 2, 3], months=MONTHS, nat=NAT, meta=META, ya=ya2, cat=1))
    assert np.allclose(before, after)
