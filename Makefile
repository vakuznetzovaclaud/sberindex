# Полный прогон от открытых данных до таблиц. Долгие шаги кэшируются: повторный запуск берёт уже скачанное и размеченное.
# Окружение: python -m venv ~/.venvs/sberindex-prognoz && pip install -r requirements.txt && pip install -e .;
# для разметки текста — Ollama с моделью из configs/base.yaml (llm.model).
PY ?= ~/.venvs/sberindex-prognoz/bin/python
CATS := 0 1 2 3 4 5
FAST := snaive snaive_growth panel_factor_panel panel lgbm lgbm_global naive_sa ses_sa theta_sa
TEXT := lgbm_text lgbm_nowcast_text lgbm_text_regex
FM := tirex2 timesfm25_sa tirex2_sa chronos2_sa chronos2_sa_cross chronos2_sa_cov chronos2_sa_cov_cross
FM_EXTRA := tirex2_raw timesfm25 chronos2_sa_ft chronos2_sa_ft_lr1e5
TIMECAST := lstm_sa patchtst_sa           # нейросети библиотеки TimeCast, глобально по всем МО
PROPHET := prophet_default prophet_timecast prophet_g0 prophet_g1 prophet_g2 prophet_g3 prophet_g4 prophet_g5 prophet_g6 prophet_g7

.PHONY: all data texts forecasts prophet evaluate site test

all: data texts forecasts prophet evaluate site

data:                       ## панель МО, справочник, население, национальные ряды портала
	$(PY) -c "from ews import panel, rosstat; panel.download(); rosstat.oktmo(); panel.build()"
	$(PY) -c "from ews import national; national.download('consumer-spending'); national.download('ver-izmenenie-trat-po-kategoriyam')"

texts:                      ## акты о ЧС (реестр, PDF, OCR, извлечение фактов), новости МЧС, Telegram для паводка и официальные каналы регионов
	$(PY) scripts/build_acts.py ocr
	$(PY) scripts/build_acts.py extract 3
	$(PY) -c "from ews import events; events.build()"
	$(PY) scripts/build_mchs.py 4
	$(PY) -c "from ews import news_classify; news_classify.label_all(3); news_classify.fetch_bodies()"
	$(PY) -c "from ews import news_classify as nc, news_telegram as nt; nc.label_posts(nt.flood_posts(), name='tg_flood', pattern=nc.FLOOD_WORDS, until='2024-04-30')"
	$(PY) -c "from ews import news_official; news_official.collect_all()"

forecasts:                  ## прогнозный стенд: 11 точек 2024 г. × h 1–3 × 2 016 МО × 6 категорий
	for c in $(CATS); do $(PY) scripts/run_forecasts.py $$c $(FAST) $(TEXT) $(FM) $(TIMECAST) || exit 1; done
	$(PY) scripts/run_forecasts.py 0 $(FM_EXTRA) tcn_sa
	$(PY) scripts/run_forecasts.py 1 timesfm25

prophet:                    ## Prophet: по умолчанию на всех МО, TimeCast и сетка настроек на выборке 400 МО
	for c in $(CATS); do $(PY) scripts/run_forecasts.py $$c $(PROPHET) && $(PY) scripts/run_forecasts.py $$c prophet_tuned || exit 1; done

evaluate:                   ## таблицы, рисунки, отчёт
	$(PY) scripts/eval_forecasts.py
	$(PY) scripts/eval_lambda.py
	$(PY) scripts/eval_text_value.py
	$(PY) scripts/eval_event_adjustment.py ens_main llm
	$(PY) scripts/eval_detectors.py 3
	$(PY) scripts/eval_detectors_real.py llm
	$(PY) scripts/offline_breaks.py
	$(PY) scripts/weekly_polygon.py
	$(PY) scripts/dose_response.py lgbm
	$(PY) scripts/dose_response.py lgbm z
	$(PY) scripts/eval_warning.py llm
	$(PY) scripts/case_flood2024.py
	$(PY) scripts/eval_detectors_flood.py
	$(PY) scripts/eval_events_2024.py
	$(PY) scripts/eval_alarm_clusters.py
	$(PY) scripts/figures.py
	$(PY) scripts/build_report.py

site:                       ## страница-презентация и её PDF-версия (для PDF нужен Chrome)
	$(PY) scripts/build_site.py
	$(PY) scripts/print_landing.py

test:
	$(PY) -m pytest -q tests
