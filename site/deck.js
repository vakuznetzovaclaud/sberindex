// Графики презентации site/deck.html: те же функции, что на странице (site/charts.js), и те же данные DATA.
(function () {
  var $ = CH.$, C = CH.C, bars = CH.bars, lines = CH.lines, table = CH.table, fmt = CH.fmt;
  var F = DATA.forecast, all = F.mae[F.categories[0]][F.windows[0]];
  var keep = ["ens_main", "ens_fm_sa", "timesfm25_sa", "tirex2_sa", "chronos2_sa_cross", "lgbm", "panel", "snaive_growth", "prophet_default"];
  var mt = DATA.forecast.table[F.categories[0]], pick = ["ансамбль", "TimesFM-2.5*", "наивный + прирост г/г", "Prophet"];
  table($("#d-metrics"), pick.map(function (n) { return mt.filter(function (r) { return r.name === n; })[0]; }).filter(Boolean),
    [{k: "name", t: "Модель"}, {k: "MAE", t: "MAE", n: 1, f: function (v) { return fmt(v); }}, {k: "MAPE, %", t: "MAPE, %", n: 1, f: function (v) { return fmt(v, 2); }},
     {k: "WAPE, %", t: "WAPE, %", n: 1, f: function (v) { return fmt(v, 2); }}, {k: "MASE", t: "MASE", n: 1, f: function (v) { return fmt(v, 2); }},
     {k: "R² г/г", t: "R² г/г", n: 1, f: function (v) { return fmt(v, 2); }}]);
  bars($("#d-mae"), all.filter(function (r) { return keep.indexOf(r.model) >= 0; }).map(function (r) {
    return {label: r.name, value: r.mae, bold: r.family === "ens", color: CH.famColor(r.family), red: r.family === "prophet"}; }), {unit: " ₽"});
  var B = F.bycat.MAE, short = CH.short;
  table($("#d-bycat"), B.rows, [{k: "name", t: "Модель"}].concat(F.categories.map(function (c) { return {k: c, t: short[c] || c, n: 1,
    f: function (v) { return v === B.best[c] ? "<b>" + fmt(v) + "</b>" : fmt(v); }}; })));
  bars($("#d-fm"), DATA.fm.inputs.map(function (r) { return {label: r.label, value: r.mae, color: r.good ? C.blue : null, bold: r.good}; }), {unit: " ₽"});
  var S = DATA.detect.synthetic[DATA.detect.shapes[0]].slice().sort(function (a, b) { return b["VUS-PR"] - a["VUS-PR"]; }).slice(0, 9);
  bars($("#d-det"), S.map(function (r) { return {label: r.det, value: r["VUS-PR"], color: CH.famColor(r.family),
    bold: r.family === "sel"}; }), {digits: 3});
  var cpal = [C.pink, C.blue, C.ink, C.green, C.mu, C.ink], cs = DATA.detect.curve.series[Object.keys(DATA.detect.curve.series)[0]];
  lines($("#d-curve"), DATA.detect.curve.levels, cs.map(function (s, i) { return {name: s.name, values: s.values, color: cpal[i], bold: i === 0, dash: i === 5}; }), {digits: 2, height: 300});
  var mp = DATA.text.dose.filter(function (r) { return r["Маркетплейсы"].med != null; });
  bars($("#d-dose"), mp.map(function (r) { return {label: r.group.replace("акты с установленной причиной", "все акты с причиной") + " (" + r.n + ")", value: r["Маркетплейсы"].med,
    color: r["Маркетплейсы"].p != null && r["Маркетплейсы"].p < 0.05 ? C.yellow : null, bold: r["Маркетплейсы"].p != null && r["Маркетплейсы"].p < 0.05}; }), {signed: true, digits: 1, unit: "%"});
  bars($("#d-flood"), DATA["case"].by_source.map(function (b) { return {label: b.label, value: b.value, color: b.label.indexOf("сайтах") > 0 ? C.green : null,
    bold: b.label.indexOf("сайтах") > 0}; }), {signed: true, digits: 1, unit: "%"});
  var bk = DATA.detect.breaks, bpal = [C.ink, C.blue, C.pink, C.green, C.mu, C.ink];
  lines($("#d-breaks"), bk.labels, bk.series.map(function (s, i) { return {name: s.name, values: s.values, color: bpal[i], bold: i === 0, dash: i === 5}; }), {lo: 0, height: 170});
  // скользящее начало: строки — последний наблюдённый месяц, клетки — цели на 1–3 месяца вперёд
  var mon = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"], cw = 48, rh = 22, x0 = 196, y0 = 34, s = "";
  mon.forEach(function (m, k) { s += '<text x="' + (x0 + k * cw + cw / 2) + '" y="22" text-anchor="middle" font-size="13" fill="#6d6d6d">' + m + '</text>'; });
  for (var r = 0; r < 11; r++) {
    var y = y0 + r * rh + (r >= 6 ? 12 : 0);
    s += '<text x="' + (x0 - 12) + '" y="' + (y + 15) + '" text-anchor="end" font-size="13">' + mon[r] + ' 2024</text>';
    for (var k = 0; k < 12; k++) {
      var h = k - r, fill = h === 0 ? C.ink : h >= 1 && h <= 3 ? ["", "#7F9FFF", "#7F9FFFaa", "#7F9FFF66"][h] : k < r ? "#12121214" : "none";
      s += '<rect x="' + (x0 + k * cw + 2) + '" y="' + (y + 2) + '" width="' + (cw - 4) + '" height="' + (rh - 4) + '" fill="' + fill + '" stroke="' + (fill === "none" ? "#1212121f" : "none") + '"/>';
    }
  }
  s += '<text x="8" y="' + (y0 + 3 * rh + 4) + '" font-size="12" font-weight="800" fill="#6d6d6d">РАЗРАБОТКА</text><text x="8" y="' + (y0 + 8.5 * rh + 16) + '" font-size="12" font-weight="800" fill="#6d6d6d">КОНТРОЛЬ</text>';
  s += '<rect x="' + x0 + '" y="' + (y0 + 11 * rh + 22) + '" width="14" height="14" fill="#121212"/><text x="' + (x0 + 20) + '" y="' + (y0 + 11 * rh + 34) + '" font-size="13">последний наблюдённый месяц</text>';
  s += '<rect x="' + (x0 + 250) + '" y="' + (y0 + 11 * rh + 22) + '" width="14" height="14" fill="#7F9FFF"/><text x="' + (x0 + 270) + '" y="' + (y0 + 11 * rh + 34) + '" font-size="13">прогноз на 1, 2, 3 месяца вперёд</text>';
  $("#d-roll").innerHTML = '<svg viewBox="0 0 ' + (x0 + 12 * cw + 4) + ' ' + (y0 + 11 * rh + 46) + '" width="100%" font-family="Nunito Sans, Arial, sans-serif">' + s + '</svg>';
})();
