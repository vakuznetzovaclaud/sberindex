// Лендинг «Где искать шок»: данные DATA собирает scripts/build_site.py из итоговых таблиц. Графики — собственный SVG.
(function () {
  var $ = CH.$, C = CH.C, fmt = CH.fmt, W = CH.W, bars = CH.bars, lines = CH.lines, table = CH.table, buttons = CH.buttons, bindTips = CH.bindTips;

  // ---------- содержание и тикер ----------
  function renderTOC() {
    var items = [["01", "Прогноз", "Ошибка вдвое меньше Prophet", "m01", "blue"], ["02", "Прогноз", "Фундаментальные модели и подготовка входа", "m02", "blue"],
      ["03", "Обнаружение", "Детекторы на искусственных шоках", "m03", "pink"], ["04", "Обнаружение", "Массовые сдвиги в реальных данных", "m04", "pink"],
      ["05", "Текст", "378 актов — " + DATA.nums.n_events + " событий с датой и списком районов", "m05", "yellow"], ["06", "Текст", "Режим ЧС и траты на маркетплейсах", "m06", "yellow"],
      ["07", "Система", "Лента тревог 2024 года", "m07", "green"], ["08", "Система", "Паводок 2024: текст назвал районы, которые детекторы пропустили", "m08", "green"], ["09", "Система", "Повторить одной командой", "m09", "yellow"]];
    $("#toc").outerHTML = items.map(function (t, i) { return '<li class="' + t[4] + '" style="--d:' + i + '"><a href="#' + t[3] + '"><span class="n">' + t[0] + '</span><span class="r">' + t[1] + '</span><span class="t">' + t[2] + '</span><span class="ar">→</span></a></li>'; }).join("");
  }
  function renderTicker(N) {
    var one = '<span>ошибка на <u>' + N.vs_prophet + '%</u> ниже Prophet</span><i>✦</i><span>точнее Prophet в <em>' + N.better_share + '%</em> муниципалитетов</span><i>✦</i><span><b>' + N.sel_rec + '%</b> искусственных шоков ловит детектор</span><i>✦</i><span><b>378</b> актов о ЧС</span><i>✦</i><span><u>87 тысяч</u> новостей МЧС</span><i>✦</i><span><b>' + N.fl_site_mp_med + '%</b> к прогнозу — покупки в районах, названных МЧС в паводок</span><i>✦</i>';
    // две одинаковые половины: сдвиг на ровно половину ширины даёт бесшовный повтор; строка короче предела ширины
    // анимируемого слоя в браузерах (около 8 тыс. пикселей), иначе часть слов не рисуется
    $("#ticker").innerHTML = one + one;
  }

  // ---------- прогноз ----------
  function renderForecast(D) {
    var cat = D.categories[0], win = D.windows[0];
    var draw = function () {
      var rows = (D.mae[cat][win] || []).map(function (r) { return {label: r.name, value: r.mae, bold: r.family === "ens",
        color: CH.famColor(r.family), red: r.family === "prophet", note: r.name + "<br>MAE " + fmt(r.mae) + " руб."}; });
      bars($("#fc-bars"), rows, {unit: " ₽"});
      table($("#fc-table"), D.table[cat], D.table_cols.map(function (c) { return {k: c.k, t: c.t, n: c.n, f: c.n ? function (v) { return fmt(v, c.d); } : null}; }));
    };
    buttons($("#fc-cat"), D.categories, cat, function (v) { cat = v; draw(); });
    buttons($("#fc-win"), D.windows, win, function (v) { win = v; draw(); });
    draw();
    var bm = Object.keys(D.bycat)[0], short = CH.short;
    var drawB = function () { var B = D.bycat[bm], dg = bm === "MAE" ? 0 : 2;
      table($("#fc-bycat"), B.rows, [{k: "name", t: "Модель"}].concat(D.categories.map(function (c) { return {k: c, t: short[c] || c, n: 1,
        f: function (v) { return v === B.best[c] ? "<b>" + fmt(v, dg) + "</b>" : fmt(v, dg); }}; }))); };
    buttons($("#fc-bm"), Object.keys(D.bycat), bm, function (v) { bm = v; drawB(); }); drawB();
    var pal = [C.mu, C.pink, C.ink, C.green, C.blue];
    lines($("#fc-month"), D.month.labels, D.month.series.map(function (s, i) { return {name: s.name, values: s.values, bold: s.bold, color: s.bold ? C.blue : pal[i % pal.length]}; }));
    table($("#fc-prophet"), D.prophet, [{k: "cat", t: "Категория"}, {k: "default", t: "Prophet", n: 1, f: function (v) { return fmt(v); }},
      {k: "timecast", t: "Prophet (TimeCast)", n: 1, f: function (v) { return fmt(v); }}, {k: "tuned", t: "Prophet (подбор)", n: 1, f: function (v, r) { return r.best === "tuned" ? "<b>" + fmt(v) + "</b>" : fmt(v); }},
      {k: "ens", t: "Ансамбль", n: 1, f: function (v, r) { return r.best === "ens" ? "<b>" + fmt(v) + "</b>" : fmt(v); }}]);
  }

  function renderFM(D) {
    bars($("#fm-in"), D.inputs.map(function (r) { return {label: r.label, value: r.mae, color: r.good ? C.blue : null, bold: r.good}; }), {unit: " ₽"});
    var pal = [C.mu, C.pink, C.blue, C.green];
    lines($("#fm-ctx"), D.context.labels, D.context.series.map(function (s, i) { return {name: s.name, values: s.values, color: pal[i % 4], bold: i >= 2}; }), {digits: 2, ref: 1});
  }

  // ---------- обнаружение ----------
  function renderDetect(D) {
    var m = D.metrics[0], sh = D.shapes[0];
    var draw = function () { bars($("#dt-bars"), D.synthetic[sh].map(function (r) { return {label: r.det, value: r[m], color: CH.famColor(r.family),
      bold: r.family === "sel", note: r.det + "<br>VUS-PR " + fmt(r["VUS-PR"], 3) + " · полнота@3% " + fmt(r["полнота@3%"], 3) + " · NAB " + fmt(r.NAB, 1)}; })
      .sort(function (a, b) { return b.value - a.value; }), {digits: m === "NAB" ? 0 : 3}); };
    buttons($("#dt-sh"), D.shapes, sh, function (v) { sh = v; draw(); });
    buttons($("#dt-m"), D.metrics, m, function (v) { m = v; draw(); }); draw();
    var cm = Object.keys(D.curve.series)[0], cpal = [C.pink, C.blue, C.ink, C.green, C.mu, C.ink];
    var drawC = function () { lines($("#dt-curve"), D.curve.levels, D.curve.series[cm].map(function (s, i) { return {name: s.name, values: s.values, color: cpal[i], bold: i === 0, dash: i === 5}; }), {digits: 2}); };
    buttons($("#dt-cm"), Object.keys(D.curve.series), cm, function (v) { cm = v; drawC(); }); drawC();
    bars($("#dt-text"), D.text.map(function (r) { return {label: r.label, value: r.vus, color: r.label.indexOf("без текста") === 0 ? null : C.pink}; }).sort(function (a, b) { return a.value - b.value; }), {digits: 3});
    var pal = [C.ink, C.blue, C.pink, C.green, C.mu, C.ink];
    lines($("#dt-breaks"), D.breaks.labels, D.breaks.series.map(function (s, i) { return {name: s.name, values: s.values, color: pal[i % pal.length], bold: i === 0, dash: i === 5}; }), {lo: 0});
    table($("#dt-week"), D.weekly, [{k: "week", t: "Неделя"}, {k: "cat", t: "Категория"}, {k: "yoy", t: "г/г, %", n: 1, f: function (v) { return fmt(v, 1); }},
      {k: "z", t: "z", n: 1, f: function (v) { return fmt(v, 1); }}, {k: "why", t: "Что произошло", f: function (v) { return v || '<span class="mu">причина не установлена</span>'; }}], "wk");
  }

  // ---------- текст ----------
  function renderPipeline() {
    var steps = [["Портал актов", "652 акта регионов о режимах ЧС за 2023–2024 годы через открытый API; в анализе 378 — без отмен и регионов без данных о тратах. Из них 260 вводят или меняют режим (51 — правки актов прошлых лет, не новые события), остальные событиями не являются: выплаты, резервный фонд, правила поведения.", '<path class="ln" pathLength="1" d="M20 100V20h120l40 30v50z"/><path class="ln w" pathLength="1" d="M40 45h90M40 65h110M40 85h70"/>'],
      ["Распознавание", "PDF → текст: Apple Vision, запасной вариант — Tesseract.", '<rect class="bar" style="--b:0" x="30" y="40" width="30" height="60"/><rect class="bar" style="--b:1" x="80" y="20" width="30" height="80"/><rect class="bar" style="--b:2" x="130" y="55" width="30" height="45"/><path class="ln w" pathLength="1" d="M20 108h180"/>'],
      ["Факты по схеме", "GigaChat3.1 выписывает режим, причину, даты и МО дословно из текста.", '<path class="ln" pathLength="1" d="M20 30h60M20 60h90M20 90h50"/><path class="ln w" pathLength="1" d="M130 30h80M150 60h60M120 90h90"/><circle class="pt" cx="110" cy="60" r="7"/>'],
      ["Правила", "Категорию причины ставят прозрачные правила; изменения старых актов — не события.", '<path class="ln" pathLength="1" d="M20 60h50l20-35h60l20 35h50"/><path class="ln w" pathLength="1" d="M90 25l30 70"/><circle class="pt" cx="200" cy="60" r="7"/>'],
      ["Привязка к МО", "Справочник СберИндекса и ОКТМО Росстата: пункт → район; регион не путается с городом.", '<circle class="pt" cx="60" cy="40" r="7"/><circle class="pt" cx="150" cy="30" r="7"/><circle class="pt" cx="110" cy="90" r="7"/><path class="ln" pathLength="1" d="M60 40L110 90L150 30L200 70"/>'],
      ["Новости МЧС и Telegram", "87 тысяч новостей 69 регионов: четыре вопроса «да/нет» модели, правило против сводок и призывов.", '<path class="ln" pathLength="1" d="M20 90L60 70L100 80L140 40L180 50L220 20"/><path class="ln w" pathLength="1" d="M20 100L60 95L100 98L140 85L180 90L220 70"/><circle class="pt" cx="220" cy="20" r="7"/>']];
    $("#pipeline").innerHTML = steps.map(function (s, i) { return '<div class="sk yellow" style="--d:' + i + '"><div class="sk-vis"><span class="sk-n">0' + (i + 1) + '</span><svg viewBox="0 0 240 120">' + s[2] + '</svg></div><div class="sk-body"><h4>' + s[0] + '</h4><p>' + s[1] + '</p></div></div>'; }).join("");
  }
  function renderText(D) {
    var cat = D.categories[0];
    var draw = function () { bars($("#tx-bars"), D.dose.filter(function (r) { return r[cat].med != null; }).map(function (r) { return {label: r.group + (W($("#tx-bars")) > 500 ? " (" + r.n + ")" : ""), value: r[cat].med,
      color: r[cat].p != null && r[cat].p < 0.05 ? C.yellow : null, bold: r[cat].p != null && r[cat].p < 0.05, note: r.group + ", актов: " + r.n + "<br>медиана " + fmt(r[cat].med, 1) + "%, p = " + fmt(r[cat].p, 3)}; }), {signed: true, digits: 1, unit: "%"}); };
    buttons($("#tx-cat"), D.categories, cat, function (v) { cat = v; draw(); }); draw();
  }

  // ---------- система ----------
  function renderFeed(D) {
    var r = Math.PI / 180, p1 = 52 * r, p2 = 64 * r, n = (Math.cos(p1) - Math.cos(p2)) / (p2 - p1), G = Math.cos(p1) / n + p1, l0 = 100 * r;
    var proj = function (la, lo) { var rho = G - la * r, th = n * ((lo < 0 ? lo + 360 : lo) * r - l0); return [rho * Math.sin(th), -rho * Math.cos(th)]; };
    var host = $("#fd-map"), w = W(host), h = Math.round(w * 0.42), xy = D.map.map(function (p) { return proj(p.lat, p.lon); });
    var xs = xy.map(function (v) { return v[0]; }), ys = xy.map(function (v) { return v[1]; });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys), k = Math.min((w - 20) / (x1 - x0), (h - 20) / (y1 - y0));
    var order = D.map.map(function (p, i) { return i; }).sort(function (a, b) { var s = {none: 0, alert: 1, hit: 2}; return s[D.map[a].state] - s[D.map[b].state]; });
    var s = '<svg viewBox="0 0 ' + w + ' ' + h + '" width="100%">';
    order.forEach(function (i) { var p = D.map[i], X = 10 + (xy[i][0] - x0) * k, Y = 10 + (xy[i][1] - y0) * k;
      s += p.state === "none" ? '<circle cx="' + X + '" cy="' + Y + '" r="1.5" fill="#1212123a" data-t="' + i + '"/>' :
        '<circle cx="' + X + '" cy="' + Y + '" r="' + (p.state === "hit" ? 4 : 3) + '" fill="' + (p.state === "hit" ? C.red : C.green) + '" stroke="' + C.ink + '" stroke-width=".7" data-t="' + i + '"/>'; });
    host.innerHTML = s + "</svg>"; bindTips(host, D.map);
    var sel = $("#fd-reg"); sel.innerHTML = '<option value="">Все регионы</option>' + D.regions.map(function (x) { return "<option>" + x + "</option>"; }).join("");
    var mode = D.filters[0], reg = "";
    var draw = function () {
      var all = D.items.filter(function (e) { return !reg || e.region === reg; }), items = all.filter(function (e) { return mode === D.filters[1] || e.hit; });
      $("#fd-count").textContent = (mode === D.filters[0] ? "Карточек с провалом: " + items.length + " из " + all.length : "Карточек: " + items.length.toLocaleString("ru-RU")) +
        (items.length > 120 ? " · показаны первые 120" : "") + " · одна карточка — один текст и дата; акт на весь регион — одна карточка на все его МО";
      $("#fd-list").innerHTML = items.slice(0, 120).map(function (e) {
        var where = e.n_mo === 1 ? e.mos[0] : e.n_mo + " МО: " + e.mos.slice(0, 3).join(", ") + (e.n_mo > 3 ? "…" : "");
        return '<div class="ep' + (e.hit ? " hit" : e.acts ? " act" : "") + '"><div><div class="t">' + where + ' <span class="s">· ' + e.region + '</span></div><div class="s">' + e.dates + ' · ' + e.cause + (e.acts ? " · акт" : "") + (e.news ? " · новостей: " + e.news : "") + '</div><div class="s" style="margin-top:4px;color:#333">' + e.texts + '</div>' +
          (e.hit ? '<div class="s" style="margin-top:4px;color:#121212"><b>Провал трат на маркетплейсах:</b> ' + e.hit_mos.join(", ") + (e.n_hit > e.hit_mos.length ? " и ещё " + (e.n_hit - e.hit_mos.length) : "") + '</div>' : "") + '</div>' +
          '<div>' + (e.hit ? '<div class="lb">от эпизода до конца месяца провала</div><div class="p">' + e.lead + ' дн.</div>' : '<div class="lb">провала не было</div>') + '</div></div>'; }).join("");
    };
    buttons($("#fd-f"), D.filters, mode, function (v) { mode = v; draw(); }); sel.onchange = function (e) { reg = e.target.value; draw(); }; draw();
  }

  function renderCase(D) {
    table($("#fl-src"), D.sources, [{k: "src", t: "Источники"}, {k: "n", t: "Сообщений", n: 1}, {k: "mo", t: "МО", n: 1},
      {k: "all", t: "«Все»: ρ (p)", n: 1, f: function (v, r) { return (r.all_p < .05 ? "<b>" : "") + fmt(v, 2) + " (" + fmt(r.all_p, 3) + ")" + (r.all_p < .05 ? "</b>" : ""); }},
      {k: "mp", t: "Маркетплейсы: ρ (p)", n: 1, f: function (v, r) { return (r.mp_p < .05 ? "<b>" : "") + fmt(v, 2) + " (" + fmt(r.mp_p, 3) + ")" + (r.mp_p < .05 ? "</b>" : ""); }}]);
    var host = $("#fl-chart");
    window.onStep = function (id, st) {
      if (id !== "flood") return;
      if (st <= 2) {
        var tl = D.timeline; $("#fl-title").textContent = "Когда что стало известно";
        var rows = []; tl.forEach(function (t) { rows.push({label: t.region + ": сайт МЧС", value: t.first, color: C.green}); rows.push({label: t.region + ": акт опубликован", value: t.act, color: st >= 1 ? C.yellow : null}); });
        rows.push({label: "конец апреля: траты за месяц", value: D.end, color: st >= 2 ? C.ink : null, bold: st >= 2});
        bars(host, rows, {unit: " дн."}); $("#fl-note").textContent = "Дни от первого сообщения о подтоплении на сайте МЧС (" + D.t0 + " 2024 года). Список районов известен до выхода трат за апрель, поэтому подобрать группу под провал задним числом нельзя.";
      } else {
        $("#fl-title").textContent = "Маркетплейсы: отклик трат, медиана по МО, %";
        bars(host, D.bars.map(function (b) { return {label: b.label, value: b.value, color: b.value < 0 ? C.ink : C.green}; }), {signed: true, digits: 1, unit: "%"});
        $("#fl-note").textContent = "Провал маркетплейсов в апреле там, где о подтоплении писали чаще, и отскок в мае.";
      }
      host.querySelectorAll(".bw").forEach(function (b) { b.style.transform = "none"; });
    };
    window.onStep("flood", 0);
    // для печати второй график разбора — отдельно: в PDF нет прокрутки, которая его переключает
    bars($("#fl-print"), D.bars.map(function (b) { return {label: b.label, value: b.value, color: b.value < 0 ? C.ink : C.green}; }), {signed: true, digits: 1, unit: "%"});
  }

  function renderAll() {
    renderTicker(DATA.nums); renderForecast(DATA.forecast); renderFM(DATA.fm); renderDetect(DATA.detect); renderPipeline(); renderText(DATA.text); renderFeed(DATA.feed); renderCase(DATA.case);
  }
  renderTOC(); renderAll();
  var rw = innerWidth; addEventListener("resize", function () { if (Math.abs(innerWidth - rw) > 80) { rw = innerWidth; renderAll(); } });

  // ---------- появление блоков, счётчики, прокрутка ----------
  var reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  function fmtc(v) { return Math.round(v).toString().replace(/\B(?=(\d{3})+(?!\d))/g, "\u00a0"); }
  function count(el) {
    if (el.dataset.done) return; el.dataset.done = 1;
    var end = +String(el.dataset.count).replace(/\s/g, "").replace(",", "."), t0; if (isNaN(end)) return;
    if (reduce) { el.textContent = fmtc(end); return; }
    requestAnimationFrame(function step(ts) { t0 = t0 || ts; var p = Math.min(1, (ts - t0) / 1600), e = 1 - Math.pow(1 - p, 3); el.textContent = fmtc(end * e); if (p < 1) requestAnimationFrame(step); });
  }
  // печать: счётчики сразу в итоговое значение, все блоки видимы
  function finalize() { document.querySelectorAll("[data-count]").forEach(function (el) { el.dataset.done = 1; el.textContent = fmtc(+String(el.dataset.count).replace(/\s/g, "").replace(",", ".")); });
    document.querySelectorAll("[data-reveal]").forEach(function (el) { el.classList.add("in"); });
    document.querySelectorAll(".chart svg .bw").forEach(function (b) { b.style.transform = "none"; });
    if (window.onStep) window.onStep("flood", 2); }
  addEventListener("beforeprint", finalize);
  if (matchMedia("print").matches || /HeadlessChrome/.test(navigator.userAgent)) finalize();
  var io = new IntersectionObserver(function (es) {
    es.forEach(function (e) { if (!e.isIntersecting) return; e.target.classList.add("in"); e.target.querySelectorAll("[data-count]").forEach(count); if (e.target.dataset.count) count(e.target); io.unobserve(e.target); });
  }, {threshold: .12, rootMargin: "0px 0px -6% 0px"});
  document.querySelectorAll("[data-reveal]").forEach(function (el) { io.observe(el); });
  setTimeout(function () { document.querySelectorAll(".cover .ml>span").forEach(function (s) { s.classList.add("go"); }); }, 60);
  var prog = $(".progress i"), blocks = [].slice.call(document.querySelectorAll("[data-par]")), tick = false;
  var pending = [].slice.call(document.querySelectorAll("[data-reveal]"));
  function revealPassed() {
    // страховка к наблюдателю: при быстрой прокрутке короткие блоки могут проскочить — раскрываем всё, что уже выше низа экрана
    var vh = innerHeight;
    pending = pending.filter(function (el) {
      if (el.classList.contains("in")) return false;
      if (el.getBoundingClientRect().top < vh) { el.classList.add("in"); el.querySelectorAll("[data-count]").forEach(count); if (el.dataset.count) count(el); return false; }
      return true;
    });
  }
  function frame() {
    var h = document.documentElement, vh = innerHeight;
    if (prog) prog.style.width = 100 * h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight) + "%";
    if (!reduce && innerWidth > 760) blocks.forEach(function (b) { var r = b.parentNode.getBoundingClientRect(); if (r.bottom < -200 || r.top > vh + 200) return; b.style.setProperty("--py", (-(r.top + r.height / 2 - vh / 2) * parseFloat(b.dataset.par)).toFixed(1) + "px"); });
    tick = false;
  }
  addEventListener("scroll", function () { revealPassed(); if (!tick) { tick = true; requestAnimationFrame(frame); } }, {passive: true}); frame(); revealPassed();
  document.querySelectorAll(".scrolly").forEach(function (sc) {
    var steps = sc.querySelectorAll(".step");
    var so = new IntersectionObserver(function (es) { es.forEach(function (e) { if (!e.isIntersecting) return; steps.forEach(function (s) { s.classList.toggle("on", s === e.target); });
      sc.dataset.active = e.target.dataset.step; if (window.onStep) window.onStep(sc.id, +e.target.dataset.step); }); }, {rootMargin: innerWidth < 1000 ? "-62% 0px -34% 0px" : "-45% 0px -45% 0px"});
    steps.forEach(function (s) { so.observe(s); }); if (steps[0]) steps[0].classList.add("on");
  });
  var h1 = $(".cover h1"), col = h1 && h1.parentNode;
  function fit() {
    if (!h1) return; h1.style.fontSize = "";
    var sp = [].slice.call(h1.querySelectorAll(".ml>span")); sp.forEach(function (s) { s.style.display = "inline-block"; });
    var widest = Math.max.apply(null, sp.map(function (s) { return s.scrollWidth; }));
    var max = col.clientWidth - 8, fs = parseFloat(getComputedStyle(h1).fontSize);
    h1.style.fontSize = Math.min(fs * max * .97 / widest, 240) + "px";
    var lead = col.lastElementChild, k = 0; lead.style.marginTop = "0";
    // граница — низ сетки обложки: колонка растягивается вместе с содержимым и сама переполнения не покажет
    var bottom = function () { return Math.min(col.getBoundingClientRect().bottom, col.parentNode.getBoundingClientRect().bottom) - 4; };
    while (innerWidth > 1000 && lead.getBoundingClientRect().bottom > bottom() && k++ < 40 && parseFloat(h1.style.fontSize) > 80) h1.style.fontSize = parseFloat(h1.style.fontSize) - 6 + "px";
    lead.style.marginTop = "";
  }
  fit(); addEventListener("resize", fit); if (document.fonts) document.fonts.ready.then(fit);
})();
