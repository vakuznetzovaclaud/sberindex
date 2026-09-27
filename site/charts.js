// Графики страницы и презентации: собственный SVG без библиотек. Общие для site/mag.js и site/deck.js.
(function () {
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var C = {ink: "#121212", red: "#ED0F00", pink: "#F46DE4", yellow: "#FFE762", blue: "#7F9FFF", green: "#41E786", grey: "#1212122e", mu: "#6d6d6d"};
  var fmt = function (v, d) { if (v == null || isNaN(v)) return "—"; return Number(v).toLocaleString("ru-RU", {minimumFractionDigits: d || 0, maximumFractionDigits: d || 0}).replace("-", "−"); };
  var tip = $("#tip");
  function showTip(e, h) { if (!tip) return; tip.innerHTML = h; tip.style.opacity = 1; tip.style.left = Math.min(e.clientX + 14, innerWidth - 330) + "px"; tip.style.top = e.clientY + 14 + "px"; }
  function hideTip() { if (tip) tip.style.opacity = 0; }
  function bindTips(host, rows) { host.querySelectorAll("[data-t]").forEach(function (n) { var r = rows[+n.dataset.t]; if (r && r.note) { n.addEventListener("mousemove", function (e) { showTip(e, r.note); }); n.addEventListener("mouseleave", hideTip); } }); }
  function W(host) { return Math.max(300, host.clientWidth || 800); }
  function clip(t, px) { var n = Math.floor(px / 7.4); return t.length > n ? t.slice(0, n - 1) + "…" : t; }

  // горизонтальные столбцы; signed — от нуля в обе стороны
  function bars(host, rows, o) {
    o = o || {}; var w = W(host), rh = 30, left = Math.min(w * (w < 500 ? 0.52 : 0.44), Math.max.apply(null, rows.map(function (r) { return r.label.length * (r.bold ? 8.6 : 7.8); })) + 16);
    var H = rows.length * rh + 8, vals = rows.map(function (r) { return r.value; });
    var mx = o.signed ? Math.max.apply(null, vals.map(Math.abs)) * 1.18 : Math.max.apply(null, vals) * 1.12;
    var span = w - left - 64, zero = o.signed ? left + span / 2 : left, k = o.signed ? span / 2 / mx : span / mx;
    var s = '<svg viewBox="0 0 ' + w + ' ' + H + '" width="100%" role="img">';
    if (o.signed || o.ref != null) { var xr = o.signed ? zero : left + o.ref * k; s += '<line x1="' + xr + '" x2="' + xr + '" y1="0" y2="' + H + '" stroke="' + C.ink + '"/>'; }
    rows.forEach(function (r, i) {
      var y = i * rh + 4, x0 = r.value < 0 ? zero + r.value * k : zero, bw = Math.max(1.5, Math.abs(r.value) * k);
      s += '<text x="' + (left - 10) + '" y="' + (y + 17) + '" text-anchor="end" font-size="13.5" font-weight="' + (r.bold ? 900 : 600) + '"' + (r.red ? ' style="fill:' + C.red + '"' : '') + '><title>' + r.label + '</title>' + clip(r.label, (left - 12) * (r.bold ? 7.4 / 8.4 : 1)) + '</text>';
      s += '<rect class="bw" style="--b:' + i + '" x="' + x0 + '" y="' + (y + 4) + '" width="' + bw + '" height="' + (rh - 12) + '" fill="' + (r.color || C.grey) + '" stroke="' + C.ink + '" stroke-width="' + (r.color ? 1 : 0) + '" data-t="' + i + '"/>';
      var tx = r.value < 0 ? zero + 6 : x0 + bw + 6;          // у отрицательных подпись справа от нуля — не наезжает на названия
      s += '<text x="' + tx + '" y="' + (y + 17) + '" font-size="13" font-weight="800">' + fmt(r.value, o.digits) + (o.unit || "") + '</text>';
    });
    host.innerHTML = s + "</svg>"; bindTips(host, rows);
    var ch = host.closest(".chart"); if (ch && ch.classList.contains("in")) host.querySelectorAll(".bw").forEach(function (b) { b.style.transform = "none"; });
  }

  function lines(host, labels, series, o) {
    o = o || {}; var w = W(host), H = o.height || 260, L = 54, R = 28, T = 14, B = 30;
    var all = [].concat.apply([], series.map(function (s) { return s.values; })).filter(function (v) { return v != null; });
    var lo = o.lo != null ? o.lo : Math.min.apply(null, all.concat(o.ref != null ? [o.ref] : [])), hi = Math.max.apply(null, all.concat(o.ref != null ? [o.ref] : []));
    var st = (hi - lo) / 4, mg = Math.pow(10, Math.floor(Math.log10(st))), e = st / mg;   // шаг сетки 1/2/5 × 10^k
    st = (e < 1.5 ? 1 : e < 3 ? 2 : e < 7 ? 5 : 10) * mg; lo = Math.floor(lo / st) * st; hi = Math.ceil(hi / st) * st;
    var nt = Math.round((hi - lo) / st), dg = o.digits != null ? o.digits : st < 1 ? 2 : 0;
    var x = function (i) { return L + (w - L - R) * i / (labels.length - 1); }, y = function (v) { return T + (H - T - B) * (1 - (v - lo) / (hi - lo)); };
    var s = '<svg viewBox="0 0 ' + w + ' ' + H + '" width="100%" role="img">';
    for (var k = 0; k <= nt; k++) { var v = lo + st * k; s += '<line x1="' + L + '" x2="' + (w - R) + '" y1="' + y(v) + '" y2="' + y(v) + '" stroke="#1212121f"/><text x="' + (L - 8) + '" y="' + (y(v) + 4) + '" text-anchor="end" font-size="11.5" class="mu">' + fmt(v, dg) + '</text>'; }
    if (o.ref != null) s += '<line x1="' + L + '" x2="' + (w - R) + '" y1="' + y(o.ref) + '" y2="' + y(o.ref) + '" stroke="' + C.ink + '" stroke-dasharray="4 4"/>';
    var step = Math.ceil(labels.length * 56 / (w - L - R));
    labels.forEach(function (l, i) { if (i % step === 0) s += '<text x="' + x(i) + '" y="' + (H - 8) + '" text-anchor="middle" font-size="11.5" class="mu">' + l + '</text>'; });
    series.forEach(function (se) {
      var pts = se.values.map(function (v, i) { return v == null ? null : x(i) + "," + y(v); }).filter(Boolean).join(" ");
      s += '<polyline points="' + pts + '" fill="none" stroke="' + se.color + '" stroke-width="' + (se.bold ? 3.4 : 1.8) + '"' + (se.dash ? ' stroke-dasharray="6 4"' : '') + '/>';
      se.values.forEach(function (v, i) { if (v != null) s += '<circle cx="' + x(i) + '" cy="' + y(v) + '" r="' + (se.bold ? 4 : 2.6) + '" fill="' + se.color + '" stroke="' + C.ink + '" stroke-width=".8"/>'; });
    });
    s += '<line class="guide" x1="0" x2="0" y1="' + T + '" y2="' + (H - B) + '" stroke="' + C.ink + '" stroke-width="1" opacity="0"/>';
    var cw = (w - L - R) / Math.max(1, labels.length - 1);
    labels.forEach(function (l, i) { s += '<rect class="hot" data-i="' + i + '" x="' + (x(i) - cw / 2) + '" y="' + T + '" width="' + cw + '" height="' + (H - T - B) + '" fill="transparent"/>'; });
    var leg = series.map(function (se) { return '<span style="display:inline-flex;align-items:center;gap:7px;margin:0 16px 6px 0;font-size:13px;font-weight:700"><svg width="22" height="8"><line x1="0" x2="22" y1="4" y2="4" stroke="' + se.color + '" stroke-width="' + (se.bold ? 4 : 2.4) + '"' + (se.dash ? ' stroke-dasharray="6 4"' : '') + '/></svg>' + se.name + '</span>'; }).join("");
    host.innerHTML = '<div>' + leg + '</div>' + s + '</svg>';
    var guide = host.querySelector(".guide");
    host.querySelectorAll(".hot").forEach(function (r) {
      var i = +r.dataset.i;
      r.addEventListener("mousemove", function (ev) { guide.setAttribute("x1", x(i)); guide.setAttribute("x2", x(i)); guide.setAttribute("opacity", ".35");
        showTip(ev, "<b>" + labels[i] + "</b><br>" + series.slice().sort(function (a, b) { return (a.values[i] == null) - (b.values[i] == null) || a.values[i] - b.values[i]; })
          .map(function (se) { return '<span style="color:' + (se.color === C.yellow ? C.ink : se.color) + '">●</span> ' + se.name + ": " + fmt(se.values[i], dg); }).join("<br>")); });
      r.addEventListener("mouseleave", function () { guide.setAttribute("opacity", "0"); hideTip(); });
    });
  }

  function table(host, rows, cols, cls) {
    var h = '<table class="t' + (cls ? " " + cls : "") + '"><tr>' + cols.map(function (c) { return '<th class="' + (c.n ? "n" : "") + '">' + c.t + '</th>'; }).join("") + '</tr>';
    rows.forEach(function (r) { h += '<tr>' + cols.map(function (c) { var v = r[c.k]; return '<td class="' + (c.n ? "n" : "") + '">' + (c.f ? c.f(v, r) : (v == null ? "—" : v)) + '</td>'; }).join("") + '</tr>'; });
    host.innerHTML = h + '</table>';
  }

  function buttons(host, items, active, pick) {
    host.innerHTML = ""; items.forEach(function (it) { var b = document.createElement("button"); b.textContent = it; if (it === active) b.className = "on";
      b.onclick = function () { host.querySelectorAll("button").forEach(function (x) { x.className = ""; }); b.className = "on"; pick(it); }; host.appendChild(b); });
  }

  var short = {"Все категории": "Все", "Продовольствие": "Продукты", "Общественное питание": "Общепит"};   // подписи столбцов категорий
  // цвет столбца по семейству модели или детектора: ансамбли, FM, выбранный детектор, GLR; остальные — серые
  function famColor(f) { return f === "ens" ? C.blue : f === "fm" ? "#7F9FFF66" : f === "sel" ? C.pink : f === "glr" ? "#F46DE488" : null; }
  window.CH = {$: $, C: C, fmt: fmt, W: W, bars: bars, lines: lines, table: table, buttons: buttons, bindTips: bindTips, showTip: showTip, hideTip: hideTip, short: short, famColor: famColor};
})();
