// YiAgents ECharts integration — themed, offline, no CDN.
//
// Reads CSS custom properties at draw time so charts follow the dual-theme
// system automatically. All instances are tracked so theme switches and route
// changes can dispose + redraw cleanly. Exposed as window.YiCharts.
//
// The keyword-lean analysis mirrors verdictTilt() in app.js — values are
// derived from real char counts and keyword densities, never fabricated scores.

(function () {
  "use strict";

  // ---- rating vocabulary (kept in sync with rating.py 5-tier scale) ----
  var RATING_ORDER = ["Buy", "Overweight", "Hold", "Underweight", "Sell"];
  var RATING_VAL = { Buy: 5, Overweight: 4, Hold: 3, Underweight: 2, Sell: 1 };
  var RATING_KEY = { Buy: "buy", Overweight: "over", Hold: "hold", Underweight: "under", Sell: "sell" };

  // ---- keyword sets for wording-lean analysis (mirrors app.js verdictTilt) ----
  var BULL_RE = /bullish|overweight|\bbuy\b|upside|\blong\b|optimistic|compelling|attractive|constructive|favor(?:able|s)?/gi;
  var BEAR_RE = /bearish|underweight|\bsell\b|downside|\bshort\b|overvalued|pessimistic|caution|deteriorat|\brisk\b/gi;
  var CAUTION_RE = /caution|risk|drawdown|stop.?loss|downside|volatil|exposure|hedge|protect|cut|reduce|limit/gi;

  function countMatches(text, re) { return ((text || "").match(re) || []).length; }
  function fmtK(n) { n = n || 0; return n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n); }
  function escapeHTML(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  // ---- read CSS custom properties into a theme snapshot ----
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function theme() {
    return {
      ink1: cssVar("--ink-1"),
      ink2: cssVar("--ink-2"),
      ink3: cssVar("--ink-3"),
      ink4: cssVar("--ink-4"),
      grid: cssVar("--grid"),
      panel: cssVar("--panel-solid"),
      border: cssVar("--border-2"),
      accent: cssVar("--accent"),
      accent2: cssVar("--accent-2"),
      accentDim: cssVar("--accent-dim"),
      buy: cssVar("--r-buy"),
      over: cssVar("--r-over"),
      hold: cssVar("--r-hold"),
      under: cssVar("--r-under"),
      sell: cssVar("--r-sell"),
      bull: cssVar("--bull"),
      bear: cssVar("--bear"),
      neutral: cssVar("--neutral"),
      font: cssVar("--font"),
      mono: cssVar("--mono")
    };
  }

  function ratingColor(th, rating) {
    return { Buy: th.buy, Overweight: th.over, Hold: th.hold, Underweight: th.under, Sell: th.sell }[rating] || th.neutral;
  }

  // ---- shared axis / tooltip / legend builders ----
  function tooltipStyle(th) {
    return {
      backgroundColor: th.panel,
      borderColor: th.border,
      borderWidth: 1,
      textStyle: { color: th.ink1, fontFamily: th.font, fontSize: 12 },
      extraCssText: "border-radius:8px;box-shadow:" + (cssVar("--shadow-float") || "0 8px 24px rgba(0,0,0,.4)") + ";"
    };
  }

  function splitArea(th) {
    return { show: false };
  }

  function splitLine(th) {
    return { show: true, lineStyle: { color: th.grid, type: "dashed" } };
  }

  function axisLine(th) {
    return { show: true, lineStyle: { color: th.grid } };
  }

  // ---- instance registry: { container -> { chart, redraw } } ----
  var registry = new Map();

  function register(container, chart, redraw) {
    if (!container) return chart;
    // dispose previous instance on this container if any
    var prev = registry.get(container.id || container);
    if (prev && prev.chart) { try { prev.chart.dispose(); } catch (_) {} }
    registry.set(container.id || container, { chart: chart, redraw: redraw });
    return chart;
  }

  // ---- public API ----

  /**
   * Rating distribution donut chart for the home page.
   * @param {HTMLElement} el - container div
   * @param {Array} tickers - [{latest_rating,...}]
   */
  function drawRatingDist(el, tickers) {
    if (!el || !window.echarts) return null;
    var th = theme();
    var counts = {}; RATING_ORDER.forEach(function (r) { counts[r] = 0; });
    var total = 0;
    (tickers || []).forEach(function (x) {
      if (x.latest_rating && counts[x.latest_rating] != null) { counts[x.latest_rating]++; total++; }
    });
    if (!total) { el.innerHTML = ""; return null; }

    var data = RATING_ORDER.filter(function (r) { return counts[r] > 0; }).map(function (r) {
      return { name: r, value: counts[r], itemStyle: { color: ratingColor(th, r) } };
    });

    var chart = echarts.init(el, null, { renderer: "canvas" });
    var opt = {
      tooltip: Object.assign(triggerItem(tooltipStyle(th)), {
        formatter: function (p) { return p.name + " · " + p.value + " (" + p.percent + "%)"; }
      }),
      legend: {
        orient: "vertical", right: 10, top: "center",
        textStyle: { color: th.ink2, fontSize: 12 },
        itemWidth: 10, itemHeight: 10, itemGap: 10
      },
      series: [{
        type: "pie", radius: ["48%", "78%"], center: ["38%", "50%"],
        avoidLabelOverlap: true, padAngle: 2,
        itemStyle: { borderRadius: 6, borderColor: th.panel, borderWidth: 2 },
        label: { show: false },
        emphasis: { scale: true, scaleSize: 6, label: { show: true, fontSize: 14, fontWeight: "bold", color: th.ink1, formatter: "{b}\n{c}" } },
        data: data
      }],
      graphic: [{
        type: "text", left: "38%", top: "50%",
        style: { text: String(total), fill: th.ink1, fontSize: 26, fontWeight: "bold", textAlign: "center" },
        z: 10
      }, {
        type: "text", left: "38%", top: "60%",
        style: { text: (window.t ? window.t("dist_tickers") : "tickers"), fill: th.ink3, fontSize: 11, textAlign: "center" },
        z: 10
      }]
    };
    chart.setOption(opt);
    return register(el, chart, function () { drawRatingDist(el, tickers); });
  }

  /**
   * Rating trend step-line for the detail page.
   * @param {HTMLElement} el
   * @param {Array} dateRatings - [{date, rating}]
   * @param {Function} onClick - callback(date, rating)
   */
  function drawRatingTrend(el, dateRatings, onClick) {
    if (!el || !window.echarts) return null;
    var th = theme();
    var drs = (dateRatings || []).filter(function (dr) { return dr.rating && RATING_VAL[dr.rating]; });
    if (!drs.length) { el.style.display = "none"; return null; }

    var dates = drs.map(function (dr) { return dr.date; });
    var vals = drs.map(function (dr) { return RATING_VAL[dr.rating]; });
    var colors = drs.map(function (dr) { return ratingColor(th, dr.rating); });

    var chart = echarts.init(el, null, { renderer: "canvas" });
    var opt = {
      tooltip: Object.assign(triggerAxis(tooltipStyle(th)), {
        formatter: function (params) {
          var p = params[0];
          var label = RATING_ORDER[5 - p.value] || "—";
          return p.axisValue + "<br/>" + label;
        }
      }),
      grid: { left: 36, right: 20, top: 20, bottom: 28 },
      xAxis: {
        type: "category", data: dates, boundaryGap: false,
        axisLine: axisLine(th), axisTick: { show: false },
        axisLabel: { color: th.ink3, fontSize: 11, rotate: dates.length > 6 ? 30 : 0 }
      },
      yAxis: {
        type: "value", min: 0.5, max: 5.5, interval: 1,
        axisLine: { show: false }, axisTick: { show: false },
        splitLine: splitLine(th),
        axisLabel: {
          color: th.ink3, fontSize: 11,
          formatter: function (v) { return ({ 1: "Sell", 2: "Under", 3: "Hold", 4: "Over", 5: "Buy" })[v] || ""; }
        }
      },
      series: [{
        type: "line", step: "middle", smooth: false, symbol: "circle", symbolSize: 10,
        lineStyle: { color: th.accent, width: 2.5 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: hexA(th.accent, 0.22) },
            { offset: 1, color: hexA(th.accent, 0.01) }
          ])
        },
        itemStyle: { color: th.accent, borderColor: th.panel, borderWidth: 2 },
        data: vals.map(function (v, i) { return { value: v, itemStyle: { color: colors[i] } }; })
      }]
    };
    chart.setOption(opt);
    chart.on("click", function (params) {
      if (params.componentType === "series" && onClick) {
        onClick(dates[params.dataIndex], RATING_ORDER[5 - vals[params.dataIndex]]);
      }
    });
    return register(el, chart, function () { drawRatingTrend(el, dateRatings, onClick); });
  }

  /**
   * Node performance horizontal bar chart (wall_seconds + token dual axis).
   * @param {HTMLElement} el
   * @param {Object} nodePerf - {nodes:{name:{wall_seconds,tokens_in,...}}, totals:{}}
   */
  function drawNodePerf(el, nodePerf) {
    if (!el || !window.echarts || !nodePerf || !nodePerf.nodes) return null;
    var th = theme();
    var entries = Object.entries(nodePerf.nodes).map(function (e) {
      var v = e[1];
      return {
        name: e[0], wall: v.wall_seconds || 0,
        tokIn: v.tokens_in || 0, tokOut: v.tokens_out || 0, tokReason: v.tokens_reasoning || 0,
        tok: (v.tokens_in || 0) + (v.tokens_out || 0) + (v.tokens_reasoning || 0)
      };
    }).filter(function (e) { return e.wall > 0; }).sort(function (a, b) { return a.wall - b.wall; }); // ascending for horizontal bar
    if (!entries.length) { el.style.display = "none"; return null; }

    var chart = echarts.init(el, null, { renderer: "canvas" });
    var opt = {
      tooltip: Object.assign(triggerItem(tooltipStyle(th)), {
        formatter: function (p) {
          var d = entries[p.dataIndex];
          return "<b>" + escapeHTML(d.name) + "</b><br/>" +
            "Wall: " + d.wall.toFixed(1) + "s<br/>" +
            "Tokens: " + d.tok.toLocaleString() + "<br/>" +
            "<span style=\"color:" + th.ink3 + ";font-size:11px\">  in " + d.tokIn.toLocaleString() + " · out " + d.tokOut.toLocaleString() + " · reason " + d.tokReason.toLocaleString() + "</span>";
        }
      }),
      grid: { left: 10, right: 60, top: 24, bottom: 10, containLabel: true },
      legend: { show: false },
      xAxis: {
        type: "value", axisLine: { show: false }, axisTick: { show: false },
        splitLine: splitLine(th),
        axisLabel: { color: th.ink3, fontSize: 11, formatter: "{value}s" }
      },
      yAxis: {
        type: "category", data: entries.map(function (e) { return e.name; }),
        axisLine: axisLine(th), axisTick: { show: false },
        axisLabel: { color: th.ink2, fontSize: 11 }
      },
      series: [{
        type: "bar", data: entries.map(function (e) { return e.wall; }),
        barMaxWidth: 22, barCategoryGap: "40%",
        itemStyle: {
          borderRadius: [0, 5, 5, 0],
          color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
            { offset: 0, color: hexA(th.accent, 0.5) },
            { offset: 1, color: th.accent }
          ])
        },
        label: { show: true, position: "right", color: th.ink3, fontSize: 11, formatter: function (p) { return p.value.toFixed(1) + "s"; } }
      }]
    };
    chart.setOption(opt);
    return register(el, chart, function () { drawNodePerf(el, nodePerf); });
  }

  /**
   * Bull ↔ Bear debate balance gauge.
   * @param {HTMLElement} el
   * @param {Object} debate - {bull_history, bear_history, judge_decision}
   */
  function drawDebateBalance(el, debate) {
    if (!el || !window.echarts) return null;
    var th = theme();
    var bull = (debate.bull_history || "").length;
    var bear = (debate.bear_history || "").length;
    var sum = bull + bear || 1;
    var bullPct = Math.round((bull / sum) * 100);

    var chart = echarts.init(el, null, { renderer: "canvas" });
    var opt = {
      tooltip: Object.assign(triggerItem(tooltipStyle(th)), {
        formatter: function () {
          return "Bull: " + fmtK(bull) + " chars (" + bullPct + "%)<br/>" +
            "Bear: " + fmtK(bear) + " chars (" + (100 - bullPct) + "%)";
        }
      }),
      series: [{
        type: "gauge", startAngle: 180, endAngle: 0, min: 0, max: 100,
        radius: "92%", center: ["50%", "78%"],
        splitNumber: 4,
        progress: { show: true, width: 14, roundCap: true,
          itemStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
              { offset: 0, color: th.bear }, { offset: 0.5, color: th.neutral }, { offset: 1, color: th.bull }
            ])
          }
        },
        pointer: { show: true, length: "55%", width: 4, itemStyle: { color: th.ink2 } },
        axisLine: { roundCap: true, lineStyle: { width: 14, color: [[1, th.grid]] } },
        axisTick: { show: false },
        splitLine: { show: true, length: 8, lineStyle: { color: th.ink4, width: 1 } },
        axisLabel: { show: false },
        anchor: { show: true, size: 8, itemStyle: { color: th.ink3 } },
        title: { show: false },
        detail: {
          valueAnimation: true, offsetCenter: [0, "-12%"],
          formatter: function (v) { return "{a|" + Math.round(v) + "%}{b|Bull}"; },
          rich: {
            a: { fontSize: 28, fontWeight: "bold", color: bullPct > 55 ? th.bull : (bullPct < 45 ? th.bear : th.neutral), fontFamily: th.font },
            b: { fontSize: 12, color: th.ink3, padding: [4, 0, 0, 4] }
          }
        },
        data: [{ value: bullPct }]
      }],
      graphic: [{
        type: "text", left: "12%", bottom: "6%",
        style: { text: "Bear " + (100 - bullPct) + "%", fill: th.bear, fontSize: 12, textAlign: "center" }
      }, {
        type: "text", right: "12%", bottom: "6%",
        style: { text: bullPct + "% Bull", fill: th.bull, fontSize: 12, textAlign: "center" }
      }]
    };
    chart.setOption(opt);
    return register(el, chart, function () { drawDebateBalance(el, debate); });
  }

  /**
   * Risk three-way radar chart.
   * Axes: Volume, Bullish Lean, Bearish Lean, Caution Level — per debater.
   * @param {HTMLElement} el
   * @param {Object} risk - {aggressive_history, neutral_history, conservative_history, judge_decision}
   */
  function drawRiskRadar(el, risk) {
    if (!el || !window.echarts) return null;
    var th = theme();

    function metrics(text) {
      var len = (text || "").length;
      var bull = countMatches(text, BULL_RE);
      var bear = countMatches(text, BEAR_RE);
      var caution = countMatches(text, CAUTION_RE);
      return { vol: len, bull: bull, bear: bear, caution: caution };
    }

    var aggr = metrics(risk.aggressive_history);
    var neut = metrics(risk.neutral_history);
    var cons = metrics(risk.conservative_history);

    // normalize each axis to 0-100
    var maxVol = Math.max(aggr.vol, neut.vol, cons.vol, 1);
    var maxBull = Math.max(aggr.bull, neut.bull, cons.bull, 1);
    var maxBear = Math.max(aggr.bear, neut.bear, cons.bear, 1);
    var maxCaution = Math.max(aggr.caution, neut.caution, cons.caution, 1);

    var labels = (window.t)
      ? [window.t("chart_axis_volume"), window.t("chart_axis_bull"), window.t("chart_axis_bear"), window.t("chart_axis_caution")]
      : ["Volume", "Bullish", "Bearish", "Caution"];

    var chart = echarts.init(el, null, { renderer: "canvas" });
    var opt = {
      tooltip: Object.assign(triggerItem(tooltipStyle(th))),
      legend: {
        bottom: 0, itemWidth: 10, itemHeight: 10,
        textStyle: { color: th.ink2, fontSize: 11 }
      },
      radar: {
        center: ["50%", "46%"], radius: "60%",
        indicator: [
          { name: labels[0], max: 100 }, { name: labels[1], max: 100 },
          { name: labels[2], max: 100 }, { name: labels[3], max: 100 }
        ],
        axisName: { color: th.ink3, fontSize: 11 },
        splitLine: splitLine(th),
        splitArea: { show: true, areaStyle: { color: [hexA(th.panel, 0), hexA(th.ink4, 0.04)] } },
        axisLine: axisLine(th)
      },
      series: [{
        type: "radar",
        symbolSize: 5,
        data: [
          {
            value: [pct(aggr.vol, maxVol), pct(aggr.bull, maxBull), pct(aggr.bear, maxBear), pct(aggr.caution, maxCaution)],
            name: window.t ? window.t("sub_aggressive") : "Aggressive",
            itemStyle: { color: th.bear }, areaStyle: { color: hexA(th.bear, 0.15) },
            lineStyle: { color: th.bear, width: 2 }
          },
          {
            value: [pct(neut.vol, maxVol), pct(neut.bull, maxBull), pct(neut.bear, maxBear), pct(neut.caution, maxCaution)],
            name: window.t ? window.t("sub_neutral") : "Neutral",
            itemStyle: { color: th.neutral }, areaStyle: { color: hexA(th.neutral, 0.12) },
            lineStyle: { color: th.neutral, width: 2 }
          },
          {
            value: [pct(cons.vol, maxVol), pct(cons.bull, maxBull), pct(cons.bear, maxBear), pct(cons.caution, maxCaution)],
            name: window.t ? window.t("sub_conservative") : "Conservative",
            itemStyle: { color: th.buy }, areaStyle: { color: hexA(th.buy, 0.12) },
            lineStyle: { color: th.buy, width: 2 }
          }
        ]
      }]
    };
    chart.setOption(opt);
    return register(el, chart, function () { drawRiskRadar(el, risk); });
  }

  // ---- lifecycle: resize, theme switch, route change ----

  function resizeAll() {
    registry.forEach(function (entry) {
      if (entry.chart) { try { entry.chart.resize(); } catch (_) {} }
    });
  }

  function redrawAll() {
    registry.forEach(function (entry) {
      if (entry.redraw) { try { entry.redraw(); } catch (_) {} }
    });
  }

  function disposeAll() {
    registry.forEach(function (entry) {
      if (entry.chart) { try { entry.chart.dispose(); } catch (_) {} }
    });
    registry.clear();
  }

  // resize on window resize (debounced)
  var resizeTimer = null;
  window.addEventListener("resize", function () {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(resizeAll, 200);
  });

  // ---- helpers ----

  function triggerItem(base) { base.trigger = "item"; return base; }
  function triggerAxis(base) { base.trigger = "axis"; return base; }

  function pct(v, max) { return max ? Math.round((v / max) * 100) : 0; }

  // add alpha to a hex color string (#rrggbb → rgba)
  function hexA(hex, alpha) {
    if (!hex) return "rgba(91,142,240," + alpha + ")";
    var h = hex.replace("#", "");
    if (h.length !== 6) return hex;
    var r = parseInt(h.slice(0, 2), 16);
    var g = parseInt(h.slice(2, 4), 16);
    var b = parseInt(h.slice(4, 6), 16);
    return "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
  }

  // ---- expose ----
  window.YiCharts = {
    drawRatingDist: drawRatingDist,
    drawRatingTrend: drawRatingTrend,
    drawNodePerf: drawNodePerf,
    drawDebateBalance: drawDebateBalance,
    drawRiskRadar: drawRiskRadar,
    resizeAll: resizeAll,
    redrawAll: redrawAll,
    disposeAll: disposeAll
  };
})();
