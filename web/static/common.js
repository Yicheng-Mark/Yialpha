// YiAgents shared utilities — single home for helpers used by BOTH app.js and
// charts.js (they were previously copy-pasted with "keep in sync" comments).
//
// Loaded before app.js/charts.js (see index.html script order). Exposed as
// window.YiUtil so the no-build IIFE modules stay framework-free.

(function () {
  "use strict";

  // ---- keyword sets for wording-lean analysis ----
  // The lean is a heuristic from prose keywords, shown in the UI as "wording
  // lean" (措辞), never as a computed bull/bear score — debate state carries
  // no score.
  var BULL_RE = /bullish|overweight|\bbuy\b|upside|\blong\b|optimistic|compelling|attractive|constructive|favor(?:able|s)?/gi;
  var BEAR_RE = /bearish|underweight|\bsell\b|downside|\bshort\b|overvalued|pessimistic|caution|deteriorat|\brisk\b/gi;
  var CAUTION_RE = /caution|risk|drawdown|stop.?loss|downside|volatil|exposure|hedge|protect|cut|reduce|limit/gi;

  function countMatches(text, re) { return ((text || "").match(re) || []).length; }

  function fmtK(n) {
    n = n || 0;
    return n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n);
  }

  // Heuristic lean from prose keywords ("bullish" | "bearish" | "neutral").
  function verdictTilt(text) {
    if (!text) return "neutral";
    var low = text.toLowerCase();
    var bull = countMatches(low, BULL_RE);
    var bear = countMatches(low, BEAR_RE);
    if (bull > bear * 1.3) return "bullish";
    if (bear > bull * 1.3) return "bearish";
    return "neutral";
  }

  function escapeHTML(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  window.YiUtil = {
    BULL_RE: BULL_RE,
    BEAR_RE: BEAR_RE,
    CAUTION_RE: CAUTION_RE,
    countMatches: countMatches,
    fmtK: fmtK,
    verdictTilt: verdictTilt,
    escapeHTML: escapeHTML,
  };
})();
