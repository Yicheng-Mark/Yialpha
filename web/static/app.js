// YiAlpha SPA router. Vanilla JS, no framework, no build step.
//
// Routes (hash-based so the server only needs to serve index.html at "/"):
//   #/                       ticker list (home, with client-side filter)
//   #/t/<ticker>             ticker detail (dates + report download)
//   #/t/<ticker>/<date>      full report view
//   #/new                    new-analysis form → task monitor
//   #/task/<id>              task monitor (polls /api/tasks/<id> every 4s)
//   #/compare                multi-ticker rating comparison (chart + matrix)
//   #/health                 preflight self-check
//
// Shared helpers (escapeHTML / verdictTilt / fmtK / countMatches) live in
// common.js (window.YiUtil) — the only cross-file duplication left is none.
//
// Agent markdown is rendered with marked.js and passed through the vendored
// DOMPurify allowlist before insertion. The
// report language itself follows the 🌐 toggle (window.lang() is sent on submit
// and routed to the run_robust child via YIALPHA_OUTPUT_LANGUAGE); there is no
// in-browser post-translation. Only static chrome goes through t() / data-i18n.
//
// Accessibility: interactive elements get a global :focus-visible ring (CSS),
// decorative emoji are aria-hidden, chart canvases carry role="img" +
// aria-label text alternatives, and the shared tooltip answers focusin as well
// as pointerover. Printing expands every <details> via beforeprint.

(function () {
  "use strict";

  var view = function () { return document.getElementById("view"); };
  let pollHandle = null;

  // Shared helpers (single source of truth in common.js).
  var esc = window.YiUtil.escapeHTML;
  var countMatches = window.YiUtil.countMatches;
  var fmtK = window.YiUtil.fmtK;
  var verdictTilt = window.YiUtil.verdictTilt;

  // ----------------------------- helpers ----------------------------------

  async function fetchJSON(url, opts) {
    const r = await fetch(url, opts);
    if (!r.ok) {
      let detail = r.statusText;
      try { detail = (await r.json()).detail || detail; } catch (_) { /* keep */ }
      const e = new Error(`${r.status}: ${detail}`);
      e.status = r.status;
      throw e;
    }
    return r.json();
  }

  // Mirror of cli.utils.is_valid_ticker_input for live client-side validation.
  function validTicker(v) {
    v = (v || "").trim();
    if (!v) return true;
    if (v.length > 32) return false;
    return [...v].every((c) => /[A-Za-z0-9._\-\^=]/.test(c));
  }

  const VERDICT_KEY = { bullish: "verdict_bull", bearish: "verdict_bear", neutral: "verdict_neutral" };

  // Single shared tooltip — enhances only; every value is also shown directly.
  // Pointer events cover the mouse; focusin/focusout cover keyboard users
  // (elements that carry data-tip are focusable, e.g. the overlay KPI tiles).
  let tipEl = null;
  function tipShow(primary, sub) {
    if (!tipEl) { tipEl = document.createElement("div"); tipEl.className = "tooltip"; document.body.appendChild(tipEl); }
    tipEl.textContent = "";
    const a = document.createElement("span"); a.className = "t-val"; a.textContent = primary; tipEl.appendChild(a);
    if (sub) { const b = document.createElement("span"); b.className = "t-sub"; b.textContent = sub; tipEl.appendChild(b); }
    tipEl.classList.add("show");
  }
  function tipAt(x, y) { if (!tipEl) return; tipEl.style.left = Math.min(x + 14, window.innerWidth - 280) + "px"; tipEl.style.top = (y + 18) + "px"; }
  function tipAtRect(el) {
    if (!tipEl || !el) return;
    const r = el.getBoundingClientRect();
    tipAt(r.left, r.bottom);
  }
  function tipHide() { if (tipEl) tipEl.classList.remove("show"); }
  document.addEventListener("pointerover", (e) => {
    const el = e.target.closest && e.target.closest("[data-tip]");
    if (!el) return;
    tipShow(el.getAttribute("data-tip"), el.getAttribute("data-tip-sub") || "");
    tipAt(e.clientX, e.clientY);
  });
  document.addEventListener("pointermove", (e) => { if (tipEl && tipEl.classList.contains("show")) tipAt(e.clientX, e.clientY); });
  document.addEventListener("pointerout", (e) => { if (e.target.closest && e.target.closest("[data-tip]")) tipHide(); });
  document.addEventListener("focusin", (e) => {
    const el = e.target.closest && e.target.closest("[data-tip]");
    if (!el) return;
    tipShow(el.getAttribute("data-tip"), el.getAttribute("data-tip-sub") || "");
    tipAtRect(el);
  });
  document.addEventListener("focusout", (e) => { if (e.target.closest && e.target.closest("[data-tip]")) tipHide(); });

  const MARKDOWN_SANITIZE_CONFIG = Object.freeze({
    // Reports need only the elements emitted by ordinary Markdown. An explicit
    // allowlist excludes SVG/MathML, forms, media, embedded documents and every
    // other active-content surface even if marked preserves raw HTML.
    ALLOWED_TAGS: [
      "a", "blockquote", "br", "code", "del", "em", "h1", "h2", "h3",
      "h4", "h5", "h6", "hr", "kbd", "li", "ol", "p", "pre", "s",
      "span", "strong", "sub", "sup", "table", "tbody", "td", "tfoot",
      "th", "thead", "tr", "ul",
    ],
    ALLOWED_ATTR: ["align", "colspan", "href", "rowspan", "start", "title"],
    FORBID_TAGS: ["math", "script", "style", "svg", "template"],
    FORBID_ATTR: ["formaction", "srcdoc", "style", "xlink:href"],
    ALLOW_ARIA_ATTR: false,
    ALLOW_DATA_ATTR: false,
    SANITIZE_DOM: true,
    SANITIZE_NAMED_PROPS: true,
    // Permit web/mail links and local anchors/paths only. This rejects
    // javascript:, data:, vbscript:, protocol-relative URLs and obfuscated
    // variants after the browser has decoded the attribute value.
    ALLOWED_URI_REGEXP: /^(?:(?:https?):\/\/[^\u0000-\u0020]*|mailto:[^\u0000-\u0020]*|(?:\/(?!\/)|\.{1,2}\/|#|\?)[^:]*|[a-z0-9._~-]+(?:[/?#][^:]*)?)$/i,
  });

  // Render untrusted model/news Markdown to inert, allowlisted HTML. If either
  // dependency is unavailable or rejects the input, fail closed to plain text.
  function md(text) {
    const t = (text || "").trim();
    if (!t) return "";
    if (!window.marked || !window.DOMPurify || typeof window.DOMPurify.sanitize !== "function") {
      return esc(t);
    }
    try {
      return window.DOMPurify.sanitize(window.marked.parse(t), MARKDOWN_SANITIZE_CONFIG);
    } catch (_) {
      return esc(t);
    }
  }

  function safeReportURL(value) {
    const url = String(value || "");
    // Task responses should point to our hash router only. Attribute escaping
    // remains mandatory even after this scheme/origin restriction.
    return url.startsWith("#/t/") ? url : null;
  }

  function errorBox(msg) {
    return `<div class="card"><p class="pill-err"><span aria-hidden="true">⚠ </span>${esc(msg)}</p>
      <p><a class="btn" href="#/">${t("common_back")}</a> <button class="btn" id="err-retry">${t("common_retry")}</button></p></div>`;
  }

  // Error paths keep a retry affordance: re-running route() re-fetches the
  // current hash's data (CSP forbids inline onclick, so bind after insert).
  function renderError(e) {
    view().innerHTML = errorBox(e && e.message ? e.message : String(e));
    const rb = document.getElementById("err-retry");
    if (rb) rb.addEventListener("click", () => route());
  }

  // ----------------------------- router -----------------------------------

  // Highlight the nav link matching the current top-level route.
  function syncNav(parts) {
    const top = parts.length ? parts[0] : "";
    document.querySelectorAll(".nav a").forEach((a) => {
      const href = (a.getAttribute("href") || "").replace(/^#\//, "");
      const active = (top === "" && href === "") || (top !== "" && href === top);
      a.classList.toggle("active", active);
    });
  }

  function route() {
    stopPoll();
    if (window.YiCharts) YiCharts.disposeAll(); // clear charts from previous view
    const raw = location.hash.replace(/^#/, "");
    const parts = raw.split("/").filter(Boolean); // ['t','AAPL','2026-07-03']
    syncNav(parts);
    view().innerHTML = `<p class="muted" role="status">${t("common_loading")}</p>`;

    if (parts.length === 0 || parts[0] === "") return renderHome();
    if (parts[0] === "new") return renderNew();
    if (parts[0] === "health") return renderHealth();
    if (parts[0] === "compare") return renderCompare();
    if (parts[0] === "accuracy") return renderAccuracy();
    if (parts[0] === "task" && parts[1]) return renderTask(decodeURIComponent(parts[1]));
    if (parts[0] === "t" && parts[1]) {
      const ticker = decodeURIComponent(parts[1]);
      if (parts[2]) return renderReport(ticker, decodeURIComponent(parts[2]));
      return renderDetail(ticker);
    }
    renderError("not found");
  }

  window.addEventListener("hashchange", route);
  document.addEventListener("langchange", () => {
    // Re-render current view so dynamic i18n strings update on toggle.
    route();
  });

  // ----------------------------- home -------------------------------------

  function homeSkeleton() {
    const card = `<div class="sk-card"><div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div></div>`;
    return `<div class="section-head"><h1 class="page-title">${t("home_title")}</h1></div><div class="sk-grid">${card.repeat(8)}</div>`;
  }

  // Text summary of rating counts — the donut canvas's text alternative.
  function distAriaLabel(tickers) {
    const counts = {};
    tickers.forEach((x) => { if (x.latest_rating) counts[x.latest_rating] = (counts[x.latest_rating] || 0) + 1; });
    const parts = Object.entries(counts).map(([r, n]) => `${r} ${n}`);
    return `${t("chart_dist_title")}: ${parts.length ? parts.join(", ") : "—"}`;
  }

  async function renderHome() {
    view().innerHTML = homeSkeleton();
    let data;
    try { data = await fetchJSON("/api/tickers"); }
    catch (e) { renderError(e); return; }

    const list = data.tickers || [];
    if (!list.length) {
      view().innerHTML = `
        <div class="empty-state">
          <div class="empty-icon" aria-hidden="true">📊</div>
          <p class="empty-title">${t("home_empty_title")}</p>
          <p class="empty-desc">${t("home_empty")}</p>
          <a class="btn btn-primary" href="#/new">${t("home_empty_cta")}</a>
        </div>`;
      return;
    }

    const RATINGS = ["Buy", "Overweight", "Hold", "Underweight", "Sell"];
    const activeRatings = new Set(); // empty = all ratings pass the filter

    view().innerHTML = `
      <div class="section-head"><h1 class="page-title">${t("home_title")}</h1></div>
      <p class="page-sub">${t("home_sub")}</p>
      ${statRowHTML(list)}
      <div class="filter-bar" id="filter-bar">
        <input class="filter-input" id="filter-text" type="search"
               placeholder="${esc(t("home_filter_ph"))}" aria-label="${esc(t("home_filter_label"))}" />
        ${RATINGS.map((r) => `
          <button type="button" class="filter-chip f-${esc(r)}" data-rating="${esc(r)}" aria-pressed="false">
            <span class="sw" aria-hidden="true"></span>${esc(r)}
          </button>`).join("")}
      </div>
      <div class="dashboard-row">
        <div class="chart-panel">
          <div class="chart-head"><div class="subhead">${t("chart_dist_title")}</div><span class="total" id="dist-total"></span></div>
          <div id="chart-dist" class="chart-area chart-area-md" role="img"></div>
        </div>
      </div>
      <div class="grid" id="home-grid"></div>`;

    // ---- client-side filter: text substring + rating chips; re-renders the
    // grid and the donut from the same cached list (stats stay global). ----
    const gridEl = document.getElementById("home-grid");
    const totalEl = document.getElementById("dist-total");
    const textEl = document.getElementById("filter-text");
    const distEl = document.getElementById("chart-dist");

    function filtered() {
      const q = (textEl.value || "").trim().toLowerCase();
      return list.filter((x) =>
        (!q || x.ticker.toLowerCase().includes(q))
        && (!activeRatings.size || activeRatings.has(x.latest_rating)));
    }

    function applyFilter() {
      const cur = filtered();
      gridEl.innerHTML = cur.length ? cur.map((x) => `
          <a class="card ticker-card" href="#/t/${encodeURIComponent(x.ticker)}">
            <div class="card-accent ${cssRatingClass(x.latest_rating)}"></div>
            ${x.latest_rating ? ratingBadge(x.latest_rating) : ""}
            <div class="ticker">${esc(x.ticker)}</div>
            <div class="meta">${t("home_latest")} ${esc(x.latest_date)} · ${esc(x.run_count)} ${t("home_runs")}</div>
          </a>`).join("")
        : `<p class="filter-none">${t("home_filter_none")}</p>`;
      totalEl.textContent = `${cur.length} ${t("dist_tickers")}`;
      distEl.setAttribute("aria-label", distAriaLabel(cur));
      if (window.YiCharts) YiCharts.drawRatingDist(distEl, cur);
    }

    textEl.addEventListener("input", applyFilter);
    document.querySelectorAll("#filter-bar .filter-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const r = chip.getAttribute("data-rating");
        if (activeRatings.has(r)) activeRatings.delete(r);
        else activeRatings.add(r);
        chip.classList.toggle("active", activeRatings.has(r));
        chip.setAttribute("aria-pressed", activeRatings.has(r) ? "true" : "false");
        applyFilter();
      });
    });

    applyFilter();
  }

  // Home: top-level stat overview cards (tickers, buy share, latest run date).
  function statRowHTML(tickers) {
    const total = tickers.length;
    const buys = tickers.filter((x) => x.latest_rating === "Buy" || x.latest_rating === "Overweight").length;
    const buyPct = total ? Math.round((buys / total) * 100) : 0;
    const latest = tickers.reduce((m, x) => (x.latest_date && x.latest_date > m ? x.latest_date : m), "");
    const card = (icon, k, v, cls) =>
      `<div class="stat-card"><div class="stat-icon" aria-hidden="true">${icon}</div><div class="stat-k">${esc(k)}</div><div class="stat-v${cls ? " " + cls : ""}">${esc(v)}</div></div>`;
    return `<div class="stat-row">
      ${card("📁", t("home_stat_tickers"), total, "")}
      ${card("📈", t("home_stat_buy"), buyPct + "%", "accent")}
      ${card("🕐", t("home_stat_latest"), latest || "—", "")}
    </div>`;
  }

  // ----------------------------- detail -----------------------------------

  // Skeleton mirroring the detail layout (sidebar + chart) so the route
  // doesn't flash a bare "loading" line.
  function detailSkeleton() {
    return `<div class="sk-detail">
      <div class="sk-card sk-side">
        <div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div>
        <div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div>
      </div>
      <div class="sk-card sk-chart">
        <div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div>
      </div>
    </div>`;
  }

  async function renderDetail(ticker) {
    view().innerHTML = detailSkeleton();
    let data;
    try { data = await fetchJSON(`/api/tickers/${encodeURIComponent(ticker)}/runs`); }
    catch (e) { renderError(e); return; }

    const drs = (data.date_ratings && data.date_ratings.length)
      ? data.date_ratings : (data.dates || []).map((d) => ({ date: d }));
    const reports = data.reports || [];
    const dateItems = drs.length
      ? drs.map((dr) => `<li><a class="date-pill" href="#/t/${encodeURIComponent(ticker)}/${encodeURIComponent(String(dr.date || ""))}">
            <span>${esc(dr.date)}</span>${dr.rating ? ratingBadge(dr.rating) : ""}</a></li>`).join("")
      : `<li class="muted">${t("detail_no_dates")}</li>`;
    // Two-line download links: a human label plus the dir's wall-clock stamp —
    // the raw <TICKER>_<stamp> dir name is machine bookkeeping, not UI copy.
    const repItems = reports.length
      ? reports.map((r) =>
          `<li><a class="rep-link" href="/reports/${encodeURIComponent(r.dir)}/complete_report.md" target="_blank" rel="noopener">
             <span class="rl-main"><span aria-hidden="true">📜 </span>${t("detail_report_file")}${r.complete ? "" : esc(t("detail_rep_incomplete"))}</span>
             <span class="rl-sub">${esc(fmtStamp(r.dir))}</span>
           </a></li>`).join("")
      : `<li class="muted">${t("detail_no_reports")}</li>`;

    // Text alternative for the trend canvas: first → last rating.
    const rated = drs.filter((dr) => dr.rating);
    const trendAria = rated.length
      ? `${t("chart_trend_title")}: ${rated[0].date} ${rated[0].rating} → ${rated[rated.length - 1].date} ${rated[rated.length - 1].rating} (${rated.length} ${t("home_runs")})`
      : t("chart_trend_title");

    // Right column: latest-run quick card (rating + overlay summary + CTA)
    // replaces the old inert "pick a date" placeholder. No dates → keep the
    // explanatory empty card.
    const latestDate = drs.length ? String(drs[drs.length - 1].date || "") : "";
    const latestRated = [...drs].reverse().find((dr) => dr.rating);
    const latestCard = latestDate ? `
        <div class="card latest-card">
          <div class="lc-head">
            ${latestRated ? ratingBadge(latestRated.rating, "rating-lg") : ""}
            <div class="lc-date">${esc(latestDate)} · ${drs.length} ${t("home_runs")}</div>
          </div>
          <div class="lc-kpis" id="lc-kpis"><span class="spinner" aria-hidden="true"></span></div>
          <a class="btn btn-primary lc-btn" href="#/t/${encodeURIComponent(ticker)}/${encodeURIComponent(latestDate)}">${t("detail_view_report")}</a>
        </div>` : `<div class="card muted">${t("detail_no_dates")}</div>`;

    view().innerHTML = `
      <p><a href="#/" class="muted">${t("common_back")}</a></p>
      <div class="report-head">
        <div>
          <div class="ticker-big">${esc(ticker)}</div>
          <div class="company">${drs.length} ${t("home_runs")}</div>
        </div>
        <span class="date-tag">${drs.length ? (t("home_latest") + " " + esc(drs[drs.length - 1].date)) : ""}</span>
      </div>
      ${(drs.some((dr) => dr.rating)) ? `
      <div class="chart-panel">
        <div class="chart-head"><div class="subhead">${t("chart_trend_title")}</div><span class="total">${drs.length} ${t("home_runs")}</span></div>
        <div id="chart-trend" class="chart-area chart-area-trend" role="img" aria-label="${esc(trendAria)}"></div>
      </div>` : ""}
      <div class="cols">
        <div class="side">
          <h3>${t("detail_dates")}</h3>
          <ul>${dateItems}</ul>
          <h3>${t("detail_reports")}</h3>
          <ul>${repItems}</ul>
        </div>
        ${latestCard}
      </div>`;

    // initialise the rating trend chart after DOM is ready
    var trendEl = document.getElementById("chart-trend");
    if (trendEl) {
      YiCharts.drawRatingTrend(trendEl, drs, function (date) {
        location.hash = "#/t/" + encodeURIComponent(ticker) + "/" + date;
      });
    }

    // Fill the quick card's overlay mini-KPIs asynchronously so the page
    // renders immediately. Guard against route changes: bail when the
    // placeholder element is gone or replaced (a re-render creates a new one).
    if (latestDate) {
      const kpisEl = document.getElementById("lc-kpis");
      fetchJSON(`/api/tickers/${encodeURIComponent(ticker)}/runs/${encodeURIComponent(latestDate)}`)
        .then((run) => {
          const el = document.getElementById("lc-kpis");
          if (!el || el !== kpisEl) return; // user navigated away / re-rendered
          if (run && run.overlay) el.innerHTML = miniOverlay(run.overlay);
          else el.remove(); // no overlay for this run — collapse the row
        })
        .catch(() => {
          const el = document.getElementById("lc-kpis");
          if (el === kpisEl) el.remove(); // degrade silently; CTA still works
        });
    }
  }

  // ----------------------------- report -----------------------------------

  function ratingBadge(rating, cls) {
    const r = (rating || "Hold").replace(/[^A-Za-z]/g, "");
    return `<span class="rating-badge rating-${esc(r)}${cls ? " " + cls : ""}">${esc(rating || "Hold")}</span>`;
  }

  // Map rating to CSS class for card accent bars (matches --r-* tokens).
  function cssRatingClass(rating) {
    const key = { Buy: "buy", Overweight: "over", Hold: "hold", Underweight: "under", Sell: "sell" };
    return key[rating] || "neutral";
  }

  // Report: severity of the overlay Action / Regime word (shared by the
  // report-page KPI grid and the detail-page latest-run mini card).
  function actionSev(action) {
    const a = String(action || "").toLowerCase();
    return /exit|close|stop|flatten/.test(a) ? "bad" : /reduce|trim|cut|decrease|lower/.test(a) ? "warn" : "good";
  }
  function regimeSev(regime) {
    const r = String(regime || "").toLowerCase();
    return /crash|extreme/.test(r) ? "bad" : /stress|watch|elevated|high/.test(r) ? "warn" : "good";
  }

  // Report: KPI tiles for the quantitative risk overlay. Tiles are focusable
  // and carry a data-tip explanation (shown by the shared tooltip on hover
  // AND on keyboard focus).
  function overlayKPI(ov) {
    if (!ov) return `<div class="kpi-grid"><p class="muted kpi-empty">${t("report_overlay_none")}</p></div>`;
    const num = (k, v, cls, tip) => `<div class="kpi-tile" tabindex="0" data-tip="${esc(tip || "")}">
        <div class="k">${esc(k)}</div><div class="v${cls ? " " + cls : ""}">${esc(v == null || v === "" ? "—" : v)}</div></div>`;
    const pill = (k, v, sev, tip) => `<div class="kpi-tile" tabindex="0" data-tip="${esc(tip || "")}">
        <div class="k">${esc(k)}</div><span class="pill st-${sev}">${esc(v == null || v === "" ? "—" : v)}</span></div>`;
    const rat = ov.rationale ? `<div class="rationale"><b>${t("report_rationale")}</b> ${esc(ov.rationale)}</div>` : "";
    // Perp-only tiles (crypto_perp runs render these; absent on stock/spot).
    const perpTiles = [
      ov.suggested_leverage && num(t("kpi_leverage"), "≤ " + ov.suggested_leverage + "x", "", t("tip_leverage")),
      ov.liquidation_price && num(t("kpi_liq"), ov.liquidation_price, "bad", t("tip_liq")),
      ov.funding_note && num(t("kpi_funding"), ov.funding_note, "", t("tip_funding")),
    ].filter(Boolean).join("");
    return `<div class="kpi-grid">
      ${pill("Action", ov.action, actionSev(ov.action), t("tip_action"))}
      ${num("Target Weight", ov.target_weight, "accent", t("tip_weight"))}
      ${num("Stop Loss", ov.stop_loss, "", t("tip_stop"))}
      ${num("Entry Reference", ov.entry, "", t("tip_entry"))}
      ${perpTiles}
      ${pill("Regime", ov.regime, regimeSev(ov.regime), t("tip_regime"))}
      ${rat}
    </div>`;
  }

  // Detail page: compact overlay summary for the latest-run quick card.
  function miniOverlay(ov) {
    const cell = (k, v, sev) => `<div class="lc-kpi">
      <div class="k">${esc(k)}</div><div class="v${sev ? " st-" + sev : ""}">${esc(v == null || v === "" ? "—" : v)}</div></div>`;
    return `${cell("Action", ov.action, actionSev(ov.action))}
      ${cell("Target Weight", ov.target_weight, "")}
      ${cell("Regime", ov.regime, regimeSev(ov.regime))}`;
  }

  // Detail page: report-dir wall-clock stamp → readable "YYYY-MM-DD HH:mm".
  // Dir names look like <TICKER>_<YYYYMMDD>_<HHMMSS>; unparseable names fall
  // through unchanged so the link text never goes blank.
  function fmtStamp(dir) {
    const name = String(dir || "");
    const parts = name.split("_");
    if (parts.length !== 3) return name;
    const d = parts[1], tm = parts[2];
    if (!/^\d{8}$/.test(d) || !/^\d{6}$/.test(tm)) return name;
    return d.slice(0, 4) + "-" + d.slice(4, 6) + "-" + d.slice(6, 8) + " " + tm.slice(0, 2) + ":" + tm.slice(2, 4);
  }

  // Report: degraded-run banner from the router's data-quality sentinels.
  // A run where core categories served NO_DATA sentinels must not look
  // identical to a fully-fed one in the UI (same contract as the ⚠ banner
  // the risk overlay appends to the decision text).
  function dataQualityBanner(dq) {
    if (!dq) return "";
    const core = dq.core_sentinel_count || 0;
    const opt = dq.optional_sentinel_count || 0;
    const stale = dq.stale_cache_count || 0;
    if (!core && !opt && !stale) return "";
    const rows = (dq.sentinels || []).map(e =>
      `<li><code>${esc(e.method || "")}</code> <span class="muted">(${esc(e.kind || "")})</span> ${esc(e.detail || "")}</li>`).join("");
    const parts = [];
    if (core) parts.push('<span aria-hidden="true">⚠ </span>' + t("dq_core").replace("{n}", core));
    if (stale) parts.push(t("dq_stale").replace("{n}", stale));
    if (opt) parts.push(t("dq_optional").replace("{n}", opt));
    return `<div class="dq-banner${core ? " dq-degraded" : ""}" role="alert">
      <div class="dq-title">${parts.join("<br>")}</div>
      ${rows ? `<ul class="dq-list">${rows}</ul>` : ""}
    </div>`;
  }

  // Report: node-perf chart container (ECharts draws into it after render).
  function perfChartContainer(nodePerf) {
    if (!nodePerf || !nodePerf.nodes) return "";
    const total = (nodePerf.totals && nodePerf.totals.wall_seconds) || 0;
    // Text alternative: slowest node + total wall clock.
    const entries = Object.entries(nodePerf.nodes)
      .map(([name, v]) => [name, (v && v.wall_seconds) || 0])
      .sort((a, b) => b[1] - a[1]);
    const slowest = entries.length ? `${entries[0][0]} ${entries[0][1].toFixed(0)}s` : "—";
    const aria = `${t("chart_perf_title")}: ${t("report_perf_total")} ${total.toFixed(0)}s, ${slowest}`;
    return `<div class="chart-panel tight">
      <div class="chart-head"><div class="subhead">${t("chart_perf_title")}</div><span class="total">${t("report_perf_total")} ${total.toFixed(0)}s</span></div>
      <div id="chart-perf" class="chart-area chart-area-perf" role="img" aria-label="${esc(aria)}"></div>
    </div>`;
  }

  // Bull ↔ Bear debate gauge + verdict — ECharts gauge container.
  function debateChartContainer(deb) {
    const bull = deb.bull_history || "", bear = deb.bear_history || "";
    const bc = bull.length, rc = bear.length, sum = bc + rc || 1;
    const bullPct = Math.round((bc / sum) * 100);
    const bullSeg = countMatches(bull, /Bull\s+Analyst/gi);
    const bearSeg = countMatches(bear, /Bear\s+Analyst/gi);
    const tilt = verdictTilt(deb.judge_decision || "");
    const rounds = (bullSeg || bearSeg) ? `${t("debate_rounds")}: Bull ${bullSeg} · Bear ${bearSeg}` : "";
    const aria = `Bull ${bullPct}% / Bear ${100 - bullPct}%`;
    return `<div class="debate-viz">
      <div id="chart-debate" class="chart-area chart-area-gauge" role="img" aria-label="${esc(aria)}"></div>
      <div class="bal-meta">
        <span>Bull ${fmtK(bc)} ${t("debate_chars")}</span>
        ${rounds ? `<span>${rounds}</span>` : ""}
        <span>${fmtK(rc)} ${t("debate_chars")} Bear</span>
      </div>
      <div class="verdict-row">
        <span class="verdict-tag ${tilt}">${t("debate_verdict")}: ${t(VERDICT_KEY[tilt])}</span>
        <span class="verdict-note">${t("debate_verdict_note")}</span>
      </div>
    </div>`;
  }

  // Risk three-way radar — ECharts radar container + verdict.
  function riskChartContainer(risk) {
    const tilt = verdictTilt(risk.judge_decision || "");
    const roles = [t("sub_aggressive"), t("sub_neutral"), t("sub_conservative")].join(" / ");
    const aria = `${t("sec_risk")}: ${roles}`;
    return `<div class="risk-viz">
      <div id="chart-risk" class="chart-area chart-area-radar" role="img" aria-label="${esc(aria)}"></div>
      <div class="verdict-row flush">
        <span class="verdict-tag ${tilt}">${t("debate_verdict")}: ${t(VERDICT_KEY[tilt])}</span>
        <span class="verdict-note">${t("debate_verdict_note")}</span>
      </div>
    </div>`;
  }

  // Asset-class badge for the report head: stock stays unbadged (the default
  // visual), crypto venues get a small pill so a BTCUSDT perp run is
  // distinguishable from its spot twin at a glance.
  function assetBadge(assetType) {
    const a = String(assetType || "stock");
    if (a === "stock") return "";
    const label = a === "crypto_perp" ? "PERP" : a === "crypto_spot" ? "SPOT" : "CRYPTO";
    return `<span class="asset-badge" tabindex="0" data-tip="${t("tip_asset_badge")}">${label}</span>`;
  }

  async function renderReport(ticker, date) {
    view().innerHTML = `<div class="sk-card sk-report"><div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div></div>`;
    let run;
    try { run = await fetchJSON(`/api/tickers/${encodeURIComponent(ticker)}/runs/${encodeURIComponent(date)}`); }
    catch (e) { renderError(e); return; }

    const s = run.sections || {};
    const deb = s.investment_debate || {};
    const risk = s.risk_debate || {};

    const analysts = [
      [t("sub_market"), s.market_report],
      [t("sub_sentiment"), s.sentiment_report],
      [t("sub_news"), s.news_report],
      [t("sub_fundamentals"), s.fundamentals_report],
    ].filter(([, body]) => body);

    view().innerHTML = `
      <p><a href="#/t/${encodeURIComponent(ticker)}" class="muted">${t("common_back")} ${esc(ticker)}</a></p>
      <div class="report-head">
        <div>
          <div class="ticker-big">${esc(ticker)} ${assetBadge(run.asset_type)}</div>
          <div class="company">${esc(run.company_of_interest || "")}</div>
        </div>
        ${ratingBadge(run.rating, "rating-lg")}
        <span class="date-tag">${esc(run.trade_date)}</span>
      </div>

      ${dataQualityBanner(run.data_quality)}

      <div class="subhead">${t("report_overlay")}</div>
      ${overlayKPI(run.overlay)}

      ${perfChartContainer(run.node_perf)}

      <details class="section" open><summary><span class="caret" aria-hidden="true"></span><span class="num" aria-hidden="true">1</span>${t("sec_analysts")}</summary>
        <div class="section-body">
          ${analysts.length ? analysts.map(([sub, body]) =>
            `<div class="subhead">${esc(sub)}</div><div class="md">${md(body)}</div>`).join("")
            : `<p class="muted">—</p>`}
        </div>
      </details>

      <details class="section"><summary><span class="caret" aria-hidden="true"></span><span class="num" aria-hidden="true">2</span>${t("sec_research")}</summary>
        <div class="section-body">
          ${debateChartContainer(deb)}
          <div class="subhead">${t("sub_bull")}</div><div class="md">${md(deb.bull_history)}</div>
          <div class="subhead">${t("sub_bear")}</div><div class="md">${md(deb.bear_history)}</div>
          <div class="subhead">${t("sub_manager")}</div><div class="md">${md(deb.judge_decision)}</div>
        </div>
      </details>

      <details class="section"><summary><span class="caret" aria-hidden="true"></span><span class="num" aria-hidden="true">3</span>${t("sec_trader")}</summary>
        <div class="section-body"><div class="md">${md(s.trader_decision)}</div></div>
      </details>

      <details class="section"><summary><span class="caret" aria-hidden="true"></span><span class="num" aria-hidden="true">4</span>${t("sec_risk")}</summary>
        <div class="section-body">
          ${riskChartContainer(risk)}
          <div class="subhead">${t("sub_aggressive")}</div><div class="md">${md(risk.aggressive_history)}</div>
          <div class="subhead">${t("sub_conservative")}</div><div class="md">${md(risk.conservative_history)}</div>
          <div class="subhead">${t("sub_neutral")}</div><div class="md">${md(risk.neutral_history)}</div>
          <div class="subhead">${t("sub_risk_judge")}</div><div class="md">${md(risk.judge_decision)}</div>
        </div>
      </details>

      <details class="section" open><summary><span class="caret" aria-hidden="true"></span><span class="num" aria-hidden="true">5</span>${t("sec_pm")}</summary>
        <div class="section-body"><div class="md">${md(s.final_trade_decision)}</div></div>
      </details>`;

    // initialise ECharts instances after DOM is ready
    var perfEl = document.getElementById("chart-perf");
    if (perfEl && run.node_perf) YiCharts.drawNodePerf(perfEl, run.node_perf);
    var debEl = document.getElementById("chart-debate");
    if (debEl) YiCharts.drawDebateBalance(debEl, deb);
    var riskEl = document.getElementById("chart-risk");
    if (riskEl) YiCharts.drawRiskRadar(riskEl, risk);
  }

  // ----------------------------- compare ----------------------------------

  function compareSkeleton() {
    return `<div class="sk-compare">
      <div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div>
    </div>`;
  }

  // Multi-ticker rating comparison: step-line chart + rating matrix table.
  // Backend contract: GET /api/compare → {tickers: [{ticker, date_ratings}]}.
  async function renderCompare() {
    view().innerHTML = compareSkeleton();
    let data;
    try { data = await fetchJSON("/api/compare"); }
    catch (e) { renderError(e); return; }

    // Only tickers with at least one readable rating can be compared.
    const series = ((data.tickers || [])
      .map((x) => ({ ticker: x.ticker, date_ratings: (x.date_ratings || []).filter((dr) => dr.rating) })))
      .filter((x) => x.date_ratings.length);

    if (!series.length) {
      view().innerHTML = `
        <div class="empty-state">
          <div class="empty-icon" aria-hidden="true">📊</div>
          <p class="empty-title">${t("compare_title")}</p>
          <p class="empty-desc">${t("compare_empty")}</p>
          <a class="btn btn-primary" href="#/new">${t("home_empty_cta")}</a>
        </div>`;
      return;
    }

    const dates = [...new Set(series.flatMap((s) => s.date_ratings.map((dr) => dr.date)))].sort();
    const SHORT = { Buy: "Buy", Overweight: "Over", Hold: "Hold", Underweight: "Under", Sell: "Sell" };

    const headerCells = dates.map((d) => `<th>${esc(d)}</th>`).join("");
    const bodyRows = series.map((s, i) => {
      const cat = "cat-" + ((i % 6) + 1);
      const byDate = Object.fromEntries(s.date_ratings.map((dr) => [dr.date, dr.rating]));
      const cells = dates.map((d) => {
        const r = byDate[d];
        return r
          ? `<td><a href="#/t/${encodeURIComponent(s.ticker)}/${encodeURIComponent(d)}"><span class="rt-dot ${cssRatingClass(r)}" aria-hidden="true"></span>${esc(SHORT[r] || r)}</a></td>`
          : `<td><span class="rt-dot na" aria-hidden="true"></span><span class="dim">—</span></td>`;
      }).join("");
      return `<tr><th><span class="cat-dot ${cat}" aria-hidden="true"></span><a href="#/t/${encodeURIComponent(s.ticker)}">${esc(s.ticker)}</a></th>${cells}</tr>`;
    }).join("");

    const aria = `${t("compare_chart_title")}: ${series.map((s) => {
      const last = s.date_ratings[s.date_ratings.length - 1];
      return `${s.ticker} ${last.rating}`;
    }).join(", ")}`;

    view().innerHTML = `
      <p><a href="#/" class="muted">${t("common_back")}</a></p>
      <h1 class="page-title">${t("compare_title")}</h1>
      <p class="page-sub">${t("compare_sub")}</p>
      <div class="chart-panel">
        <div class="chart-head"><div class="subhead">${t("compare_chart_title")}</div><span class="total">${series.length} ${t("dist_tickers")}</span></div>
        <div id="chart-compare" class="chart-area chart-area-compare" role="img" aria-label="${esc(aria)}"></div>
      </div>
      <div class="cmp-table-wrap">
        <div class="subhead">${t("compare_table_title")}</div>
        <table class="cmp-table">
          <thead><tr><th>${esc(t("report_company"))}</th>${headerCells}</tr></thead>
          <tbody>${bodyRows}</tbody>
        </table>
      </div>`;

    const el = document.getElementById("chart-compare");
    if (el && window.YiCharts) {
      YiCharts.drawRatingCompare(el, series, function (ticker, date) {
        location.hash = "#/t/" + encodeURIComponent(ticker) + "/" + encodeURIComponent(date);
      });
    }
  }

  // Rating↔outcome accuracy: serves the verify-history artifact. Every number
  // is shown WITH its sample size — a hit rate without n misleads.
  // Backend contract: GET /api/accuracy → {available, direction, hold_*,
  // by_rating, by_ticker, total_runs, scored, pending, holding_days, ...}.
  async function renderAccuracy() {
    view().innerHTML = `<p class="muted" role="status">${t("common_loading")}</p>`;
    let data;
    try { data = await fetchJSON("/api/accuracy"); }
    catch (e) { renderError(e); return; }

    if (!data.available) {
      view().innerHTML = `
        <p><a href="#/" class="muted">${t("common_back")}</a></p>
        <div class="empty-state">
          <div class="empty-icon" aria-hidden="true">🎯</div>
          <p class="empty-title">${t("accuracy_empty_title")}</p>
          <p class="empty-desc">${t("accuracy_empty")}</p>
          <p><code>yialpha verify-history</code></p>
        </div>`;
      return;
    }

    const pct = (x) => (x == null ? "—" : (100 * x).toFixed(1) + "%");
    const sgn = (x) => (x == null ? "—" : (x >= 0 ? "+" : "") + (100 * x).toFixed(2) + "%");
    const d = data.direction || {};
    const header = `
      <tr>
        <th>${esc(t("accuracy_col_rating"))}</th>
        <th>${esc(t("accuracy_col_n"))}</th>
        <th>${esc(t("accuracy_col_mean"))}</th>
        <th>${esc(t("accuracy_col_dirn"))}</th>
        <th>${esc(t("accuracy_col_hitrate"))}</th>
      </tr>`;
    const row = (label, b) => `
      <tr>
        <th>${esc(label)}</th>
        <td>${b.n}</td>
        <td>${sgn(b.mean_return)}</td>
        <td>${b.directional_n}</td>
        <td>${b.directional_n ? pct(b.hit_rate) : "—"}</td>
      </tr>`;

    const ratingRows = Object.entries(data.by_rating || {}).map(([k, b]) => row(k, b)).join("");
    const tickerRows = Object.entries(data.by_ticker || {}).map(([k, b]) => row(k, b)).join("");
    const scanned = t("accuracy_scanned")
      .replace("{total}", data.total_runs).replace("{scored}", data.scored)
      .replace("{pending}", data.pending).replace("{days}", data.holding_days);
    const holdLine = data.hold_n
      ? "<p>" + t("accuracy_hold").replace("{n}", data.hold_n).replace("{ret}", sgn(data.hold_mean_return)) + "</p>"
      : "";

    view().innerHTML = `
      <p><a href="#/" class="muted">${t("common_back")}</a></p>
      <h1 class="page-title">${t("accuracy_title")}</h1>
      <p class="page-sub">${t("accuracy_sub")}</p>
      <p class="muted">${esc(scanned)} · ${esc(t("accuracy_generated"))}: ${esc(data.generated_at || "—")}</p>
      <div class="card">
        <div class="subhead">${t("accuracy_direction")}</div>
        <p><strong>${d.hits || 0}/${d.n || 0} (${pct(d.hit_rate)})</strong></p>
        ${holdLine}
      </div>
      <div class="cmp-table-wrap">
        <div class="subhead">${t("accuracy_by_rating")}</div>
        <table class="cmp-table"><thead>${header}</thead><tbody>${ratingRows}</tbody></table>
      </div>
      <div class="cmp-table-wrap">
        <div class="subhead">${t("accuracy_by_ticker")}</div>
        <table class="cmp-table"><thead>${header}</thead><tbody>${tickerRows}</tbody></table>
      </div>`;
  }

  // ----------------------------- new analysis -----------------------------

  function renderNew() {
    const today = new Date().toISOString().slice(0, 10);
    view().innerHTML = `
      <p><a href="#/" class="muted">${t("common_back")}</a></p>
      <h1 class="page-title">${t("new_title")}</h1>
      <p class="page-sub">${t("new_sub")}</p>
      <form class="form card" id="new-form">
        <div class="field">
          <label for="f-ticker">${t("new_ticker")}</label>
          <input id="f-ticker" placeholder="${esc(t("new_ticker_ph"))}" autocomplete="off" />
          <div class="hint">${t("new_ticker_hint")}</div>
          <div class="err" id="f-ticker-err"></div>
        </div>
        <div class="field">
          <label for="f-date">${t("new_date")}</label>
          <input id="f-date" type="date" value="${today}" max="${today}" />
          <div class="err" id="f-date-err"></div>
        </div>
        <div class="field">
          <label for="f-asset">${t("new_asset")}</label>
          <select id="f-asset">
            <option value="auto" data-i18n="new_asset_auto">${t("new_asset_auto")}</option>
            <option value="stock" data-i18n="new_asset_stock">${t("new_asset_stock")}</option>
            <option value="crypto" data-i18n="new_asset_crypto">${t("new_asset_crypto")}</option>
            <option value="crypto_spot" data-i18n="new_asset_crypto_spot">${t("new_asset_crypto_spot")}</option>
            <option value="crypto_perp" data-i18n="new_asset_crypto_perp">${t("new_asset_crypto_perp")}</option>
          </select>
        </div>
        <button class="btn btn-primary" type="submit" id="f-submit">${t("new_submit")}</button>
        <div class="err form-err" id="f-form-err"></div>
      </form>`;

    const form = document.getElementById("new-form");
    const tickerInput = document.getElementById("f-ticker");
    const tickerErr = document.getElementById("f-ticker-err");
    tickerInput.addEventListener("input", () => {
      const bad = tickerInput.value.trim() && !validTicker(tickerInput.value);
      tickerErr.textContent = bad ? t("new_invalid") : "";
    });

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const ticker = tickerInput.value.trim();
      const date = document.getElementById("f-date").value;
      const asset = document.getElementById("f-asset").value;
      const dateErr = document.getElementById("f-date-err");
      const formErr = document.getElementById("f-form-err");
      dateErr.textContent = ""; formErr.textContent = "";

      if (!ticker) { tickerErr.textContent = t("new_invalid"); return; }
      if (!validTicker(ticker)) { tickerErr.textContent = t("new_invalid"); return; }
      if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) { dateErr.textContent = t("new_missing_date"); return; }

      const btn = document.getElementById("f-submit");
      btn.disabled = true; btn.textContent = "…";
      try {
        const res = await fetchJSON("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ticker, date, asset_type: asset, language: window.lang() }),
        });
        location.hash = `#/task/${res.task_id}`;
      } catch (err) {
        btn.disabled = false; btn.textContent = t("new_submit");
        formErr.textContent = err.status === 409 ? t("new_busy") : err.message;
      }
    });
  }

  // ----------------------------- task monitor -----------------------------

  function stopPoll() {
    if (pollHandle) { clearInterval(pollHandle); pollHandle = null; }
  }

  // ~8–10 min runs: fill to 90% at the 9-minute mark; the exact finish time
  // is unknown, so cap below 100% until the task actually completes.
  const TASK_FULL_S = 540;

  async function renderTask(taskId) {
    const draw = (st) => {
      const statusKey = { running: "task_status_running", done: "task_status_done",
        error: "task_status_error", pending: "task_status_running" }[st.status] || "task_status_running";
      const statusCls = st.status === "done" ? "pill-ok" : (st.status === "error" ? "pill-err" : "");
      const attempt = st.max_attempts ? `${st.attempt}/${st.max_attempts}` : (st.attempt || "—");
      const spinner = st.status === "running" ? '<span class="spinner" aria-hidden="true"></span>' : "";
      const reportURL = safeReportURL(st.report_url);
      const reportLink = reportURL
        ? `<p class="task-report"><a class="btn btn-primary" href="${esc(reportURL)}">${t("task_view_report")}</a></p>` : "";
      const elapsedS = st.elapsed_s != null ? Math.round(st.elapsed_s) : 0;
      // Drop the ETA row once the promised window has passed — a stale
      // "~8–10 min" past 10 minutes reads as a broken promise.
      const etaRow = (st.status === "running" && elapsedS < 600)
        ? `<div class="row"><span class="muted">${t("task_eta")}</span><span class="muted">~8–10 min</span></div>` : "";
      const pct = st.status === "done" ? 100 : Math.min(90, Math.round((elapsedS / TASK_FULL_S) * 100));
      const logTail = (st.log_tail && st.log_tail.length)
        ? `<div class="subhead">${t("task_log_tail")}</div><div class="log-tail">${esc(st.log_tail.join("\n"))}</div>` : "";
      view().innerHTML = `
        <p><a href="#/" class="muted">${t("common_back")}</a></p>
        <h1 class="page-title">${t("task_title")}</h1>
        <div class="task-card">
          <div class="row"><span class="muted">${t("report_company")}</span><strong>${esc(st.ticker)}</strong></div>
          <div class="row"><span class="muted">${t("new_date")}</span><strong>${esc(st.date)}</strong></div>
          <div class="row"><span class="muted">${t("task_status")}</span><span class="${statusCls}">${spinner} ${t(statusKey)}</span></div>
          <div class="row"><span class="muted">${t("task_attempt")}</span><strong>${esc(attempt)}</strong></div>
          <div class="row"><span class="muted">${t("task_elapsed")}</span><strong>${elapsedS ? elapsedS + "s" : "—"}</strong></div>
          ${etaRow}
          <div class="progress" role="progressbar" aria-label="${esc(t("task_progress"))}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}">
            <div class="progress-bar${st.status === "done" ? " full" : ""}" id="task-progress-bar"></div>
          </div>
          ${st.error ? `<div class="row"><span class="muted">error</span><span class="pill-err">${esc(st.error)}</span></div>` : ""}
          ${reportLink}
          ${logTail}
        </div>`;
      // Dynamic width set via the style property (template stays inline-style-free).
      const bar = document.getElementById("task-progress-bar");
      if (bar) bar.style.width = pct + "%";
    };

    const poll = async () => {
      try {
        const st = await fetchJSON(`/api/tasks/${encodeURIComponent(taskId)}`);
        draw(st);
        if (st.status === "done") { stopPoll(); /* report_url shown, user clicks */ }
        else if (st.status === "error") { stopPoll(); }
        return st.status;
      } catch (e) {
        stopPoll();
        renderError(e);
        return "error";
      }
    };

    draw({ status: "pending", ticker: "…", date: "…", elapsed_s: 0,
      attempt: 0, max_attempts: 0 });
    // Re-arm polling only while the task is still in flight. The old
    // unconditional `if (!pollHandle)` re-armed even after the first poll
    // returned done/error (stopPoll had just cleared it), firing one extra
    // /api/tasks request per finished task.
    const firstStatus = await poll();
    if (firstStatus === "running") pollHandle = setInterval(poll, 4000);
  }

  // ----------------------------- health -----------------------------------

  async function renderHealth() {
    view().innerHTML = `
      <p><a href="#/" class="muted">${t("common_back")}</a></p>
      <h1 class="page-title">${t("health_title")}</h1>
      <p class="page-sub">${t("health_sub")}</p>
      <div class="card" id="health-box"><p class="muted" role="status">${t("health_checking")}</p></div>`;

    const box = document.getElementById("health-box");
    let data;
    try { data = await fetchJSON("/api/health"); }
    catch (e) { box.innerHTML = `<p class="pill-err"><span aria-hidden="true">⚠ </span>${esc(e.message)}</p>`; return; }

    const rows = (data.checks || []).map((c) =>
      `<div class="check-row">
        <span class="dot ${c.ok ? "dot-ok" : "dot-no"}" aria-hidden="true"></span>
        <span class="sr-only">${c.ok ? "OK" : "FAIL"}:</span>
        <span>${esc(c.name)}</span>
        ${c.hint ? `<span class="muted check-hint">${esc(c.hint)}</span>` : ""}
       </div>`).join("");
    box.innerHTML = `
      <p><strong class="${data.ok ? "pill-ok" : "pill-err"}">
        <span aria-hidden="true">${data.ok ? "✅ " : "⚠ "}</span>${data.ok ? "OK" : t("common_error")}</strong></p>
      ${rows}
      <p class="health-actions"><button class="btn" id="health-redo">${t("health_refresh")}</button></p>`;
    document.getElementById("health-redo").addEventListener("click", renderHealth);
  }

  // ----------------------------- theme ------------------------------------

  const THEME_KEY = "yialpha_theme";
  const themeBtn = () => document.getElementById("theme-toggle");

  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") || "dark";
  }

  function setTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    localStorage.setItem(THEME_KEY, t);
    const btn = themeBtn();
    if (btn) btn.textContent = t === "dark" ? "🌙" : "☀️";
    // Redraw all ECharts so they pick up the new theme's CSS colors.
    // CSS vars update synchronously on setAttribute, but redraw must wait
    // a frame for the browser to recompute computed styles.
    requestAnimationFrame(() => { if (window.YiCharts) YiCharts.redrawAll(); });
  }

  // Scripts load at </body>, so #theme-toggle already exists — bind directly
  // rather than waiting for DOMContentLoaded (which may have already fired).
  function initTheme() {
    const btn = themeBtn();
    if (!btn) return;
    btn.textContent = currentTheme() === "dark" ? "🌙" : "☀️";
    btn.addEventListener("click", () => setTheme(currentTheme() === "dark" ? "light" : "dark"));
  }
  initTheme();

  // ----------------------------- print ------------------------------------

  // Reports are documents: expand every collapsed <details> for the print
  // pipeline and restore the previous state afterwards.
  let printedDetails = [];
  window.addEventListener("beforeprint", () => {
    printedDetails = [...document.querySelectorAll("details.section:not([open])")];
    printedDetails.forEach((d) => { d.open = true; });
  });
  window.addEventListener("afterprint", () => {
    printedDetails.forEach((d) => { d.open = false; });
    printedDetails = [];
  });

  // ----------------------------- boot -------------------------------------

  // Skip-to-content: keyboard users land on this button first; Enter moves
  // focus into #view (tabindex="-1" makes the container focusable, not tabbable).
  const skipBtn = document.getElementById("skip-link");
  if (skipBtn) skipBtn.addEventListener("click", () => {
    const v = view();
    if (v) v.focus();
  });

  route();
})();
