// YiAgents SPA router. Vanilla JS, no framework, no build step.
//
// Routes (hash-based so the server only needs to serve index.html at "/"):
//   #/                       ticker list (home)
//   #/t/<ticker>             ticker detail (dates + report download)
//   #/t/<ticker>/<date>      full report view
//   #/new                    new-analysis form → task monitor
//   #/task/<id>              task monitor (polls /api/tasks/<id> every 4s)
//   #/health                 preflight self-check
//
// Agent markdown is rendered verbatim with the offline marked.js copy; it is
// never translated. Only static chrome goes through t() / data-i18n.

(function () {
  "use strict";

  const view = () => document.getElementById("view");
  let pollHandle = null;

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

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // ----- rating + debate helpers (data is real: char counts, segment counts,
  // and a keyword *wording-lean* — labeled as such, never a numeric score) -----
  function fmtK(n) { n = n || 0; return n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n); }
  function countMatches(text, re) { return ((text || "").match(re) || []).length; }

  // Heuristic lean from prose keywords. Shown in the UI as "wording lean"
  // (措辞), never as a computed bull/bear score — debate state carries no score.
  function verdictTilt(text) {
    if (!text) return "neutral";
    const low = text.toLowerCase();
    const bull = countMatches(low, /bullish|overweight|\bbuy\b|upside|\blong\b|optimistic|compelling|attractive|constructive|favor(?:able|s)?/g);
    const bear = countMatches(low, /bearish|underweight|\bsell\b|downside|\bshort\b|overvalued|pessimistic|caution|deteriorat|\brisk\b/g);
    if (bull > bear * 1.3) return "bullish";
    if (bear > bull * 1.3) return "bearish";
    return "neutral";
  }
  const VERDICT_KEY = { bullish: "verdict_bull", bearish: "verdict_bear", neutral: "verdict_neutral" };

  // Single shared tooltip — enhances only; every value is also shown directly.
  let tipEl = null;
  function tipShow(primary, sub) {
    if (!tipEl) { tipEl = document.createElement("div"); tipEl.className = "tooltip"; document.body.appendChild(tipEl); }
    tipEl.textContent = "";
    const a = document.createElement("span"); a.className = "t-val"; a.textContent = primary; tipEl.appendChild(a);
    if (sub) { const b = document.createElement("span"); b.className = "t-sub"; b.textContent = sub; tipEl.appendChild(b); }
    tipEl.classList.add("show");
  }
  function tipAt(x, y) { if (!tipEl) return; tipEl.style.left = Math.min(x + 14, window.innerWidth - 280) + "px"; tipEl.style.top = (y + 18) + "px"; }
  function tipHide() { if (tipEl) tipEl.classList.remove("show"); }
  document.addEventListener("pointerover", (e) => {
    const el = e.target.closest && e.target.closest("[data-tip]");
    if (!el) return;
    tipShow(el.getAttribute("data-tip"), el.getAttribute("data-tip-sub") || "");
    tipAt(e.clientX, e.clientY);
  });
  document.addEventListener("pointermove", (e) => { if (tipEl && tipEl.classList.contains("show")) tipAt(e.clientX, e.clientY); });
  document.addEventListener("pointerout", (e) => { if (e.target.closest && e.target.closest("[data-tip]")) tipHide(); });

  // Render markdown to HTML. Agent output is trusted (our own LLM calls).
  function md(text) {
    const t = (text || "").trim();
    if (!t) return "";
    try { return window.marked.parse(t); } catch (_) { return esc(t); }
  }

  function fmtAgo(epoch) {
    if (!epoch) return "";
    const s = (Date.now() / 1000) - epoch;
    if (s < 60) return Math.max(1, Math.round(s)) + "s";
    if (s < 3600) return Math.round(s / 60) + "m";
    return Math.round(s / 3600) + "h";
  }

  function errorBox(msg) {
    return `<div class="card"><p class="pill-err">⚠ ${esc(msg)}</p>
      <p><a class="btn" href="#/">${t("common_back")}</a></p></div>`;
  }

  // ----------------------------- router -----------------------------------

  function route() {
    stopPoll();
    const raw = location.hash.replace(/^#/, "");
    const parts = raw.split("/").filter(Boolean); // ['t','AAPL','2026-07-03']
    view().innerHTML = `<p class="muted">${t("common_loading")}</p>`;

    if (parts.length === 0 || parts[0] === "") return renderHome();
    if (parts[0] === "new") return renderNew();
    if (parts[0] === "health") return renderHealth();
    if (parts[0] === "task" && parts[1]) return renderTask(decodeURIComponent(parts[1]));
    if (parts[0] === "t" && parts[1]) {
      const ticker = decodeURIComponent(parts[1]);
      if (parts[2]) return renderReport(ticker, decodeURIComponent(parts[2]));
      return renderDetail(ticker);
    }
    view().innerHTML = errorBox("not found");
  }

  window.addEventListener("hashchange", route);
  document.addEventListener("langchange", () => {
    // Re-render current view so dynamic i18n strings update on toggle.
    route();
  });

  // ----------------------------- home -------------------------------------

  function homeSkeleton() {
    const card = `<div class="sk-card"><div class="skeleton" style="width:38%"></div><div class="skeleton" style="width:72%;height:20px"></div><div class="skeleton" style="width:54%"></div></div>`;
    return `<div class="section-head"><h1 class="page-title">${t("home_title")}</h1></div><div class="sk-grid">${card.repeat(8)}</div>`;
  }

  async function renderHome() {
    view().innerHTML = homeSkeleton();
    let data;
    try { data = await fetchJSON("/api/tickers"); }
    catch (e) { view().innerHTML = errorBox(e.message); return; }

    const list = data.tickers || [];
    if (!list.length) {
      view().innerHTML = `
        <h1 class="page-title">${t("home_title")}</h1>
        <p class="muted">${t("home_empty")}</p>`;
      return;
    }
    view().innerHTML = `
      <div class="section-head"><h1 class="page-title">${t("home_title")}</h1></div>
      <p class="page-sub">${t("home_sub")}</p>
      ${ratingDistHTML(list)}
      <div class="grid">
        ${list.map((x) => `
          <a class="card" href="#/t/${encodeURIComponent(x.ticker)}">
            ${x.latest_rating ? ratingBadge(x.latest_rating) : ""}
            <div class="ticker">${esc(x.ticker)}</div>
            <div class="meta">${t("home_latest")} ${esc(x.latest_date)} · ${x.run_count} ${t("home_runs")}</div>
          </a>`).join("")}
      </div>`;
  }

  // ----------------------------- detail -----------------------------------

  async function renderDetail(ticker) {
    let data;
    try { data = await fetchJSON(`/api/tickers/${encodeURIComponent(ticker)}/runs`); }
    catch (e) { view().innerHTML = errorBox(e.message); return; }

    const drs = (data.date_ratings && data.date_ratings.length)
      ? data.date_ratings : (data.dates || []).map((d) => ({ date: d }));
    const reports = data.reports || [];
    const dateItems = drs.length
      ? drs.map((dr) => `<li><a class="date-pill" href="#/t/${encodeURIComponent(ticker)}/${dr.date}">
            <span>${dr.date}</span>${dr.rating ? ratingBadge(dr.rating) : ""}</a></li>`).join("")
      : `<li class="muted">${t("detail_no_dates")}</li>`;
    const repItems = reports.length
      ? reports.map((r) =>
          `<li><a href="/reports/${encodeURIComponent(r.dir)}/complete_report.md" target="_blank" rel="noopener">📜 ${esc(r.dir)}</a>
             ${r.complete ? "" : '<span class="muted"> (incomplete)</span>'}</li>`).join("")
      : `<li class="muted">${t("detail_no_reports")}</li>`;

    view().innerHTML = `
      <p><a href="#/" class="muted">${t("common_back")}</a></p>
      <div class="report-head">
        <div>
          <div class="ticker-big">${esc(ticker)}</div>
          <div class="company">${drs.length} ${t("home_runs")}</div>
        </div>
        <span class="date-tag">${drs.length ? (t("home_latest") + " " + drs[drs.length - 1].date) : ""}</span>
      </div>
      <div class="cols">
        <div class="side">
          <h3>${t("detail_dates")}</h3>
          <ul>${dateItems}</ul>
          <h3 style="margin-top:18px">${t("detail_reports")}</h3>
          <ul>${repItems}</ul>
        </div>
        <div class="card muted">${drs.length ? t("detail_pick_date") : t("detail_no_dates")}</div>
      </div>`;
  }

  // ----------------------------- report -----------------------------------

  function ratingBadge(rating, cls) {
    const r = (rating || "Hold").replace(/[^A-Za-z]/g, "");
    return `<span class="rating-badge rating-${esc(r)}${cls ? " " + cls : ""}">${esc(rating || "Hold")}</span>`;
  }

  // Home: 5-tier distribution summary bar (stacked segments, 2px surface gaps).
  function ratingDistHTML(tickers) {
    const order = ["Buy", "Overweight", "Hold", "Underweight", "Sell"];
    const key = { Buy: "buy", Overweight: "over", Hold: "hold", Underweight: "under", Sell: "sell" };
    const counts = {}; order.forEach((r) => (counts[r] = 0));
    let total = 0;
    tickers.forEach((x) => { if (x.latest_rating && counts[x.latest_rating] != null) { counts[x.latest_rating]++; total++; } });
    if (!total) return "";
    const present = order.filter((r) => counts[r] > 0);
    const segs = present.map((r) => {
      const pct = (counts[r] / total) * 100;
      const label = pct >= 11 ? `<span class="seg-label">${counts[r]}</span>` : "";
      return `<div class="dist-seg" style="flex:${counts[r]};background:var(--r-${key[r]})" data-tip="${esc(r)} · ${counts[r]} (${pct.toFixed(0)}%)">${label}</div>`;
    }).join("");
    const legend = present.map((r) =>
      `<span class="li"><span class="sw" style="background:var(--r-${key[r]})"></span>${esc(r)} <span class="dim">${counts[r]}</span></span>`).join("");
    return `<div class="dist-wrap">
      <div class="dist-title"><h2>${t("dist_title")}</h2><span class="total">${total} ${t("dist_tickers")}</span></div>
      <div class="dist-bar">${segs}</div>
      <div class="dist-legend">${legend}</div>
    </div>`;
  }

  // Report: KPI tiles for the quantitative risk overlay.
  function overlayKPI(ov) {
    if (!ov) return `<div class="kpi-grid"><p class="muted" style="grid-column:1/-1;margin:4px 0">${t("report_overlay_none")}</p></div>`;
    const num = (k, v, cls) => `<div class="kpi-tile"><div class="k">${esc(k)}</div><div class="v${cls ? " " + cls : ""}">${esc(v == null || v === "" ? "—" : v)}</div></div>`;
    const pill = (k, v, sev) => `<div class="kpi-tile"><div class="k">${esc(k)}</div><span class="pill st-${sev}">${esc(v == null || v === "" ? "—" : v)}</span></div>`;
    const act = String(ov.action || "").toLowerCase();
    const actSev = /exit|close|stop|flatten/.test(act) ? "bad" : /reduce|trim|cut|decrease|lower/.test(act) ? "warn" : "good";
    const reg = String(ov.regime || "").toLowerCase();
    const regSev = /crash|extreme/.test(reg) ? "bad" : /stress|watch|elevated|high/.test(reg) ? "warn" : "good";
    const rat = ov.rationale ? `<div class="rationale"><b>${t("report_rationale")}</b> ${esc(ov.rationale)}</div>` : "";
    return `<div class="kpi-grid">
      ${pill("Action", ov.action, actSev)}
      ${num("Target Weight", ov.target_weight, "accent")}
      ${num("Stop Loss", ov.stop_loss)}
      ${num("Entry Reference", ov.entry)}
      ${pill("Regime", ov.regime, regSev)}
      ${rat}
    </div>`;
  }

  function perfChart(nodePerf) {
    if (!nodePerf || !nodePerf.nodes) return "";
    const entries = Object.entries(nodePerf.nodes)
      .map(([name, v]) => ({ name, wall: v.wall_seconds || 0,
        tok: (v.tokens_in || 0) + (v.tokens_out || 0) + (v.tokens_reasoning || 0) }))
      .filter((e) => e.wall > 0)
      .sort((a, b) => b.wall - a.wall);
    if (!entries.length) return "";
    const max = entries[0].wall;
    const total = entries.reduce((s, e) => s + e.wall, 0);
    const rows = entries.map((e) => `
      <div class="perf-row" data-tip="${esc(e.name)}" data-tip-sub="${e.wall.toFixed(1)}s · ${e.tok.toLocaleString()} tok · ${(e.wall / total * 100).toFixed(0)}%">
        <div class="perf-name">${esc(e.name)}</div>
        <div class="perf-track"><div class="perf-bar" style="width:${(e.wall / max * 100).toFixed(1)}%"></div></div>
        <div class="perf-val">${e.wall.toFixed(1)}s</div>
        <div class="perf-tok">${e.tok.toLocaleString()} tok</div>
      </div>`).join("");
    return `<div class="perf-wrap">
      <div class="perf-head"><div class="subhead" style="margin:0">${t("report_perf")}</div><span class="total">${t("report_perf_total")} ${total.toFixed(0)}s</span></div>
      ${rows}
    </div>`;
  }

  // Bull ↔ Bear balance from character volume + segment counts, plus a
  // keyword wording-lean of the Research Manager verdict (labeled as such).
  function debateHTML(deb) {
    const bull = deb.bull_history || "", bear = deb.bear_history || "";
    const bc = bull.length, rc = bear.length, sum = bc + rc || 1;
    const bullPct = Math.round((bc / sum) * 100);
    const bullSeg = countMatches(bull, /Bull\s+Analyst/gi);
    const bearSeg = countMatches(bear, /Bear\s+Analyst/gi);
    const tilt = verdictTilt(deb.judge_decision || "");
    const rounds = (bullSeg || bearSeg) ? `${t("debate_rounds")}: Bull ${bullSeg} · Bear ${bearSeg}` : "";
    return `<div class="debate">
      <div class="balance">
        <span class="bal-side bull">Bull ${bullPct}%</span>
        <div class="bal-track">
          <div class="bal-fill bull" style="width:${bullPct}%"></div>
          <div class="bal-fill bear" style="width:${100 - bullPct}%"></div>
          <div class="bal-mid"></div>
        </div>
        <span class="bal-side bear">${100 - bullPct}% Bear</span>
      </div>
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

  // Risk three-way overview: per-debater char volume + wording lean, verdict below.
  function riskTrioHTML(risk) {
    const cols = [
      { cls: "aggressive", role: t("sub_aggressive"), text: risk.aggressive_history || "" },
      { cls: "neutral-c", role: t("sub_neutral"), text: risk.neutral_history || "" },
      { cls: "conservative", role: t("sub_conservative"), text: risk.conservative_history || "" },
    ];
    const tilt = verdictTilt(risk.judge_decision || "");
    const colsHTML = cols.map((c) => {
      const ct = verdictTilt(c.text);
      return `<div class="trio-col ${c.cls}">
        <div class="role">${esc(c.role)}</div>
        <div class="name">${fmtK(c.text.length)} ${t("debate_chars")}</div>
        <span class="verdict-tag ${ct}" style="font-size:11px;padding:2px 9px">${t(VERDICT_KEY[ct])}</span>
      </div>`;
    }).join("");
    return `<div class="trio" style="margin-bottom:8px">${colsHTML}</div>
      <div class="verdict-row" style="border-top:1px solid var(--grid);padding-top:12px;margin-top:14px">
        <span class="verdict-tag ${tilt}">${t("debate_verdict")}: ${t(VERDICT_KEY[tilt])}</span>
        <span class="verdict-note">${t("debate_verdict_note")}</span>
      </div>`;
  }

  async function renderReport(ticker, date) {
    view().innerHTML = `<div class="sk-card" style="max-width:560px"><div class="skeleton" style="width:46%"></div><div class="skeleton" style="width:78%;height:22px"></div><div class="skeleton"></div><div class="skeleton" style="width:88%"></div></div>`;
    let run;
    try { run = await fetchJSON(`/api/tickers/${encodeURIComponent(ticker)}/runs/${date}`); }
    catch (e) { view().innerHTML = errorBox(e.message); return; }

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
          <div class="ticker-big">${esc(ticker)}</div>
          <div class="company">${esc(run.company_of_interest || "")}</div>
        </div>
        ${ratingBadge(run.rating, "rating-lg")}
        <span class="date-tag">${esc(run.trade_date)}</span>
      </div>

      <div class="subhead">${t("report_overlay")}</div>
      ${overlayKPI(run.overlay)}

      ${perfChart(run.node_perf)}

      <details class="section" open><summary><span class="caret"></span><span class="num">1</span>${t("sec_analysts")}</summary>
        <div class="section-body">
          ${analysts.length ? analysts.map(([sub, body]) =>
            `<div class="subhead">${esc(sub)}</div><div class="md">${md(body)}</div>`).join("")
            : `<p class="muted">—</p>`}
        </div>
      </details>

      <details class="section"><summary><span class="caret"></span><span class="num">2</span>${t("sec_research")}</summary>
        <div class="section-body">
          ${debateHTML(deb)}
          <div class="subhead">${t("sub_bull")}</div><div class="md">${md(deb.bull_history)}</div>
          <div class="subhead">${t("sub_bear")}</div><div class="md">${md(deb.bear_history)}</div>
          <div class="subhead">${t("sub_manager")}</div><div class="md">${md(deb.judge_decision)}</div>
        </div>
      </details>

      <details class="section"><summary><span class="caret"></span><span class="num">3</span>${t("sec_trader")}</summary>
        <div class="section-body"><div class="md">${md(s.trader_decision)}</div></div>
      </details>

      <details class="section"><summary><span class="caret"></span><span class="num">4</span>${t("sec_risk")}</summary>
        <div class="section-body">
          ${riskTrioHTML(risk)}
          <div class="subhead">${t("sub_aggressive")}</div><div class="md">${md(risk.aggressive_history)}</div>
          <div class="subhead">${t("sub_conservative")}</div><div class="md">${md(risk.conservative_history)}</div>
          <div class="subhead">${t("sub_neutral")}</div><div class="md">${md(risk.neutral_history)}</div>
          <div class="subhead">${t("sub_risk_judge")}</div><div class="md">${md(risk.judge_decision)}</div>
        </div>
      </details>

      <details class="section" open><summary><span class="caret"></span><span class="num">5</span>${t("sec_pm")}</summary>
        <div class="section-body"><div class="md">${md(s.final_trade_decision)}</div></div>
      </details>`;
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
          <label>${t("new_ticker")}</label>
          <input id="f-ticker" placeholder="${esc(t("new_ticker_ph"))}" autocomplete="off" />
          <div class="hint">${t("new_ticker_hint")}</div>
          <div class="err" id="f-ticker-err"></div>
        </div>
        <div class="field">
          <label>${t("new_date")}</label>
          <input id="f-date" type="date" value="${today}" />
          <div class="err" id="f-date-err"></div>
        </div>
        <div class="field">
          <label>${t("new_asset")}</label>
          <select id="f-asset">
            <option value="auto" data-i18n="new_asset_auto">${t("new_asset_auto")}</option>
            <option value="stock" data-i18n="new_asset_stock">${t("new_asset_stock")}</option>
            <option value="crypto" data-i18n="new_asset_crypto">${t("new_asset_crypto")}</option>
            <option value="crypto_perp" data-i18n="new_asset_crypto_perp">${t("new_asset_crypto_perp")}</option>
          </select>
        </div>
        <button class="btn btn-primary" type="submit" id="f-submit">${t("new_submit")}</button>
        <div class="err" id="f-form-err" style="margin-top:10px"></div>
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
          body: JSON.stringify({ ticker, date, asset_type: asset }),
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

  async function renderTask(taskId) {
    const draw = (st) => {
      const statusKey = { running: "task_status_running", done: "task_status_done",
        error: "task_status_error", pending: "task_status_running" }[st.status] || "task_status_running";
      const statusCls = st.status === "done" ? "pill-ok" : (st.status === "error" ? "pill-err" : "");
      const attempt = st.max_attempts ? `${st.attempt}/${st.max_attempts}` : (st.attempt || "—");
      const spinner = st.status === "running" ? '<span class="spinner"></span>' : "";
      const reportLink = st.report_url
        ? `<p style="margin-top:14px"><a class="btn btn-primary" href="${st.report_url}">${t("task_view_report")}</a></p>` : "";
      const logTail = (st.log_tail && st.log_tail.length)
        ? `<div class="subhead">${t("task_log_tail")}</div><div class="log-tail">${esc(st.log_tail.join("\n"))}</div>` : "";
      view().innerHTML = `
        <p><a href="#/" class="muted">${t("common_back")}</a></p>
        <h1 class="page-title">${t("task_title")}</h1>
        <div class="task-card">
          <div class="row"><span class="muted">${t("report_company")}</span><strong>${esc(st.ticker)}</strong></div>
          <div class="row"><span class="muted">${t("new_date")}</span><strong>${esc(st.date)}</strong></div>
          <div class="row"><span class="muted">${t("task_status")}</span><span class="${statusCls}">${spinner} ${t(statusKey)}</span></div>
          <div class="row"><span class="muted">${t("task_attempt")}</span><strong>${attempt}</strong></div>
          <div class="row"><span class="muted">${t("task_elapsed")}</span><strong>${st.elapsed_s != null ? Math.round(st.elapsed_s) + "s" : "—"}</strong></div>
          <div class="row"><span class="muted">${t("task_eta")}</span><span class="muted">~8–10 min</span></div>
          ${st.error ? `<div class="row"><span class="muted">error</span><span class="pill-err">${esc(st.error)}</span></div>` : ""}
          ${reportLink}
          ${logTail}
        </div>`;
    };

    const poll = async () => {
      try {
        const st = await fetchJSON(`/api/tasks/${encodeURIComponent(taskId)}`);
        draw(st);
        if (st.status === "done") { stopPoll(); /* report_url shown, user clicks */ }
        else if (st.status === "error") { stopPoll(); }
      } catch (e) {
        stopPoll();
        view().innerHTML = errorBox(e.message);
      }
    };

    draw({ status: "pending", ticker: "…", date: "…", elapsed_s: 0,
      attempt: 0, max_attempts: 0 });
    await poll();
    if (!pollHandle) pollHandle = setInterval(poll, 4000);
  }

  // ----------------------------- health -----------------------------------

  async function renderHealth() {
    view().innerHTML = `
      <p><a href="#/" class="muted">${t("common_back")}</a></p>
      <h1 class="page-title">${t("health_title")}</h1>
      <p class="page-sub">${t("health_sub")}</p>
      <div class="card" id="health-box"><p class="muted">${t("health_checking")}</p></div>`;

    const box = document.getElementById("health-box");
    let data;
    try { data = await fetchJSON("/api/health"); }
    catch (e) { box.innerHTML = `<p class="pill-err">⚠ ${esc(e.message)}</p>`; return; }

    const rows = (data.checks || []).map((c) =>
      `<div class="check-row">
        <span class="dot ${c.ok ? "dot-ok" : "dot-no"}"></span>
        <span>${esc(c.name)}</span>
        ${c.hint ? `<span class="muted" style="margin-left:auto">${esc(c.hint)}</span>` : ""}
       </div>`).join("");
    box.innerHTML = `
      <p><strong class="${data.ok ? "pill-ok" : "pill-err"}">
        ${data.ok ? "✅ OK" : "⚠ " + t("common_error")}</strong></p>
      ${rows}
      <p style="margin-top:14px"><button class="btn" id="health-redo">${t("health_refresh")}</button></p>`;
    document.getElementById("health-redo").addEventListener("click", renderHealth);
  }

  // ----------------------------- boot -------------------------------------

  route();
})();
