// i18n for YiAgents web. Static chrome is translated here via data-i18n / t().
// The 🌐 toggle ALSO sets the report language: window.lang() is sent on submit
// and routed to the run_robust child via YIAGENTS_OUTPUT_LANGUAGE, so the
// existing get_language_instruction() localizes every agent's output. Agent
// markdown is still rendered as-is — there is no in-browser post-translation.
//
// Usage: elements with data-i18n="key" get textContent rewritten on language
// switch; data-i18n-ph="key" rewrites placeholder. Dynamic strings in app.js
// go through window.t("key"). Choice persists in localStorage.

(function () {
  "use strict";

  const DICT = {
    zh: {
      app_title: "YiAgents 决策可视化",
      nav_home: "历史报告",
      nav_new: "提交新分析",
      nav_health: "环境自检",
      lang_btn: "🌐 EN",
      footer_note: "报告语言跟随 🌐 切换",

      common_loading: "加载中…",
      common_error: "出错了",
      common_retry: "重试",
      common_back: "← 返回",

      // home
      home_title: "历史分析报告",
      home_sub: "点击 ticker 查看历次分析；左侧为分析日期，右侧可下载原始报告。",
      home_empty: "还没有任何报告。去「提交新分析」跑一次？",
      home_latest: "最近",
      home_runs: "次分析",

      // detail
      detail_dates: "分析日期",
      detail_reports: "报告目录（下载）",
      detail_no_reports: "无报告目录",
      detail_no_dates: "该 ticker 无已落盘的完整分析。",
      detail_pick_date: "选择左侧日期查看报告",

      // report
      report_rating: "评级",
      report_company: "标的",
      report_overlay: "量化风控叠加",
      report_overlay_none: "本次分析未开启量化风控层（无 overlay 段）。",
      sec_analysts: "分析师团队",
      sec_research: "Bull / Bear 辩论 + Research Manager 裁决",
      sec_trader: "Trader 方案",
      sec_risk: "风控三方辩论",
      sec_pm: "Portfolio Manager 最终决策",
      sub_market: "Market Analyst",
      sub_sentiment: "Sentiment Analyst",
      sub_news: "News Analyst",
      sub_fundamentals: "Fundamentals Analyst",
      sub_bull: "Bull Researcher",
      sub_bear: "Bear Researcher",
      sub_manager: "Research Manager 裁决",
      sub_aggressive: "激进派 Risk Aggressor",
      sub_conservative: "保守派 Risk Conservator",
      sub_neutral: "中立派 Risk Moderator",
      sub_risk_judge: "风控裁决 Risk Judge",
      report_download: "下载 complete_report.md",
      report_perf: "节点墙钟",
      report_perf_total: "总耗时",
      report_rationale: "理由",
      dist_title: "评级分布",
      dist_tickers: "个标的",
      debate_chars: "字符",
      debate_rounds: "发言段数",
      debate_verdict: "裁决倾向",
      debate_verdict_note: "（基于措辞关键词，非数值评分）",
      verdict_bull: "偏多",
      verdict_bear: "偏空",
      verdict_neutral: "中性",

      // new analysis form
      new_title: "提交新分析",
      new_sub: "后台跑一次完整多智能体决策（约 8–10 分钟）。同一时刻只允许一个任务。",
      new_ticker: "Ticker",
      new_ticker_ph: "如 AAPL、0700.HK、BTC-USD",
      new_ticker_hint: "支持的字符：字母/数字/._-^=（Yahoo 符号）。",
      new_date: "分析日期",
      new_asset: "资产类型",
      new_asset_auto: "自动识别（auto）",
      new_asset_stock: "股票（stock）",
      new_asset_crypto: "加密现货（crypto）",
      new_asset_crypto_perp: "Binance 永续（crypto_perp）",
      new_submit: "开始分析",
      new_invalid: "ticker 不合法（仅字母/数字/._-^=，≤32 字符）。",
      new_missing_date: "请填日期（YYYY-MM-DD）。",
      new_busy: "当前已有任务在跑，请等它结束。",

      // task monitor
      task_title: "任务监控",
      task_status: "状态",
      task_status_running: "运行中",
      task_status_done: "完成",
      task_status_error: "失败",
      task_elapsed: "已耗时",
      task_eta: "预估",
      task_attempt: "尝试",
      task_view_report: "查看报告 →",
      task_log_tail: "run_robust 日志尾",
      task_pending: "等待启动…",

      // health
      health_title: "环境自检",
      health_sub: "复刻 preflight 五项（依赖 / key / 代理 / yfinance / DeepSeek）。含网络探测，稍等数秒。",
      health_refresh: "重新检测",
      health_checking: "检测中…",
    },

    en: {
      app_title: "YiAgents Decision Viewer",
      nav_home: "Reports",
      nav_new: "New Analysis",
      nav_health: "Health Check",
      lang_btn: "🌐 中",
      footer_note: "report language follows the 🌐 toggle",

      common_loading: "Loading…",
      common_error: "Something went wrong",
      common_retry: "Retry",
      common_back: "← Back",

      home_title: "Past Analyses",
      home_sub: "Click a ticker to browse its runs; dates on the left, downloadable raw reports on the right.",
      home_empty: "No reports yet. Run one under “New Analysis”.",
      home_latest: "latest",
      home_runs: "runs",

      detail_dates: "Analysis dates",
      detail_reports: "Report dirs (download)",
      detail_no_reports: "no report dirs",
      detail_no_dates: "No completed analysis on disk for this ticker.",
      detail_pick_date: "Pick a date on the left to view its report",

      report_rating: "Rating",
      report_company: "Ticker",
      report_overlay: "Quantitative Risk Overlay",
      report_overlay_none: "Risk overlay was off for this run (no overlay section).",
      sec_analysts: "Analyst Team",
      sec_research: "Bull / Bear Debate + Research Manager",
      sec_trader: "Trader Plan",
      sec_risk: "Risk Three-way Debate",
      sec_pm: "Portfolio Manager — Final Decision",
      sub_market: "Market Analyst",
      sub_sentiment: "Sentiment Analyst",
      sub_news: "News Analyst",
      sub_fundamentals: "Fundamentals Analyst",
      sub_bull: "Bull Researcher",
      sub_bear: "Bear Researcher",
      sub_manager: "Research Manager Verdict",
      sub_aggressive: "Risk Aggressor",
      sub_conservative: "Risk Conservator",
      sub_neutral: "Risk Moderator",
      sub_risk_judge: "Risk Judge Verdict",
      report_download: "Download complete_report.md",
      report_perf: "Node wall-clock",
      report_perf_total: "total",
      report_rationale: "Rationale",
      dist_title: "Rating distribution",
      dist_tickers: "tickers",
      debate_chars: "chars",
      debate_rounds: "segments",
      debate_verdict: "Verdict lean",
      debate_verdict_note: "(keyword wording lean, not a score)",
      verdict_bull: "bullish",
      verdict_bear: "bearish",
      verdict_neutral: "neutral",

      new_title: "New Analysis",
      new_sub: "Runs a full multi-agent decision in the background (~8–10 min). One task at a time.",
      new_ticker: "Ticker",
      new_ticker_ph: "e.g. AAPL, 0700.HK, BTC-USD",
      new_ticker_hint: "Allowed chars: letters/digits/._-^= (Yahoo symbols).",
      new_date: "Analysis date",
      new_asset: "Asset type",
      new_asset_auto: "Auto-detect (auto)",
      new_asset_stock: "Stock",
      new_asset_crypto: "Crypto spot",
      new_asset_crypto_perp: "Binance perp",
      new_submit: "Run analysis",
      new_invalid: "Invalid ticker (letters/digits/._-^= only, ≤32 chars).",
      new_missing_date: "Please enter a date (YYYY-MM-DD).",
      new_busy: "A task is already running — wait for it to finish.",

      task_title: "Task Monitor",
      task_status: "Status",
      task_status_running: "running",
      task_status_done: "done",
      task_status_error: "error",
      task_elapsed: "Elapsed",
      task_eta: "ETA",
      task_attempt: "Attempt",
      task_view_report: "View report →",
      task_log_tail: "run_robust log tail",
      task_pending: "pending start…",

      health_title: "Environment Check",
      health_sub: "Replicates the 5 preflight checks (deps / key / proxy / yfinance / DeepSeek). Network probes take a few seconds.",
      health_refresh: "Re-run checks",
      health_checking: "Checking…",
    },
  };

  const STORE_KEY = "yiagents_lang";
  let current = localStorage.getItem(STORE_KEY);
  if (current !== "zh" && current !== "en") current = "zh";

  function applyAll(lang) {
    current = lang;
    document.documentElement.lang = lang;
    const d = DICT[lang];
    document.querySelectorAll("[data-i18n]").forEach((e) => {
      const k = e.getAttribute("data-i18n");
      if (d[k] != null) e.textContent = d[k];
    });
    document.querySelectorAll("[data-i18n-ph]").forEach((e) => {
      const k = e.getAttribute("data-i18n-ph");
      if (d[k] != null) e.setAttribute("placeholder", d[k]);
    });
    localStorage.setItem(STORE_KEY, lang);
    document.dispatchEvent(new CustomEvent("langchange", { detail: { lang } }));
  }

  function toggle() {
    applyAll(current === "zh" ? "en" : "zh");
  }

  // Translate a dynamic key from the current language's dictionary.
  window.t = function (key) {
    const v = DICT[current] && DICT[current][key];
    return v != null ? v : key;
  };
  window.lang = () => current;

  document.addEventListener("DOMContentLoaded", () => {
    applyAll(current);
    const btn = document.getElementById("lang-toggle");
    if (btn) btn.addEventListener("click", toggle);
  });

  // Expose for app.js to force a re-render of dynamic i18n text on toggle.
  window.applyLang = applyAll;
})();
