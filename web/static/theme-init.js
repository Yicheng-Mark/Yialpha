(function () {
  "use strict";

  var theme = null;
  try {
    theme = localStorage.getItem("yiagents_theme");
  } catch (_) {
    // Storage may be unavailable in hardened/private browser contexts.
  }
  if (theme !== "light" && theme !== "dark") {
    theme = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }
  document.documentElement.setAttribute("data-theme", theme);
})();
