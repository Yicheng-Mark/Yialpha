"""PR5 — news contract-angle source swap, typed soft-miss fallback, ETF branch.

Pins:
  * the router's vendor chain FALLS THROUGH on an empty yfinance result
    (the typed soft-miss that the old "No news found" success-string
    masked);
  * the perp CONTRACT angle is served by an exact-symbol open-web search
    (live) / an honest unavailable placeholder (historical), never by the
    stock-news vendors that cannot answer a MUUSDT query;
  * the dual-angle token budgets (article cap / char cap);
  * high-confidence ETF detection and the fund-framing nudge;
  * the TradFi-perp trading-session wording (no false "24/7" claim).
"""

from __future__ import annotations

from datetime import date

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

import yialpha.agents.analysts.fundamentals_analyst as fa
import yialpha.agents.analysts.news_analyst as news
from yialpha.dataflows.errors import NoMarketDataError
from yialpha.graph.routing import is_exchange_traded_fund


# ---------------------------------------------------------------------------
# Typed soft-miss: empty yfinance falls through to the next vendor
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_empty_yfinance_falls_through_to_alpha_vantage(monkeypatch):
    from yialpha.dataflows import interface

    def empty_yf(ticker, start_date, end_date):
        raise NoMarketDataError(ticker, ticker, "no news items")

    def av_news(ticker, start_date, end_date):
        return "AV NEWS PAYLOAD"

    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_news"], "yfinance", empty_yf
    )
    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_news"], "alpha_vantage", av_news
    )
    out = interface.route_to_vendor("get_news", "AAPL", "2026-01-01", "2026-01-08")
    assert out == "AV NEWS PAYLOAD"


@pytest.mark.unit
def test_all_vendors_empty_returns_no_data_sentinel(monkeypatch):
    from yialpha.dataflows import interface

    def empty_yf(ticker, start_date, end_date):
        raise NoMarketDataError(ticker, ticker, "no news items")

    def empty_av(ticker, start_date, end_date):
        raise NoMarketDataError(ticker, ticker, "no coverage")

    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_news"], "yfinance", empty_yf
    )
    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_news"], "alpha_vantage", empty_av
    )
    out = interface.route_to_vendor("get_news", "AAPL", "2026-01-01", "2026-01-08")
    assert isinstance(out, str)
    assert out.startswith("NO_DATA")  # the router's explicit empty sentinel


@pytest.mark.unit
def test_alpha_vantage_empty_feed_is_typed_no_data(monkeypatch):
    # A 200 body of {"feed": []} is a SOFT MISS, not a success: the vendor
    # fn raises the typed no-data error so the router records KIND_NO_DATA
    # and falls through — instead of masking the fallback and counting an
    # empty answer as a core success.
    import yialpha.dataflows.alpha_vantage_news as avn

    monkeypatch.setattr(
        avn, "_make_api_request", lambda fn, params: '{"feed": []}'
    )
    with pytest.raises(NoMarketDataError, match="empty news feed"):
        avn.get_news("AAPL", "2026-01-01", "2026-01-08")


@pytest.mark.unit
def test_alpha_vantage_empty_feed_falls_through_to_yfinance(monkeypatch):
    from yialpha.dataflows import interface

    def empty_av(ticker, start_date, end_date):  # noqa: ARG001

        # Simulate the vendor's own soft-miss: the empty feed check raised.
        raise NoMarketDataError(ticker, ticker, "empty news feed")

    def yf_news(ticker, start_date, end_date):  # noqa: ARG001
        return "YF NEWS PAYLOAD"

    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_news"], "alpha_vantage", empty_av
    )
    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_news"], "yfinance", yf_news
    )
    out = interface.route_to_vendor("get_news", "AAPL", "2026-01-01", "2026-01-08")
    assert out == "YF NEWS PAYLOAD"


# ---------------------------------------------------------------------------
# Contract-angle source swap
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_contract_angle_live_uses_exact_symbol_web_search(monkeypatch):
    import yialpha.dataflows.tavily as tavily

    seen = {}

    def fake_search(query, max_results=5, scope="news", days=None):
        seen.update(query=query, max_results=max_results, scope=scope, days=days)
        return "1. [MUUSDT contract noted](https://example.com)"

    monkeypatch.setattr(tavily, "get_web_search", fake_search)
    monkeypatch.setattr(news, "is_historical_date", lambda _d: False)
    out = news._fetch_perp_contract_news("MUUSDT", date.today().isoformat())
    assert out.startswith("1. [MUUSDT contract noted](https://example.com)")
    assert seen["query"] == '"MUUSDT" Binance perpetual futures'
    assert seen["scope"] == "news"       # charges the news budget, not another
    assert seen["max_results"] == 5
    # 7-day window: the contract angle matches the company angle's lookback
    # (Tavily news-topic semantics), disclosed in the block.
    assert seen["days"] == 7
    assert "7-day news-topic window" in out


@pytest.mark.unit
def test_contract_angle_unexpected_exception_is_isolated(monkeypatch):
    # Per-angle fault isolation (mirrors the company angle): an unexpected
    # exception degrades to a placeholder instead of killing the node.
    import yialpha.dataflows.tavily as tavily

    def boom(**kwargs):  # noqa: ARG001
        raise RuntimeError("unexpected tavily path")

    monkeypatch.setattr(tavily, "get_web_search", boom)
    monkeypatch.setattr(news, "is_historical_date", lambda _d: False)
    out = news._fetch_perp_contract_news("MUUSDT", date.today().isoformat())
    assert out == "<contract-angle news unavailable: RuntimeError>"


@pytest.mark.unit
def test_contract_angle_historical_is_unavailable_placeholder(monkeypatch):
    monkeypatch.setattr(news, "is_historical_date", lambda _d: True)
    out = news._fetch_perp_contract_news("MUUSDT", "2020-01-10")
    assert out.startswith("<unavailable: contract-side news")
    assert "no as-of boundary" in out


@pytest.mark.unit
def test_contract_angle_char_capped(monkeypatch):
    import yialpha.dataflows.tavily as tavily

    digest = "\n".join(f"{i}. [item {i}](https://x/{i}) " + "y" * 300
                       for i in range(1, 30))
    monkeypatch.setattr(
        tavily, "get_web_search", lambda **k: digest
    )
    monkeypatch.setattr(news, "is_historical_date", lambda _d: False)
    out = news._fetch_perp_contract_news("MUUSDT", date.today().isoformat())
    assert len(out) < len(digest)
    assert "truncated by token budget" in out


@pytest.mark.unit
def test_company_angle_article_cap():
    articles = "\n\n".join(
        f"### Headline {i} (source: P)\nbody {i}\nLink: https://x/{i}"
        for i in range(1, 11)
    )
    block = f"## MU News, from 2026-01-01 to 2026-01-08:\n\n{articles}\n"
    out = news._cap_articles(block, news._MAX_COMPANY_ARTICLES)
    assert "Headline 8" in out
    assert "Headline 9" not in out
    assert "Capped to the first" in out
    # Non-article shapes (placeholders) pass through untouched.
    assert news._cap_articles("<no coverage>", 8) == "<no coverage>"


# ---------------------------------------------------------------------------
# ETF detection + fund framing
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_etf_detection_truth_table():
    assert is_exchange_traded_fund("stock", "SPY") is True
    assert is_exchange_traded_fund("stock", "QQQ") is True
    assert is_exchange_traded_fund("crypto_perp", "SPYUSDT") is True
    assert is_exchange_traded_fund("crypto_perp", "TQQQUSDT") is True
    assert is_exchange_traded_fund("crypto_perp", "MUUSDT") is False
    assert is_exchange_traded_fund("stock", "AAPL") is False
    assert is_exchange_traded_fund("crypto_perp", "BTCUSDT") is False


class _PromptCaptureLLM(Runnable):
    def __init__(self):
        super().__init__()
        self.prompt = None

    def invoke(self, inp, config=None, **kwargs):  # noqa: ARG002
        self.prompt = inp
        return AIMessage(content="MOCK REPORT", tool_calls=[])

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self


def _fundamentals_state(ticker, asset_type):
    return {
        "trade_date": date.today().isoformat(),
        "company_of_interest": ticker,
        "asset_type": asset_type,
        "instrument_context": "CTX",
        "messages": [HumanMessage(content="analyze")],
    }


def _capture_fundamentals_prompt(state):
    from yialpha.dataflows import config as cfgmod

    orig = cfgmod.get_config()
    llm = _PromptCaptureLLM()
    try:
        cfgmod.set_config({**orig, "web_search_enabled": False})
        fa.create_fundamentals_analyst(llm)(state)
    finally:
        cfgmod.set_config(orig)
    return str(llm.prompt)


@pytest.mark.unit
def test_etf_perp_gets_fund_framing_and_perp_nudge():
    prompt = _capture_fundamentals_prompt(
        _fundamentals_state("SPYUSDT", "crypto_perp")
    )
    assert "EXCHANGE-TRADED FUND" in prompt
    assert "volatility decay" in prompt
    # The stock-perp nudge still applies alongside (SPY is an equity perp).
    assert "tokenized-stock perpetual" in prompt


@pytest.mark.unit
def test_etf_plain_stock_gets_fund_framing():
    prompt = _capture_fundamentals_prompt(_fundamentals_state("SPY", "stock"))
    assert "EXCHANGE-TRADED FUND" in prompt
    assert "tokenized-stock perpetual" not in prompt


@pytest.mark.unit
def test_company_stock_gets_no_fund_nudge():
    prompt = _capture_fundamentals_prompt(_fundamentals_state("AAPL", "stock"))
    assert "EXCHANGE-TRADED FUND" not in prompt


@pytest.mark.unit
def test_stock_perp_nudge_no_false_247_claim():
    prompt = _capture_fundamentals_prompt(
        _fundamentals_state("MUUSDT", "crypto_perp")
    )
    # The false claim is gone; sessions wording points at Binance's specs.
    assert "trades 24/7 while" not in prompt
    assert "published trading sessions" in prompt
    assert "NOT the 24/7 crypto calendar" in prompt
