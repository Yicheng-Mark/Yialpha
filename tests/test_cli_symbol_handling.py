"""CLI symbol validation/classification must agree with the data path.

Regressions for #980 (validation rejected GC=F), #981 (BTCUSD misclassified as
stock), #982 (BTC-USDT accepted but unpriceable on Yahoo).
"""
import pytest

from yialpha.cli.models import AssetType
from yialpha.cli.utils import (
    detect_asset_type,
    is_canonical_ymd,
    is_valid_ticker_input,
    normalize_ticker_symbol,
)
from yialpha.dataflows.symbol_utils import normalize_symbol


# --- #982: stablecoin-quoted crypto normalizes to Yahoo's -USD pair ---
@pytest.mark.parametrize("raw,expected", [
    ("BTCUSD", "BTC-USD"),
    ("BTCUSDT", "BTC-USD"),
    ("BTC-USDT", "BTC-USD"),
    ("BTC-USDC", "BTC-USD"),
    ("ethusdt", "ETH-USD"),
    # non-crypto must be untouched (former forex/CFD/alias mapping was
    # removed with the US+crypto-only policy — they pass through and are
    # rejected by the market gate instead)
    ("AAPL", "AAPL"),
    ("600519.SS", "600519.SS"),
    ("EURUSD", "EURUSD"),
    ("XAUUSD", "XAUUSD"),
])
def test_normalize_symbol_crypto_and_passthrough(raw, expected):
    assert normalize_symbol(raw) == expected


# --- #980: validation accepts Yahoo futures/forex symbols ---
@pytest.mark.parametrize("value,ok", [
    ("GC=F", True),
    ("EURUSD=X", True),
    ("AAPL", True),
    ("0700.HK", True),
    ("^GSPC", True),
    ("", True),                 # empty -> defaults to SPY downstream
    ("bad symbol!", False),     # space + '!' rejected
    ("A" * 40, False),          # too long
])
def test_ticker_input_validation(value, ok):
    assert is_valid_ticker_input(value) is ok


# --- market gate layering in the interactive prompt (US+crypto only) -------
@pytest.mark.parametrize("value,kind", [
    ("AAPL", None),            # supported -> no message
    ("BRK.B", None),
    ("BTC-USD", None),
    ("", None),                # empty defaults to SPY downstream
    ("0700.HK", "market"),     # charset-valid but unsupported market
    ("600519.SS", "market"),
    ("EURUSD", "market"),
    ("XAUUSD", "market"),
    ("bad symbol!", "charset"),  # a typo reads as a typo, not a policy hit
])
def test_ticker_prompt_error_layering(value, kind):
    from yialpha.cli.utils import _ticker_prompt_error

    msg = _ticker_prompt_error(value)
    if kind is None:
        assert msg is None
    elif kind == "charset":
        assert msg is not None and "e.g. AAPL" in msg
    else:
        assert msg is not None and "only" in msg  # policy message


# --- round-11: shared canonical-ymd gate (run_robust --date + /api/analyze) ---
@pytest.mark.parametrize("value,ok", [
    ("2026-09-01", True),
    ("2026-9-1", False),       # non-zero-padded: bare strptime accepts it
    ("2026/09/01", False),     # wrong separator
    ("garbage", False),
    ("2026-09-31", False),     # strptime rejects the impossible day
    ("", False),
])
def test_is_canonical_ymd(value, ok):
    """The predicate both parents of the run_robust subprocess apply: accept
    exactly what the child's --date guard accepts (strftime round-trip,
    zero-padded ISO) — a task accepted upstream would flip to exit 2 seconds
    later in the subprocess."""
    assert is_canonical_ymd(value) is ok


# --- #981/#982: asset-type classified on the canonical symbol ---
@pytest.mark.parametrize("raw,expected", [
    ("BTCUSD", AssetType.CRYPTO),
    ("BTC-USDT", AssetType.CRYPTO),
    ("BTC-USD", AssetType.CRYPTO),
    ("ETHUSD", AssetType.CRYPTO),
    # Compact USDT/USDC forms for bases outside the Yahoo whitelist
    # (tokenized-stock perps + unlisted alts) must not fall through to STOCK.
    ("MUUSDT", AssetType.CRYPTO),
    ("SPCXUSDT", AssetType.CRYPTO),
    ("PEPEUSDT", AssetType.CRYPTO),
    ("1000PEPEUSDT", AssetType.CRYPTO),
    ("pepe-usdc", AssetType.CRYPTO),
    # Guards: forex canonicalizes to PAIR=X (never ends in USDT/USDC), and
    # plain equities/indexes/A-shares stay STOCK.
    ("EURUSD", AssetType.STOCK),
    ("CHFUSD", AssetType.STOCK),
    ("AAPL", AssetType.STOCK),
    ("GC=F", AssetType.STOCK),
    ("600519.SS", AssetType.STOCK),
])
def test_detect_asset_type(raw, expected):
    assert detect_asset_type(raw) == expected


# --- interactive analyze: --asset-type override reaches perp runs ---------
# crypto_perp is never auto-detected (BTCUSDT -> CRYPTO spot by design), so
# the override is the ONLY way `yialpha analyze` reaches a perp run.


def _mock_interactive(monkeypatch):
    """Patch the interactive prompts; also record the asset_type handed to
    select_analysts so tests can assert the override/detection propagates
    into the analyst filter (the perp Fundamentals gating depends on it).

    A crypto_perp selection also warms the live equity-perp listing cache
    before the analyst filter. The CLI imports that warmer lazily *inside*
    get_user_selections, so the stub has to go on the vendor module (a patch
    on yialpha.cli.main would never be looked up); left real, this test sent
    a Binance exchangeInfo request from the suite (round-16 network guard).
    """
    import yialpha.cli.main as cli_main
    from yialpha.cli.models import AnalystType
    from yialpha.dataflows import binance as binance_mod

    seen = {}
    warmed: list[int] = []

    def fake_select_analysts(asset_type, ticker=""):
        seen["asset_type"] = asset_type
        seen["ticker"] = ticker
        return [AnalystType.MARKET, AnalystType.SOCIAL, AnalystType.NEWS]

    monkeypatch.setattr(cli_main, "get_ticker", lambda: "BTCUSDT")
    monkeypatch.setattr(cli_main, "get_analysis_date", lambda: "2026-01-10")
    monkeypatch.setattr(cli_main, "ask_output_language", lambda: "中文")
    monkeypatch.setattr(cli_main, "select_analysts", fake_select_analysts)
    monkeypatch.setattr(cli_main, "select_research_depth", lambda: "quick")
    monkeypatch.setattr(
        cli_main, "select_llm_provider",
        lambda: ("deepseek", "https://api.deepseek.com"),
    )
    monkeypatch.setattr(cli_main, "ensure_api_key", lambda p: None)
    monkeypatch.setattr(
        binance_mod, "warm_equity_perp_bases", lambda: warmed.append(1),
    )
    seen["warm"] = warmed
    monkeypatch.setenv("YIALPHA_OUTPUT_LANGUAGE", "中文")
    monkeypatch.setenv("YIALPHA_QUICK_THINK_LLM", "deepseek-chat")
    monkeypatch.setenv("YIALPHA_DEEP_THINK_LLM", "deepseek-reasoner")
    return cli_main, seen


def test_get_user_selections_asset_type_override_to_perp(monkeypatch):
    cli_main, seen = _mock_interactive(monkeypatch)
    selections = cli_main.get_user_selections(asset_type_override="crypto_perp")
    assert selections["asset_type"] == "crypto_perp"
    # The override must reach the analyst filter — not the detected spot type.
    assert seen["asset_type"] == AssetType.CRYPTO_PERP
    # A perp run warms the equity-perp listing exactly once, before the
    # analyst filter consults it.
    assert len(seen["warm"]) == 1


def test_get_user_selections_auto_keeps_detection(monkeypatch):
    cli_main, seen = _mock_interactive(monkeypatch)
    selections = cli_main.get_user_selections(asset_type_override="auto")
    # BTCUSDT detects as the Yahoo spot crypto pair — the historical default.
    assert selections["asset_type"] == "crypto"
    assert seen["asset_type"] == AssetType.CRYPTO
    # Spot crypto has no equity-perp filter to prime: no warming request.
    assert seen["warm"] == []


def test_get_user_selections_invalid_override_fails_fast(monkeypatch):
    import typer

    cli_main, _ = _mock_interactive(monkeypatch)
    with pytest.raises(typer.Exit):
        cli_main.get_user_selections(asset_type_override="nasdaq_perp")


def test_cli_normalize_delegates_to_data_layer():
    # CLI must produce the same canonical symbol the data path will price.
    for raw in ("muusdt", "BTCUSD", "btc-usdt", "AAPL"):
        assert normalize_ticker_symbol(raw) == normalize_symbol(raw)
