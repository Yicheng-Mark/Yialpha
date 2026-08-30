"""CLI symbol validation/classification must agree with the data path.

Regressions for #980 (validation rejected GC=F), #981 (BTCUSD misclassified as
stock), #982 (BTC-USDT accepted but unpriceable on Yahoo).
"""
import pytest

from yialpha.cli.models import AssetType
from yialpha.cli.utils import detect_asset_type, is_valid_ticker_input, normalize_ticker_symbol
from yialpha.dataflows.symbol_utils import normalize_symbol


# --- #982: stablecoin-quoted crypto normalizes to Yahoo's -USD pair ---
@pytest.mark.parametrize("raw,expected", [
    ("BTCUSD", "BTC-USD"),
    ("BTCUSDT", "BTC-USD"),
    ("BTC-USDT", "BTC-USD"),
    ("BTC-USDC", "BTC-USD"),
    ("ethusdt", "ETH-USD"),
    # non-crypto must be untouched
    ("AAPL", "AAPL"),
    ("GC=F", "GC=F"),
    ("600519.SS", "600519.SS"),
    ("EURUSD", "EURUSD=X"),
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
    into the analyst filter (the perp Fundamentals gating depends on it)."""
    import yialpha.cli.main as cli_main
    from yialpha.cli.models import AnalystType

    seen = {}

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


def test_get_user_selections_auto_keeps_detection(monkeypatch):
    cli_main, seen = _mock_interactive(monkeypatch)
    selections = cli_main.get_user_selections(asset_type_override="auto")
    # BTCUSDT detects as the Yahoo spot crypto pair — the historical default.
    assert selections["asset_type"] == "crypto"
    assert seen["asset_type"] == AssetType.CRYPTO


def test_get_user_selections_invalid_override_fails_fast(monkeypatch):
    import typer

    cli_main, _ = _mock_interactive(monkeypatch)
    with pytest.raises(typer.Exit):
        cli_main.get_user_selections(asset_type_override="nasdaq_perp")


def test_cli_normalize_delegates_to_data_layer():
    # CLI must produce the same canonical symbol the data path will price.
    for raw in ("XAUUSD", "BTCUSD", "btc-usdt", "AAPL"):
        assert normalize_ticker_symbol(raw) == normalize_symbol(raw)
