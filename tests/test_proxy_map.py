"""dataflows.utils.proxy_map: shared proxy resolution for US/quote sources.

Regression guard for the divergence that ``binance`` read only HTTP_PROXY/
HTTPS_PROXY while ``sec_edgar`` also fell back to ALL_PROXY — a user setting
only ALL_PROXY got proxied SEC traffic but *unproxied* Binance traffic (hang
or IP leak). Both vendors now resolve through this one helper.
"""
import os
import unittest

from yialpha.dataflows.utils import proxy_map

_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")


class TestProxyMap(unittest.TestCase):
    def setUp(self):
        self._saved = {v: os.environ.get(v) for v in _VARS}
        for v in _VARS:
            os.environ.pop(v, None)

    def tearDown(self):
        for v, val in self._saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val

    def test_nothing_set_yields_empty(self):
        # No proxy env vars set -> empty dict (no None values; requests'
        # proxies contract is Mapping[str, str], and {} means "no proxy").
        self.assertEqual(proxy_map(), {})

    def test_scheme_specific_used(self):
        os.environ["HTTP_PROXY"] = "http://h:8080"
        os.environ["HTTPS_PROXY"] = "http://s:8080"
        self.assertEqual(
            proxy_map(), {"http": "http://h:8080", "https": "http://s:8080"}
        )

    def test_all_proxy_fallback_for_both_schemes(self):
        # The regression: only ALL_PROXY set must proxy BOTH http and https.
        os.environ["ALL_PROXY"] = "socks5h://127.0.0.1:1080"
        self.assertEqual(
            proxy_map(),
            {"http": "socks5h://127.0.0.1:1080", "https": "socks5h://127.0.0.1:1080"},
        )

    def test_scheme_specific_takes_precedence_over_all_proxy(self):
        os.environ["HTTP_PROXY"] = "http://h:8080"
        os.environ["ALL_PROXY"] = "socks5h://127.0.0.1:1080"
        m = proxy_map()
        self.assertEqual(m["http"], "http://h:8080")  # scheme-specific wins
        self.assertEqual(m["https"], "socks5h://127.0.0.1:1080")  # ALL_PROXY fallback


class TestSharedAcrossVendors(unittest.TestCase):
    """binance and sec_edgar resolve proxies through the same helper now."""

    def test_no_divergent_local_proxy_helpers(self):
        import yialpha.dataflows.binance as bn
        import yialpha.dataflows.sec_edgar as se

        # The divergent local _proxies() helpers were removed in favour of the
        # shared utils.proxy_map(); their continued absence is the contract that
        # the two vendors can never drift apart again.
        self.assertFalse(hasattr(bn, "_proxies"))
        self.assertFalse(hasattr(se, "_proxies"))


if __name__ == "__main__":
    unittest.main()
