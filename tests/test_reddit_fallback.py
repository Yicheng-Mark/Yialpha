"""Tests for the Reddit fetcher: RSS feed parsing + 429 backoff, the OAuth-API-
first path (token caching, negative cache, engagement fields, RSS fallback on
every failure mode), and chunked-transfer error handling (#1024)."""

from __future__ import annotations

import base64
import http.client
import json
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from yiagents.dataflows import reddit


@pytest.fixture(autouse=True)
def _clean_reddit_oauth_state(monkeypatch):
    """Every test starts with no Reddit OAuth creds and an empty token cache.

    Guards against a local ``.env`` leaking ``REDDIT_CLIENT_ID`` / ``_SECRET``
    into the test process (which would flip the dispatcher onto the JSON path
    and break the RSS-default assertions). Individual OAuth tests set their own
    creds explicitly via ``monkeypatch.setenv``.
    """
    reddit._token_cache = None
    for var in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"):
        monkeypatch.delenv(var, raising=False)
    yield


def _set_creds(monkeypatch, client_id="cid", secret="sec"):
    """Convenience: arm the OAuth dispatcher for one test."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", client_id)
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", secret)

_SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>NVDA earnings beat, stock pops</title>
    <published>2026-05-20T14:30:00+00:00</published>
    <content type="html">&lt;!-- SC_OFF --&gt;&lt;div class="md"&gt;&lt;p&gt;Great &lt;b&gt;quarter&lt;/b&gt; for NVDA&amp;#39;s datacenter unit.&lt;/p&gt;&lt;/div&gt;&lt;!-- SC_ON --&gt;</content>
  </entry>
  <entry>
    <title>Is NVDA overvalued?</title>
    <published>2026-05-19T09:00:00Z</published>
    <content type="html">&lt;p&gt;Forward P/E discussion&lt;/p&gt;</content>
  </entry>
</feed>
"""


def _resp(read_fn):
    """A minimal context-manager response whose read() runs ``read_fn``."""
    class _Resp:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *a):
            return False

        def read(self_inner):
            return read_fn()
    return _Resp()


def _atom_resp():
    return _resp(lambda: _SAMPLE_ATOM.encode("utf-8"))


def _json_resp(obj):
    """A response whose read() returns ``obj`` JSON-encoded (OAuth token /
    search payloads)."""
    return _resp(lambda: json.dumps(obj).encode("utf-8"))


def _token_resp():
    return _json_resp({"access_token": "abc", "token_type": "bearer", "expires_in": 3600})


def _raise(exc):
    def _r():
        raise exc
    return _resp(_r)


@pytest.mark.unit
class TestIsoToTimestamp:
    def test_parses_offset_and_z(self):
        assert reddit._iso_to_timestamp("2026-05-20T14:30:00+00:00") > 0
        assert reddit._iso_to_timestamp("2026-05-19T09:00:00Z") > 0

    def test_none_and_garbage_return_none(self):
        assert reddit._iso_to_timestamp(None) is None
        assert reddit._iso_to_timestamp("not-a-date") is None


@pytest.mark.unit
class TestStripHtml:
    def test_extracts_between_sc_markers_and_unescapes(self):
        raw = "<!-- SC_OFF --><div class=\"md\"><p>Great <b>quarter</b> &amp; more</p></div><!-- SC_ON -->"
        assert reddit._strip_html(raw) == "Great quarter & more"

    def test_empty(self):
        assert reddit._strip_html("") == ""


@pytest.mark.unit
class TestRssParsing:
    def test_parses_atom_entries(self):
        with patch.object(reddit, "urlopen", return_value=_atom_resp()):
            posts = reddit._fetch_subreddit_rss("NVDA", "stocks", limit=5, timeout=5.0)
        assert len(posts) == 2
        assert posts[0]["title"] == "NVDA earnings beat, stock pops"
        assert posts[0]["source"] == "rss"
        assert posts[0]["score"] is None
        assert posts[0]["num_comments"] is None
        assert posts[0]["created_utc"] > 0
        assert "datacenter unit" in posts[0]["selftext"]

    def test_malformed_xml_fails_open(self):
        with patch.object(reddit, "urlopen", return_value=_resp(lambda: b"<<not xml>>")):
            assert reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0) == []


@pytest.mark.unit
class TestFetchSubredditIsRssFirst:
    """Without OAuth creds (the keyless default) the per-subreddit fetch goes
    straight to RSS — it must not touch the network or the JSON path."""

    def test_delegates_to_rss_without_touching_json(self):
        sentinel = [{"title": "x", "source": "rss", "score": None,
                     "num_comments": None, "created_utc": None, "selftext": ""}]
        with patch.object(reddit, "_fetch_subreddit_rss", return_value=sentinel) as rss, \
             patch.object(reddit, "urlopen",
                          side_effect=AssertionError("JSON endpoint must not be called")):
            out = reddit._fetch_subreddit("NVDA", "stocks", 5, 5.0)
        rss.assert_called_once()
        assert out is sentinel


@pytest.mark.unit
class TestJsonPathFallsBackToRss:
    """The OAuth JSON path still degrades to RSS on an API error (a 403 from the
    authenticated endpoint is handled the same way as the legacy public-JSON
    WAF 403 of #862, which oauth.reddit.com sidesteps entirely)."""

    def test_403_triggers_rss(self, monkeypatch):
        _set_creds(monkeypatch)
        err = HTTPError("url", 403, "Blocked", {}, None)
        rss_posts = [{"title": "x", "source": "rss", "score": None,
                      "num_comments": None, "created_utc": None, "selftext": ""}]
        # urlopen: token fetch (OK) then search (403 raised by urlopen itself).
        with patch.object(reddit, "urlopen", side_effect=[_token_resp(), err]), \
             patch.object(reddit, "_fetch_subreddit_rss", return_value=rss_posts) as rss:
            out = reddit._fetch_subreddit_json("NVDA", "stocks", 5, 5.0)
        rss.assert_called_once()
        assert out and out[0]["source"] == "rss"


@pytest.mark.unit
class TestRss429Backoff:
    def test_429_then_success_retries_once(self):
        err = HTTPError("url", 429, "Too Many Requests", {}, None)
        with patch.object(reddit, "urlopen", side_effect=[err, _atom_resp()]) as op, \
             patch.object(reddit.time, "sleep") as slept:
            posts = reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        assert op.call_count == 2          # original + exactly one retry
        slept.assert_called_once()         # backed off before retrying
        assert len(posts) == 2

    def test_429_twice_gives_up_after_one_retry(self):
        err = HTTPError("url", 429, "Too Many Requests", {}, None)
        with patch.object(reddit, "urlopen", side_effect=[err, err]) as op, \
             patch.object(reddit.time, "sleep"):
            posts = reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        assert op.call_count == 2          # one retry, then gives up cleanly
        assert posts == []

    def test_retry_after_header_is_honoured(self):
        err = HTTPError("url", 429, "Too Many Requests", {"Retry-After": "12"}, None)
        with patch.object(reddit, "urlopen", side_effect=[err, _atom_resp()]), \
             patch.object(reddit.time, "sleep") as slept:
            reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0)
        slept.assert_called_once_with(12.0)


@pytest.mark.unit
class TestChunkedTransferErrorsHandled:
    """IncompleteRead/RemoteDisconnected come from http.client and are NOT
    OSErrors, so they were previously uncaught and crashed the pipeline (#1024)."""

    def test_rss_incomplete_read_degrades_to_empty(self):
        with patch.object(reddit, "urlopen", return_value=_raise(http.client.IncompleteRead(b""))):
            assert reddit._fetch_subreddit_rss("NVDA", "stocks", 5, 5.0) == []

    def test_json_incomplete_read_falls_back_to_rss(self, monkeypatch):
        _set_creds(monkeypatch)
        # token fetch (OK) then search body chunked-fragments mid-read.
        with patch.object(reddit, "urlopen",
                          side_effect=[_token_resp(),
                                       _raise(http.client.IncompleteRead(b""))]), \
             patch.object(reddit, "_fetch_subreddit_rss", return_value=[]) as rss:
            reddit._fetch_subreddit_json("NVDA", "stocks", 5, 5.0)
        rss.assert_called_once()


@pytest.mark.unit
class TestFormatterHandlesRssPosts:
    def test_rss_posts_omit_fake_counts_and_note_source(self):
        rss_posts = [{
            "title": "NVDA pops", "score": None, "num_comments": None,
            "created_utc": reddit._iso_to_timestamp("2026-05-20T14:30:00Z"),
            "selftext": "great quarter", "source": "rss",
        }]
        with patch.object(reddit, "_fetch_subreddit", return_value=rss_posts):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",), inter_request_delay=0)
        assert "via RSS feed" in out
        assert "↑" not in out  # no fake score arrow
        assert "NVDA pops" in out
        assert "great quarter" in out

    def test_json_posts_still_show_counts(self):
        json_posts = [{
            "title": "NVDA pops", "score": 1234, "num_comments": 56,
            "created_utc": reddit._iso_to_timestamp("2026-05-20T14:30:00Z"),
            "selftext": "",
        }]
        with patch.object(reddit, "_fetch_subreddit", return_value=json_posts):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks",), inter_request_delay=0)
        assert "1234↑" in out
        assert "56c" in out
        assert "via RSS" not in out


@pytest.mark.unit
class TestOauthTokenAndPath:
    """The OAuth-API-first path: token caching, negative cache, engagement
    fields, and RSS fallback on every failure mode. ``_clean_reddit_oauth_state``
    clears creds/cache before each test; OAuth tests arm creds via ``_set_creds``.
    """

    def test_no_creds_returns_none_and_skips_network(self):
        # autouse fixture has cleared creds → no token, no network call.
        with patch.object(reddit, "urlopen",
                          side_effect=AssertionError("must not hit network without creds")):
            assert reddit._get_oauth_token(5.0) is None

    def test_token_cached_across_calls(self, monkeypatch):
        _set_creds(monkeypatch)
        with patch.object(reddit, "urlopen", side_effect=[_token_resp(), _token_resp()]) as op:
            t1 = reddit._get_oauth_token(5.0)
            t2 = reddit._get_oauth_token(5.0)
        assert t1 == "abc" and t2 == "abc"
        assert op.call_count == 1  # second call served from in-memory cache

    def test_token_failure_negative_caches(self, monkeypatch):
        """A token-fetch failure (token endpoint 429) is negative-cached so a
        multi-subreddit batch coasts on RSS instead of hammering the endpoint."""
        _set_creds(monkeypatch)
        err = HTTPError("url", 429, "Too Many Requests", {}, None)
        with patch.object(reddit, "urlopen", side_effect=[err, err]) as op:
            assert reddit._get_oauth_token(5.0) is None
            assert reddit._get_oauth_token(5.0) is None
        assert op.call_count == 1  # second call short-circuited by negative cache

    def test_request_oauth_token_sends_basic_auth_and_body(self, monkeypatch):
        monkeypatch.setenv("REDDIT_USER_AGENT", "my-app/1.0")
        captured = {}

        def _fake_urlopen(req, timeout):
            captured["method"] = req.get_method()
            captured["ua"] = req.get_header("User-agent")
            captured["auth"] = req.get_header("Authorization")
            captured["data"] = req.data
            return _resp(lambda: b'{"access_token":"tok","expires_in":3600}')

        with patch.object(reddit, "urlopen", side_effect=_fake_urlopen):
            token, exp = reddit._request_oauth_token(("cid", "sec"), 5.0)
        assert token == "tok" and exp == 3600
        assert captured["method"] == "POST"
        assert captured["ua"] == "my-app/1.0"
        assert captured["auth"] == "Basic " + base64.b64encode(b"cid:sec").decode()
        assert captured["data"] == b"grant_type=client_credentials"

    def test_oauth_search_success_returns_engagement(self, monkeypatch):
        _set_creds(monkeypatch)
        search = {"data": {"children": [
            {"data": {"title": "NVDA pops", "score": 1234, "num_comments": 56,
                      "created_utc": 1716215400.0, "selftext": "great quarter"}},
        ]}}
        with patch.object(reddit, "urlopen",
                          side_effect=[_token_resp(), _json_resp(search)]) as op:
            posts = reddit._fetch_subreddit_json("NVDA", "stocks", 5, 5.0)
        assert op.call_count == 2  # token endpoint + search endpoint
        assert len(posts) == 1
        assert posts[0]["title"] == "NVDA pops"
        assert posts[0]["score"] == 1234
        assert posts[0]["num_comments"] == 56
        assert posts[0]["source"] == "oauth"

    def test_oauth_search_429_falls_back_to_rss(self, monkeypatch):
        _set_creds(monkeypatch)
        err = HTTPError("url", 429, "Too Many Requests", {}, None)
        rss_posts = [{"title": "x", "source": "rss", "score": None,
                      "num_comments": None, "created_utc": None, "selftext": ""}]
        with patch.object(reddit, "urlopen", side_effect=[_token_resp(), err]), \
             patch.object(reddit, "_fetch_subreddit_rss", return_value=rss_posts) as rss:
            out = reddit._fetch_subreddit_json("NVDA", "stocks", 5, 5.0)
        rss.assert_called_once()
        assert out and out[0]["source"] == "rss"

    def test_fetch_subreddit_json_no_creds_skips_urlopen(self):
        rss_posts = [{"title": "x", "source": "rss", "score": None,
                      "num_comments": None, "created_utc": None, "selftext": ""}]
        with patch.object(reddit, "urlopen",
                          side_effect=AssertionError("no token fetch without creds")), \
             patch.object(reddit, "_fetch_subreddit_rss", return_value=rss_posts) as rss:
            out = reddit._fetch_subreddit_json("NVDA", "stocks", 5, 5.0)
        rss.assert_called_once()
        assert out[0]["source"] == "rss"

    def test_fetch_subreddit_uses_json_when_creds_present(self, monkeypatch):
        _set_creds(monkeypatch)
        sentinel = [{"title": "x", "source": "oauth", "score": 1, "num_comments": 0,
                     "created_utc": None, "selftext": ""}]
        with patch.object(reddit, "_fetch_subreddit_json", return_value=sentinel) as js, \
             patch.object(reddit, "_fetch_subreddit_rss",
                          side_effect=AssertionError("RSS must not run when creds present")):
            out = reddit._fetch_subreddit("NVDA", "stocks", 5, 5.0)
        js.assert_called_once()
        assert out is sentinel

    def test_mixed_batch_renders_both_headers(self, monkeypatch):
        """One OAuth sub + one RSS sub in the same call: each renders its own
        header (engagement vs. 'via RSS feed') correctly."""
        oauth_posts = [{"title": "OA", "score": 1234, "num_comments": 56,
                        "created_utc": 1716215400.0, "selftext": "", "source": "oauth"}]
        rss_posts = [{"title": "RX", "score": None, "num_comments": None,
                      "created_utc": 1716215400.0, "selftext": "", "source": "rss"}]

        def _fake_fetch(ticker, sub, limit, timeout):
            return oauth_posts if sub == "stocks" else rss_posts

        with patch.object(reddit, "_fetch_subreddit", side_effect=_fake_fetch):
            out = reddit.fetch_reddit_posts("NVDA", subreddits=("stocks", "investing"),
                                            inter_request_delay=0)
        assert "1234↑" in out and "56c" in out          # OAuth engagement shown
        assert "via RSS feed" in out                     # RSS sub noted
        stocks_hdr = next(line for line in out.splitlines() if line.startswith("r/stocks —"))
        investing_hdr = next(line for line in out.splitlines() if line.startswith("r/investing —"))
        assert "via RSS" not in stocks_hdr
        assert "via RSS" in investing_hdr
