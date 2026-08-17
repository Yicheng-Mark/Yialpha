"""Append-only markdown decision log for YiAgents."""

import logging
import re
from contextlib import nullcontext
from datetime import date
from pathlib import Path
from typing import Any

from yiagents.agents.utils.rating import parse_rating
from yiagents.batch.locks import FileLock

logger = logging.getLogger(__name__)


class TradingMemoryLog:
    """Append-only markdown log of trading decisions and reflections."""

    # HTML comment: cannot appear in LLM prose output, safe as a hard delimiter
    _SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"
    # Precompiled patterns — avoids re-compilation on every load_entries() call
    _DECISION_RE = re.compile(r"DECISION:\n(.*?)(?=\nREFLECTION:|\Z)", re.DOTALL)
    _REFLECTION_RE = re.compile(r"REFLECTION:\n(.*?)$", re.DOTALL)

    @staticmethod
    def _tag_keyed_fields(fields: list[str]) -> dict[str, str]:
        """The ``key=value`` fields of a split tag line (``asset=``,
        ``known=``) — everything that is NOT positional."""
        out: dict[str, str] = {}
        for f in fields:
            if "=" in f:
                k, v = f.split("=", 1)
                out[k.strip()] = v.strip()
        return out

    @staticmethod
    def _is_pending(fields: list[str]) -> bool:
        """Pending marker as an exact field — tolerant of trailing keyed
        fields (``... | pending | asset=crypto_perp]``)."""
        return any(f == "pending" for f in fields)

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self._log_path: Path | None = None
        # Persistent reflection is deliberately opt-in in the application
        # default.  Explicit, minimal configs used by library callers remain
        # backwards compatible: providing a path without ``memory_enabled``
        # still enables the log.
        path = cfg.get("memory_log_path") if cfg.get("memory_enabled", True) else None
        if path:
            self._log_path = Path(path).expanduser()
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        # Optional cap on resolved entries. None disables rotation.
        self._max_entries = cfg.get("memory_log_max_entries")
        # Serialize the read-modify-write write paths so concurrent ticker
        # workers (and separate processes) can't lost-update each other. The
        # read path (get_past_context -> load_entries) needs no lock: writes use
        # an atomic os.replace, so a reader sees either the old or new file,
        # never a torn one. nullcontext when there's no file or locking is off
        # keeps single-ticker runs uncontended.
        if self._log_path and cfg.get("batch_memory_lock", True):
            self._lock: FileLock | nullcontext[None] = FileLock(self._log_path)
        else:
            self._lock = nullcontext()

    # --- Write path (Phase A) ---

    def store_decision(
        self,
        ticker: str,
        trade_date: str,
        final_trade_decision: str,
        asset_type: str | None = None,
    ) -> None:
        """Append pending entry at end of propagate(). No LLM call.

        ``asset_type`` rides the tag as a trailing keyed field
        (``asset=crypto_perp``) so perp and spot decisions on the SAME ticker
        stay distinguishable in the log and in the injected past context;
        omitted keeps the legacy asset-less tag byte-identical.
        """
        if not self._log_path:
            return
        with self._lock:
            # Idempotency guard: fast raw-text scan instead of full parse. The
            # scan + append must be atomic, else two workers both pass the
            # check and double-append the same pending decision.
            if self._log_path.exists():
                raw = self._log_path.read_text(encoding="utf-8")
                for line in raw.splitlines():
                    if not line.startswith(f"[{trade_date} | {ticker} |"):
                        continue
                    if line.startswith("[") and line.endswith("]"):
                        fields = [f.strip() for f in line[1:-1].split("|")]
                        if self._is_pending(fields):
                            return
            rating = parse_rating(final_trade_decision)
            tag = f"[{trade_date} | {ticker} | {rating} | pending"
            if asset_type:
                tag += f" | asset={asset_type}"
            tag += "]"
            entry = f"{tag}\n\nDECISION:\n{final_trade_decision}{self._SEPARATOR}"
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(entry)

    # --- Read path (Phase A) ---

    def load_entries(self) -> list[dict]:
        """Parse all entries from log. Returns list of dicts."""
        if not self._log_path or not self._log_path.exists():
            return []
        text = self._log_path.read_text(encoding="utf-8")
        raw_entries = [e.strip() for e in text.split(self._SEPARATOR) if e.strip()]
        entries = []
        for raw in raw_entries:
            parsed = self._parse_entry(raw)
            if parsed:
                entries.append(parsed)
        return entries

    def get_pending_entries(self) -> list[dict]:
        """Return entries with outcome:pending (for Phase B)."""
        return [e for e in self.load_entries() if e.get("pending")]

    def get_past_context(
        self,
        ticker: str,
        n_same: int = 5,
        n_cross: int = 3,
        *,
        as_of_date: str | None = None,
        asset_type: str | None = None,
    ) -> str:
        """Return causal past context for agent prompt injection.

        When ``as_of_date`` is supplied, an entry is visible only when both its
        decision date and the date on which its outcome became known precede (or
        equal, for the outcome) the simulated date.  Legacy resolved entries do
        not carry an outcome-availability date, so they are excluded from
        historical runs instead of being trusted and potentially leaking a
        reflection produced with present-day prices.

        ``asset_type`` (when supplied) restricts the SAME-ticker section to
        entries recorded for that asset: a crypto_perp run of BTCUSDT must not
        inherit the spot run's lessons (different venue, different funding
        regime). Legacy entries without an asset tag match any asset type —
        they predate the split and are the only history that exists.
        """
        entries = [e for e in self.load_entries() if not e.get("pending")]
        if as_of_date is not None:
            try:
                cutoff = date.fromisoformat(str(as_of_date)[:10])
            except (TypeError, ValueError):
                # Fail closed (no history injected), but the whole accumulated
                # lesson history being dropped is a major degradation — the
                # caller should find out their date format is wrong.
                logger.warning(
                    "memory: unparseable as_of_date %r — injecting no past "
                    "context instead of risking a look-ahead leak",
                    as_of_date,
                )
                return ""

            causal_entries = []
            for entry in entries:
                try:
                    decision_date = date.fromisoformat(str(entry["date"])[:10])
                    available_date = date.fromisoformat(
                        str(entry["available_date"])[:10]
                    )
                except (KeyError, TypeError, ValueError):
                    # Fail closed: an undated reflection cannot be proven to
                    # have existed at the requested point in time.
                    continue
                if decision_date < cutoff and available_date <= cutoff:
                    causal_entries.append(entry)
            entries = causal_entries
        if not entries:
            return ""

        same: list[dict[str, Any]] = []
        cross: list[dict[str, Any]] = []
        for e in reversed(entries):
            if len(same) >= n_same and len(cross) >= n_cross:
                break
            if e["ticker"] == ticker:
                entry_asset = e.get("asset")
                asset_ok = (
                    asset_type is None
                    or entry_asset is None
                    or entry_asset == asset_type
                )
                if asset_ok and len(same) < n_same:
                    same.append(e)
            elif e["ticker"] != ticker and len(cross) < n_cross:
                cross.append(e)

        if not same and not cross:
            return ""

        parts = []
        if same:
            parts.append(f"Past analyses of {ticker} (most recent first):")
            parts.extend(self._format_full(e) for e in same)
        if cross:
            parts.append("Recent cross-ticker lessons:")
            parts.extend(self._format_reflection_only(e) for e in cross)
        return "\n\n".join(parts)

    # --- Update path (Phase B) ---

    def update_with_outcome(
        self,
        ticker: str,
        trade_date: str,
        raw_return: float,
        alpha_return: float,
        holding_days: int,
        reflection: str,
        available_date: str | None = None,
    ) -> None:
        """Replace pending tag and append REFLECTION section using atomic write.

        Finds the first pending entry matching (trade_date, ticker), updates
        its tag with return figures, and appends a REFLECTION section.  Uses
        a temp-file + os.replace() so a crash mid-write never corrupts the log.
        """
        if not self._log_path or not self._log_path.exists():
            return

        with self._lock:
            text = self._log_path.read_text(encoding="utf-8")
            blocks = text.split(self._SEPARATOR)

            pending_prefix = f"[{trade_date} | {ticker} |"
            raw_pct = f"{raw_return:+.1%}"
            alpha_pct = f"{alpha_return:+.1%}"

            updated = False
            new_blocks = []
            for block in blocks:
                stripped = block.strip()
                if not stripped:
                    new_blocks.append(block)
                    continue

                lines = stripped.splitlines()
                tag_line = lines[0].strip()

                fields = (
                    [f.strip() for f in tag_line[1:-1].split("|")]
                    if tag_line.startswith("[") and tag_line.endswith("]")
                    else []
                )
                if (
                    not updated
                    and tag_line.startswith(pending_prefix)
                    and self._is_pending(fields)
                ):
                    # Parse rating from the existing pending tag; keep any
                    # keyed fields (asset=...) so resolution never strips the
                    # entry's venue identity.
                    positional = [f for f in fields if "=" not in f]
                    rating = positional[2] if len(positional) > 2 else ""
                    keyed = self._tag_keyed_fields(fields)
                    new_tag = (
                        f"[{trade_date} | {ticker} | {rating}"
                        f" | {raw_pct} | {alpha_pct} | {holding_days}d"
                    )
                    if available_date:
                        new_tag += f" | known={available_date}"
                    if keyed.get("asset"):
                        new_tag += f" | asset={keyed['asset']}"
                    new_tag += "]"
                    rest = "\n".join(lines[1:])
                    new_blocks.append(
                        f"{new_tag}\n\n{rest.lstrip()}\n\nREFLECTION:\n{reflection}"
                    )
                    updated = True
                else:
                    new_blocks.append(block)

            if not updated:
                return

            new_blocks = self._apply_rotation(new_blocks)
            new_text = self._SEPARATOR.join(new_blocks)
            tmp_path = self._log_path.with_suffix(".tmp")
            tmp_path.write_text(new_text, encoding="utf-8")
            tmp_path.replace(self._log_path)

    def batch_update_with_outcomes(self, updates: list[dict]) -> None:
        """Apply multiple outcome updates in a single read + atomic write.

        Each element of updates must have keys: ticker, trade_date,
        raw_return, alpha_return, holding_days, reflection.
        """
        if not self._log_path or not self._log_path.exists() or not updates:
            return

        with self._lock:
            text = self._log_path.read_text(encoding="utf-8")
            blocks = text.split(self._SEPARATOR)

            # Build lookup keyed by (trade_date, ticker) for O(1) dispatch
            update_map = {(u["trade_date"], u["ticker"]): u for u in updates}

            new_blocks = []
            for block in blocks:
                stripped = block.strip()
                if not stripped:
                    new_blocks.append(block)
                    continue

                lines = stripped.splitlines()
                tag_line = lines[0].strip()

                matched = False
                for (trade_date, ticker), upd in list(update_map.items()):
                    pending_prefix = f"[{trade_date} | {ticker} |"
                    fields = (
                        [f.strip() for f in tag_line[1:-1].split("|")]
                        if tag_line.startswith("[") and tag_line.endswith("]")
                        else []
                    )
                    if tag_line.startswith(pending_prefix) and self._is_pending(fields):
                        positional = [f for f in fields if "=" not in f]
                        rating = positional[2] if len(positional) > 2 else ""
                        keyed = self._tag_keyed_fields(fields)
                        raw_pct = f"{upd['raw_return']:+.1%}"
                        alpha_pct = f"{upd['alpha_return']:+.1%}"
                        new_tag = (
                            f"[{trade_date} | {ticker} | {rating}"
                            f" | {raw_pct} | {alpha_pct} | {upd['holding_days']}d"
                        )
                        if upd.get("available_date"):
                            new_tag += f" | known={upd['available_date']}"
                        if keyed.get("asset"):
                            new_tag += f" | asset={keyed['asset']}"
                        new_tag += "]"
                        rest = "\n".join(lines[1:])
                        new_blocks.append(
                            f"{new_tag}\n\n{rest.lstrip()}\n\nREFLECTION:\n{upd['reflection']}"
                        )
                        del update_map[(trade_date, ticker)]
                        matched = True
                        break

                if not matched:
                    new_blocks.append(block)

            new_blocks = self._apply_rotation(new_blocks)
            new_text = self._SEPARATOR.join(new_blocks)
            tmp_path = self._log_path.with_suffix(".tmp")
            tmp_path.write_text(new_text, encoding="utf-8")
            tmp_path.replace(self._log_path)

    # --- Helpers ---

    def _apply_rotation(self, blocks: list[str]) -> list[str]:
        """Drop oldest resolved blocks when their count exceeds max_entries.

        Pending blocks are always kept (they represent unprocessed work).
        Returns ``blocks`` unchanged when rotation is disabled or under cap.
        """
        if not self._max_entries or self._max_entries <= 0:
            return blocks

        # Tag each block with (kept, is_resolved) by parsing tag-line markers.
        decisions = []
        for block in blocks:
            stripped = block.strip()
            if not stripped:
                decisions.append((block, False))
                continue
            tag_line = stripped.splitlines()[0].strip()
            is_resolved = (
                tag_line.startswith("[")
                and tag_line.endswith("]")
                and not tag_line.endswith("| pending]")
            )
            decisions.append((block, is_resolved))

        resolved_count = sum(1 for _, r in decisions if r)
        if resolved_count <= self._max_entries:
            return blocks

        to_drop = resolved_count - self._max_entries
        kept: list[str] = []
        for block, is_resolved in decisions:
            if is_resolved and to_drop > 0:
                to_drop -= 1
                continue
            kept.append(block)
        return kept

    def _parse_entry(self, raw: str) -> dict | None:
        lines = raw.strip().splitlines()
        if not lines:
            return None
        tag_line = lines[0].strip()
        if not (tag_line.startswith("[") and tag_line.endswith("]")):
            return None
        fields = [f.strip() for f in tag_line[1:-1].split("|")]
        keyed = self._tag_keyed_fields(fields)
        positional = [f for f in fields if "=" not in f]
        if len(positional) < 4:
            return None
        entry = {
            "date": positional[0],
            "ticker": positional[1],
            "rating": positional[2],
            "pending": positional[3] == "pending",
            "raw": positional[3] if positional[3] != "pending" else None,
            "alpha": positional[4] if len(positional) > 4 else None,
            "holding": positional[5] if len(positional) > 5 else None,
            # Keyed fields: known= is the PIT outcome-availability date,
            # asset= is the venue identity (perp vs spot on the same ticker).
            "available_date": keyed.get("known"),
            "asset": keyed.get("asset"),
        }
        body = "\n".join(lines[1:]).strip()
        decision_match = self._DECISION_RE.search(body)
        reflection_match = self._REFLECTION_RE.search(body)
        entry["decision"] = decision_match.group(1).strip() if decision_match else ""
        entry["reflection"] = reflection_match.group(1).strip() if reflection_match else ""
        return entry

    def _format_full(self, e: dict) -> str:
        raw = e["raw"] or "n/a"
        alpha = e["alpha"] or "n/a"
        holding = e["holding"] or "n/a"
        tag = f"[{e['date']} | {e['ticker']} | {e['rating']} | {raw} | {alpha} | {holding}]"
        if e.get("asset"):
            tag = tag[:-1] + f" | asset={e['asset']}]"
        parts = [tag, f"DECISION:\n{e['decision']}"]
        if e["reflection"]:
            parts.append(f"REFLECTION:\n{e['reflection']}")
        return "\n\n".join(parts)

    def _format_reflection_only(self, e: dict) -> str:
        tag = f"[{e['date']} | {e['ticker']} | {e['rating']} | {e['raw'] or 'n/a'}]"
        if e.get("asset"):
            tag = tag[:-1] + f" | asset={e['asset']}]"
        if e["reflection"]:
            return f"{tag}\n{e['reflection']}"
        text = e["decision"][:300]
        suffix = "..." if len(e["decision"]) > 300 else ""
        return f"{tag}\n{text}{suffix}"
