"""B0 offline validation CLI. Import bootstrap uses no project package."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Avoid source-tree bytecode writes before the scoped runtime audit starts.
sys.dont_write_bytecode = True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="P0 shadow validation CLI: B0 offline fixtures, B1 real capture, B2 natural-expiry scoring; no LLM")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("offline-demo", help="create a new synthetic cohort under the system temporary directory")
    demo.add_argument("--root", required=True, type=Path)
    demo.add_argument("--cohort-id", required=True)
    demo.add_argument("--scenario", choices=("baseline", "missing-funding"), default="baseline")
    inspect = commands.add_parser("inspect", help="read an identified cohort and write an independent audit; never score")
    inspect.add_argument("--root", required=True, type=Path)
    inspect.add_argument("--cohort-id", required=True)
    capture = commands.add_parser("capture", help="B1: form fixed-input process-validation samples with real vendor snapshots")
    capture.add_argument("--root", required=True, type=Path)
    capture.add_argument("--cohort-id", required=True)
    capture.add_argument("--groups", default="C-BTC,C-ETH,C-MU",
                         help="comma-separated group ids; U-MU-WE only on an actual UTC weekend day")
    score = commands.add_parser("score", help="B2: natural-expiry scoring at the real current UTC clock")
    score.add_argument("--root", required=True, type=Path)
    score.add_argument("--cohort-id", required=True)
    args = parser.parse_args(argv)
    # Unified bootstrap (scripts/_bootstrap.py) — kept INSIDE main(), not
    # hoisted to module level: this CLI defers every project import until
    # after argument parsing (see the module docstring), and module level
    # sets sys.dont_write_bytecode before any import could write __pycache__.
    # Only the spelling is unified; position and laziness are preserved.
    # GBK-console note: the final json.dumps(ensure_ascii=False) below can
    # carry non-GBK characters from cohort payloads — without the UTF-8
    # reconfigure the LAST print dies on UnicodeEncodeError after every step
    # already succeeded (and the script exits non-zero despite doing its job).
    _here = str(Path(__file__).resolve().parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)
    import _bootstrap

    _bootstrap.bootstrap()  # project-root sys.path + UTF-8 console (Windows GBK)
    from scripts.p0_shadow.guard import GuardError
    from scripts.p0_shadow.runner import (
        capture_cohort,
        inspect_cohort,
        run_offline_demo,
        score_cohort,
    )

    try:
        if args.command == "offline-demo":
            result = run_offline_demo(args.root, cohort_id=args.cohort_id, scenario=args.scenario)
        elif args.command == "capture":
            groups = tuple(group.strip() for group in args.groups.split(",") if group.strip())
            result = capture_cohort(args.root, cohort_id=args.cohort_id, groups=groups)
        elif args.command == "score":
            result = score_cohort(args.root, cohort_id=args.cohort_id)
        else:
            result = inspect_cohort(args.root, cohort_id=args.cohort_id)
    except (GuardError, ValueError, FileExistsError) as exc:
        parser.exit(2, f"P0 guard rejected operation: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
