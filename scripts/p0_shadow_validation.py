"""B0 offline validation CLI. Import bootstrap uses no project package."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Avoid source-tree bytecode writes before the scoped runtime audit starts.
sys.dont_write_bytecode = True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P0 B0 offline fixtures / read-only cohort inspection; no live collector")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("offline-demo", help="create a new synthetic cohort under the system temporary directory")
    demo.add_argument("--root", required=True, type=Path)
    demo.add_argument("--cohort-id", required=True)
    demo.add_argument("--scenario", choices=("baseline", "missing-funding"), default="baseline")
    inspect = commands.add_parser("inspect", help="read an identified cohort and write an independent audit; never score")
    inspect.add_argument("--root", required=True, type=Path)
    inspect.add_argument("--cohort-id", required=True)
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project))
    from scripts.p0_shadow.guard import GuardError
    from scripts.p0_shadow.runner import inspect_cohort, run_offline_demo

    try:
        if args.command == "offline-demo":
            result = run_offline_demo(args.root, cohort_id=args.cohort_id, scenario=args.scenario)
        else:
            result = inspect_cohort(args.root, cohort_id=args.cohort_id)
    except (GuardError, ValueError, FileExistsError) as exc:
        parser.exit(2, f"P0 guard rejected operation: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
