"""One successful train job per GitHub user per week."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Any


def within_quota(
    actor: str,
    runs: list[dict[str, Any]],
    *,
    now: datetime | None = None,
    limit: int = 1,
    window_days: int = 7,
) -> bool:
    """True when ``actor`` has fewer than ``limit`` successful train runs in the window."""
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    cutoff = moment - timedelta(days=window_days)
    count = 0
    for run in runs:
        run_actor = (run.get("actor") or {}).get("login") if isinstance(run.get("actor"), dict) else run.get("actor")
        if run_actor != actor:
            continue
        if run.get("conclusion") != "success":
            continue
        created = run.get("created_at") or ""
        try:
            stamp = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        except ValueError:
            continue
        if stamp >= cutoff:
            count += 1
    return count < limit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refuse when this user already trained this week")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--runs", help="JSON file of workflow runs. Reads stdin when omitted and not a tty.")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--window-days", type=int, default=7)
    args = parser.parse_args(argv)
    if args.runs:
        payload = json.loads(open(args.runs, encoding="utf-8").read())
    else:
        import sys

        payload = json.loads(sys.stdin.read() or "{}")
    runs = payload.get("workflow_runs", payload if isinstance(payload, list) else [])
    ok = within_quota(args.actor, runs, limit=args.limit, window_days=args.window_days)
    print(json.dumps({"ok": ok, "actor": args.actor}))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
