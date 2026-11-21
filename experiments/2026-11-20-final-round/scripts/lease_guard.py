#!/usr/bin/env python3
"""Run one complete eight-GPU process block inside a fixed Linux lease.

Adapted from the Codex package (A100-12h-experiment-plan); margins and the per-block
budget are now parameters, and the budget is optional.

Usage: python3 lease_guard.py --lease-end 2026-11-21T04:00:00+08:00 \
         [--budget-sec 240] [--no-start-min 60] [--no-compute-min 40] -- ./adapter --process-id B0001

Times refer to the provider's actual, timezone-aware lease expiry. New work
closes --no-start-min minutes before that time and all GPU work stops
--no-compute-min minutes before it, leaving time to verify copies off the machine.
Without --budget-sec a block only has to start before the no-start cutoff. The wrapper signals only the
process group it creates. The adapter must keep all local workers in this group.
Use an outer job step/cgroup for workers launched elsewhere or detached into
another session/group. This wrapper does not control provider billing or the
lease. Exit 75 = not launched, 124 = timeout, 125 = cleanup incomplete, 130/143
= interrupted. Otherwise returns the command's exit code (including collisions
with these reserved codes); the final stderr message identifies the reason.
"""
import argparse
import datetime as dt
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


TERM_GRACE_SEC = 3.0
POLL_SEC = 0.1


def say(message):
    print(f"[lease_guard] {message}", file=sys.stderr, flush=True)


def timestamp(value):
    try:
        result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use a timezone-aware ISO 8601 timestamp") from exc
    if result.utcoffset() is None:
        raise argparse.ArgumentTypeError("lease end must include the provider's UTC offset or Z")
    return result.astimezone(dt.timezone.utc)


def positive_seconds(value):
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("budget must be a positive number of seconds") from exc
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("budget must be finite and positive")
    return result


def live_members(group):
    """Find live processes in our session+group; skip the unreaped leader zombie."""
    members = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            # comm may contain spaces and ')'; fields beginning with state follow the last ')'.
            stat = (path / "stat").read_bytes().rsplit(b")", 1)[1].split()
            if stat[0] not in (b"Z", b"X") and int(stat[2]) == group and int(stat[3]) == group:
                members.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def send_group(group, sig):
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass


def leader_exited(pid):
    # Keep the leader's PID reserved until all cleanup signals have been sent.
    return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def finish_group(child, hard_remaining):
    if live_members(child.pid):
        send_group(child.pid, signal.SIGTERM)
        grace_end = time.monotonic() + TERM_GRACE_SEC
        while time.monotonic() < grace_end and hard_remaining() > 0 and live_members(child.pid):
            time.sleep(min(POLL_SEC, max(0, hard_remaining())))
        # Kill the group atomically after grace. Handles workers spawned during shutdown.
        send_group(child.pid, signal.SIGKILL)
    check_end = time.monotonic() + 2.0
    while live_members(child.pid) and time.monotonic() < check_end:
        send_group(child.pid, signal.SIGKILL)
        time.sleep(0.05)
    remaining = live_members(child.pid)
    if remaining:
        say(f"cleanup incomplete; still-live process IDs {remaining}. Check the outer job step.")
        return False
    try:
        child.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        say("cleanup incomplete; process-group leader has not exited")
        return False
    return True


def run(args):
    if not sys.platform.startswith("linux") or not Path("/proc/self/stat").is_file():
        say("requires Linux with procfs")
        return 125
    cancel = []
    def handle(signum, _frame):
        if not cancel:
            cancel.append(signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handle)

    started = time.monotonic()
    end_utc = args.lease_end.timestamp()
    remaining = end_utc - time.time()
    no_start_sec, no_compute_sec = args.no_start_min * 60.0, args.no_compute_min * 60.0
    hard_utc = end_utc - no_compute_sec
    hard_monotonic = started + remaining - no_compute_sec
    def hard_remaining():
        return min(hard_utc - time.time(), hard_monotonic - time.monotonic())
    budget = args.budget_sec if args.budget_sec is not None else 0.0
    if remaining <= no_start_sec or budget > hard_remaining():
        say(f"not launched: lease remaining {remaining:.1f}s, compute remaining {hard_remaining():.1f}s, "
            f"requested block budget {budget:g}s")
        return 75
    if cancel:
        return 128 + cancel[0]
    child = None
    block_limit = started + args.budget_sec if args.budget_sec is not None else float("inf")
    try:
        child = subprocess.Popen(args.command, start_new_session=True)
    except FileNotFoundError:
        say("command or its interpreter was not found")
        return 127
    except OSError as exc:
        say(f"launch failed: {exc}")
        return 126
    say(f"started group {child.pid}; budget={'none' if args.budget_sec is None else f'{args.budget_sec:g}s'}; "
        f"compute cutoff UTC={dt.datetime.fromtimestamp(hard_utc, dt.timezone.utc).isoformat()}")
    result = None
    try:
        while True:
            if cancel:
                result = 128 + cancel[0]
                break
            left = min(block_limit - time.monotonic(), hard_remaining())
            if left <= 0:
                result = 124
                break
            if leader_exited(child.pid):
                break
            time.sleep(min(left, POLL_SEC))
    finally:
        cleaned = finish_group(child, hard_remaining)
    if not cleaned:
        return 125
    if cancel:
        say(f"interrupted by signal {cancel[0]}; process group stopped")
        return 128 + cancel[0]
    if result == 124:
        say("timed out; process group stopped")
        return 124
    code = child.returncode if child.returncode >= 0 else 128 - child.returncode
    say(f"command exited {code}; process group stopped")
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lease-end", type=timestamp, required=True)
    parser.add_argument("--budget-sec", type=positive_seconds)
    parser.add_argument("--no-start-min", type=float, default=60.0)
    parser.add_argument("--no-compute-min", type=float, default=40.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("specify the block command after --")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
