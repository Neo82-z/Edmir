#!/usr/bin/env python3
"""SHA-256 manifests for freezing builds and for verifying results copied off the machine.

  python3 manifest.py create  --root results/ --out results/MANIFEST.json
  python3 manifest.py verify  --root copy_of_results/ --manifest copy_of_results/MANIFEST.json

create writes the manifest to a temporary file, renames it into place, then writes MANIFEST.done;
a copy is complete only when MANIFEST.done is present and verify passes. verify reports missing,
changed and extra files and exits 1 on any difference.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import sha256  # noqa: E402

SKIP = {"MANIFEST.json", "MANIFEST.done"}


def walk(root):
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name not in SKIP and not path.name.endswith(".tmp"):
            yield path


def create(root, out):
    entries = {str(p.relative_to(root)): dict(sha256=sha256(p), bytes=p.stat().st_size) for p in walk(root)}
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(dict(schema="manifest.v1", files=entries), indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, out)
    out.with_name("MANIFEST.done").write_text(sha256(out) + "\n", encoding="utf-8")
    print(f"{len(entries)} files, {sum(e['bytes'] for e in entries.values())} bytes -> {out}")
    return 0


def verify(root, manifest):
    done = manifest.with_name("MANIFEST.done")
    problems = []
    if not done.exists():
        problems.append("MANIFEST.done missing: the copy may be partial")
    elif done.read_text(encoding="utf-8").strip() != sha256(manifest):
        problems.append("MANIFEST.done does not match MANIFEST.json")
    files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    present = {str(p.relative_to(root)): p for p in walk(root)}
    for name, entry in sorted(files.items()):
        p = present.get(name)
        if p is None:
            problems.append(f"missing {name}")
        elif p.stat().st_size != entry["bytes"] or sha256(p) != entry["sha256"]:
            problems.append(f"changed {name}")
    problems += [f"extra {name}" for name in sorted(set(present) - set(files))]
    for line in problems:
        print(line)
    print(f"{len(files)} files in manifest; {len(problems)} problems")
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--root", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--root", type=Path, required=True)
    v.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    return create(args.root, args.out) if args.cmd == "create" else verify(args.root, args.manifest)


if __name__ == "__main__":
    sys.exit(main())
