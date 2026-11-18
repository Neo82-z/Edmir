"""Verify finalized experiment artifacts against the remote manifest."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
recovered = root / "recovered"
manifest = json.loads((recovered / "recovery-manifest.json").read_text())
errors = []
total = 0
for name, expected in manifest["files"].items():
    rel = Path(name)
    if rel.is_absolute() or ".." in rel.parts or "cache" in rel.parts:
        raise ValueError("Unexpected recovery path: " + name)
    path = recovered / rel
    if not path.is_file():
        errors.append({"file": name, "error": "missing"})
        continue
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    size = path.stat().st_size
    total += size
    if digest != expected["sha256"] or size != expected["bytes"]:
        errors.append({"file": name, "error": "size or digest mismatch"})
receipt = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               instance_id=manifest["instance_id"], files=len(manifest["files"]),
               bytes=total, verified=not errors, errors=errors)
(root / "recovery-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
raise SystemExit(bool(errors))
