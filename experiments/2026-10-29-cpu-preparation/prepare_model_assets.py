"""Download fixed public model revisions on a zero-GPU Pod and verify every shard.

The default is a plan only. --download requires an explicit persistent destination.
This program never creates, starts, stops or changes any cloud instance.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

SELECTED = (
    "zai-org/GLM-4.7-Flash",
    "Qwen/Qwen3.6-35B-A3B",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
)


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Persistent SSD directory, e.g. /workspace/edmir-assets")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--model", choices=SELECTED, action="append")
    args = parser.parse_args()
    manifest_path = Path(__file__).with_name("metadata-audit.json")
    audit = json.loads(manifest_path.read_text())
    wanted = set(args.model or SELECTED)
    models = [row for row in audit["models"] if row["id"] in wanted]
    assert len(models) == len(wanted)
    for row in models:
        assert row["weight_file_bytes"] > 0 and len(row["revision"]) == 40
        for shard in row["weight_files"]:
            assert len(shard.get("lfs", {}).get("sha256", "")) == 64
    total = sum(row["weight_file_bytes"] for row in models)
    print(json.dumps({"mode": "download" if args.download else "plan_only",
                      "weight_bytes": total,
                      "models": [{k: row[k] for k in ("id", "revision", "weight_file_bytes")} for row in models]},
                     ensure_ascii=False), flush=True)
    if not args.download:
        return
    if args.root is None or not args.root.is_absolute():
        parser.error("--download requires an absolute --root on persistent storage")
    if sys.platform != "linux":
        parser.error("download target must be the Linux preparation Pod")
    if list(Path("/dev").glob("nvidia[0-9]*")):
        raise SystemExit("Refusing preparation: GPU device nodes are visible; verify zero-GPU allocation first")
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / ".download.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = args.root / "hf-hub"
        records = args.root / "manifests"
        records.mkdir(exist_ok=True)
        # Keep Xet temporary/cache data on the same persistent volume as the files.
        os.environ["HF_HOME"] = str(args.root / "hf-home")
        os.environ["HF_XET_CACHE"] = str(args.root / "hf-xet")
        from huggingface_hub import snapshot_download
        import huggingface_hub
        free = shutil.disk_usage(args.root).free
        already_present = 0
        for row in models:
            model_cache = cache / ("models--" + row["id"].replace("/", "--")) / "blobs"
            for shard in row["weight_files"]:
                blob = model_cache / shard["lfs"]["sha256"]
                if blob.is_file() and blob.stat().st_size == shard["size"]:
                    already_present += shard["size"]
        if free < total - already_present + 30 * 1024**3:
            raise SystemExit("Insufficient free SSD space for remaining weights plus 30 GiB working reserve")
        for row in models:
            record = {"id": row["id"], "revision": row["revision"], "status": "downloading",
                      "started_unix": time.time(), "huggingface_hub": huggingface_hub.__version__,
                      "input_manifest_sha256": sha256(manifest_path), "verified_shards": []}
            target = records / (row["id"].replace("/", "--") + ".json")
            save(target, record)
            try:
                directory = Path(snapshot_download(
                    repo_id=row["id"], revision=row["revision"], cache_dir=str(cache), max_workers=4,
                    allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.jinja", "*.jinja2",
                                    "tokenizer*", "README.md", "LICENSE*"],
                ))
                record.update(status="verifying", snapshot=str(directory))
                save(target, record)
                for shard in row["weight_files"]:
                    path = directory / shard["rfilename"]
                    if path.stat().st_size != shard["size"] or sha256(path) != shard["lfs"]["sha256"]:
                        raise ValueError("Shard validation failed: " + shard["rfilename"])
                    record["verified_shards"].append(shard["rfilename"])
                    save(target, record)
                for name in ("config.json", "model.safetensors.index.json"):
                    expected = row.get(name + "_sha256")
                    if expected and sha256(directory / name) != expected:
                        raise ValueError("Metadata validation failed: " + name)
                record.update(status="complete", completed_unix=time.time())
                save(target, record)
                print(json.dumps({"id": row["id"], "status": "complete", "snapshot": str(directory)}), flush=True)
            except Exception as exc:
                record.update(status="failed", error_type=type(exc).__name__)
                save(target, record)
                raise


if __name__ == "__main__":
    main()
