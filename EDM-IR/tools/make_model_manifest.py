#!/usr/bin/env python3
"""Pin a downloaded model directory as the manifest the collector verifies. CPU only.

The GLM manifest was built from Hugging Face metadata. For the other models this tool hashes
the local files and cross-checks them with what the repository already pins: the config hash
in models/SOURCES.json, and the upstream LFS digests of the weight shards when a list is given.

  EDMIR_MOE_MODEL=QWEN35 python3 make_model_manifest.py --model-dir /model \\
      --sources <package>/models/SOURCES.json --weights <package>/models/QWEN35.weights.json \\
      --out QWEN35.model-manifest.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edmir.benchmark.real_moe import MODEL, PROFILE_ID, REVISION

SMALL = ("tokenizer.json","tokenizer_config.json","generation_config.json","special_tokens_map.json",
         "chat_template.jinja","vocab.json","merges.txt","added_tokens.json","tokenizer.model")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024**2),b""):h.update(b)
    return h.hexdigest()


def build(model_dir,sources,weights):
    shards=sorted(p for p in model_dir.glob("*.safetensors"))
    if not shards:raise ValueError("no weight shards in the model directory")
    config=sha(model_dir/"config.json")
    pinned=json.loads(sources.read_text())["models"][PROFILE_ID]
    if (pinned["repo"],pinned["revision"])!=(MODEL,REVISION) or pinned["sha256"]!=config:
        raise ValueError("config.json differs from the pinned model revision")
    rows=[dict(rfilename=p.name,size=p.stat().st_size,lfs=dict(sha256=sha(p))) for p in shards]
    checked="local files only"
    if weights is not None:
        upstream=json.loads(weights.read_text())
        if (upstream["repo"],upstream["revision"])!=(MODEL,REVISION):raise ValueError("weight list is for another revision")
        expected={r["file"]:(r["bytes"],r["lfs_sha256"]) for r in upstream["shards"]}
        actual={r["rfilename"]:(r["size"],r["lfs"]["sha256"]) for r in rows}
        if actual!=expected:
            raise ValueError("weight shards differ from the upstream list: "+", ".join(
                sorted(k for k in expected.keys()|actual.keys() if expected.get(k)!=actual.get(k))))
        checked="upstream LFS digests"
    auxiliary={name:sha(model_dir/name) for name in SMALL if (model_dir/name).is_file()}
    if not all(name in auxiliary for name in ("tokenizer.json","tokenizer_config.json")):
        raise ValueError("tokenizer.json and tokenizer_config.json are required")
    return {"id":MODEL,"revision":REVISION,"weight_files":rows,"weight_file_bytes":sum(r["size"] for r in rows),
            "config.json_sha256":config,
            "model.safetensors.index.json_sha256":sha(model_dir/"model.safetensors.index.json"),
            "auxiliary_hashes":auxiliary,"weights_checked_against":checked,
            "auxiliary_pinned_from":"local files at manifest time"}


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--sources",type=Path,required=True)
    p.add_argument("--weights",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    manifest=build(a.model_dir,a.sources,a.weights)
    with a.out.open("x") as f:f.write(json.dumps(manifest,indent=1)+"\n")
    print(f"{MODEL}: {len(manifest['weight_files'])} shards, weights checked against {manifest['weights_checked_against']}")


if __name__=="__main__":
    main()
