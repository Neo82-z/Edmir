"""Read public metadata only; never download weights, execute remote code, or rent GPUs."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

MODELS = {
    "zai-org/GLM-4.7-Flash": "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
    "Qwen/Qwen3.6-35B-A3B": "995ad96eacd98c81ed38be0c5b274b04031597b0",
    "deepseek-ai/DeepSeek-V2-Lite-Chat": "85864749cd611b4353ce1decdb286193298f64c7",
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "9c7f2fbe84465e40164a94cc16cd30b6999b0cc7",
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": "1bae0e1088993b9b82de08f59de5be5a34e42170",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "92f3b1597a195b523d8d9e5700e57e4fbb8f20d3",
    "meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
}
FIELDS = (
    "model_type", "architectures", "torch_dtype", "dtype", "hidden_size",
    "moe_intermediate_size", "num_hidden_layers", "n_routed_experts",
    "num_experts", "num_experts_per_tok", "n_shared_experts",
    "first_k_dense_replace", "moe_layer_freq",
)


def read(url):
    req = urllib.request.Request(url, headers={"User-Agent": "edmir-metadata-audit"})
    with urllib.request.urlopen(req, timeout=25) as response:
        raw = response.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("metadata response exceeds 8 MiB")
    return raw


def audit_model(item):
    model, revision = item
    result = {"id": model, "revision": revision, "weights_downloaded": False}
    api = f"https://huggingface.co/api/models/{model}/revision/{revision}?blobs=true"
    result["metadata_url"] = api
    try:
        info = json.loads(read(api))
        if info["sha"] != revision:
            raise ValueError("revision mismatch")
        shards = [x for x in info.get("siblings", []) if x["rfilename"].endswith(".safetensors")]
        result.update(gated=info.get("gated"), safetensors=info.get("safetensors"))
        result["weight_files"] = [
            {k: x[k] for k in ("rfilename", "size", "lfs") if k in x} for x in shards
        ]
        sizes = [x.get("size") for x in shards]
        result["weight_file_bytes"] = sum(sizes) if sizes and all(isinstance(s, int) for s in sizes) else None
        for filename in ("config.json", "model.safetensors.index.json"):
            url = f"https://huggingface.co/{model}/resolve/{revision}/{filename}"
            try:
                raw = read(url)
                obj = json.loads(raw)
                result[filename + "_sha256"] = hashlib.sha256(raw).hexdigest()
                if filename == "config.json":
                    text = obj.get("text_config", obj)
                    result["text_config"] = {k: text[k] for k in FIELDS if k in text}
                    result["quantization_config"] = obj.get("quantization_config", text.get("quantization_config"))
                else:
                    result["index_reported_total_size"] = obj.get("metadata", {}).get("total_size")
            except (urllib.error.URLError, ValueError) as exc:
                result[filename + "_error"] = str(exc)
    except (urllib.error.URLError, ValueError, KeyError) as exc:
        result["audit_error"] = str(exc)
    return result


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(audit_model, MODELS.items()))
    stack = {}
    for name, path in (("registry", "vllm/model_executor/models/registry.py"), ("cuda_requirements", "requirements/cuda.txt")):
        url = "https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/" + path
        try:
            raw = read(url)
            text = raw.decode()
            stack[name] = {"url": url, "sha256": hashlib.sha256(raw).hexdigest()}
            if name == "registry":
                terms = ("Glm4MoeLiteForCausalLM", "Qwen3_5MoeForConditionalGeneration", "DeepseekV2ForCausalLM", "Qwen3NextForCausalLM", "Llama4ForConditionalGeneration", "LlamaForCausalLM")
                stack[name]["registered_names"] = {term: term in text for term in terms}
            else:
                stack[name]["content"] = text
        except (urllib.error.URLError, ValueError) as exc:
            stack[name] = {"url": url, "error": str(exc)}
    output = {
        "checked_date": "2026-10-29",
        "status": "public metadata audit; not a built, installed or GPU-tested environment",
        "models": results,
        "stack_candidate": {"vllm": "v0.20.1", "source_checks": stack},
        "scope": {"GPU_started": False, "paid_resources_created": False, "frozen_six_opened": False},
        "size_caveat": "Use summed weight file sizes for download planning; index total_size can be incorrect. File bytes are not peak device memory.",
    }
    path = Path(__file__).with_name("metadata-audit.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    for row in results:
        print(json.dumps({k: row.get(k) for k in ("id", "weight_file_bytes", "index_reported_total_size", "gated", "audit_error")}, ensure_ascii=False))
    print("Saved", path)


if __name__ == "__main__":
    main()
