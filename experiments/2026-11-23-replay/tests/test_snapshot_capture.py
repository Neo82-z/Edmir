"""Drive the real capture hooks on CPU for every model and read the result back through the replay loader."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]/"EDM-IR"))
from snapshots import load_rank, load_receipts, snapshot_spec  # noqa: E402

DESIGN = ROOT/"design"
TOKENS = 4
BLOCK_M = 16
DEFAULT = dict(BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1,
               SPLIT_K=1, num_warps=4, num_stages=4)


def collector(profile):
    """The collector binds one model per process; import it fresh, as a process with that profile would."""
    names = [f"edmir.benchmark.{m}" for m in ("real_moe", "real_moe_config", "vllm0201_capture")]
    for name in names:
        sys.modules.pop(name, None)
    with patch.dict(os.environ, {"EDMIR_MOE_MODEL": profile}):
        return [importlib.import_module(name) for name in names][-1]


def fused_moe_stub(folder):
    fm = types.ModuleType("vllm.model_executor.layers.fused_moe.fused_moe")
    fm.__file__ = str(folder/"fused_moe.py")
    fm.get_config_file_name = lambda E, N, dtype, block_shape=None: \
        f"E={E},N={N},device_name=NVIDIA_A100-SXM4-80GB.json"
    fm.get_default_config = lambda M, E, N, K, topk, dtype, block_shape=None: dict(DEFAULT)
    return fm


def routes(spec):
    """Eight decode sources, four tokens each, top-k distinct experts per token."""
    return [[[(src*13+t*7+s*17) % spec["experts"] for s in range(spec["top_k"])] for t in range(TOKENS)]
            for src in range(8)]


def alignment(all_ids, rank, local):
    """Local-expert blocks only, padded with the sentinel slot as moe_align does."""
    flat = [e for src in all_ids for token in src for e in token]
    sentinel = len(flat)
    sorted_ids, experts = [], []
    for expert in range(local):
        slots = [i for i, e in enumerate(flat) if e == rank*local+expert]
        for start in range(0, len(slots), BLOCK_M):
            chunk = slots[start:start+BLOCK_M]
            sorted_ids += chunk+[sentinel]*(BLOCK_M-len(chunk))
            experts.append(expert)
    post = len(sorted_ids)
    return (torch.tensor(sorted_ids+[sentinel]*BLOCK_M, dtype=torch.int32),
            torch.tensor(experts+[-1], dtype=torch.int32), torch.tensor([post], dtype=torch.int32))


class SnapshotCaptureTest(unittest.TestCase):
    def tearDown(self):
        collector("GLM")

    def capture(self, profile, out, *, layout, config, tune_sha=None, gate_up=None):
        capture = collector(profile)
        name = profile+"_DECODE"
        spec = snapshot_spec(DESIGN, name)
        k, n, top_k, local = spec["k"], spec["n"], spec["top_k"], spec["local"]
        all_ids = routes(spec)
        weight = (torch.randn(gate_up or (local, n, k))*.05).bfloat16()
        hidden = (torch.randn((8*TOKENS, k))*.1).bfloat16()
        tokens = [[dict(phase="decode", request_id=f"r{src}-{t}") for t in range(TOKENS)] for src in range(8)]
        for rank in range(8):
            state = object.__new__(capture.CaptureState)
            state.torch, state.rank, state.out, state.mode = torch, rank, out, "routes"
            state.writers, state.failures = {}, 0
            state.config = dict(snapshot_replay=True, run_id="cpu-capture", source_lock_sha256="a"*64,
                                request_sha256="b"*64, native_tune_sha256=tune_sha)
            state.current = dict(step=11, block_id="diag-S32", selected=True, sizes=[TOKENS]*8,
                                 all_tokens=tokens, category="pure_decode", snapshot=name,
                                 phase_ordinal=8)
            emap = torch.tensor([e % local if e//local == rank else -1 for e in range(spec["experts"])],
                                dtype=torch.int32)
            layer = types.SimpleNamespace(layer_id=spec["layer"], expert_map=emap,
                                          runner=types.SimpleNamespace(do_naive_dispatch_combine=layout == "gathered_dp"))
            state.layer = layer
            ids = torch.tensor(all_ids, dtype=torch.int64)
            ids = ids[rank] if layout == "local" else ids.reshape(-1, top_k)
            # Production hooks run through safe(), which turns an exception into a failure receipt.
            state.safe(lambda: state.route(layer, torch.full(ids.shape, 1/top_k), ids))
            sorted_ids, experts, post = alignment(all_ids, rank, local)
            bound = dict(A=hidden, B=weight, C=torch.empty((8*TOKENS, top_k, n), dtype=torch.bfloat16),
                         A_scale=None, B_scale=None, topk_weights=None, sorted_token_ids=sorted_ids,
                         expert_ids=experts, num_tokens_post_padded=post, mul_routed_weight=False, top_k=top_k,
                         config=dict(config), compute_type=types.SimpleNamespace(name="bf16"),
                         use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False,
                         per_channel_quant=False, block_shape=None, B_bias=None)
            state.safe(lambda: state.kernel(bound))
            state.close()
        return name, spec, all_ids

    def check(self, out, name, spec, all_ids, layout):
        root = out/"snapshots"
        rows = load_receipts(root, DESIGN, name)
        for rank, row in enumerate(rows):
            payload = load_rank(root, name, row, spec)
            self.assertEqual(payload["ids"].tolist(), all_ids[rank])
            self.assertEqual(tuple(payload["expert_weights"].shape), (spec["local"], spec["n"], spec["k"]))
            self.assertEqual((row["routing_layout"], row["model"]), (layout, spec["model"]))
        return rows

    def test_every_model_with_local_routing_and_default_config(self):
        # 2026-11-17 observed local routing for GLM on this vLLM; the other models use the same layer class.
        for profile in ("GLM", "QWEN35", "DSV2L"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as d:
                out = Path(d)/"workers"
                fm = fused_moe_stub(Path(d))
                with patch.dict(sys.modules, {fm.__name__: fm}), patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("VLLM_TUNED_CONFIG_FOLDER", None)
                    name, spec, all_ids = self.capture(profile, out, layout="local", config=DEFAULT)
                rows = self.check(out, name, spec, all_ids, "local")
                self.assertEqual(rows[0]["native_tune"], {"kind": "default"})

    def test_gathered_layout_and_tuned_config_provenance(self):
        from native import SHAPES
        for profile in ("GLM", "QWEN35", "DSV2L"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as d:
                out, folder = Path(d)/"workers", Path(d)/"tuned"
                folder.mkdir()
                tuned = dict(DEFAULT, BLOCK_SIZE_N=128, num_warps=8)
                local, n, *_ = SHAPES[profile]
                # The file name the capture looks for is the one native.py writes for this model.
                path = folder/f"E={local},N={n//2},device_name=NVIDIA_A100-SXM4-80GB.json"
                path.write_text(json.dumps({"triton_version": "3.6.0", "16": dict(DEFAULT), "32": tuned}))
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                fm = fused_moe_stub(Path(d))
                with patch.dict(sys.modules, {fm.__name__: fm}), \
                        patch.dict(os.environ, {"VLLM_TUNED_CONFIG_FOLDER": str(folder)}):
                    name, spec, all_ids = self.capture(profile, out, layout="gathered_dp", config=tuned, tune_sha=digest)
                    rows = self.check(out, name, spec, all_ids, "gathered_dp")
                    self.assertEqual(rows[0]["native_tune"]["kind"], "tuned")
                    self.assertEqual(rows[0]["native_tune"]["sha256"], digest)
                    # A launch that did not use the frozen file is a capture failure, never a silent default.
                    bad = Path(d)/"bad"
                    self.capture(profile, bad, layout="local", config=DEFAULT, tune_sha=digest)
                    self.assertTrue(list(bad.glob("capture-failure-rank*.json")))
                    self.assertFalse((bad/"snapshots"/name/"rank-0.json").exists())

    def test_another_models_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            fm = fused_moe_stub(Path(d))
            with patch.dict(sys.modules, {fm.__name__: fm}):
                # A DeepSeek profile must not accept GLM-shaped gate/up weights.
                out = Path(d)/"workers"
                self.capture("DSV2L", out, layout="local", config=DEFAULT, gate_up=(8, 3072, 2048))
            self.assertTrue(list(out.glob("capture-failure-rank*.json")))
            self.assertFalse((out/"snapshots").exists())


if __name__ == "__main__":
    unittest.main()
