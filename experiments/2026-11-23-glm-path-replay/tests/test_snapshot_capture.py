"""Drive the real capture hooks on CPU and read the result back through the replay loader."""
import hashlib
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
from edmir.benchmark import vllm0201_capture as capture  # noqa: E402
from snapshots import load_rank, load_receipts  # noqa: E402

DESIGN = ROOT/"design"
TOKENS = 4
BLOCK_M = 16
DEFAULT = dict(BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128, GROUP_SIZE_M=1,
               SPLIT_K=1, num_warps=4, num_stages=4)


def fused_moe_stub(folder):
    fm = types.ModuleType("vllm.model_executor.layers.fused_moe.fused_moe")
    fm.__file__ = str(folder/"fused_moe.py")
    fm.get_config_file_name = lambda E, N, dtype, block_shape=None: \
        f"E={E},N={N},device_name=NVIDIA_A100-SXM4-80GB.json"
    fm.get_default_config = lambda M, E, N, K, topk, dtype, block_shape=None: dict(DEFAULT)
    return fm


def routes():
    """Eight decode sources, four tokens each, four distinct experts per token."""
    return [[[(src*13+t*7+s*17) % 64 for s in range(4)] for t in range(TOKENS)] for src in range(8)]


def alignment(all_ids, rank):
    """Local-expert blocks only, padded with the sentinel slot as moe_align does."""
    flat = [e for src in all_ids for token in src for e in token]
    sentinel = len(flat)
    sorted_ids, experts = [], []
    for local in range(8):
        slots = [i for i, e in enumerate(flat) if e == rank*8+local]
        for start in range(0, len(slots), BLOCK_M):
            chunk = slots[start:start+BLOCK_M]
            sorted_ids += chunk+[sentinel]*(BLOCK_M-len(chunk))
            experts.append(local)
    post = len(sorted_ids)
    return (torch.tensor(sorted_ids+[sentinel]*BLOCK_M, dtype=torch.int32),
            torch.tensor(experts+[-1], dtype=torch.int32), torch.tensor([post], dtype=torch.int32))


class SnapshotCaptureTest(unittest.TestCase):
    def capture(self, out, *, layout, config, tune_sha=None):
        all_ids = routes()
        weight = (torch.randn((8, 3072, 2048))*.05).bfloat16()
        hidden = (torch.randn((8*TOKENS, 2048))*.1).bfloat16()
        tokens = [[dict(phase="decode", request_id=f"r{src}-{t}") for t in range(TOKENS)] for src in range(8)]
        for rank in range(8):
            state = object.__new__(capture.CaptureState)
            state.torch, state.rank, state.out, state.mode = torch, rank, out, "routes"
            state.writers, state.failures = {}, 0
            state.config = dict(snapshot_replay=True, run_id="cpu-capture", source_lock_sha256="a"*64,
                                request_sha256="b"*64, native_tune_sha256=tune_sha)
            state.current = dict(step=11, block_id="diag-S32", selected=True, sizes=[TOKENS]*8,
                                 all_tokens=tokens, category="pure_decode", snapshot="GLM_DECODE",
                                 phase_ordinal=8)
            emap = torch.tensor([e % 8 if e//8 == rank else -1 for e in range(64)], dtype=torch.int32)
            layer = types.SimpleNamespace(layer_id=23, expert_map=emap,
                                          runner=types.SimpleNamespace(do_naive_dispatch_combine=layout == "gathered_dp"))
            state.layer = layer
            ids = torch.tensor(all_ids, dtype=torch.int64)
            if layout == "local":
                ids = ids[rank]
            else:
                ids = ids.reshape(-1, 4)
            # Production hooks run through safe(), which turns an exception into a failure receipt.
            state.safe(lambda: state.route(layer, torch.full(ids.shape, .25), ids))
            sorted_ids, experts, post = alignment(all_ids, rank)
            bound = dict(A=hidden, B=weight, C=torch.empty((8*TOKENS, 4, 3072), dtype=torch.bfloat16),
                         A_scale=None, B_scale=None, topk_weights=None, sorted_token_ids=sorted_ids,
                         expert_ids=experts, num_tokens_post_padded=post, mul_routed_weight=False, top_k=4,
                         config=dict(config), compute_type=types.SimpleNamespace(name="bf16"),
                         use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False,
                         per_channel_quant=False, block_shape=None, B_bias=None)
            state.safe(lambda: state.kernel(bound))
            state.close()
        return all_ids

    def check(self, out, all_ids, layout):
        root = out/"snapshots"
        rows = load_receipts(root, DESIGN, "GLM_DECODE")
        for rank, row in enumerate(rows):
            payload = load_rank(root, "GLM_DECODE", row)
            self.assertEqual(payload["ids"].tolist(), all_ids[rank])
            self.assertEqual(row["routing_layout"], layout)
        return rows

    def test_local_routing_as_observed_on_2026_11_17_and_default_config(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/"workers"
            fm = fused_moe_stub(Path(d))
            with patch.dict(sys.modules, {fm.__name__: fm}), patch.dict(os.environ, {}, clear=False):
                os.environ.pop("VLLM_TUNED_CONFIG_FOLDER", None)
                all_ids = self.capture(out, layout="local", config=DEFAULT)
            rows = self.check(out, all_ids, "local")
            self.assertEqual(rows[0]["native_tune"], {"kind": "default"})

    def test_gathered_layout_and_tuned_config_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            out, folder = Path(d)/"workers", Path(d)/"tuned"
            folder.mkdir()
            tuned = dict(DEFAULT, BLOCK_SIZE_N=128, num_warps=8)
            path = folder/"E=8,N=1536,device_name=NVIDIA_A100-SXM4-80GB.json"
            path.write_text(json.dumps({"triton_version": "3.6.0", "16": dict(DEFAULT), "32": tuned}))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            fm = fused_moe_stub(Path(d))
            with patch.dict(sys.modules, {fm.__name__: fm}), \
                    patch.dict(os.environ, {"VLLM_TUNED_CONFIG_FOLDER": str(folder)}):
                all_ids = self.capture(out, layout="gathered_dp", config=tuned, tune_sha=digest)
                rows = self.check(out, all_ids, "gathered_dp")
                self.assertEqual(rows[0]["native_tune"]["kind"], "tuned")
                self.assertEqual(rows[0]["native_tune"]["sha256"], digest)
                # A launch that did not use the frozen file is a capture failure, never a silent default.
                bad = Path(d)/"bad"
                self.capture(bad, layout="local", config=DEFAULT, tune_sha=digest)
                self.assertTrue(list(bad.glob("capture-failure-rank*.json")))
                self.assertFalse((bad/"snapshots"/"GLM_DECODE"/"rank-0.json").exists())


if __name__ == "__main__":
    unittest.main()
