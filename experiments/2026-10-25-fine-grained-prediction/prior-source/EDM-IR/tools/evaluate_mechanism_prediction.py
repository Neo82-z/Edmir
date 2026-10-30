#!/usr/bin/env python3
"""Clean, calibrate, expose failures, and issue conditional local predictions.

No GPU execution. Requires a new output directory; never edits original data.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clean_mechanism_data import clean_collection, read, sha
from independent_selector_validation import digest
from edmir.compiler.mechanism_prediction import (
    MechanismModel, VARIANTS, HOSTS, basis, conditional_policy_latency, critical_path_us, select_prediction,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO/"experiments/2026-10-21-runpod-mechanism-collection/remote-run-01"
DEFAULT_DESIGN = REPO/"experiments/2026-10-24-mechanism-prediction/design-02.json"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def prediction(model, row, scope):
    result = model.predict(row["shape"], row["family"], context_id=row["context_id"])
    return {**result, "config_id": row["config_id"], "shape": row["shape"],
            "candidate_id": row["candidate_id"], "family": row["family"],
            "observed_us": row["observed_us"], "process_medians_us": row["process_medians_us"],
            "error_us": result["predicted_us"]-row["observed_us"],
            "absolute_percentage_error": abs(result["predicted_us"]/row["observed_us"]-1),
            "quality_flags": row["quality_flags"], "eligible_for_calibration": row["eligible_for_calibration"],
            "scope": scope}


def folds(primary, variant, *, include_flagged=False):
    predictions, audits = [], []
    for shape in sorted({tuple(r["shape"]) for r in primary}):
        train = [r for r in primary if tuple(r["shape"]) != shape
                 and (include_flagged or r["eligible_for_calibration"])]
        test = [r for r in primary if tuple(r["shape"]) == shape]
        model = MechanismModel.fit(train, variant=variant, include_flagged=include_flagged)
        assert shape not in model.training_shapes
        assert not set(model.training_config_ids) & {r["config_id"] for r in test}
        audits.append({"held_out_shape": shape, "model": model.to_dict()})
        for row in sorted(test, key=lambda r: r["candidate_id"]):
            predictions.append(prediction(model, row, "whole_shape_leave_out"))
    return predictions, audits


def comparisons(predictions):
    groups = defaultdict(list)
    for row in predictions:
        groups[tuple(row["shape"])].append(row)
    decisions, pairs = [], []
    for shape, all_rows in sorted(groups.items()):
        rows = [r for r in all_rows if not r["quality_flags"]]
        missing_hosts = sorted(set(HOSTS)-{r["family"] for r in all_rows})
        for scope in ("all_measured_stable", "compiler_host_vs_fused", "fused_only"):
            selected = [r for r in rows if scope == "all_measured_stable" or
                        (r["family"] == "host" or r["family"] not in HOSTS) and scope == "compiler_host_vs_fused"
                        or r["family"] not in HOSTS and scope == "fused_only"]
            if len(selected) < 2:
                continue
            name = select_prediction(selected)
            actual = min(selected, key=lambda r: (r["observed_us"], r["candidate_id"]))
            chosen = next(r for r in selected if r["candidate_id"] == name)
            decisions.append({"shape": shape, "scope": scope, "predicted_winner": name,
                "measured_subset_winner": actual["candidate_id"],
                "choice_loss_us": chosen["observed_us"]-actual["observed_us"],
                "choice_loss_fraction": chosen["observed_us"]/actual["observed_us"]-1,
                "measured_candidates": len(selected), "missing_host_families": missing_hosts,
                "quality_excluded": [r["candidate_id"] for r in all_rows if r["quality_flags"]],
                "full_candidate_oracle": False})
        for a, b in combinations(sorted(rows, key=lambda r: r["candidate_id"]), 2):
            actual_delta = b["observed_us"]-a["observed_us"]
            predicted_delta = b["predicted_us"]-a["predicted_us"]
            observed_best = min((a, b), key=lambda r: (r["observed_us"], r["candidate_id"]))
            predicted_best = min((a, b), key=lambda r: (r["predicted_us"], r["candidate_id"]))
            wrong = predicted_best["candidate_id"] != observed_best["candidate_id"]
            deltas = [vb-va for va, vb in zip(a["process_medians_us"], b["process_medians_us"])]
            categories = []
            if a["family"] in HOSTS or b["family"] in HOSTS:
                categories.append("host_vs_host" if a["family"] in HOSTS and b["family"] in HOSTS else "host_vs_fused")
            else:
                ga, gb = a["geometry"], b["geometry"]
                if ga["q"] == gb["q"]:
                    categories.append("grouped_N_same_q")
                if ga["g"] == gb["g"]:
                    categories.append("split_K_same_group")
                if not categories:
                    categories.append("joint_q_g")
            pairs.append({"shape": shape, "a": a["candidate_id"], "b": b["candidate_id"],
                "observed_b_minus_a_us": actual_delta, "predicted_b_minus_a_us": predicted_delta,
                "paired_process_deltas_us": deltas,
                "all_three_process_directions_agree": all(v > 0 for v in deltas) or all(v < 0 for v in deltas),
                "wrong_direction": wrong,
                "choice_loss_fraction": predicted_best["observed_us"]/observed_best["observed_us"]-1,
                "choice_loss_us": predicted_best["observed_us"]-observed_best["observed_us"],
                "categories": categories,
                "term_b_minus_a_us": {k: b["contributions_us"].get(k, 0)-a["contributions_us"].get(k, 0)
                                       for k in sorted(a["contributions_us"].keys() | b["contributions_us"].keys())}})
    return decisions, pairs


def metrics(predictions, design):
    stable = [r for r in predictions if not r["quality_flags"]]
    groups = defaultdict(list)
    for row in stable:
        groups[tuple(row["shape"])].append(row["absolute_percentage_error"])
    decisions, pairs = comparisons(predictions)
    macro = statistics.mean(statistics.mean(v) for v in groups.values())
    maximum = max(r["absolute_percentage_error"] for r in stable)
    severe = [p for p in pairs if p["wrong_direction"] and p["choice_loss_fraction"] > .10]
    accepted = (macro <= design["macro_shape_mean_absolute_percentage_error_max"]
                and maximum <= design["max_candidate_absolute_percentage_error_max"]
                and len(severe) <= design["wrong_pair_decisions_with_over_10_percent_loss_max"])
    return {"stable_candidates": len(stable), "shapes": len(groups),
            "macro_shape_mean_absolute_percentage_error": macro,
            "mean_absolute_error_us": statistics.mean(abs(r["error_us"]) for r in stable),
            "max_absolute_percentage_error": maximum,
            "wrong_pair_directions": sum(p["wrong_direction"] for p in pairs), "pairs": len(pairs),
            "wrong_pairs_over_10_percent_loss": len(severe),
            "development_acceptance_passed": accepted, "compiler_admission": False,
            "decision_metrics": {s: {"wrong": sum(d["choice_loss_us"] > 0 for d in decisions if d["scope"] == s),
                "count": sum(d["scope"] == s for d in decisions),
                "mean_choice_loss_fraction": statistics.mean(d["choice_loss_fraction"] for d in decisions if d["scope"] == s),
                "max_choice_loss_fraction": max(d["choice_loss_fraction"] for d in decisions if d["scope"] == s)}
                for s in sorted({d["scope"] for d in decisions})}}


def profile_exposure(root):
    audit_file = REPO/"experiments/2026-10-24-fused-dependency-audit/results.json"
    audit = read(audit_file)
    old = audit["captures"]["prepared-identity-full"]
    new = audit["captures"]["prepared-identity-compute_only"]
    ancestors = {int(k): v for k, v in old["kernel_ancestors"].items()}
    restricted = {int(k): v for k, v in new["kernel_ancestors"].items()}
    output = {"scope": "Profile-only fixed-duration DAG counterfactual. Never subtracted from unprofiled wall time.",
              "interpretation": "Exposure of queue ordering under observed kernel durations; not measured communication exposure or new-policy speedup.",
              "physical_ECR": None, "physical_EDMR": None,
              "missing_identification": ["fused-kernel communication/compute decomposition", "physical NVLink/HBM bytes", "semaphore stall duration"],
              "audit_sha256": sha(audit_file), "captures": {}}
    for label, info in sorted(audit["trace_consistency"].items()):
        db = root/"server-backup/work"/f"trace-{label}/timeline.sqlite"
        if sha(db) != info["sqlite_sha256"]:
            raise ValueError("profile identity changed")
        with sqlite3.connect(db.as_uri()+"?mode=ro", uri=True) as conn:
            rows = conn.execute("SELECT k.start,k.end,k.deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE ? ORDER BY k.start", ("%lsa_cutlass_gemm<%",)).fetchall()
        by_device = {d: [r for r in rows if r[2] == d] for d in range(4)}
        if any(len(v) != 484 for v in by_device.values()):
            raise ValueError("profile kernel count changed")
        per_iteration = []
        for iteration in range(100):
            times = {}
            for d in range(4):
                ids = [i for i, _, dest in old["op_order"] if dest == d]
                times.update({i: by_device[d][84+4*iteration+j][:2] for j, i in enumerate(ids)})
            if any(times[i][1] > times[j][0] for j in ancestors for i in ancestors[j]):
                raise ValueError("observed execution violates audited precedences")
            durations = {i: (end-start)/1000 for i, (start, end) in times.items()}
            full = critical_path_us(ancestors, durations)
            relaxed = critical_path_us(restricted, durations)
            envelope = (max(end for _, end in times.values())-min(start for start, _ in times.values()))/1000
            if not 0 <= relaxed <= full <= envelope+1e-6:
                raise ValueError("invalid exposure relation")
            per_iteration.append({"iteration": iteration, "observed_gpu_envelope_us": envelope,
                "old_graph_fixed_duration_cp_us": full, "restricted_graph_fixed_duration_cp_us": relaxed,
                "conditional_queue_exposure_us": full-relaxed, "exposure_fraction_of_old_cp": (full-relaxed)/full})
        output["captures"][label] = {"sqlite_sha256": sha(db), "per_iteration": per_iteration,
            "medians": {key: statistics.median(r[key] for r in per_iteration) for key in per_iteration[0] if key != "iteration"}}
    return output


def scenarios(predictions, design):
    output = []
    for r in predictions:
        if r["family"] in HOSTS:
            continue
        hosts = [p for p in predictions if p["shape"] == r["shape"] and p["family"] in HOSTS and not p["quality_flags"]]
        host = min(hosts, key=lambda p: p["observed_us"], default=None)
        output.append({"shape": r["shape"], "candidate_id": r["candidate_id"],
            "old_policy_heldout_prediction_us": r["predicted_us"],
            "old_policy_observed_us": r["observed_us"],
            "comparison_host": host["candidate_id"] if host else None,
            "comparison_host_observed_us": host["observed_us"] if host else None,
            "scenarios": [{"nonscaling_fraction": a, "service_multiplier": b,
                "from_heldout_prediction_us": conditional_policy_latency(r["predicted_us"], a, b),
                "from_observed_anchor_us": conditional_policy_latency(r["observed_us"], a, b)}
                for a in design["a"] for b in design["b"]],
            "status": "Assumption sensitivity only. Old residuals and readiness changes carry over; not new-policy validation."})
    return output


def structural_aliases(configs, variants):
    output = []
    for variant in variants:
        by_features = defaultdict(list)
        for row in configs:
            if row["family"] in HOSTS:
                by_features[(row["family"], basis(row["shape"], row["family"], variant)[1])].append(row)
        for (family, features), rows in by_features.items():
            for a, b in combinations(rows, 2):
                if a["shape"] == b["shape"]:
                    continue
                output.append({"variant": variant, "family": family, "features": features,
                    "shape_a": a["shape"], "shape_b": b["shape"],
                    "observed_a_us": a["observed_us"], "observed_b_us": b["observed_us"],
                    "unavoidable_max_absolute_error_us": abs(a["observed_us"]-b["observed_us"])/2,
                    "unavoidable_max_relative_error": abs(a["observed_us"]-b["observed_us"])/(a["observed_us"]+b["observed_us"]),
                    "reason": "Identical input vectors force identical predictions under any coefficient optimizer."})
    return output


def report(out, manifest, results, predictions, exposure, elapsed, selected_variant):
    coupled = results[selected_variant]
    decisions, pairs = comparisons(predictions)
    text = ["# 小范围数据清理、成本分解与预测", "",
        "本次仅在本地分析。原始数据未改写；没有 GPU 实测、神经网络训练或超参数搜索。少量非负系数标定仍属于从数据估参。",
        "已看过的数据用于开发，按整个形状留出可以检查泄漏和插值失败，但不是独立泛化验证。", "",
        "## 数据", "",
        f"共 {manifest['all_processes']} 个进程、{manifest['all_samples']:,} 次计时；主测 282 个进程，预热诊断 24 个进程分开保留。",
        f"69 个 prepared/merged 主配置中 {manifest['calibration_configurations']} 个满足既定稳定性规则；6 个完整配置隔离标记，不删除任何慢样本。",
        "192 个封存文件、136 个源文件、20 个输入/参考二进制哈希核对通过；全部原始 stdout、计时统计、实现/算法身份、输入身份和区组完整性重验通过。",
        "另核验了 63 个编译后二进制及归档 NCCL 动态库。两个 NCCL 头文件只有原构建记录的哈希，本地归档未包含它们；不声称完成全部外部构建依赖的重新验证。",
        "三个进程的中位数再取中位数为标签。100 次迭代不是 100 个独立重复。短 K 的 200 次预热诊断不替换主测标签。", "",
        "## 模型", "",
        "fused 共享 7 个固定成本项：固定项、K/q 分区深度、grouped-N × 分区深度、CTA 数、唯一 A 大小、A 的逻辑重读量、split-K 额外串行输出读写量。",
        "四种 host 各自标定固定项、远端 A 字节、逻辑计算量；不把 merged 的 PM 行 GEMM 当成 per-source 的 M 行 GEMM。第二轮仅对已知 CUTLASS 的两种 host 增加 K/32 循环深度，修复相同 MK/MKN 却不同 K 的表示冲突；没有猜测 cuBLAS 内部 tile。",
        "上述字节项是逻辑计数。系数解释端到端时间关联，不能当作实测带宽、占用率或信号量等待时间。固定 1e-6 数值正则；每个模型最多枚举 128 个非负解面，检查 KKT。",
        "所有候选共享同一形状留出边界。强 host 与 fused 分别评估，也单独检查 fused 内部排序。平局按候选 ID，遍历顺序不改变选择。", "",
        "## 留出结果", "", "| 固定模型 | 按形状平均绝对相对误差 | 最大候选误差 | 错误方向/候选对 | 错选损失 >10% 的候选对 |", "|---|---:|---:|---:|---:|"]
    for name, r in results.items():
        text.append(f"| {name} | {r['macro_shape_mean_absolute_percentage_error']:.1%} | {r['max_absolute_percentage_error']:.1%} | {r['wrong_pair_directions']}/{r['pairs']} | {r['wrong_pairs_over_10_percent_loss']} |")
    text += ["", "reduced_counts 是本轮固定的删项对照，不是旧模型的跨版本复现；不能用这张表宣称战胜此前不同数据身份上的预测器。",
        "开发验收要求：按形状平均误差 ≤10%、最大误差 ≤25%、错选损失超过 10% 的候选对为 0。",
        f"本轮修正版 {selected_variant} 验收：**{'通过' if coupled['development_acceptance_passed'] else '未通过'}**。即使通过也不自动进入编译器决策。第二轮修正看过第一轮结果，因此不是独立验证。", "",
        "| 决策范围 | 错选形状 | 平均选择损失 | 最坏选择损失 |", "|---|---:|---:|---:|"]
    for name, r in coupled["decision_metrics"].items():
        text.append(f"| {name} | {r['wrong']}/{r['count']} | {r['mean_choice_loss_fraction']:.1%} | {r['max_choice_loss_fraction']:.1%} |")
    text += ["", "geometry 阶段没有测齐强 host；表中最优只限该形状已测且稳定的候选。候选间极小差异也计入原始误选数，不能宣称统计显著。", "",
        "## M512 与大 N：数值预测", "", "单位 μs。预测完全排除该形状所有候选；实测列不是训练输入。", "",
        "| M,K,N | 候选 | 留出预测 | 实测 | 误差 | 质量标记 |", "|---|---|---:|---:|---:|---|"]
    for r in sorted(predictions, key=lambda r: (r["shape"], r["candidate_id"])):
        if r["shape"] not in ([4,512,4096,512], [4,64,4096,2048]):
            continue
        text.append(f"| {','.join(map(str,r['shape'][1:]))} | {r['candidate_id']} | {r['predicted_us']:.1f} | {r['observed_us']:.1f} | {r['error_us']:+.1f} | {','.join(r['quality_flags']) or '无'} |")
    text += ["", "## 不能被清理掩盖的模型错误", "", "以下按实际损失排序，所有候选对和成本项差异见 pair-errors.json。", "",
        "| M,K,N | 候选 A / B | 预测 B−A μs | 实测 B−A μs | 错选损失 |", "|---|---|---:|---:|---:|"]
    for p in sorted((p for p in pairs if p["wrong_direction"]), key=lambda p: -p["choice_loss_fraction"])[:12]:
        text.append(f"| {','.join(map(str,p['shape'][1:]))} | {p['a']} / {p['b']} | {p['predicted_b_minus_a_us']:+.1f} | {p['observed_b_minus_a_us']:+.1f} | {p['choice_loss_fraction']:.1%} |")
    text += ["", "## 条件暴露与新策略预测", "",
        "对三个旧 profiler 捕获，保留各 kernel 原时长，重算旧图和精简依赖图的最长路径。差值是**固定时长假设下的队列依赖暴露**，不等于可实现的加速，也不是物理通信暴露。", "",
        "| 捕获 | 旧图路径 μs | 精简图路径 μs | 条件暴露 μs |", "|---|---:|---:|---:|"]
    for name, row in exposure["captures"].items():
        r = row["medians"]
        text.append(f"| {name} | {r['old_graph_fixed_duration_cp_us']:.1f} | {r['restricted_graph_fixed_duration_cp_us']:.1f} | {r['conditional_queue_exposure_us']:.1f} |")
    text += ["", "各列独立取中位数；暴露列先逐轮相减，再取中位数，因此不必等于前两列中位数之差。",
        "新策略使用显式情景 T′=T[a+(1−a)(4/11)b]，a 为不随路径缩短的耗时占比，b 为并发改变后的服务时间倍率；二者尚未测得。",
        "policy-scenarios.json 分开保存以留出预测和旧实测为锚的数值。a=0、b=1 是理想化假设，不能据此宣称 2.75 倍加速。物理 ECR/EDMR 保持未知。",
        "情景包含 b=3 的争用增加情形：并行度增加也可能抹去收益。枚举范围不是统计置信区间。",
        "当前修复后的源码、输入就绪策略与新依赖图均没有 GPU 标签，旧数据不能验证新运行时。", "",
        "## 复现与文件", "", "在仓库根目录执行（输出目录必须不存在）：", "", "```sh",
        "python3 EDM-IR/tools/evaluate_mechanism_prediction.py --out /tmp/edmir-mechanism-review", "```", "",
        "- normalized-processes.jsonl：306 条原始记录及来源；保留全部 samples_ms。",
        "- configurations.json / dataset-manifest.json：标签、三次进程中位数、质量标记及哈希身份。",
        "- models.json / fold-models.json：全量标定及每个留出模型、系数、秩、KKT、训练 ID。",
        "- heldout-predictions.json / calibration-predictions.json：数值、成本项、外推和误差。",
        "- pair-errors.json / decisions.json：双向 host/fused、q、grouped-N 错误清单。",
        "- structural-aliases.json：旧计数表示的相同输入冲突，以及任何优化器都无法消除的误差下界。",
        "- exposure.json / policy-scenarios.json：条件暴露与新策略假设，不混入主测标签。",
        "- quality-sensitivity.json：含被标记配置的标定对照，明确标为敏感性分析。",
        f"", f"本次生成耗时 {elapsed:.2f} 秒；无外部依赖安装。软件执行耗时只用于检查本地流程，不代表 GPU 性能。", ""]
    (out/"README.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if args.out.exists():
        parser.error("output directory must not exist")
    design = read(args.design)
    rows, configs, manifest = clean_collection(args.data)
    primary = [r for r in configs if r["primary_scope"]]
    stable = [r for r in primary if r["eligible_for_calibration"]]
    variants = [v for v in VARIANTS if v in design["models"]]
    selected_variant = design.get("selected_variant", "coupled_counts")
    if not variants or selected_variant not in variants:
        raise ValueError("design must explicitly specify supported variants and selection")
    models, fold_models, heldout, calibrated, results = {}, {}, {}, {}, {}
    for variant in variants:
        model = MechanismModel.fit(stable, variant=variant)
        models[variant] = model.to_dict()
        heldout[variant], fold_models[variant] = folds(primary, variant)
        calibrated[variant] = [prediction(model, r, "in_sample_development") for r in primary]
        results[variant] = metrics(heldout[variant], design["development_acceptance"])
    sensitivity, sensitivity_models = folds(primary, selected_variant, include_flagged=True)
    exposure = profile_exposure(args.data)
    args.out.mkdir(parents=True)
    (args.out/"normalized-processes.jsonl").write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False)+"\n" for r in rows))
    write(args.out/"design.json", design)
    write(args.out/"dataset-manifest.json", manifest)
    write(args.out/"configurations.json", configs)
    write(args.out/"models.json", models)
    write(args.out/"fold-models.json", fold_models)
    write(args.out/"heldout-predictions.json", heldout)
    write(args.out/"calibration-predictions.json", calibrated)
    write(args.out/"metrics.json", results)
    decisions, pairs = comparisons(heldout[selected_variant])
    write(args.out/"decisions.json", decisions)
    write(args.out/"pair-errors.json", pairs)
    write(args.out/"structural-aliases.json", structural_aliases(stable, variants))
    write(args.out/"quality-sensitivity.json", {"scope": "All primary labels, including flagged configurations; NOT stable-only evidence.",
        "predictions": sensitivity, "fold_models": sensitivity_models,
        "scored_on_same_stable_subset": metrics(sensitivity, design["development_acceptance"])})
    write(args.out/"exposure.json", exposure)
    write(args.out/"policy-scenarios.json", scenarios(heldout[selected_variant], design["new_policy_scenarios"]))
    report(args.out, manifest, results, heldout[selected_variant], exposure, time.monotonic()-started, selected_variant)
    source_files = [Path(__file__), Path(__file__).with_name("clean_mechanism_data.py"),
                    REPO/"EDM-IR/edmir/compiler/mechanism_prediction.py", args.design,
                    *[Path(__file__).with_name(f) for f in ("run_mechanism_gpu.py", "run_selector_gpu.py", "independent_selector_validation.py")]]
    write(args.out/"analysis-seal.json", {"design_sha256": digest(design),
        "source_sha256": {str(f.relative_to(REPO)) if f.is_relative_to(REPO) else str(f): sha(f) for f in source_files},
        "output_sha256": {f.name: sha(f) for f in sorted(args.out.iterdir()) if f.is_file()},
        "gpu_runs": 0, "neural_training_runs": 0, "hyperparameter_searches": 0,
        "independent_validation": False, "compiler_admission": False})
    print(json.dumps({"out": str(args.out.resolve()), "metrics": results,
                      "elapsed_seconds": time.monotonic()-started}, indent=2))


if __name__ == "__main__":
    main()
