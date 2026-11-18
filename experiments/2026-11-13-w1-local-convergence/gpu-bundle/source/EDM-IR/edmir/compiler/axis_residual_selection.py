"""Fixed, training-only axis-transfer controls for the archival P4 scorer.

These are empirical end-to-end residual transfers, not GPU service estimates.
No coefficients are fit. Existing K corrections retain precedence. A bounded
M extension is explicitly unsupported extrapolation, never automatic admission.
"""
from __future__ import annotations

from bisect import bisect_left
from copy import deepcopy
import json
import math

from . import curvature_transfer_selection as base

SCHEMA = "edmir.axis-residual-control.v1"
MODES = ("axis_brackets", "axis_M_extension", "M_work_extension")
AXES = ((1, "M"), (3, "N"))


def _key(row, axis):
    shape = list(row["shape"])
    shape[axis] = None
    return json.dumps([row["execution_context"], shape, row["candidate_id"],
                       base.stage.curves.baseline.old.precision_key(row),
                       row.get("implementation_id")], separators=(",", ":"))


def assemble(saved_model, training_rows, mode="axis_brackets"):
    if mode not in MODES:
        raise ValueError("unknown fixed control")
    surfaces = base.stage.curves.baseline.old.valid_rows(training_rows, "train")
    if {tuple(s[0]["shape"]) for s in surfaces} != {tuple(s) for s in saved_model["training_shapes"]}:
        raise ValueError("exact original training-fold shapes required")
    expected=set()
    for key,points in saved_model['curves'].items():
        _,p,m,n,candidate,*_=json.loads(key)
        expected.update(((p,m,v['k'],n),candidate,v['source_id']) for v in points)
    observed={(tuple(r['shape']),r['candidate_id'],r['source_id']) for s in surfaces for r in s}
    if observed != expected:
        raise ValueError('all original training candidates required')
    index = {}
    for surface in surfaces:
        old = base.predict(saved_model, surface)
        for row in surface:
            # Verify the same observed training anchors already stored in base.
            stored = saved_model["curves"][base.stage.curves.curve_key(row)]
            knot = next(k for k in stored if k["k"] == row["shape"][2])
            if knot["source_id"] != row["source_id"] or knot["latency_us"] != row["latency_us"]:
                raise ValueError("training observation differs from saved model")
            prior = old["curve_details"][row["candidate_id"]]["prior_us"]
            for axis, _ in AXES:
                index.setdefault(_key(row, axis), []).append({
                    "x": row["shape"][axis], "residual_us": row["latency_us"] - prior,
                    "latency_us": row["latency_us"],
                    "shape": list(row["shape"]), "source_id": row["source_id"]})
    for points in index.values():
        points.sort(key=lambda p: p["x"])
        if len(points) > 256 or len({p["x"] for p in points}) != len(points):
            raise ValueError("bounded unique axis anchors required")
    return {"schema": SCHEMA, "mode": mode, "base": deepcopy(saved_model), "index": index,
            "training_shapes": deepcopy(saved_model["training_shapes"]),
            "compiler_admission": False, "optimizer": None}


def _bracket(points, x, *, extend=False):
    if len(points)>256 or type(x) is not int or x<=0:
        raise ValueError('bounded axis query required')
    if any(type(p['x']) is not int or p['x']<=0 or
           not math.isfinite(p['residual_us']) or not math.isfinite(p['latency_us']) or p['latency_us']<=0 for p in points):
        raise ValueError('invalid axis observations')
    if len(points) < 2:
        return None
    xs = [p["x"] for p in points]
    if xs != sorted(set(xs)):
        raise ValueError('strictly ordered axis anchors required')
    i = bisect_left(xs, x)
    if i < len(xs) and xs[i] == x:
        return None  # Existing K route owns exact training points.
    extrapolated = False
    if 0 < i < len(xs):
        pair = points[i-1:i+1]
    elif extend and i == len(xs) and x <= 2 * xs[-1]:
        pair, extrapolated = points[-2:], True
    else:
        return None
    a, b = pair
    alpha = (x-a["x"])/(b["x"]-a["x"])
    weights = [1-alpha, alpha]
    return {"residual_us": math.fsum(w*p["residual_us"] for w,p in zip(weights,pair)),
            "weights": weights, "anchors": pair, "extrapolated": extrapolated}


def predict(model, rows):
    if model.get("schema") != SCHEMA or model.get("mode") not in MODES:
        raise ValueError("unknown axis model")
    p = base.predict(model["base"], rows)
    p["axis_details"] = {}
    for row in sorted(rows, key=lambda r: r["candidate_id"]):
        name = row["candidate_id"]
        if name not in p["predicted_log_us"]:
            continue
        detail = p["curve_details"][name]
        if detail["curve_applied"] or detail["host_interval_applied"]:
            p["axis_details"][name] = {"applied": False, "reason": "existing_K_route"}
            continue
        transfers = []
        for axis, label in (() if model["mode"] == "M_work_extension" else AXES):
            hit = _bracket(model["index"].get(_key(row,axis), []), row["shape"][axis])
            if hit:
                transfers.append({"axis": label, **hit})
        if not transfers and model["mode"] in ("axis_M_extension", "M_work_extension"):
            hit = _bracket(model["index"].get(_key(row,1), []), row["shape"][1], extend=True)
            if hit and (model["mode"] != "M_work_extension" or hit['extrapolated']):
                transfers.append({"axis": "M", **hit})
        if not transfers:
            p["axis_details"][name] = {"applied": False, "reason": "no_axis_support"}
            continue
        value = detail["prior_us"] + math.fsum(t["residual_us"] for t in transfers)/len(transfers)
        absolute_host = row['is_host'] and model['mode'] == 'M_work_extension'
        if absolute_host:
            t = transfers[0]
            value = math.fsum(w*a['latency_us'] for w,a in zip(t['weights'],t['anchors']))
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid axis prediction; do not clip or silently revert")
        p["predicted_log_us"][name] = math.log(value)
        detail['base_prediction_us'] = detail['prediction_us']
        detail['prediction_us'] = value
        detail['axis_transfer_applied'] = True
        p["axis_details"][name] = {"applied": True, "prediction_us": value, "transfers": transfers,
                                   "absolute_host_work_extension": absolute_host,
                                   "not_identified_component_services": True}
        if any(t["extrapolated"] for t in transfers):
            p["support"][name] = {"supported": False, "reason": "unverified_M_axis_extrapolation"}
    p["provisional_choice"] = base.stage.curves.baseline.canonical_choice(p["predicted_log_us"]) if p["predicted_log_us"] else None
    p["requires_validation"] = (not p["predicted_log_us"] or
        any(not v["supported"] for v in p["support"].values()) or
        any(v != "failed" for v in p["excluded"].values()))
    p["chosen"] = None if p["requires_validation"] else p["provisional_choice"]
    p["compiler_admission"] = False
    return p
