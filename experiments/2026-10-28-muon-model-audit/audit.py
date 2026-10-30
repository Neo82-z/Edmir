"""Read-only diagnostics of saved scoring models; no new selector training.

Tests capacity on training observations, affine null-space ambiguity on P8
held-out shapes, score-objective incompatibility, and P4 support gates.
The free-score fit is a label-space diagnostic, never a deployable model.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import time

import numpy as np
import torch
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'EDM-IR'))
from edmir.compiler import precision_candidate_selection as scorer

OUT = Path(__file__).parent
SOURCE = ROOT / 'experiments/2026-10-28-precision-scoring'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_stats(model, rows):
    predicted, actual = [], []
    for shape in sorted({tuple(r['shape']) for r in rows}):
        surface = [r for r in rows if tuple(r['shape']) == shape]
        scores = scorer.predict(model, surface)['scores']
        for r in surface:
            predicted.append(scores[r['candidate_id']])
            actual.append(math.log(r['latency_us']))
    e = np.asarray(predicted) - actual
    return {'log_rmse': float(np.sqrt(np.mean(e**2))),
            'mape': float(np.mean(np.abs(np.expm1(e)))),
            'max_relative_error': float(np.max(np.abs(np.expm1(e))))}


def support_audit(data, saved):
    items = []
    for fold in range(4):
        m = saved[fold]['model']
        train = [r for r in data if r['shape'] in m['training_shapes']]
        test = [r for r in data if r['shape'] not in m['training_shapes']]
        x = (np.array([scorer.feature_vector(r) for r in train]) - m['means']) / m['scales']
        xt = (np.array([scorer.feature_vector(r) for r in test]) - m['means']) / m['scales']
        a, at = np.c_[x, np.ones(len(x))], np.c_[xt, np.ones(len(xt))]
        u, sv, vt = np.linalg.svd(a, full_matrices=True)
        tol = sv[0] * max(a.shape) * np.finfo(float).eps
        rank = int(np.sum(sv > tol))
        null = vt[rank:].T
        y = np.log([r['latency_us'] for r in train])
        coef = np.linalg.lstsq(a, y, rcond=None)[0]
        tests = []
        for r, v in zip(test, at):
            z = null @ (null.T @ v)
            norm2 = float(z @ z)
            # 50% latency perturbation fixed before checking held-out labels.
            delta = math.log(1.5) * z / norm2 if norm2 > 1e-20 else np.zeros_like(z)
            family = r['candidate_id'].split('@')[0]
            tests.append({'candidate_id': r['candidate_id'],
                'same_candidate_training_count': sum(t['candidate_id'] == r['candidate_id'] for t in train),
                'q_training_counts': {str(q): sum(not t['is_host'] and scorer.parse_schedule_family(t['candidate_id'].split('@')[0])[0] == q for t in train) for q in (2,4,8,16)},
                'normalized_augmented_feature_null_component_norm': math.sqrt(norm2),
                'affine_null_perturbation': delta.tolist(),
                'max_training_score_change': float(np.max(np.abs(a @ delta))),
                'this_candidate_score_change': float(v @ delta),
                'this_candidate_latency_multiplier': math.exp(float(v @ delta)),
                'other_test_score_changes': (at @ delta).tolist()})
            if norm2 > 1e-20:
                assert np.max(np.abs(a @ delta)) < 1e-9
                assert abs(v @ delta - math.log(1.5)) < 1e-9
        items.append({'fold': fold, 'heldout_shape': test[0]['shape'],
            'train_rows': len(train), 'train_shapes': len(m['training_shapes']),
            'affine_parameter_count': a.shape[1], 'train_design_rank': rank,
            'nullity': a.shape[1] - rank,
            'unregularized_affine_train_max_log_error': float(np.max(np.abs(a @ coef-y))),
            'train_fit': prediction_stats(m, train), 'heldout_fit': prediction_stats(m, test),
            'candidate_ambiguity': tests})
    return items


def loss_audit(rows):
    surfaces = scorer.valid_rows(rows, 'train')
    items = []
    for surface in surfaces:
        n = len(surface)
        target = torch.tensor([[math.log(r['latency_us']) for r in surface]], dtype=torch.float64)
        mask = torch.ones_like(target, dtype=torch.bool)
        hosts = sum(r['is_host'] for r in surface)
        weights = torch.tensor([[.5/(hosts if r['is_host'] else n-hosts) for r in surface]], dtype=torch.float64)
        perfect = target.clone().requires_grad_(True)
        total, comp = scorer.losses(perfect, target, mask, weights, ce_weight=1.)
        grads = {k: torch.autograd.grad(v, perfect, retain_graph=True)[0].detach().numpy().ravel().tolist()
                 for k,v in comp.items() if k in ('anchor','expected_regret','gap','cross_entropy')}
        q = torch.softmax(-target/.05, dim=1).detach().numpy().ravel()
        regret = np.expm1(target.numpy().ravel()-target.min().item())
        analytic = q*(q@regret-regret)/.05
        assert np.max(np.abs(analytic-grads['expected_regret'])) < 1e-12
        assert max(abs(v) for k in ('anchor','gap','cross_entropy') for v in grads[k]) < 1e-12
        def fn(v):
            s = torch.tensor(v[None,:], dtype=torch.float64, requires_grad=True)
            loss, _ = scorer.losses(s,target,mask,weights,ce_weight=1.)
            g = torch.autograd.grad(loss,s)[0]
            return float(loss.detach()), g.detach().numpy().ravel()
        # Only optimizes independent label-space scores, not features/parameters.
        opt = minimize(fn, target.numpy().ravel(), jac=True, method='L-BFGS-B',
                       options={'maxiter':200,'maxfun':400,'ftol':1e-14,'gtol':1e-10})
        before, after = fn(target.numpy().ravel())[0], fn(opt.x)[0]
        assert after <= before + 1e-10
        items.append({'shape':surface[0]['shape'],'candidates':[r['candidate_id'] for r in surface],
            'observed_log_us':target.numpy().ravel().tolist(),
            'perfect_latency_component_gradients':grads,
            'objective_at_perfect_latency':before,'free_score_objective':after,
            'free_score_delta_log':(opt.x-target.numpy().ravel()).tolist(),
            'free_score_gradient_norm':float(np.linalg.norm(fn(opt.x)[1])),
            'free_score_solver_success':bool(opt.success),'free_score_solver_message':str(opt.message),
            'free_score_is_global_optimum_claim':False,
            'free_score_is_predictor':False})
    return items


def main():
    start = time.monotonic()
    paths = [SOURCE/'datasets-with-evidence-contracts.json',
             ROOT/'EDM-IR/edmir/compiler/precision_candidate_selection.py',
             SOURCE/'run-01/results.json']
    paths += sorted((SOURCE/'run-01').glob('p*-*-fold*.json'))
    protected = json.loads((ROOT/'experiments/2026-10-28-host-decision-audit/protected-before.json').read_text())
    before = {str(p.relative_to(ROOT)):digest(p) for p in paths}
    assert all(digest(ROOT/p) == h for p,h in protected.items())
    data = json.loads(paths[0].read_text())
    result = {'kind':'retrospective diagnosis, not model selection', 'sources':before,
        'article_urls':['https://spaces.ac.cn/archives/11985','https://spaces.ac.cn/archives/11964'],
        'new_optimizer_training':False,'frozen_six_used':False,'new_gpu_cost':0,
        'parameter_count':377, 'data_geometry':{},'p8_heldout_audits':{}}
    for ds in ('p4','p8'):
        rows = data[ds]['rows']
        x = np.array([scorer.feature_vector(r) for r in rows])
        result['data_geometry'][ds] = {'rows':len(rows),'shape_count':len({tuple(r['shape']) for r in rows}),
            'feature_count':x.shape[1], 'constant_features':[n for n,c in zip(list(scorer.base.FEATURES)+list(scorer.EXTRA),np.ptp(x,axis=0)) if c<1e-12],
            'affine_design_rank':int(np.linalg.matrix_rank(np.c_[x,np.ones(len(x))])),
            'exact_feature_collisions':len(x)-len(np.unique(x,axis=0))}
    for optimizer in ('adamw','dion3'):
        saved = [json.loads((SOURCE/f'run-01/p8-{optimizer}-ce1-fold{i}.json').read_text()) for i in range(4)]
        result['p8_heldout_audits'][optimizer] = support_audit(data['p8']['rows'],saved)
    result['p8_loss_audit'] = loss_audit(data['p8']['rows'])
    previous = json.loads((SOURCE/'run-01/results.json').read_text())
    result['p4_support_gates'] = {}
    result['p4_fit_audits'] = {}
    for variant in ('adamw-ce1','dion3-ce1'):
        decisions = previous['datasets']['p4'][variant]['decisions']
        result['p4_support_gates'][variant] = {'unsupported_shape_count':sum(r['residual_geometric_support'] is False for r in decisions),
            'errors':[r for r in decisions if r['regret']>0],
            'max_regret_record':max(decisions,key=lambda r:r['regret'])}
        fits = []
        for fold in range(4):
            model = json.loads((SOURCE/f'run-01/p4-{variant}-fold{fold}.json').read_text())['model']
            train = [r for r in data['p4']['rows'] if r['shape'] in model['training_shapes']]
            test = [r for r in data['p4']['rows'] if r['shape'] not in model['training_shapes']]
            fits.append({'fold':fold, 'train_fit':prediction_stats(model,train),
                         'heldout_fit':prediction_stats(model,test)})
            case = [r for r in test if r['shape']==[4,512,4096,512]]
            if case:
                scores = scorer.predict(model,case)['scores']
                prior = scorer.prior_model.prior_scores(model['prior'],case)
                fits[-1]['m512_prior_and_residual'] = [
                    {'candidate_id':r['candidate_id'],'measured_us':r['latency_us'],
                     'prior_us':math.exp(v),'predicted_us':math.exp(scores[r['candidate_id']])}
                    for r,v in zip(case,prior)]
        result['p4_fit_audits'][variant] = fits
    assert all(digest(ROOT/p) == h for p,h in {**before,**protected}.items())
    result['all_sources_and_protected_files_unchanged'] = True
    result['seconds'] = time.monotonic()-start
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'seconds':result['seconds'],'p4':result['data_geometry']['p4'],
                     'p8':result['data_geometry']['p8'],'unchanged':True},indent=2))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
