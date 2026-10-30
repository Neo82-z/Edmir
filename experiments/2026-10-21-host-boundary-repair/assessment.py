"""Promotion gate: average gains cannot override boundary regressions."""
N_SHAPE=(4,64,4096,2048)
M_SHAPES={(4,512,k,512) for k in (2048,4096,8192)}


def assess(result,review):
    primary=result['primary']['decisions'];stress=result['stress']['decisions']
    for rows in (primary,stress):
        if not (M_SHAPES|{N_SHAPE}) <= {tuple(r['shape']) for r in rows}:
            raise ValueError('missing requested evaluation shapes')
    deleted={tuple(r['shape']):r for r in review['deletion']}
    if not (M_SHAPES|{N_SHAPE}) <= set(deleted):raise ValueError('missing deletion tests')
    if any(len(deleted[s]['deletions'])!=8 for s in M_SHAPES|{N_SHAPE}):raise ValueError('incomplete deletion tests')
    boundary=review['boundary_stress']
    if tuple(boundary['shape'])!=N_SHAPE or len(boundary['records'])!=34:raise ValueError('incomplete boundary stress')
    old={tuple(r['shape']):r for r in review['previous_attacks']}
    if N_SHAPE not in old:raise ValueError('missing previous adversarial replay')
    checks={
        'unchanged_primary_gates_and_M512_exact_choices':result['gate']['passed'],
        'M512_all_training_deletions_preserve_best':all(d['regret']<=1e-12 for s in M_SHAPES for d in deleted[s]['deletions']),
        'N_all_training_deletions_preserve_host':all(d['choice']=='host' for d in deleted[N_SHAPE]['deletions']),
        'N_previous_attack_resisted':old[N_SHAPE]['new_choice']=='host',
        'N_one_percent_stress_all_correct':all(r['regret']<=1e-12 for r in boundary['records']),
    }
    return {'candidate_accepted':all(checks.values()),'checks':checks,
            'interpretation':'Retrospective engineering checks only. Passing is not independent GPU validation.'}
