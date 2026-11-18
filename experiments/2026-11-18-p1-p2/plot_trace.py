"""Show the first captured iteration, not a hand-picked best overlap window."""
from pathlib import Path
import json
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent


def main():
    audit = json.loads((HERE / 'trace-analysis.json').read_text())
    if not audit['all_four_coverage_complete']:
        raise ValueError('Trace coverage must pass before plotting')
    cases = [x for x in audit['cases'] if x['family'] in ('pipeline_serial', 'pipeline')]
    cases.sort(key=lambda x: 0 if x['family'] == 'pipeline_serial' else 1)
    colors = {'NCCL kernel': '#d27d43', 'GEMM kernel': '#467aa5', 'Other kernel': '#7e8b86'}
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, constrained_layout=True)
    windows = []
    for ax, case in zip(axes, cases):
        path = HERE / 'recovered/results/diagnostic' / case['case'] / 'timeline.sqlite'
        conn = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        strings = dict(conn.execute('select id,value from StringIds'))
        phases = []
        for row in conn.execute('select * from NVTX_EVENTS where end>start'):
            r = dict(row)
            text = r.get('text') or strings.get(r.get('textId'), '')
            if text.startswith('edm.phase:runtime.execute'):
                phases.append(r)
        phases.sort(key=lambda x: x['start'])
        assert len(phases) == 3
        start, end = phases[0]['start'], phases[1]['start']
        rows = [dict(x) for x in conn.execute(
            'select k.*,s.value kernel from CUPTI_ACTIVITY_KIND_KERNEL k '
            'join StringIds s on k.demangledName=s.id where k.start>=? and k.end<=?', (start, end))]
        assert {x['deviceId'] for x in rows} == set(range(8))
        for gpu in range(8):
            for category, color in colors.items():
                intervals = []
                for r in rows:
                    k = r['kernel'].lower()
                    kind = 'NCCL kernel' if k.removeprefix('void ').startswith(('nccldevkernel_', 'ncclkernel_')) else 'GEMM kernel' if ('cutlass' in k or 'gemm' in k) else 'Other kernel'
                    if r['deviceId'] == gpu and kind == category:
                        intervals.append(((r['start'] - start) / 1000, (r['end'] - r['start']) / 1000))
                ax.broken_barh(intervals, (gpu - .32, .64), facecolors=color, linewidth=0)
        ax.set_yticks(range(8), [f'GPU {i}' for i in range(8)])
        ax.invert_yaxis()
        ax.set_title('Serialized dependencies' if case['family'] == 'pipeline_serial' else 'Pipelined dependencies')
        ax.grid(axis='x', alpha=.2)
        windows.append({'case': case['case'], 'selection': 'first of exactly three captured iterations',
                        'start_ns': start, 'next_iteration_start_ns': end,
                        'kernel_count': len(rows), 'device_ids': sorted({x['deviceId'] for x in rows})})
        conn.close()
    axes[-1].set_xlabel('Time from first CPU execute-range start (µs; profiler enabled)')
    axes[0].legend(handles=[Patch(color=v, label=k) for k, v in colors.items()],
                   ncol=3, loc='lower right', bbox_to_anchor=(1, 1.13), fontsize=9)
    fig.suptitle('P1 diagnostic timeline: all eight GPUs, first captured execution')
    fig.text(.01, -.025, 'Kernel intervals only; memory-copy/API activity is omitted. NCCL intervals may contain waiting.\n'
             'Temporal overlap is not a causal hidden-communication fraction. Performance comes from separate unprofiled runs.', fontsize=8)
    out = HERE / 'figures'
    out.mkdir(exist_ok=True)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(out / f'p1-timeline.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    (HERE / 'timeline-plot-windows.json').write_text(json.dumps(windows, indent=2) + '\n')


if __name__ == '__main__':
    main()
