"""Plot every frozen shape and every primary paired block; never refit scores."""
from pathlib import Path
import json
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / 'analysis.json').read_text())
    folder = HERE / 'figures'
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for ax, (key, pair) in zip(axes, data['p1']['pairs'].items()):
        a, b = key.split('__')
        for row in pair['blocks']:
            ax.plot([0, 1], [row['a_us'], row['b_us']], 'o-',
                    color='#6486b5', alpha=.65, linewidth=1, markersize=4)
        av = st.median(r['a_us'] for r in pair['blocks'])
        bv = st.median(r['b_us'] for r in pair['blocks'])
        ax.plot([0, 1], [av, bv], 's-', color='#1d293d', linewidth=2, markersize=6,
                label='Median across blocks')
        labels = ['Serialized', 'Pipelined'] if a == 'pipeline_serial' else ['q4 / g1', 'q4 / g2']
        ax.set_xticks([0, 1], labels)
        ax.set_ylabel('Unprofiled program latency (µs)')
        ax.set_title(f"Median paired speedup: {pair['median_speedup']:.3f}×")
        ax.grid(axis='y', alpha=.2)
        ax.set_xlim(-.25, 1.25)
        ax.legend(loc='best', fontsize=8)
    fig.suptitle('P1: six paired blocks, three seeds, reversed execution order')
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(folder / f'p1-paired.{ext}', dpi=180)
    plt.close(fig)

    tags = sorted(data['p2'])
    methods = [('model_top1', 'Frozen score: top 1', '#b44f4f'),
               ('model_top3', 'Frozen score: top 3 + measurement', '#e8a464'),
               ('fixed_q4_g2', 'Fixed q4 / g2', '#478f8a')]
    fig, (ax, host_ax) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, constrained_layout=True)
    xs = np.arange(len(tags)); width = .22
    for j, (name, label, color) in enumerate(methods):
        ys = [data['p2'][tag]['evaluations'][name]['aggregate_regret_pct'] for tag in tags]
        bars = ax.bar(xs + (j - 1) * width, ys, width, label=label, color=color)
        ax.bar_label(bars, labels=[f'{v:.2f}' for v in ys], padding=3, fontsize=8)
    ax.set_title('(a) Frozen scoring and fixed fused baseline')
    ax.set_ylabel('Selection regret (%)')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc='upper right', fontsize=8)
    ax.margins(y=.17)
    for j, (name, label, color) in enumerate([
            ('independent_host_choice', 'Measured merged-host choice', '#6486b5'),
            ('strong_host_oracle', 'Best host on verification (hindsight)', '#9daabe')]):
        ys = [data['p2'][tag]['evaluations'][name]['aggregate_regret_pct'] for tag in tags]
        host_ax.bar(xs + (j - .5) * .27, ys, .27, label=label, color=color)
    host_ax.set_title('(b) Host controls on the same six shapes (different y-axis scale)')
    host_ax.set_xticks(xs, [tag + '\n' + '/'.join(map(str, data['p2'][tag]['shape'][1:])) for tag in tags])
    host_ax.set_xlabel('Frozen shape: M / K / N (P = 8)')
    host_ax.set_ylabel('Selection regret (%)')
    host_ax.set_ylim(bottom=0)
    host_ax.grid(axis='y', alpha=.2)
    host_ax.set_axisbelow(True)
    host_ax.legend(loc='upper left', fontsize=8)
    fig.text(.01, -.035, 'Oracle includes all numerically eligible measured candidates. Verification uses three process medians; shortlist choice uses a separate block.\n'
             'Frozen numerical-head transfer diagnostic; strict deployment admission is separate. Fixed-order top-3 equals fixed q4/g2 here.', fontsize=8)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(folder / f'p2-frozen-shapes.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
