"""Plot the descriptive quality/latency points from analyze_service_phase.py."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    summary = json.loads(args.summary.read_text())
    args.output_dir.mkdir(parents=True)
    order = ('full', 'strict-prefix', 'blend-window10')
    labels = {'full': 'Full Prefill', 'strict-prefix': 'Strict Prefix', 'blend-window10': 'CacheBlend window 10%'}
    colors = {'full': '#596579', 'strict-prefix': '#157f6e', 'blend-window10': '#c4483b'}
    full_acc = summary['full']['accuracy_percent']
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    fig.subplots_adjust(left=.12, right=.98, bottom=.23, top=.88)
    for arm in order:
        item = summary[arm]
        x = item['ttft_with_context_s']['p95']
        y = item['accuracy_percent']
        lower_bound = full_acc - item['paired_session_bootstrap']['one_sided_upper95_pp']
        if lower_bound < y:
            ax.errorbar(x, y, yerr=[[y - lower_bound], [0]], fmt='none', ecolor=colors[arm],
                        elinewidth=1.6, capsize=4, zorder=2)
        ax.scatter(x, y, s=100, color=colors[arm], edgecolor='white', linewidth=1.0, zorder=3,
                   label=f"{labels[arm]} ({item['correct']}/120; P95={x:.3f}s)")
        display = labels[arm]
        if arm == 'blend-window10':
            display += f"\n95% upper loss={item['paired_session_bootstrap']['one_sided_upper95_pp']:.1f} pp"
        offsets = {'full': (-92, 10), 'strict-prefix': (8, -25), 'blend-window10': (-165, 4)}
        ax.annotate(display, (x, y), xytext=offsets[arm], textcoords='offset points', fontsize=9)
    prefix_p95 = summary['strict-prefix']['ttft_with_context_s']['p95']
    ax.axhline(full_acc - 1.0, color='#a66a00', linestyle=':', linewidth=1.6,
               label='Full accuracy − 1 pp')
    ax.axvline(prefix_p95, color=colors['strict-prefix'], linestyle='--', linewidth=1.1,
               alpha=.75, label='Strict Prefix P95')
    ax.set_xlabel('TTFT P95 including context RPC (seconds; lower is better)')
    ax.set_ylabel('Structured-answer accuracy (%)')
    ax.set_ylim(95.0, 98.9)
    ax.set_xlim(.85, 2.95)
    ax.set_title('Qwen3-8B host-instrumented calibration replay (n=120 per arm)')
    ax.grid(True, color='#d8dde5', linewidth=.7, alpha=.8)
    ax.set_axisbelow(True)
    ax.legend(loc='best', fontsize=8, frameon=True)
    fig.text(.12, .035,
             'Downward whisker: paired-session bootstrap one-sided 95% loss bound. Calibration only; fixed arm order;\n'
             'host phase hooks span multi-token generate calls. Not a holdout or production gate.',
             fontsize=8, color='#4f5968', va='bottom')
    for suffix in ('svg', 'png'):
        fig.savefig(args.output_dir / f'quality-latency.{suffix}', dpi=180 if suffix == 'png' else None,
                    bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'output_dir': str(args.output_dir.resolve()),
                      'files': ['quality-latency.svg', 'quality-latency.png']}))


if __name__ == '__main__':
    main()
