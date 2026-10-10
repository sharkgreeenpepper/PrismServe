"""Compare quality and strict-prefix-normalized P95 across two Qwen3 models."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary-8b', type=Path, required=True)
    parser.add_argument('--summary-14b', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    summaries = {
        'Qwen3-8B': json.loads(args.summary_8b.read_text()),
        'Qwen3-14B': json.loads(args.summary_14b.read_text()),
    }
    args.output_dir.mkdir(parents=True)
    arms = ('full', 'strict-prefix', 'blend-window10')
    labels = {'full': 'Full Prefill', 'strict-prefix': 'Strict Prefix',
              'blend-window10': 'Blend 10%'}
    colors = {'full': '#596579', 'strict-prefix': '#157f6e', 'blend-window10': '#c4483b'}
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=False)
    fig.subplots_adjust(left=.08, right=.985, bottom=.20, top=.84, wspace=.22)
    for ax, (model, summary) in zip(axes, summaries.items()):
        strict_p95 = summary['strict-prefix']['ttft_with_context_s']['p95']
        full_acc = summary['full']['accuracy_percent']
        lower_bounds = [full_acc - summary[arm]['paired_session_bootstrap']['one_sided_upper95_pp']
                        for arm in arms]
        accuracies = [summary[arm]['accuracy_percent'] for arm in arms]
        for arm in arms:
            item = summary[arm]
            x = item['ttft_with_context_s']['p95'] / strict_p95
            y = item['accuracy_percent']
            lower = full_acc - item['paired_session_bootstrap']['one_sided_upper95_pp']
            if lower < y:
                ax.errorbar(x, y, yerr=[[y - lower], [0]], fmt='none', ecolor=colors[arm],
                            elinewidth=1.5, capsize=4, zorder=2)
            ax.scatter(x, y, s=92, color=colors[arm], edgecolor='white', linewidth=1.0,
                       zorder=3, label=f"{labels[arm]} ({item['correct']}/120)")
            ax.annotate(f"{labels[arm]}\n{x:.2f}×, {y:.1f}%", (x, y),
                        xytext={'full': (-35, 12), 'strict-prefix': (8, -26),
                                'blend-window10': (-48, -34)}[arm],
                        textcoords='offset points', fontsize=8)
        ax.axvline(1.0, color=colors['strict-prefix'], linestyle='--', linewidth=1.0,
                   alpha=.7, label='Strict Prefix P95')
        ax.axvline(.9, color='#a66a00', linestyle=':', linewidth=1.2,
                   label='10% P95 improvement target')
        ax.axhline(full_acc - 1.0, color='#8664a8', linestyle=':', linewidth=1.2,
                   label='Full accuracy − 1 pp')
        ax.set_title(model)
        ax.set_xlabel('TTFT P95 / same-model Strict Prefix P95')
        ax.set_xlim(.78, 2.9)
        ax.set_ylim(min(lower_bounds) - .4, max(accuracies) + .5)
        ax.grid(True, color='#d8dde5', linewidth=.7, alpha=.8)
        ax.set_axisbelow(True)
        ax.legend(loc='best', fontsize=7.5, frameon=True)
    axes[0].set_ylabel('Structured-answer accuracy (%)')
    fig.suptitle('Model-scale calibration: quality vs normalized TTFT P95', fontsize=12)
    fig.text(.08, .035,
             'Both panels use the same 120 synthetic calibration requests. Error whisker: paired-session bootstrap one-sided 95% loss bound.\n'
             'Each model is normalized to its own Strict Prefix P95; not a production or holdout result.',
             fontsize=8, color='#4f5968', va='bottom')
    for suffix in ('svg', 'png'):
        fig.savefig(args.output_dir / f'model-scale-quality-latency.{suffix}',
                    dpi=180 if suffix == 'png' else None, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'output_dir': str(args.output_dir.resolve()),
                      'files': ['model-scale-quality-latency.svg', 'model-scale-quality-latency.png']}))


if __name__ == '__main__':
    main()
