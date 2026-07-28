'''
    VGG16-C10: Baseline vs start_ep=50 adaptive lambda
    Spike count (x) vs Accuracy (y) trajectory
'''
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_FILE = 'results_viz/vgg_ep50_vs_baseline.png'


def parse_log(path):
    epochs, val_accs, s_counts, lambdas = [], [], [], []
    epoch_num = 0
    with open(path) as f:
        for line in f:
            if '500/500' in line and 'val_acc' in line and 's_count' in line:
                epoch_num += 1
                m_acc = re.search(r'val_acc:\s*([\d.]+)', line)
                m_sc = re.search(r's_count:\s*([\d.]+)', line)
                m_lam = re.search(r'adp_lambda:\s*([\d.eE+-]+)', line)
                if m_acc and m_sc:
                    epochs.append(epoch_num)
                    val_accs.append(float(m_acc.group(1)))
                    s_counts.append(float(m_sc.group(1)))
                    lambdas.append(float(m_lam.group(1)) if m_lam else 0.0)
    return np.array(epochs), np.array(val_accs), np.array(s_counts), np.array(lambdas)


BASELINE = '_baseline_vgg_c10_trajectory/train.log'

EXPERIMENTS = [
    # start_ep=1 best
    {'log': '_adaptive_lambda_v2/vgg_c10_su0.5_lmax1e-6/train.log',
     'label': 'start_ep=1, lmax=1e-6', 'color': '#9467bd', 'marker': 'D', 'ls': '--'},
    # start_ep=50 sweep
    {'log': '_adaptive_lambda_v2/vgg_c10_ep50_lmax1e-6/train.log',
     'label': 'ep50, lmax=1e-6', 'color': '#1f77b4', 'marker': 's'},
    {'log': '_adaptive_lambda_v2/vgg_c10_ep50_lmax3e-6/train.log',
     'label': 'ep50, lmax=3e-6', 'color': '#2ca02c', 'marker': '^'},
    {'log': '_adaptive_lambda_v2/vgg_c10_ep50_lmax5e-6/train.log',
     'label': 'ep50, lmax=5e-6', 'color': '#ff7f0e', 'marker': 'v'},
    {'log': '_adaptive_lambda_v2/vgg_c10_ep50_lmax1e-5/train.log',
     'label': 'ep50, lmax=1e-5', 'color': '#d62728', 'marker': 'p'},
    {'log': '_adaptive_lambda_v2/vgg_c10_ep50_lmax1e-4/train.log',
     'label': 'ep50, lmax=1e-4', 'color': '#8c564b', 'marker': 'X'},
]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ===== Left: Spike vs Accuracy trajectory =====
    ax = axes[0]

    # Baseline
    ep_b, acc_b, sc_b, _ = parse_log(BASELINE)
    ax.plot(sc_b / 1000, acc_b * 100, color='#333333', linewidth=1.0, alpha=0.4)
    ax.scatter(sc_b / 1000, acc_b * 100, color='#333333', s=10, alpha=0.5,
              edgecolors='none', label='Baseline (no reg)', zorder=2)
    # annotate baseline final
    ax.annotate(f'Baseline\n{acc_b[-1]*100:.1f}%\n{sc_b[-1]/1000:.0f}K',
                (sc_b[-1]/1000, acc_b[-1]*100), fontsize=8, color='#333333',
                fontweight='bold', xytext=(10, -10), textcoords='offset points')

    # Experiments
    for exp in EXPERIMENTS:
        try:
            ep, acc, sc, lam = parse_log(exp['log'])
        except FileNotFoundError:
            print(f"Skip: {exp['log']}")
            continue

        ls = exp.get('ls', '-')
        ax.plot(sc / 1000, acc * 100, color=exp['color'], linewidth=0.8, alpha=0.4, linestyle=ls)
        ax.scatter(sc / 1000, acc * 100, color=exp['color'], marker=exp['marker'],
                  s=15, alpha=0.6, edgecolors='none', label=exp['label'], zorder=3)

        # annotate final
        best_idx = np.argmax(acc)
        ax.annotate(f'{acc[best_idx]*100:.1f}%\n{sc[best_idx]/1000:.0f}K',
                    (sc[best_idx]/1000, acc[best_idx]*100), fontsize=7, color=exp['color'],
                    fontweight='bold', xytext=(8, -5), textcoords='offset points')

    ax.set_xlabel('Total Spike Count (×1000)', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('VGG16-C10 — Spike vs Accuracy Trajectory', fontsize=13)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    # Zoom into interesting region
    ax.set_xlim(0, 280)
    ax.set_ylim(10, 97)

    # ===== Right: Zoomed in on final results =====
    ax2 = axes[1]

    # Baseline final point
    ax2.axhline(y=acc_b.max()*100, color='#333333', linestyle='--', alpha=0.5, label=f'Baseline acc={acc_b.max()*100:.2f}%')
    ax2.axvline(x=sc_b[np.argmax(acc_b)]/1000, color='#333333', linestyle=':', alpha=0.3)

    # Plot best point of each experiment
    results = []
    for exp in EXPERIMENTS:
        try:
            ep, acc, sc, lam = parse_log(exp['log'])
        except FileNotFoundError:
            continue
        best_idx = np.argmax(acc)
        results.append({
            'label': exp['label'], 'color': exp['color'], 'marker': exp['marker'],
            'acc': acc[best_idx] * 100, 'spike': sc[best_idx] / 1000
        })

    # Baseline point
    best_b = np.argmax(acc_b)
    ax2.scatter(sc_b[best_b]/1000, acc_b[best_b]*100, color='#333333', s=150,
               marker='*', zorder=5, label=f'Baseline: {acc_b[best_b]*100:.2f}%, {sc_b[best_b]/1000:.0f}K')

    for r in results:
        ax2.scatter(r['spike'], r['acc'], color=r['color'], marker=r['marker'],
                   s=100, zorder=5, edgecolors='black', linewidths=0.5,
                   label=f'{r["label"]}: {r["acc"]:.2f}%, {r["spike"]:.0f}K')

    ax2.set_xlabel('Best Spike Count (×1000)', fontsize=12)
    ax2.set_ylabel('Best Validation Accuracy (%)', fontsize=12)
    ax2.set_title('VGG16-C10 — Best Accuracy vs Spike (Pareto)', fontsize=13)
    ax2.legend(fontsize=8, loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(20, 85)
    ax2.set_ylim(92.5, 95.5)

    plt.suptitle('VGG16-C10: start_ep=50 Adaptive Lambda vs Baseline',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=150, bbox_inches='tight')
    print(f'Saved: {OUT_FILE}')

    # Summary
    print('\n=== Summary ===')
    print(f'Baseline: acc={acc_b.max()*100:.2f}%, spike={sc_b[np.argmax(acc_b)]/1000:.0f}K')
    for r in results:
        print(f'{r["label"]}: acc={r["acc"]:.2f}%, spike={r["spike"]:.0f}K')


if __name__ == '__main__':
    main()
