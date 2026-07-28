'''
    Baseline only: spike count (x) vs accuracy (y) trajectory per epoch
    VGG16-C10 and R19-C10 separate subplots
'''
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_FILE = 'results_viz/baseline_spike_vs_acc.png'

BASELINES = {
    'R19-C10': '_baseline_trajectory/train.log',
    'VGG16-C10': '_baseline_vgg_c10_trajectory/train.log',
}


def parse_log(path):
    epochs, val_accs, s_counts = [], [], []
    epoch_num = 0
    with open(path) as f:
        for line in f:
            if 's_count' in line and 'val_acc' in line and '500/500' in line:
                epoch_num += 1
                m_acc = re.search(r'val_acc:\s*([\d.]+)', line)
                m_sc = re.search(r's_count:\s*([\d.]+)', line)
                if m_acc and m_sc:
                    epochs.append(epoch_num)
                    val_accs.append(float(m_acc.group(1)))
                    s_counts.append(float(m_sc.group(1)))
    return np.array(epochs), np.array(val_accs), np.array(s_counts)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors_phase = {
        'early': '#d62728',    # red (ep 1-50)
        'mid': '#ff7f0e',      # orange (ep 51-150)
        'late': '#2ca02c',     # green (ep 151-250)
        'final': '#1f77b4',    # blue (ep 251-310)
    }

    for ax, (model, logpath) in zip(axes, BASELINES.items()):
        epochs, val_accs, s_counts = parse_log(logpath)
        accs = val_accs * 100
        spikes = s_counts / 1000

        # Phase boundaries
        phases = [
            (0, 50, 'ep 1-50', colors_phase['early']),
            (50, 150, 'ep 51-150', colors_phase['mid']),
            (150, 250, 'ep 151-250', colors_phase['late']),
            (250, 310, 'ep 251-310', colors_phase['final']),
        ]

        # Plot each phase with different color
        for start, end, label, color in phases:
            mask = (epochs > start) & (epochs <= end)
            if mask.sum() == 0:
                continue
            ax.plot(spikes[mask], accs[mask], color=color, linewidth=1.0, alpha=0.5)
            ax.scatter(spikes[mask], accs[mask], color=color, s=15, alpha=0.7,
                       edgecolors='none', label=label, zorder=3)

        # Annotate key points
        # Epoch 1
        ax.annotate(f'ep1\n{accs[0]:.1f}%\n{spikes[0]:.0f}K',
                     (spikes[0], accs[0]), fontsize=8, color=colors_phase['early'],
                     fontweight='bold', xytext=(10, -15), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color=colors_phase['early'], lw=0.8))

        # Epoch 50
        idx50 = 49 if len(epochs) > 49 else len(epochs) - 1
        ax.annotate(f'ep50\n{accs[idx50]:.1f}%\n{spikes[idx50]:.0f}K',
                     (spikes[idx50], accs[idx50]), fontsize=8, color=colors_phase['early'],
                     fontweight='bold', xytext=(10, -15), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color=colors_phase['early'], lw=0.8))

        # Final epoch
        ax.annotate(f'ep{epochs[-1]}\n{accs[-1]:.1f}%\n{spikes[-1]:.0f}K',
                     (spikes[-1], accs[-1]), fontsize=8, color=colors_phase['final'],
                     fontweight='bold', xytext=(-60, -20), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color=colors_phase['final'], lw=0.8))

        # Best accuracy point
        best_idx = np.argmax(accs)
        ax.annotate(f'best ep{epochs[best_idx]}\n{accs[best_idx]:.1f}%\n{spikes[best_idx]:.0f}K',
                     (spikes[best_idx], accs[best_idx]), fontsize=8, color='black',
                     fontweight='bold', xytext=(10, 10), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

        ax.set_xlabel('Total Spike Count (×1000)', fontsize=12)
        ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
        ax.set_title(f'{model} Baseline — Spike vs Accuracy', fontsize=13)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Baseline (No Regularization) — Natural Spike Reduction During Training',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=150, bbox_inches='tight')
    print(f'Saved: {OUT_FILE}')

    # Print summary
    for model, logpath in BASELINES.items():
        epochs, val_accs, s_counts = parse_log(logpath)
        print(f'\n{model}:')
        print(f'  ep1:   acc={val_accs[0]*100:.1f}%, spike={s_counts[0]/1000:.0f}K')
        print(f'  ep50:  acc={val_accs[49]*100:.1f}%, spike={s_counts[49]/1000:.0f}K')
        print(f'  ep310: acc={val_accs[-1]*100:.1f}%, spike={s_counts[-1]/1000:.0f}K')
        print(f'  best:  acc={val_accs.max()*100:.2f}% (ep{np.argmax(val_accs)+1}), spike={s_counts[np.argmax(val_accs)]/1000:.0f}K')
        print(f'  spike reduction: {s_counts[0]/1000:.0f}K → {s_counts[-1]/1000:.0f}K ({(1-s_counts[-1]/s_counts[0])*100:.0f}%)')


if __name__ == '__main__':
    main()
