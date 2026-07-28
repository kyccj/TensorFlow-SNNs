'''
    VGG16-C10 start_ep=50: Lambda trajectory per epoch for each lmax
'''
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_FILE = 'results_viz/vgg_ep50_lambda_trajectory.png'


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


EXPERIMENTS = [
    {'log': '_adaptive_lambda_v2/vgg_c10_ep50_lmax1e-6/train.log',
     'label': 'lmax=1e-6', 'color': '#1f77b4'},
    {'log': '_adaptive_lambda_v2/vgg_c10_ep50_lmax3e-6/train.log',
     'label': 'lmax=3e-6', 'color': '#2ca02c'},
    {'log': '_adaptive_lambda_v2/vgg_c10_ep50_lmax5e-6/train.log',
     'label': 'lmax=5e-6', 'color': '#ff7f0e'},
    {'log': '_adaptive_lambda_v2/vgg_c10_ep50_lmax1e-5/train.log',
     'label': 'lmax=1e-5', 'color': '#d62728'},
    {'log': '_adaptive_lambda_v2/vgg_c10_ep50_lmax1e-4/train.log',
     'label': 'lmax=1e-4', 'color': '#8c564b'},
]


def main():
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    for exp in EXPERIMENTS:
        try:
            ep, acc, sc, lam = parse_log(exp['log'])
        except FileNotFoundError:
            print(f"Skip: {exp['log']}")
            continue

        # Top: Lambda trajectory (log scale)
        lam_plot = np.where(lam > 0, lam, np.nan)
        axes[0].plot(ep, lam_plot, color=exp['color'], linewidth=1.5, label=exp['label'])

        # Middle: Spike count
        axes[1].plot(ep, sc / 1000, color=exp['color'], linewidth=1.2, label=exp['label'])

        # Bottom: Accuracy
        axes[2].plot(ep, acc * 100, color=exp['color'], linewidth=1.0, alpha=0.8, label=exp['label'])

    # Top: Lambda
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Lambda (λ)', fontsize=12)
    axes[0].set_title('VGG16-C10 start_ep=50 — Lambda / Spike / Accuracy Trajectory', fontsize=14)
    axes[0].legend(fontsize=9, loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=50, color='red', linestyle='--', alpha=0.5, label='start_ep=50')
    axes[0].text(51, 1e-8, 'reg start', fontsize=9, color='red', alpha=0.7)

    # Middle: Spike
    axes[1].set_ylabel('Spike Count (×1000)', fontsize=12)
    axes[1].legend(fontsize=9, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=50, color='red', linestyle='--', alpha=0.5)

    # Bottom: Accuracy
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[2].legend(fontsize=9, loc='lower right')
    axes[2].grid(True, alpha=0.3)
    axes[2].axvline(x=50, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=150, bbox_inches='tight')
    print(f'Saved: {OUT_FILE}')


if __name__ == '__main__':
    main()
