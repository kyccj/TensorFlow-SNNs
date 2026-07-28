'''
    Visualize per-layer spike regularization metrics from reg_detail.csv files
    and training accuracy from train.log files.

    Usage:
        python visualize_reg_detail.py
'''

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results_viz')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Parsing helpers
# ============================================================

def parse_train_log(log_path):
    """Extract per-epoch metrics from Keras training log."""
    epochs = []
    pattern = re.compile(
        r'(\d+)/\d+ .*?'
        r'val_acc: ([\d.]+).*?'
        r'best_val_acc: ([\d.]+).*?'
        r's_count: ([\-\d.]+).*?'
        r'best_s_count: ([\-\d.]+)'
    )
    if not os.path.exists(log_path):
        return None
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append({
                    'val_acc': float(m.group(2)),
                    'best_val_acc': float(m.group(3)),
                    's_count': float(m.group(4)),
                    'best_s_count': float(m.group(5)),
                })
    if not epochs:
        return None
    return pd.DataFrame(epochs)


def parse_reg_detail(csv_path):
    """Read reg_detail.csv."""
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def parse_reg_neuron_detail(csv_path):
    """Read reg_neuron_detail.csv."""
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def find_experiments(prefix, sweep_dir_name):
    """Find all experiment folders matching a pattern."""
    experiments = {}

    # Sweep experiments
    sweep_dir = os.path.join(PROJECT_ROOT, sweep_dir_name)
    if os.path.isdir(sweep_dir):
        for d in sorted(os.listdir(sweep_dir)):
            if d.startswith('lambda_'):
                lmb_str = d.replace('lambda_', '')
                lmb = float(lmb_str)
                run_dir = os.path.join(sweep_dir, d)
                log_path = os.path.join(run_dir, 'train.log')

                # Find reg_detail.csv in experiment output dir
                csv_dirs = glob.glob(os.path.join(PROJECT_ROOT, f'{prefix}{lmb_str}*'))
                csv_path = None
                for cd in csv_dirs:
                    cp = os.path.join(cd, 'reg_detail.csv')
                    if os.path.exists(cp):
                        csv_path = cp
                        break

                neuron_csv_path = None
                for cd in csv_dirs:
                    ncp = os.path.join(cd, 'reg_neuron_detail.csv')
                    if os.path.exists(ncp):
                        neuron_csv_path = ncp
                        break

                experiments[lmb] = {
                    'log_path': log_path,
                    'csv_path': csv_path,
                    'neuron_csv_path': neuron_csv_path,
                    'label': f'λ={lmb}',
                }

    return experiments


# ============================================================
# Plotting functions
# ============================================================

def plot_accuracy_spike_vs_lambda(experiments, title_suffix, filename):
    """Bar chart: accuracy and spike count vs lambda."""
    lambdas = []
    accs = []
    s_counts = []

    for lmb in sorted(experiments.keys()):
        exp = experiments[lmb]
        log_df = parse_train_log(exp['log_path'])
        if log_df is None:
            continue
        lambdas.append(lmb)
        accs.append(log_df['best_val_acc'].iloc[-1] * 100)
        s_counts.append(log_df['best_s_count'].iloc[-1])

    if not lambdas:
        return

    x = np.arange(len(lambdas))
    labels = [f'{l:.0e}' for l in lambdas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    bars1 = ax1.bar(x, accs, color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_xlabel('Lambda (reg_spike_out_const)')
    ax1.set_ylabel('Best Val Accuracy (%)')
    ax1.set_title(f'Best Validation Accuracy {title_suffix}')
    # Annotate
    for bar, val in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    if accs:
        ax1.set_ylim(min(min(accs) - 3, 0), max(accs) + 3)

    # Spike count
    colors2 = ['tomato' if s < 0 else 'coral' for s in s_counts]
    bars2 = ax2.bar(x, s_counts, color=colors2, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_xlabel('Lambda (reg_spike_out_const)')
    ax2.set_ylabel('Best Spike Count')
    ax2.set_title(f'Best Spike Count {title_suffix}')
    for bar, val in zip(bars2, s_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(abs(v) for v in s_counts)*0.01,
                 f'{val:.0f}', ha='center', va='bottom', fontsize=7, rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f'  Saved: {filename}')


def plot_firing_rate_heatmap(experiments, title_suffix, filename):
    """Heatmap: per-layer firing rate at last epoch."""
    lambdas_sorted = sorted([l for l in experiments.keys()])
    all_data = {}

    for lmb in lambdas_sorted:
        exp = experiments[lmb]
        if exp['csv_path'] is None:
            continue
        df = parse_reg_detail(exp['csv_path'])
        if df is None:
            continue
        last_epoch = df['epoch'].max()
        last = df[df['epoch'] == last_epoch].copy()
        # skip n_in layer
        last = last[last['layer'] != 'n_in']
        all_data[lmb] = last.set_index('layer')['firing_rate']

    if not all_data:
        return

    # Build matrix
    layers = list(list(all_data.values())[0].index)
    matrix = np.zeros((len(layers), len(all_data)))
    col_labels = []
    for j, lmb in enumerate(sorted(all_data.keys())):
        col_labels.append(f'{lmb:.0e}')
        for i, layer in enumerate(layers):
            matrix[i, j] = all_data[lmb].get(layer, 0)

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels)*1.2), max(6, len(layers)*0.4)))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Layer')
    ax.set_title(f'Per-Layer Firing Rate (Last Epoch) {title_suffix}')

    # Annotate cells
    for i in range(len(layers)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            color = 'white' if val > matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label='Firing Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f'  Saved: {filename}')


def plot_sc_loss_heatmap(experiments, title_suffix, filename):
    """Heatmap: per-layer sc_loss at last epoch."""
    lambdas_sorted = sorted([l for l in experiments.keys()])
    all_data = {}

    for lmb in lambdas_sorted:
        exp = experiments[lmb]
        if exp['csv_path'] is None:
            continue
        df = parse_reg_detail(exp['csv_path'])
        if df is None:
            continue
        last_epoch = df['epoch'].max()
        last = df[df['epoch'] == last_epoch].copy()
        last = last[last['layer'] != 'n_in']
        all_data[lmb] = last.set_index('layer')['sc_loss']

    if not all_data:
        return

    layers = list(list(all_data.values())[0].index)
    matrix = np.zeros((len(layers), len(all_data)))
    col_labels = []
    for j, lmb in enumerate(sorted(all_data.keys())):
        col_labels.append(f'{lmb:.0e}')
        for i, layer in enumerate(layers):
            matrix[i, j] = all_data[lmb].get(layer, 0)

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels)*1.2), max(6, len(layers)*0.4)))
    im = ax.imshow(matrix, aspect='auto', cmap='YlGnBu', interpolation='nearest')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Layer')
    ax.set_title(f'Per-Layer sc_loss (Last Epoch) {title_suffix}')

    for i in range(len(layers)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            color = 'white' if val > matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label='sc_loss')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f'  Saved: {filename}')


def plot_firing_rate_over_epochs(experiments, title_suffix, filename, target_lambdas=None):
    """Line plot: per-layer firing rate over epochs for select lambdas."""
    if target_lambdas is None:
        # Pick a few representative lambdas
        all_lambdas = sorted(experiments.keys())
        if len(all_lambdas) <= 4:
            target_lambdas = all_lambdas
        else:
            target_lambdas = [all_lambdas[0], all_lambdas[len(all_lambdas)//3],
                              all_lambdas[2*len(all_lambdas)//3], all_lambdas[-1]]

    # Select representative layers
    select_layers = ['n_conv1', 'n_conv3', 'n_conv5_1', 'n_fc1']

    fig, axes = plt.subplots(1, len(target_lambdas), figsize=(5*len(target_lambdas), 5), sharey=True)
    if len(target_lambdas) == 1:
        axes = [axes]

    for idx, lmb in enumerate(target_lambdas):
        ax = axes[idx]
        exp = experiments.get(lmb)
        if exp is None or exp['csv_path'] is None:
            ax.set_title(f'λ={lmb:.0e}\n(no data)')
            continue

        df = parse_reg_detail(exp['csv_path'])
        if df is None:
            continue

        for layer in select_layers:
            layer_df = df[df['layer'] == layer]
            if len(layer_df) > 0:
                ax.plot(layer_df['epoch'], layer_df['firing_rate'], label=layer, linewidth=1)

        ax.set_xlabel('Epoch')
        ax.set_title(f'λ={lmb:.0e}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Firing Rate')
    fig.suptitle(f'Firing Rate Over Epochs {title_suffix}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f'  Saved: {filename}')


def plot_accuracy_over_epochs(experiments, title_suffix, filename, target_lambdas=None):
    """Line plot: validation accuracy over epochs for all lambdas."""
    if target_lambdas is None:
        target_lambdas = sorted(experiments.keys())

    fig, ax = plt.subplots(figsize=(10, 5))

    for lmb in target_lambdas:
        exp = experiments.get(lmb)
        if exp is None:
            continue
        log_df = parse_train_log(exp['log_path'])
        if log_df is None:
            continue
        epochs = range(1, len(log_df) + 1)
        ax.plot(epochs, log_df['val_acc'] * 100, label=f'λ={lmb:.0e}', linewidth=1)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title(f'Validation Accuracy Over Epochs {title_suffix}')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f'  Saved: {filename}')


def plot_spike_count_over_epochs(experiments, title_suffix, filename, target_lambdas=None):
    """Line plot: spike count over epochs for all lambdas."""
    if target_lambdas is None:
        target_lambdas = sorted(experiments.keys())

    fig, ax = plt.subplots(figsize=(10, 5))

    for lmb in target_lambdas:
        exp = experiments.get(lmb)
        if exp is None:
            continue
        log_df = parse_train_log(exp['log_path'])
        if log_df is None:
            continue
        epochs = range(1, len(log_df) + 1)
        ax.plot(epochs, log_df['s_count'], label=f'λ={lmb:.0e}', linewidth=1)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Spike Count')
    ax.set_title(f'Spike Count Over Epochs {title_suffix}')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f'  Saved: {filename}')


def plot_summary_table(experiments, baseline_exp, title_suffix, filename):
    """Summary table as a figure."""
    rows = []

    # Baseline
    if baseline_exp is not None:
        log_df = parse_train_log(baseline_exp['log_path'])
        if log_df is not None:
            rows.append({
                'Lambda': 'No Reg',
                'Best Acc (%)': f"{log_df['best_val_acc'].iloc[-1]*100:.2f}",
                'Spike Count': f"{log_df['best_s_count'].iloc[-1]:.0f}",
            })

    for lmb in sorted(experiments.keys()):
        exp = experiments[lmb]
        log_df = parse_train_log(exp['log_path'])
        if log_df is None:
            continue
        rows.append({
            'Lambda': f'{lmb:.0e}',
            'Best Acc (%)': f"{log_df['best_val_acc'].iloc[-1]*100:.2f}",
            'Spike Count': f"{log_df['best_s_count'].iloc[-1]:.0f}",
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, max(2, len(rows)*0.4 + 1)))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Highlight best accuracy row
    accs = [float(r['Best Acc (%)']) for r in rows]
    best_idx = np.argmax(accs)
    for j in range(len(df.columns)):
        table[best_idx + 1, j].set_facecolor('#d4edda')

    ax.set_title(f'Summary {title_suffix}', fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f'  Saved: {filename}')


# ============================================================
# Neuron-level WTA metric plots
# ============================================================

def _build_neuron_heatmap(experiments, metric, title_suffix, filename, cmap='YlOrRd', fmt='.3f'):
    """Generic heatmap builder for reg_neuron_detail.csv metrics."""
    lambdas_sorted = sorted(experiments.keys())
    all_data = {}

    for lmb in lambdas_sorted:
        exp = experiments[lmb]
        ncp = exp.get('neuron_csv_path')
        if ncp is None:
            continue
        df = parse_reg_neuron_detail(ncp)
        if df is None:
            continue
        last_epoch = df['epoch'].max()
        last = df[df['epoch'] == last_epoch].copy()
        last = last[last['layer'] != 'n_in']
        all_data[lmb] = last.set_index('layer')[metric]

    if not all_data:
        return

    layers = list(list(all_data.values())[0].index)
    matrix = np.zeros((len(layers), len(all_data)))
    col_labels = []
    for j, lmb in enumerate(sorted(all_data.keys())):
        col_labels.append(f'{lmb:.0e}')
        for i, layer in enumerate(layers):
            matrix[i, j] = all_data[lmb].get(layer, 0)

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels)*1.2), max(6, len(layers)*0.4)))
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, interpolation='nearest')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Layer')
    ax.set_title(f'{metric} (Last Epoch) {title_suffix}')

    for i in range(len(layers)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            color = 'white' if val > matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:{fmt}}', ha='center', va='center', fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f'  Saved: {filename}')


def plot_gini_heatmap(experiments, title_suffix, filename):
    """Heatmap: Gini coefficient per layer at last epoch."""
    _build_neuron_heatmap(experiments, 'gini', title_suffix, filename, cmap='YlOrRd')


def plot_top10_share_heatmap(experiments, title_suffix, filename):
    """Heatmap: top-10% spike share per layer at last epoch."""
    _build_neuron_heatmap(experiments, 'top10_share', title_suffix, filename, cmap='OrRd')


def plot_dead_neuron_heatmap(experiments, title_suffix, filename):
    """Heatmap: dead neuron ratio per layer at last epoch."""
    _build_neuron_heatmap(experiments, 'dead_neuron_ratio', title_suffix, filename, cmap='PuBu')


def plot_neuron_metrics_over_epochs(experiments, title_suffix, filename, target_lambdas=None):
    """Line plot: gini, dead_neuron_ratio, top10_share over epochs for select lambdas."""
    if target_lambdas is None:
        all_lambdas = sorted(experiments.keys())
        if len(all_lambdas) <= 4:
            target_lambdas = all_lambdas
        else:
            target_lambdas = [all_lambdas[0], all_lambdas[len(all_lambdas)//3],
                              all_lambdas[2*len(all_lambdas)//3], all_lambdas[-1]]

    metrics = ['gini', 'dead_neuron_ratio', 'top10_share']
    select_layers = ['n_conv1', 'n_conv3', 'n_conv5_1', 'n_fc1']

    fig, axes = plt.subplots(len(metrics), len(target_lambdas),
                             figsize=(5*len(target_lambdas), 4*len(metrics)), squeeze=False)

    for col_idx, lmb in enumerate(target_lambdas):
        exp = experiments.get(lmb)
        ncp = exp.get('neuron_csv_path') if exp else None
        df = parse_reg_neuron_detail(ncp) if ncp else None

        for row_idx, metric in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            if df is None:
                ax.set_title(f'λ={lmb:.0e}\n(no data)')
                continue

            for layer in select_layers:
                layer_df = df[df['layer'] == layer]
                if len(layer_df) > 0:
                    ax.plot(layer_df['epoch'], layer_df[metric], label=layer, linewidth=1)

            ax.set_xlabel('Epoch')
            if col_idx == 0:
                ax.set_ylabel(metric)
            ax.set_title(f'λ={lmb:.0e}')
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f'Neuron-Level WTA Metrics Over Epochs {title_suffix}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f'  Saved: {filename}')


# ============================================================
# Main
# ============================================================

def visualize_sweep(sweep_dir_name, csv_prefix, title_suffix, file_prefix,
                    baseline_dir=None, baseline_csv_prefix=None):
    """Generate all plots for one sweep."""
    print(f'\n=== {title_suffix} ===')

    experiments = find_experiments(csv_prefix, sweep_dir_name)
    if not experiments:
        print(f'  No experiments found for {sweep_dir_name}')
        return

    print(f'  Found {len(experiments)} experiments: {sorted(experiments.keys())}')

    # Baseline
    baseline_exp = None
    if baseline_dir:
        bl_log = os.path.join(PROJECT_ROOT, baseline_dir, 'train.log')
        bl_csv = None
        if baseline_csv_prefix:
            csv_dirs = glob.glob(os.path.join(PROJECT_ROOT, f'{baseline_csv_prefix}*'))
            for cd in csv_dirs:
                cp = os.path.join(cd, 'reg_detail.csv')
                if os.path.exists(cp):
                    bl_csv = cp
                    break
        if os.path.exists(bl_log):
            bl_neuron_csv = None
            if baseline_csv_prefix:
                for cd in csv_dirs:
                    ncp = os.path.join(cd, 'reg_neuron_detail.csv')
                    if os.path.exists(ncp):
                        bl_neuron_csv = ncp
                        break
            baseline_exp = {'log_path': bl_log, 'csv_path': bl_csv, 'neuron_csv_path': bl_neuron_csv, 'label': 'No Reg'}

    # Generate plots
    plot_summary_table(experiments, baseline_exp, title_suffix, f'{file_prefix}_summary.png')
    plot_accuracy_spike_vs_lambda(experiments, title_suffix, f'{file_prefix}_acc_spike_bar.png')
    plot_accuracy_over_epochs(experiments, title_suffix, f'{file_prefix}_acc_epochs.png')
    plot_spike_count_over_epochs(experiments, title_suffix, f'{file_prefix}_spike_epochs.png')
    plot_firing_rate_heatmap(experiments, title_suffix, f'{file_prefix}_firing_heatmap.png')
    plot_sc_loss_heatmap(experiments, title_suffix, f'{file_prefix}_scloss_heatmap.png')
    plot_firing_rate_over_epochs(experiments, title_suffix, f'{file_prefix}_firing_epochs.png')

    # Neuron-level WTA metric plots
    plot_gini_heatmap(experiments, title_suffix, f'{file_prefix}_gini_heatmap.png')
    plot_top10_share_heatmap(experiments, title_suffix, f'{file_prefix}_top10_heatmap.png')
    plot_dead_neuron_heatmap(experiments, title_suffix, f'{file_prefix}_dead_neuron_heatmap.png')
    plot_neuron_metrics_over_epochs(experiments, title_suffix, f'{file_prefix}_neuron_metrics_epochs.png')


def main():
    print(f'Output directory: {OUTPUT_DIR}')

    # VGG16 CIFAR10 WTA-rev sweep
    visualize_sweep(
        sweep_dir_name='_sweep_wta_rev',
        csv_prefix='EIP-SNN-26_wta-rev-lambda-',
        title_suffix='[VGG16 CIFAR10 WTA-Rev]',
        file_prefix='c10_wta_rev',
        baseline_dir='_baseline_no_reg',
        baseline_csv_prefix='EIP-SNN-26_baseline-no-reg',
    )

    # VGG16 CIFAR100 WTA-rev sweep
    visualize_sweep(
        sweep_dir_name='_sweep_wta_rev_c100',
        csv_prefix='EIP-SNN-26_wta-rev-c100-lambda-',
        title_suffix='[VGG16 CIFAR100 WTA-Rev]',
        file_prefix='c100_wta_rev',
    )

    # ResNet19 CIFAR10 WTA-rev sweep
    visualize_sweep(
        sweep_dir_name='_sweep_wta_rev_r19_c10',
        csv_prefix='EIP-SNN-26_wta-rev-r19-c10-lambda-',
        title_suffix='[ResNet19 CIFAR10 WTA-Rev]',
        file_prefix='r19_c10_wta_rev',
    )

    # ResNet19 CIFAR100 WTA-rev sweep
    visualize_sweep(
        sweep_dir_name='_sweep_wta_rev_r19_c100',
        csv_prefix='EIP-SNN-26_wta-rev-r19-c100-lambda-',
        title_suffix='[ResNet19 CIFAR100 WTA-Rev]',
        file_prefix='r19_c100_wta_rev',
    )

    print(f'\nAll plots saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
