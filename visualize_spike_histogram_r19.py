'''
    Visualize spike count histograms for ResNet19-CIFAR10:
    Baseline vs Softmax WTA vs WTA-Rev
'''
import subprocess
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'

# Experiments: config_dir (with config_sweep.py) and ckpt_dir (with checkpoint)
EXPERIMENTS = {
    'Baseline (no reg)': {
        'config_dir': '_baseline_no_reg_r19_c10',
        'ckpt_dir': 'EIP-SNN-26_baseline-no-reg-r19-c10',
    },
    'Softmax λ=1e-9': {
        'config_dir': '_r19_c10_lambda_sweep/lmb_1e-09',
        'ckpt_dir': 'r19-c10-lmb-1e-09',
    },
    'Softmax λ=1e-8': {
        'config_dir': '_r19_c10_lambda_sweep/lmb_1e-08',
        'ckpt_dir': 'r19-c10-lmb-1e-08',
    },
    'Softmax λ=1e-7': {
        'config_dir': '_r19_c10_lambda_sweep/lmb_1e-07',
        'ckpt_dir': 'r19-c10-lmb-1e-07',
    },
    'Softmax λ=3e-7': {
        'config_dir': '_r19_c10_lambda_sweep/lmb_3e-07',
        'ckpt_dir': 'r19-c10-lmb-3e-07',
    },
    'Softmax λ=5e-7': {
        'config_dir': '_r19_c10_lambda_sweep/lmb_5e-07',
        'ckpt_dir': 'r19-c10-lmb-5e-07',
    },
    'Softmax λ=1e-6': {
        'config_dir': '_r19_c10_lambda_sweep/lmb_1e-06',
        'ckpt_dir': 'r19-c10-lmb-1e-06',
    },
    'WTA-Rev λ=1e-9': {
        'config_dir': '_sweep_wta_rev_r19_c10/lambda_1e-09',
        'ckpt_dir': 'EIP-SNN-26_wta-rev-r19-c10-lambda-1e-09',
    },
    'WTA-Rev λ=5e-8': {
        'config_dir': '_sweep_wta_rev_r19_c10/lambda_5e-08',
        'ckpt_dir': 'EIP-SNN-26_wta-rev-r19-c10-lambda-5e-08',
    },
    'WTA-Rev λ=1e-7': {
        'config_dir': '_sweep_wta_rev_r19_c10/lambda_1e-07',
        'ckpt_dir': 'EIP-SNN-26_wta-rev-r19-c10-lambda-1e-07',
    },
    'WTA-Rev λ=5e-7': {
        'config_dir': '_sweep_wta_rev_r19_c10/lambda_5e-07',
        'ckpt_dir': 'EIP-SNN-26_wta-rev-r19-c10-lambda-5e-07',
    },
    'WTA-Rev λ=1e-6': {
        'config_dir': '_sweep_wta_rev_r19_c10/lambda_1e-06',
        'ckpt_dir': 'EIP-SNN-26_wta-rev-r19-c10-lambda-1e-06',
    },
}


def find_checkpoint(ckpt_dir):
    pattern = os.path.join(PROJECT_ROOT, ckpt_dir, 'ResNet19_CIFAR10', 'ep-*', '*.weights.h5')
    files = glob.glob(pattern)
    return files[0] if files else None


def run_histogram(config_dir, ckpt_path):
    env = os.environ.copy()
    env['PYTHONPATH'] = PROJECT_ROOT
    env['LD_LIBRARY_PATH'] = '/home/kyccj/miniconda3/envs/venv_1/lib:' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/miniconda3/envs/venv_1'

    result = subprocess.run(
        [PYTHON, 'analyze_spike_histogram.py', config_dir, ckpt_path],
        capture_output=True, text=True, cwd=PROJECT_ROOT, env=env, timeout=600
    )
    if result.returncode != 0:
        print(f'    STDERR: {result.stderr[-500:]}')
    return result.stdout


def parse_histogram(output):
    data = {}
    in_data = False
    for line in output.strip().split('\n'):
        if line.startswith('---'):
            in_data = True
            continue
        if not in_data:
            continue
        parts = line.split()
        if len(parts) >= 7:
            try:
                name = parts[0]
                neurons = int(parts[1])
                fracs = []
                for p in parts[2:7]:
                    fracs.append(float(p.strip('%')) / 100.0)
                mean = float(parts[7])
                data[name] = {'neurons': neurons, 'fracs': fracs, 'mean': mean}
            except (ValueError, IndexError):
                continue
    return data


def collect_all_data():
    all_data = {}
    for name, exp in EXPERIMENTS.items():
        ckpt = find_checkpoint(exp['ckpt_dir'])
        if ckpt is None:
            print(f'  [{name}] No checkpoint found in {exp["ckpt_dir"]}, skipping')
            continue
        print(f'  [{name}] Running histogram analysis...')
        output = run_histogram(exp['config_dir'], ckpt)
        parsed = parse_histogram(output)
        if parsed:
            all_data[name] = parsed
            print(f'    -> {len(parsed)} layers parsed')
        else:
            print(f'    -> FAILED to parse output')
            print(f'    stdout: {output[:300]}')
    return all_data


def plot_comparison_by_lambda(all_data, save_dir):
    """For each lambda, plot Baseline vs Softmax vs WTA-Rev"""
    lambdas = ['1e-7', '5e-7', '1e-6']
    # ResNet19 layer names (residual blocks)
    key_layers = sorted(all_data[list(all_data.keys())[0]].keys())
    # Pick representative layers
    pick_layers = []
    for l in key_layers:
        if any(k in l for k in ['conv1', 'res2a', 'res3a', 'res4a', 'fc']):
            pick_layers.append(l)
    if not pick_layers:
        pick_layers = key_layers[:5]  # fallback: first 5

    sc_labels = ['sc=0', 'sc=1', 'sc=2', 'sc=3', 'sc=4']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

    for lmb in lambdas:
        methods = [
            ('Baseline (no reg)', 'Baseline'),
            (f'Softmax λ={lmb}', 'Softmax'),
            (f'WTA-Rev λ={lmb}', 'WTA-Rev'),
        ]
        available = [(k, label) for k, label in methods if k in all_data]
        if len(available) < 2:
            continue

        fig, axes = plt.subplots(1, len(pick_layers), figsize=(4*len(pick_layers), 5))
        if len(pick_layers) == 1:
            axes = [axes]
        fig.suptitle(f'R19-C10 Spike Count Distribution (λ={lmb})', fontsize=14, fontweight='bold')

        for col, layer in enumerate(pick_layers):
            ax = axes[col]
            x = np.arange(5)
            width = 0.8 / len(available)

            for i, (key, label) in enumerate(available):
                if layer in all_data[key]:
                    fracs = all_data[key][layer]['fracs']
                    offset = (i - len(available)/2 + 0.5) * width
                    ax.bar(x + offset, [f*100 for f in fracs], width,
                           label=label, color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)

            ax.set_xlabel('Spike Count')
            ax.set_xticks(x)
            ax.set_xticklabels(sc_labels, fontsize=9)
            neurons = all_data[available[0][0]].get(layer, {}).get('neurons', '?')
            ax.set_title(f'{layer}\n({neurons} neurons)', fontsize=10)
            if col == 0:
                ax.set_ylabel('Neurons (%)')
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3)

        axes[-1].legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        path = os.path.join(save_dir, f'r19_spike_hist_lambda_{lmb.replace("-","")}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


def plot_lambda_sweep(all_data, save_dir):
    """Show how sc=0 and sc=4 ratios change across lambdas"""
    # Get all layers from first experiment
    first_key = list(all_data.keys())[0]
    all_layers = sorted(all_data[first_key].keys())

    # Pick 3 representative layers
    pick = []
    for l in all_layers:
        if 'conv1' in l and len(pick) < 1:
            pick.append(l)
    for l in all_layers:
        if ('res3' in l or 'res4' in l or 'conv4' in l) and len(pick) < 2:
            pick.append(l)
    for l in all_layers:
        if 'fc' in l and len(pick) < 3:
            pick.append(l)
    if len(pick) < 3:
        pick = all_layers[:3]

    softmax_lambdas_str = ['1e-9', '1e-8', '1e-7', '3e-7', '5e-7', '1e-6']
    wta_lambdas_str = ['1e-9', '5e-8', '1e-7', '5e-7', '1e-6']

    fig, axes = plt.subplots(2, len(pick), figsize=(5*len(pick), 8), squeeze=False)
    fig.suptitle('R19-C10: sc=0 and sc=4 Ratios across Lambda', fontsize=14, fontweight='bold')

    for col, layer in enumerate(pick):
        for row, (sc_idx, sc_name) in enumerate([(0, 'sc=0 (dead)'), (4, 'sc=4 (max fire)')]):
            ax = axes[row, col]

            # Baseline
            if 'Baseline (no reg)' in all_data and layer in all_data['Baseline (no reg)']:
                bl_val = all_data['Baseline (no reg)'][layer]['fracs'][sc_idx] * 100
                ax.axhline(y=bl_val, color='gray', linestyle='--', alpha=0.7, label='Baseline')

            # Softmax
            vals = []
            for lmb in softmax_lambdas_str:
                key = f'Softmax λ={lmb}'
                if key in all_data and layer in all_data[key]:
                    vals.append(all_data[key][layer]['fracs'][sc_idx] * 100)
                else:
                    vals.append(np.nan)
            ax.plot(range(len(softmax_lambdas_str)), vals, '-o', color='#2196F3',
                    label='Softmax', linewidth=2, markersize=6)

            # WTA-Rev
            vals = []
            for lmb in wta_lambdas_str:
                key = f'WTA-Rev λ={lmb}'
                if key in all_data and layer in all_data[key]:
                    vals.append(all_data[key][layer]['fracs'][sc_idx] * 100)
                else:
                    vals.append(np.nan)
            ax2_x = np.linspace(0, len(softmax_lambdas_str)-1, len(wta_lambdas_str))
            ax.plot(ax2_x, vals, '-^', color='#F44336',
                    label='WTA-Rev', linewidth=2, markersize=6)

            ax.set_xticks(range(len(softmax_lambdas_str)))
            ax.set_xticklabels(softmax_lambdas_str, fontsize=8)
            ax.set_xlabel('Lambda')
            ax.set_title(f'{layer} — {sc_name}', fontsize=10)
            if col == 0:
                ax.set_ylabel('%')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc='best')

    plt.tight_layout()
    path = os.path.join(save_dir, 'r19_spike_hist_lambda_sweep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_mean_spike(all_data, save_dir):
    """Mean spike count per layer"""
    first_key = list(all_data.keys())[0]
    all_layers = sorted(all_data[first_key].keys())

    lambdas_str = ['1e-7', '5e-7', '1e-6']
    method_specs = [
        ('Baseline (no reg)', 'Baseline', 'gray', '--'),
        ('Softmax λ={}', 'Softmax', '#2196F3', '-'),
        ('WTA-Rev λ={}', 'WTA-Rev', '#F44336', '-'),
    ]

    n_lmb = len(lambdas_str)
    fig, axes = plt.subplots(1, n_lmb, figsize=(6*n_lmb, 5))
    fig.suptitle('R19-C10: Mean Spike Count per Layer', fontsize=14, fontweight='bold')

    for idx, lmb in enumerate(lambdas_str):
        ax = axes[idx]
        x = range(len(all_layers))

        for template, label, color, ls in method_specs:
            if '{}' in template:
                key = template.format(lmb)
            else:
                key = template
            if key in all_data:
                means = [all_data[key].get(l, {}).get('mean', np.nan) for l in all_layers]
                ax.plot(x, means, ls, color=color, label=label, linewidth=2, alpha=0.8)

        short_names = [l.replace('n_', '') for l in all_layers]
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Mean Spike Count')
        ax.set_title(f'λ={lmb}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 2.0)

    plt.tight_layout()
    path = os.path.join(save_dir, 'r19_spike_mean_per_layer.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


if __name__ == '__main__':
    save_dir = os.path.join(PROJECT_ROOT, 'results_viz')
    os.makedirs(save_dir, exist_ok=True)

    print('Collecting R19-C10 histogram data...')
    all_data = collect_all_data()

    print(f'\nGenerating plots ({len(all_data)} experiments)...')
    if all_data:
        plot_comparison_by_lambda(all_data, save_dir)
        plot_lambda_sweep(all_data, save_dir)
        plot_mean_spike(all_data, save_dir)
    print('\nDone!')
