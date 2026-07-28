'''
    Visualize spike count histograms: Baseline vs Softmax vs MaxNorm vs +Enc
    Runs analyze_spike_histogram.py for each experiment and plots results.
'''
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'

# Experiments to compare
EXPERIMENTS = {
    'Baseline (no reg)': {
        'config_dir': '_baseline_no_reg_run2',
    },
    'Softmax λ=1e-8': {
        'config_dir': '_compare_maxnorm/softmax_lmb_1e-08',
    },
    'Softmax λ=1e-7': {
        'config_dir': '_compare_maxnorm/softmax_lmb_1e-07',
    },
    'Softmax λ=3e-7': {
        'config_dir': '_compare_maxnorm/softmax_lmb_3e-07',
    },
    'Softmax λ=1e-6': {
        'config_dir': '_compare_maxnorm/softmax_lmb_1e-06',
    },
    'MaxNorm λ=1e-8': {
        'config_dir': '_compare_maxnorm/maxnorm_lmb_1e-08',
    },
    'MaxNorm λ=1e-7': {
        'config_dir': '_compare_maxnorm/maxnorm_lmb_1e-07',
    },
    'MaxNorm λ=3e-7': {
        'config_dir': '_compare_maxnorm/maxnorm_lmb_3e-07',
    },
    'MaxNorm λ=1e-6': {
        'config_dir': '_compare_maxnorm/maxnorm_lmb_1e-06',
    },
    'MaxNorm+Enc λ=1e-8': {
        'config_dir': '_encourage/maxnorm_enc_lmb_1e-08',
    },
    'MaxNorm+Enc λ=1e-7': {
        'config_dir': '_encourage/maxnorm_enc_lmb_1e-07',
    },
    'MaxNorm+Enc λ=3e-7': {
        'config_dir': '_encourage/maxnorm_enc_lmb_3e-07',
    },
    'MaxNorm+Enc λ=1e-6': {
        'config_dir': '_encourage/maxnorm_enc_lmb_1e-06',
    },
}

def find_checkpoint(config_dir):
    import glob
    pattern = os.path.join(PROJECT_ROOT, config_dir, 'VGG16_AP_CIFAR10', 'ep-*', '*.weights.h5')
    files = glob.glob(pattern)
    if not pattern or not files:
        pattern = os.path.join(PROJECT_ROOT, config_dir, 'VGG16_AP_CIFAR10', 'ep-*', '*.weights.h5')
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
    return result.stdout


def parse_histogram(output):
    """Parse histogram output into dict: {layer_name: {sc0:%, sc1:%, ..., mean:float}}"""
    data = {}
    for line in output.strip().split('\n'):
        if line.startswith('n_'):
            parts = line.split()
            name = parts[0]
            neurons = int(parts[1])
            fracs = []
            for p in parts[2:7]:  # sc=0 through sc=4
                fracs.append(float(p.strip('%')) / 100.0)
            mean = float(parts[7])
            data[name] = {'neurons': neurons, 'fracs': fracs, 'mean': mean}
    return data


def collect_all_data():
    all_data = {}
    for name, exp in EXPERIMENTS.items():
        ckpt = find_checkpoint(exp['config_dir'])
        if ckpt is None:
            print(f'  [{name}] No checkpoint found, skipping')
            continue
        print(f'  [{name}] Running histogram analysis...')
        output = run_histogram(exp['config_dir'], ckpt)
        parsed = parse_histogram(output)
        if parsed:
            all_data[name] = parsed
            print(f'    -> {len(parsed)} layers parsed')
        else:
            print(f'    -> FAILED to parse output')
    return all_data


def plot_comparison_by_lambda(all_data, save_dir):
    """For each lambda, plot Baseline vs Softmax vs MaxNorm vs MaxNorm+Enc"""
    lambdas = ['1e-8', '1e-7', '3e-7', '1e-6']
    key_layers = ['n_conv1', 'n_conv2', 'n_conv3_2', 'n_conv5_1', 'n_fc1']
    sc_labels = ['sc=0', 'sc=1', 'sc=2', 'sc=3', 'sc=4']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

    for lmb in lambdas:
        methods = [
            ('Baseline (no reg)', 'Baseline'),
            (f'Softmax λ={lmb}', 'Softmax'),
            (f'MaxNorm λ={lmb}', 'MaxNorm'),
            (f'MaxNorm+Enc λ={lmb}', 'MaxNorm+Enc'),
        ]
        available = [(k, label) for k, label in methods if k in all_data]
        if len(available) < 2:
            continue

        fig, axes = plt.subplots(1, len(key_layers), figsize=(4*len(key_layers), 5))
        fig.suptitle(f'Spike Count Distribution (λ={lmb})', fontsize=14, fontweight='bold')

        for col, layer in enumerate(key_layers):
            ax = axes[col]
            x = np.arange(5)  # sc=0..4
            width = 0.8 / len(available)

            for i, (key, label) in enumerate(available):
                if layer in all_data[key]:
                    fracs = all_data[key][layer]['fracs']
                    offset = (i - len(available)/2 + 0.5) * width
                    bars = ax.bar(x + offset, [f*100 for f in fracs], width,
                                  label=label, color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)

            ax.set_xlabel('Spike Count')
            ax.set_xticks(x)
            ax.set_xticklabels(sc_labels, fontsize=9)
            ax.set_title(f'{layer}\n({all_data[available[0][0]][layer]["neurons"]} neurons)', fontsize=10)
            if col == 0:
                ax.set_ylabel('Neurons (%)')
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3)

        axes[-1].legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        path = os.path.join(save_dir, f'spike_hist_lambda_{lmb.replace("-","")}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}')


def plot_lambda_sweep(all_data, save_dir):
    """Show how sc=0 and sc=4 ratios change across lambdas for each method"""
    lambdas_str = ['1e-8', '1e-7', '3e-7', '1e-6']
    lambdas_val = [1e-8, 1e-7, 3e-7, 1e-6]
    key_layers = ['n_conv1', 'n_conv5_1', 'n_fc1']

    method_templates = [
        ('Softmax', 'Softmax λ={}', '#2196F3', '-o'),
        ('MaxNorm', 'MaxNorm λ={}', '#FF9800', '-s'),
        ('MaxNorm+Enc', 'MaxNorm+Enc λ={}', '#F44336', '-^'),
    ]

    fig, axes = plt.subplots(2, len(key_layers), figsize=(5*len(key_layers), 8))
    fig.suptitle('sc=0 and sc=4 Ratios across Lambda', fontsize=14, fontweight='bold')

    for col, layer in enumerate(key_layers):
        for row, (sc_idx, sc_name) in enumerate([(0, 'sc=0 (dead)'), (4, 'sc=4 (max fire)')]):
            ax = axes[row][col]

            # Baseline horizontal line
            if 'Baseline (no reg)' in all_data and layer in all_data['Baseline (no reg)']:
                bl_val = all_data['Baseline (no reg)'][layer]['fracs'][sc_idx] * 100
                ax.axhline(y=bl_val, color='gray', linestyle='--', alpha=0.7, label='Baseline')

            for method_name, template, color, marker in method_templates:
                vals = []
                for lmb in lambdas_str:
                    key = template.format(lmb)
                    if key in all_data and layer in all_data[key]:
                        vals.append(all_data[key][layer]['fracs'][sc_idx] * 100)
                    else:
                        vals.append(np.nan)
                ax.plot(range(len(lambdas_str)), vals, marker, color=color,
                        label=method_name, linewidth=2, markersize=6)

            ax.set_xticks(range(len(lambdas_str)))
            ax.set_xticklabels(lambdas_str, fontsize=9)
            ax.set_xlabel('Lambda')
            ax.set_title(f'{layer} — {sc_name}', fontsize=10)
            if col == 0:
                ax.set_ylabel('%')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc='best')

    plt.tight_layout()
    path = os.path.join(save_dir, 'spike_hist_lambda_sweep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_mean_spike(all_data, save_dir):
    """Mean spike count per layer for each method"""
    lambdas_str = ['1e-8', '1e-7', '3e-7', '1e-6']
    all_layers = ['n_conv1', 'n_conv1_1', 'n_conv2', 'n_conv2_1',
                  'n_conv3', 'n_conv3_1', 'n_conv3_2',
                  'n_conv4', 'n_conv4_1', 'n_conv4_2',
                  'n_conv5', 'n_conv5_1', 'n_conv5_2',
                  'n_fc1', 'n_fc2']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Mean Spike Count per Layer', fontsize=14, fontweight='bold')

    method_specs = [
        ('Baseline (no reg)', 'Baseline', 'gray', '--'),
        ('Softmax λ={}', 'Softmax', '#2196F3', '-'),
        ('MaxNorm λ={}', 'MaxNorm', '#FF9800', '-'),
        ('MaxNorm+Enc λ={}', 'MaxNorm+Enc', '#F44336', '-'),
    ]

    for idx, lmb in enumerate(lambdas_str):
        ax = axes[idx // 2][idx % 2]
        x = range(len(all_layers))

        for template, label, color, ls in method_specs:
            if '{}' in template:
                key = template.format(lmb)
            else:
                key = template
            if key in all_data:
                means = [all_data[key].get(l, {}).get('mean', np.nan) for l in all_layers]
                ax.plot(x, means, ls, color=color, label=label, linewidth=2, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([l.replace('n_', '') for l in all_layers], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean Spike Count')
        ax.set_title(f'λ={lmb}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 2.0)

    plt.tight_layout()
    path = os.path.join(save_dir, 'spike_mean_per_layer.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


if __name__ == '__main__':
    save_dir = os.path.join(PROJECT_ROOT, 'results_viz')
    os.makedirs(save_dir, exist_ok=True)

    print('Collecting histogram data...')
    all_data = collect_all_data()

    print(f'\nGenerating plots ({len(all_data)} experiments)...')
    plot_comparison_by_lambda(all_data, save_dir)
    plot_lambda_sweep(all_data, save_dir)
    plot_mean_spike(all_data, save_dir)
    print('\nDone!')
