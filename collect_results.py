#!/usr/bin/env python3
"""
Collect comprehensive experiment results from all SNN training experiments.
Parses train.log files and config_sweep.py to extract metrics and metadata.
"""

import os
import re
import glob

BASE = "/home/kyccj/PycharmProjects/TensorFlow-SNNs"


def parse_last_epoch_line(log_path):
    """Parse the last epoch-end line from train.log.
    Returns dict with metrics or None if not found.
    """
    if not os.path.isfile(log_path):
        return None

    last_line = None
    epoch_num = None
    total_epochs = None

    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Track "Epoch N/M" lines
                m_ep = re.match(r'Epoch\s+(\d+)/(\d+)', line)
                if m_ep:
                    epoch_num = int(m_ep.group(1))
                    total_epochs = int(m_ep.group(2))
                # Look for completed epoch lines (500/500 with val_acc and s_count)
                if '500/500' in line and 'val_acc' in line and 's_count' in line:
                    last_line = line
    except Exception as e:
        return None

    if last_line is None:
        return None

    result = {}
    result['epochs'] = total_epochs if total_epochs else '?'

    # Parse metrics from the line using regex
    # Format: loss: 0.6421 - acc: 0.9343 - ... - val_loss: 0.6342 - val_acc: 0.9466 - ...
    patterns = {
        'train_loss': r'(?<!\w)loss:\s*([\d.]+)',
        'train_acc': r'(?<!\w)acc:\s*([\d.]+)',
        'val_loss': r'val_loss:\s*([\d.]+)',
        'val_acc': r'val_acc:\s*([\d.]+)',
        'best_val_acc': r'best_val_acc:\s*([\d.]+)',
        's_count': r'(?<!best_)s_count:\s*([\d.]+)',
        'best_s_count': r'best_s_count:\s*([\d.]+)',
    }

    for key, pat in patterns.items():
        m = re.search(pat, last_line)
        if m:
            result[key] = m.group(1)
        else:
            result[key] = 'N/A'

    return result


def parse_config(config_path):
    """Parse config_sweep.py to extract model, dataset, exp_set_name."""
    result = {'model': 'N/A', 'dataset': 'N/A', 'exp_set_name': 'N/A'}
    if not os.path.isfile(config_path):
        return result

    try:
        with open(config_path, 'r') as f:
            content = f.read()

        # Find uncommented conf.model lines (the last uncommented one wins)
        for m in re.finditer(r"^conf\.model\s*=\s*['\"]([^'\"]+)['\"]", content, re.MULTILINE):
            result['model'] = m.group(1)

        # Find uncommented conf.dataset lines
        for m in re.finditer(r"^conf\.dataset\s*=\s*['\"]([^'\"]+)['\"]", content, re.MULTILINE):
            result['dataset'] = m.group(1)

        # Find uncommented conf.exp_set_name lines
        for m in re.finditer(r"^conf\.exp_set_name\s*=\s*['\"]([^'\"]+)['\"]", content, re.MULTILINE):
            result['exp_set_name'] = m.group(1)

    except Exception:
        pass

    return result


def find_checkpoints(exp_dir):
    """Find checkpoint files (*.weights.h5) in the experiment directory."""
    ckpts = glob.glob(os.path.join(exp_dir, '**', '*.weights.h5'), recursive=True)
    if ckpts:
        # Return the latest one (by name, which includes epoch number)
        ckpts.sort()
        return ckpts[-1]
    return 'N/A'


def collect_experiment(exp_dir, category):
    """Collect results for a single experiment directory."""
    log_path = os.path.join(exp_dir, 'train.log')
    config_path = os.path.join(exp_dir, 'config_sweep.py')

    metrics = parse_last_epoch_line(log_path)
    if metrics is None:
        return None

    config = parse_config(config_path)
    ckpt = find_checkpoints(exp_dir)

    rel_dir = os.path.relpath(exp_dir, BASE)

    return {
        'dir': rel_dir,
        'category': category,
        'model': config['model'],
        'dataset': config['dataset'],
        'epochs': metrics['epochs'],
        'train_loss': metrics['train_loss'],
        'train_acc': metrics['train_acc'],
        'val_loss': metrics['val_loss'],
        'val_acc': metrics['val_acc'],
        'best_val_acc': metrics['best_val_acc'],
        's_count': metrics['s_count'],
        'best_s_count': metrics['best_s_count'],
        'ckpt_path': ckpt,
    }


def collect_from_dirs(dir_patterns, category):
    """Collect from a list of directory patterns. Each can be a glob or explicit path."""
    results = []
    for pattern in dir_patterns:
        full_pattern = os.path.join(BASE, pattern)
        dirs = sorted(glob.glob(full_pattern))
        if not dirs:
            # Try as exact path
            if os.path.isdir(full_pattern):
                dirs = [full_pattern]
        for d in dirs:
            if os.path.isdir(d):
                r = collect_experiment(d, category)
                if r:
                    results.append(r)
    return results


def print_results(results, category_name):
    """Print results for a category in tabular format."""
    if not results:
        print(f"\n{'='*120}")
        print(f"  {category_name}: NO RESULTS FOUND")
        print(f"{'='*120}")
        return

    print(f"\n{'='*160}")
    print(f"  {category_name} ({len(results)} experiments)")
    print(f"{'='*160}")

    # Header
    header = f"{'DIR':<60} | {'MODEL':<12} | {'DATASET':<10} | {'EP':>4} | {'TRAIN_LOSS':>10} | {'TRAIN_ACC':>9} | {'VAL_LOSS':>10} | {'VAL_ACC':>8} | {'BEST_VAL':>8} | {'S_COUNT':>12} | {'BEST_SC':>12} | CKPT_PATH"
    print(header)
    print('-' * len(header) + '-' * 40)

    for r in results:
        ckpt_short = r['ckpt_path'] if r['ckpt_path'] == 'N/A' else os.path.relpath(r['ckpt_path'], BASE)
        line = f"{r['dir']:<60} | {r['model']:<12} | {r['dataset']:<10} | {r['epochs']:>4} | {r['train_loss']:>10} | {r['train_acc']:>9} | {r['val_loss']:>10} | {r['val_acc']:>8} | {r['best_val_acc']:>8} | {r['s_count']:>12} | {r['best_s_count']:>12} | {ckpt_short}"
        print(line)


def main():
    all_results = []

    # 1. Baseline (no reg) experiments
    baseline_dirs = [
        '_baseline_trajectory',
        '_baseline_vgg_c10_trajectory',
        '_baseline_no_reg',
        '_baseline_no_reg_r19_c10',
        '_baseline_no_reg_r19_c100',
        '_baseline_no_reg_vgg_c100',
        '_baseline_no_reg_vgg_c100_run2',
        '_baseline_no_reg_vgg_c100_run3',
        '_baseline_no_reg_vgg_c100_v2',
        '_baseline_no_reg_run2',
        '_baseline_no_reg_run3',
    ]
    baseline_results = collect_from_dirs(baseline_dirs, 'Baseline (no reg)')
    print_results(baseline_results, 'BASELINE (NO REG)')
    all_results.extend(baseline_results)

    # 2. Softmax WTA (original method) - lambda sweep
    softmax_dirs = [
        '_sweep_lambda/lambda_*',
        '_r19_c10_lambda_sweep/lmb_*',
        '_r19_c10_lambda_sweep_run2/lmb_*',
        '_r19_c10_lambda_sweep_run3/lmb_*',
        '_r19_c10_lambda_sweep_run4/lmb_*',
        '_vgg_c10_lambda_sweep/lmb_*',
    ]
    softmax_results = collect_from_dirs(softmax_dirs, 'Softmax WTA Lambda Sweep')
    print_results(softmax_results, 'SOFTMAX WTA LAMBDA SWEEP')
    all_results.extend(softmax_results)

    # 3. WTA-Rev sweep
    wta_rev_dirs = [
        '_sweep_wta_rev/lambda_*',
        '_sweep_wta_rev_c100/lambda_*',
        '_sweep_wta_rev_r19_c10/lambda_*',
        '_sweep_wta_rev_r19_c100/lambda_*',
    ]
    wta_rev_results = collect_from_dirs(wta_rev_dirs, 'WTA-Rev Sweep')
    print_results(wta_rev_results, 'WTA-REV SWEEP')
    all_results.extend(wta_rev_results)

    # 4. WTA Alpha sweep
    wta_alpha_dirs = [
        '_sweep_wta_alpha/alpha_*',
        '_sweep_wta_alpha_run2/alpha_*',
        '_sweep_wta_alpha_run3/alpha_*',
        '_sweep_alpha/alpha_*',
    ]
    wta_alpha_results = collect_from_dirs(wta_alpha_dirs, 'WTA Alpha Sweep')
    print_results(wta_alpha_results, 'WTA ALPHA SWEEP')
    all_results.extend(wta_alpha_results)

    # 5. MaxNorm sweep
    maxnorm_dirs = [
        '_compare_maxnorm/*',
        '_r19_c10_maxnorm/*',
        '_r19_c10_maxnorm_run2/*',
        '_r19_c100_maxnorm/*',
        '_vgg_c10_maxnorm/*',
        '_spikformer_c10_maxnorm/*',
    ]
    maxnorm_results = collect_from_dirs(maxnorm_dirs, 'MaxNorm Sweep')
    print_results(maxnorm_results, 'MAXNORM SWEEP')
    all_results.extend(maxnorm_results)

    # 6. Encourage sweep
    encourage_dirs = [
        '_encourage/*',
    ]
    encourage_results = collect_from_dirs(encourage_dirs, 'Encourage Sweep')
    print_results(encourage_results, 'ENCOURAGE SWEEP')
    all_results.extend(encourage_results)

    # 7. Entropy WTA sweep
    entropy_dirs = [
        '_entropy_sweep/*',
    ]
    entropy_results = collect_from_dirs(entropy_dirs, 'Entropy WTA Sweep')
    print_results(entropy_results, 'ENTROPY WTA SWEEP')
    all_results.extend(entropy_results)

    # 8. Adaptive lambda
    adaptive_dirs = [
        '_adaptive_lambda/*',
    ]
    adaptive_results = collect_from_dirs(adaptive_dirs, 'Adaptive Lambda')
    print_results(adaptive_results, 'ADAPTIVE LAMBDA')
    all_results.extend(adaptive_results)

    # 9. Spikformer experiments
    spikformer_dirs = [
        '_spikformer_c10_baseline',
        '_spikformer_c10_wta_rev',
        '_spikformer_c10_lmb_*',
    ]
    spikformer_results = collect_from_dirs(spikformer_dirs, 'Spikformer Experiments')
    print_results(spikformer_results, 'SPIKFORMER EXPERIMENTS')
    all_results.extend(spikformer_results)

    # Summary
    print(f"\n{'='*160}")
    print(f"  SUMMARY")
    print(f"{'='*160}")
    print(f"Total experiments collected: {len(all_results)}")

    # Group by category
    categories = {}
    for r in all_results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    for cat, exps in categories.items():
        best = max(exps, key=lambda x: float(x['best_val_acc']) if x['best_val_acc'] != 'N/A' else 0)
        print(f"  {cat}: {len(exps)} experiments, best_val_acc = {best['best_val_acc']} ({best['dir']})")

    # Also produce a CSV for easy import
    csv_path = os.path.join(BASE, 'all_experiment_results.csv')
    with open(csv_path, 'w') as f:
        f.write('category,dir,model,dataset,epochs,train_loss,train_acc,val_loss,val_acc,best_val_acc,s_count,best_s_count,ckpt_path\n')
        for r in all_results:
            ckpt_short = r['ckpt_path'] if r['ckpt_path'] == 'N/A' else os.path.relpath(r['ckpt_path'], BASE)
            f.write(f"{r['category']},{r['dir']},{r['model']},{r['dataset']},{r['epochs']},{r['train_loss']},{r['train_acc']},{r['val_loss']},{r['val_acc']},{r['best_val_acc']},{r['s_count']},{r['best_s_count']},{ckpt_short}\n")
    print(f"\nCSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
