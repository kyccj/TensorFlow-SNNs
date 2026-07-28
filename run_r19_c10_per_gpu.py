'''
    ResNet19 CIFAR10 WTA-Rev lambda sweep - per-GPU pipeline
    Each GPU runs its own run1(if needed)->run2->run3 independently.
    Usage: python run_r19_c10_per_gpu.py
'''
import subprocess
import os
import sys
import time
import threading
import re

PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/miniconda3/envs/venv_1/lib'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

LAMBDAS_BY_GPU = {
    0: 1e-9,
    1: 5e-9,
    2: 1e-8,
    3: 3e-8,
    4: 1e-7,
    5: 3e-7,
    6: 5e-7,
    7: 1e-6,
}

FIXED_ALPHA = 4
RUNS = ['run2', 'run3']  # run1 already done or finishing


def generate_config(run_dir, gpu, lmb, run_tag):
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, 'config_sweep.py')

    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        content = f.read()

    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu}"'
    )
    content = content.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")
    content = content.replace(
        'conf.reg_spike_out_alpha=3  # temperature',
        f'conf.reg_spike_out_alpha={FIXED_ALPHA}  # temperature'
    )
    content = content.replace(
        'conf.reg_spike_out_const=1E-8',
        f'conf.reg_spike_out_const={lmb}'
    )
    content = content.replace(
        '#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
        'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'
    )
    content = content.replace(
        '#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
        'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'
    )
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        f"conf.exp_set_name='r19-c10-lmb-{lmb:.0e}-{run_tag}'"
    )

    with open(config_path, 'w') as f:
        f.write(content)


def generate_main(run_dir):
    main_path = os.path.join(run_dir, 'main_sweep.py')
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')) as f:
        content = f.read()
    content = content.replace(
        'from config_snn_training import config',
        'from config_sweep import config'
    )
    with open(main_path, 'w') as f:
        f.write(content)


def is_run1_done(gpu, lmb):
    """Check if run1 for this GPU/lambda is finished."""
    log = os.path.join(PROJECT_ROOT, f'_r19_c10_lambda_sweep/lmb_{lmb:.0e}/train.log')
    if not os.path.exists(log):
        return True  # no run1, skip
    # Check if process still running
    result = subprocess.run(
        ['pgrep', '-f', f'_r19_c10_lambda_sweep/lmb_{lmb:.0e}/main_sweep'],
        capture_output=True, text=True
    )
    return result.returncode != 0  # not running = done


def gpu_pipeline(gpu, lmb):
    """Run pipeline for a single GPU: wait for run1, then run2, run3."""
    thread_name = f'GPU{gpu}-lmb{lmb:.0e}'

    # Wait for run1 to finish on this GPU
    while not is_run1_done(gpu, lmb):
        time.sleep(30)
    print(f'[{thread_name}] run1 done, starting run2/run3 pipeline', flush=True)

    for run_tag in RUNS:
        run_dir = os.path.join(PROJECT_ROOT, f'_r19_c10_lambda_sweep_{run_tag}/lmb_{lmb:.0e}')
        generate_config(run_dir, gpu, lmb, run_tag)
        generate_main(run_dir)

        log_file = os.path.join(run_dir, 'train.log')
        print(f'[{thread_name}] {run_tag} started -> {log_file}', flush=True)

        env = os.environ.copy()
        env['PYTHONPATH'] = run_dir + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
        env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
        env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/miniconda3/envs/venv_1'

        with open(log_file, 'w') as lf:
            p = subprocess.Popen(
                [PYTHON, os.path.join(run_dir, 'main_sweep.py')],
                cwd=PROJECT_ROOT,
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=env,
            )
        p.wait()
        status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'

        # Extract best acc
        best_acc = '?'
        if os.path.exists(log_file):
            with open(log_file) as f:
                accs = re.findall(r'best_val_acc: ([0-9.]+)', f.read())
                if accs:
                    best_acc = accs[-1]

        print(f'[{thread_name}] {run_tag} finished: {status} (best_acc={best_acc})', flush=True)

    print(f'[{thread_name}] ALL RUNS DONE', flush=True)


def main():
    print(f'Starting per-GPU pipelines (alpha={FIXED_ALPHA}, runs={RUNS})')
    print(f'Lambda mapping: {LAMBDAS_BY_GPU}')

    threads = []
    for gpu, lmb in LAMBDAS_BY_GPU.items():
        t = threading.Thread(target=gpu_pipeline, args=(gpu, lmb), name=f'gpu{gpu}')
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print('\n' + '='*60)
    print('  ALL GPUs ALL RUNS COMPLETED')
    print('='*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted.')
        sys.exit(1)
