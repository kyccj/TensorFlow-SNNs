'''
    ResNet19 CIFAR10 WTA-Rev lambda sweep - 3 runs (run2, run3)
    Run1 is already running. This script waits for run1 to finish,
    then launches run2 and run3 sequentially.
'''
import subprocess
import os
import time
import signal
import sys

PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/miniconda3/envs/venv_1/lib'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

LAMBDAS = [1e-9, 5e-9, 1e-8, 3e-8, 1e-7, 3e-7, 5e-7, 1e-6]
FIXED_ALPHA = 4


def make_experiments(run_tag):
    return [
        {'lmb': lmb, 'gpu': gpu, 'dir': f'_r19_c10_lambda_sweep_{run_tag}/lmb_{lmb:.0e}'}
        for gpu, lmb in enumerate(LAMBDAS)
    ]


def generate_config(exp):
    run_dir = os.path.join(PROJECT_ROOT, exp['dir'])
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, 'config_sweep.py')

    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        content = f.read()

    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{exp["gpu"]}"'
    )
    content = content.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")
    content = content.replace(
        'conf.reg_spike_out_alpha=3  # temperature',
        f'conf.reg_spike_out_alpha={FIXED_ALPHA}  # temperature'
    )
    content = content.replace(
        'conf.reg_spike_out_const=1E-8',
        f'conf.reg_spike_out_const={exp["lmb"]}'
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
        f"conf.exp_set_name='r19-c10-lmb-{exp['lmb']:.0e}-{exp['dir'].split('/')[0].split('_')[-1]}'"
    )

    with open(config_path, 'w') as f:
        f.write(content)


def generate_main(exp):
    run_dir = os.path.join(PROJECT_ROOT, exp['dir'])
    main_path = os.path.join(run_dir, 'main_sweep.py')

    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')) as f:
        content = f.read()
    content = content.replace(
        'from config_snn_training import config',
        'from config_sweep import config'
    )
    with open(main_path, 'w') as f:
        f.write(content)


def run_sweep(run_tag):
    experiments = make_experiments(run_tag)
    processes = []

    print(f'\n{"="*60}')
    print(f'  Launching {run_tag} ({len(experiments)} experiments)')
    print(f'{"="*60}')

    for exp in experiments:
        generate_config(exp)
        generate_main(exp)

        run_dir = os.path.join(PROJECT_ROOT, exp['dir'])
        log_file = os.path.join(run_dir, 'train.log')
        print(f'[GPU {exp["gpu"]}] R19 C10 WTA-Rev a={FIXED_ALPHA} lmb={exp["lmb"]} -> {log_file}')

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
        processes.append((exp, p, log_file))

    print(f'\n--- {run_tag}: {len(processes)} experiments launched ---')

    for exp, p, log_file in processes:
        p.wait()
        status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'
        print(f'[GPU {exp["gpu"]}] {run_tag} lmb={exp["lmb"]} finished: {status}')

    print(f'\n*** {run_tag} ALL DONE ***\n')


def wait_for_run1():
    """Wait for run1 processes to finish by checking if GPUs are free."""
    run1_dir = os.path.join(PROJECT_ROOT, '_r19_c10_lambda_sweep')
    if not os.path.exists(run1_dir):
        print('Run1 dir not found, skipping wait.')
        return

    print('Waiting for run1 to finish...')
    while True:
        # Check if any main_sweep.py from run1 is still running
        result = subprocess.run(
            ['pgrep', '-f', '_r19_c10_lambda_sweep/lmb_.*main_sweep'],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            # No matching processes
            break
        time.sleep(30)
        # Print a dot for progress
        print('.', end='', flush=True)

    print('\nRun1 finished!')

    # Print run1 results
    print('\n--- Run1 Results ---')
    for lmb in LAMBDAS:
        log = os.path.join(run1_dir, f'lmb_{lmb:.0e}', 'train.log')
        if os.path.exists(log):
            with open(log) as f:
                content = f.read()
            import re
            accs = re.findall(r'best_val_acc: ([0-9.]+)', content)
            scs = re.findall(r'best_s_count: ([0-9.]+)', content)
            if accs and scs:
                print(f'  lmb={lmb:.0e}: acc={accs[-1]}, sc={scs[-1]}')


def main():
    # Wait for run1
    wait_for_run1()

    # Run2
    run_sweep('run2')

    # Run3
    run_sweep('run3')

    print('\n' + '='*60)
    print('  ALL 3 RUNS COMPLETED')
    print('='*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted.')
        sys.exit(1)
