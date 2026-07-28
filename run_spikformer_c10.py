'''
    Run Spikformer CIFAR10 experiments on GPU 6, 7
    - GPU 6: baseline (no spike reg)
    - GPU 7: WTA-Rev (lambda=5e-8, alpha=7)
'''
import subprocess
import os

PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/miniconda3/envs/venv_1/lib'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    {
        'name': 'Spikformer C10 baseline',
        'dir': '_spikformer_c10_baseline',
        'exp_name': 'spikformer-c10-baseline',
        'gpu': 6,
        'reg_spike_out': False,
        'wta_rev': False,
        'lmb': None,
        'alpha': None,
    },
    {
        'name': 'Spikformer C10 WTA-Rev',
        'dir': '_spikformer_c10_wta_rev',
        'exp_name': 'spikformer-c10-wta-rev',
        'gpu': 7,
        'reg_spike_out': True,
        'wta_rev': True,
        'lmb': 5e-8,
        'alpha': 7,
    },
]


def generate_config(exp):
    run_dir = os.path.join(PROJECT_ROOT, exp['dir'])
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, 'config_sweep.py')

    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        content = f.read()

    # GPU
    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{exp["gpu"]}"'
    )

    # Spikformer + CIFAR10
    content = content.replace("conf.model='ResNet19'", "conf.model='Spikformer'")
    content = content.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")

    # spike reg
    if not exp['reg_spike_out']:
        content = content.replace("conf.reg_spike_out=True", "conf.reg_spike_out=False")
    else:
        # alpha
        if exp['alpha'] is not None:
            content = content.replace(
                'conf.reg_spike_out_alpha=3  # temperature',
                f'conf.reg_spike_out_alpha={exp["alpha"]}  # temperature'
            )
        # lambda
        if exp['lmb'] is not None:
            content = content.replace(
                'conf.reg_spike_out_const=1E-8',
                f'conf.reg_spike_out_const={exp["lmb"]}'
            )
        # wta_rev
        if exp['wta_rev']:
            content = content.replace(
                '#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
                'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'
            )

    # reg detail logging
    content = content.replace(
        '#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
        'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'
    )

    # exp name
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        f"conf.exp_set_name='{exp['exp_name']}'"
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


def main():
    processes = []

    for exp in EXPERIMENTS:
        generate_config(exp)
        generate_main(exp)

        run_dir = os.path.join(PROJECT_ROOT, exp['dir'])
        log_file = os.path.join(run_dir, 'train.log')
        print(f'[GPU {exp["gpu"]}] {exp["name"]} -> {log_file}')

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

    print(f'\n--- {len(processes)} experiments launched ---')
    print('Monitor:')
    for exp, _, _ in processes:
        print(f'  tail -f {exp["dir"]}/train.log')

    try:
        for exp, p, log_file in processes:
            p.wait()
            status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'
            print(f'[GPU {exp["gpu"]}] {exp["name"]} finished: {status}')
    except KeyboardInterrupt:
        print('\nInterrupted - terminating...')
        for _, p, _ in processes:
            p.terminate()
        for _, p, _ in processes:
            p.wait()


if __name__ == '__main__':
    main()
