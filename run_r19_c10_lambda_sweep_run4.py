'''
    ResNet19 CIFAR10 WTA-Rev lambda sweep - Run 4 (additional repetition)
    Alpha=4 fixed, same lambda values as run1-3
    GPU: 0-7
'''
import subprocess
import os

PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/miniconda3/envs/venv_1/lib'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    {'lmb': 1e-9,  'gpu': 0, 'dir': '_r19_c10_lambda_sweep_run4/lmb_1e-09'},
    {'lmb': 5e-9,  'gpu': 1, 'dir': '_r19_c10_lambda_sweep_run4/lmb_5e-09'},
    {'lmb': 1e-8,  'gpu': 2, 'dir': '_r19_c10_lambda_sweep_run4/lmb_1e-08'},
    {'lmb': 3e-8,  'gpu': 3, 'dir': '_r19_c10_lambda_sweep_run4/lmb_3e-08'},
    {'lmb': 1e-7,  'gpu': 4, 'dir': '_r19_c10_lambda_sweep_run4/lmb_1e-07'},
    {'lmb': 3e-7,  'gpu': 5, 'dir': '_r19_c10_lambda_sweep_run4/lmb_3e-07'},
    {'lmb': 5e-7,  'gpu': 6, 'dir': '_r19_c10_lambda_sweep_run4/lmb_5e-07'},
    {'lmb': 1e-6,  'gpu': 7, 'dir': '_r19_c10_lambda_sweep_run4/lmb_1e-06'},
]

FIXED_ALPHA = 4


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
        f"conf.exp_set_name='r19-c10-lmb-{exp['lmb']:.0e}-run4'"
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
        print(f'[GPU {exp["gpu"]}] R19 C10 WTA-Rev a={FIXED_ALPHA} lmb={exp["lmb"]} run4 -> {log_file}')

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

    print(f'\n--- {len(processes)} experiments launched (run4) ---')
    print('Monitor:')
    for exp, _, _ in processes:
        print(f'  tail -f {exp["dir"]}/train.log')

    try:
        for exp, p, log_file in processes:
            p.wait()
            status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'
            print(f'[GPU {exp["gpu"]}] lmb={exp["lmb"]} run4 finished: {status}')
    except KeyboardInterrupt:
        print('\nInterrupted - terminating...')
        for _, p, _ in processes:
            p.terminate()
        for _, p, _ in processes:
            p.wait()


if __name__ == '__main__':
    main()
