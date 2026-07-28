'''
    WTA revised sweep - VGG16 CIFAR100
    - wta_rev=True (reduce_mean instead of L2 norm)
    - alpha=7 (fixed, best from alpha sweep)
    - reg_spike_out_const: [1e-10, 1e-9, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6]
    - VGG16 + CIFAR100, 310 epochs
    - GPU 1~7
'''

import subprocess
import os
import sys

SWEEP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_sweep_wta_rev_c100')
LAMBDA_VALUES = [1e-10, 1e-9, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6]
GPU_IDS = list(range(1, 8))  # 1~7
PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/miniconda3/envs/venv_1/lib'


def generate_config(lmb, gpu_id, run_dir):
    config_path = os.path.join(run_dir, 'config_sweep.py')

    orig_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_snn_training.py')
    with open(orig_config, 'r') as f:
        content = f.read()

    # GPU
    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'
    )

    # alpha=7 (fixed)
    content = content.replace(
        'conf.reg_spike_out_alpha=3  # temperature',
        'conf.reg_spike_out_alpha=7  # temperature'
    )

    # lambda
    content = content.replace(
        'conf.reg_spike_out_const=1E-8',
        f'conf.reg_spike_out_const={lmb}'
    )

    # wta_rev enable
    content = content.replace(
        '#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
        'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'
    )

    # reg detail logging
    content = content.replace(
        '#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
        'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'
    )

    # VGG16 (change from ResNet19), keep CIFAR100
    content = content.replace("conf.model='ResNet19'", "conf.model='VGG16'")

    # exp name
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        f"conf.exp_set_name='EIP-SNN-26_wta-rev-c100-lambda-{lmb}'"
    )

    with open(config_path, 'w') as f:
        f.write(content)


def generate_main(run_dir):
    main_path = os.path.join(run_dir, 'main_sweep.py')

    orig_main = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main_snn_training.py')
    with open(orig_main, 'r') as f:
        content = f.read()

    content = content.replace(
        'from config_snn_training import config',
        'from config_sweep import config'
    )

    with open(main_path, 'w') as f:
        f.write(content)


def main():
    os.makedirs(SWEEP_DIR, exist_ok=True)
    project_root = os.path.dirname(os.path.abspath(__file__))
    processes = []

    for lmb, gpu_id in zip(LAMBDA_VALUES, GPU_IDS):
        run_dir = os.path.join(SWEEP_DIR, f'lambda_{lmb}')
        os.makedirs(run_dir, exist_ok=True)

        generate_config(lmb, gpu_id, run_dir)
        generate_main(run_dir)

        log_file = os.path.join(run_dir, 'train.log')
        print(f'[GPU {gpu_id}] lambda={lmb} -> {log_file}')

        env = os.environ.copy()
        env['PYTHONPATH'] = run_dir + ':' + project_root + ':' + env.get('PYTHONPATH', '')
        env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
        env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/miniconda3/envs/venv_1'

        with open(log_file, 'w') as lf:
            p = subprocess.Popen(
                [PYTHON, os.path.join(run_dir, 'main_sweep.py')],
                cwd=project_root,
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=env,
            )
        processes.append((lmb, gpu_id, p, log_file))

    print(f'\n--- {len(processes)} experiments launched ---')
    print('Monitor logs:')
    print(f'  tail -f {SWEEP_DIR}/lambda_*/train.log')
    print(f'\nKill all:')
    print(f'  kill {" ".join(str(p.pid) for _, _, p, _ in processes)}')

    try:
        for lmb, gpu_id, p, log_file in processes:
            p.wait()
            status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'
            print(f'[GPU {gpu_id}] lambda={lmb} finished: {status}')
    except KeyboardInterrupt:
        print('\nInterrupted - terminating all processes...')
        for _, _, p, _ in processes:
            p.terminate()
        for _, _, p, _ in processes:
            p.wait()
        print('All terminated.')


if __name__ == '__main__':
    main()
