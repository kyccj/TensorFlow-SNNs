'''
    Alpha sweep script
    - reg_spike_out_alpha: [0.5, 1, 1.5, 2, 3, 5, 7, 10]
    - GPU 0~7, one per alpha value
'''

import subprocess
import os
import sys
import time
import shutil
import signal

SWEEP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_sweep_alpha')
ALPHA_VALUES = [0.5, 1, 1.5, 2, 3, 5, 7, 10]
GPU_IDS = list(range(8))  # 0~7
PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'

# CUDA libs from conda env (CUDA 11.8 + cuDNN 8.9, compatible with driver 535)
CUDA_LD_PATH = '/home/kyccj/miniconda3/envs/venv_1/lib'


def generate_config(alpha, gpu_id, run_dir):
    """Generate config file with specified alpha and GPU."""
    config_path = os.path.join(run_dir, 'config_sweep.py')

    # Read original config
    orig_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_snn_training.py')
    with open(orig_config, 'r') as f:
        content = f.read()

    # Replace GPU
    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'
    )

    # Replace alpha
    content = content.replace(
        'conf.reg_spike_out_alpha=3  # temperature',
        f'conf.reg_spike_out_alpha={alpha}  # temperature'
    )

    # Switch to VGG16 + CIFAR10
    content = content.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    content = content.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")

    # Replace exp_set_name to include alpha value
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        f"conf.exp_set_name='EIP-SNN-26_sweep-alpha-{alpha}'"
    )

    with open(config_path, 'w') as f:
        f.write(content)

    return config_path


def generate_main(run_dir):
    """Generate main training script that imports from local config."""
    main_path = os.path.join(run_dir, 'main_sweep.py')

    # Read original main
    orig_main = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main_snn_training.py')
    with open(orig_main, 'r') as f:
        content = f.read()

    # Replace config import
    content = content.replace(
        'from config_snn_training import config',
        'from config_sweep import config'
    )

    with open(main_path, 'w') as f:
        f.write(content)

    return main_path


def main():
    assert len(ALPHA_VALUES) <= len(GPU_IDS), \
        f"Not enough GPUs ({len(GPU_IDS)}) for alpha values ({len(ALPHA_VALUES)})"

    # Create sweep directory
    os.makedirs(SWEEP_DIR, exist_ok=True)

    project_root = os.path.dirname(os.path.abspath(__file__))
    processes = []

    for alpha, gpu_id in zip(ALPHA_VALUES, GPU_IDS):
        run_dir = os.path.join(SWEEP_DIR, f'alpha_{alpha}')
        os.makedirs(run_dir, exist_ok=True)

        # Generate config and main script
        generate_config(alpha, gpu_id, run_dir)
        generate_main(run_dir)

        # Log file
        log_file = os.path.join(run_dir, 'train.log')

        print(f'[GPU {gpu_id}] alpha={alpha} -> {log_file}')

        # Launch training
        # run_dir first in PYTHONPATH so config_sweep is found before config_snn_training
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
        processes.append((alpha, gpu_id, p, log_file))

    print(f'\n--- {len(processes)} experiments launched ---')
    print('Monitor logs:')
    print(f'  tail -f {SWEEP_DIR}/alpha_*/train.log')
    print(f'\nKill all:')
    print(f'  kill {" ".join(str(p.pid) for _, _, p, _ in processes)}')

    # Wait for all
    try:
        for alpha, gpu_id, p, log_file in processes:
            p.wait()
            status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'
            print(f'[GPU {gpu_id}] alpha={alpha} finished: {status}')
    except KeyboardInterrupt:
        print('\nInterrupted - terminating all processes...')
        for _, _, p, _ in processes:
            p.terminate()
        for _, _, p, _ in processes:
            p.wait()
        print('All terminated.')


if __name__ == '__main__':
    main()
