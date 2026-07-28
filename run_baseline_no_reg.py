'''
    Baseline: VGG16 CIFAR10, no spike regularization
    GPU 0
'''
import subprocess
import os

PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/miniconda3/envs/venv_1/lib'
RUN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_baseline_no_reg')
GPU_ID = 0


def generate_config():
    os.makedirs(RUN_DIR, exist_ok=True)
    config_path = os.path.join(RUN_DIR, 'config_sweep.py')
    orig_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config_snn_training.py')

    with open(orig_config, 'r') as f:
        content = f.read()

    # GPU
    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{GPU_ID}"'
    )

    # VGG16 + CIFAR10
    content = content.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    content = content.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")

    # Disable spike regularization
    content = content.replace("conf.reg_spike_out=True", "conf.reg_spike_out=False")

    # Enable detail logging for spike_count comparison
    content = content.replace(
        '#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
        'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'
    )

    # exp name
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        "conf.exp_set_name='EIP-SNN-26_baseline-no-reg'"
    )

    with open(config_path, 'w') as f:
        f.write(content)


def generate_main():
    main_path = os.path.join(RUN_DIR, 'main_sweep.py')
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
    generate_config()
    generate_main()

    log_file = os.path.join(RUN_DIR, 'train.log')
    print(f'[GPU {GPU_ID}] Baseline (no reg) -> {log_file}')

    project_root = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env['PYTHONPATH'] = RUN_DIR + ':' + project_root + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/miniconda3/envs/venv_1'

    with open(log_file, 'w') as lf:
        p = subprocess.Popen(
            [PYTHON, os.path.join(RUN_DIR, 'main_sweep.py')],
            cwd=project_root,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
        )

    print(f'PID: {p.pid}')
    print(f'Monitor: tail -f {log_file}')

    try:
        p.wait()
        status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'
        print(f'Baseline finished: {status}')
    except KeyboardInterrupt:
        print('Interrupted - terminating...')
        p.terminate()
        p.wait()


if __name__ == '__main__':
    main()
