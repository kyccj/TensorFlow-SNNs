'''
    Run all missing baseline experiments (no spike regularization).
    - VGG16 CIFAR100 (GPU 0)
    - ResNet19 CIFAR10 (GPU 1)
    - ResNet19 CIFAR100 (GPU 2)

    Existing: VGG16 CIFAR10 (_baseline_no_reg) - already done
'''
import subprocess
import os
import sys

PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/miniconda3/envs/venv_1/lib'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

BASELINES = [
    {
        'name': 'VGG16 CIFAR100',
        'dir': '_baseline_no_reg_vgg_c100',
        'exp_name': 'EIP-SNN-26_baseline-no-reg-vgg-c100',
        'model': 'VGG16',
        'dataset': 'CIFAR100',
        'gpu': 0,
    },
    {
        'name': 'ResNet19 CIFAR10',
        'dir': '_baseline_no_reg_r19_c10',
        'exp_name': 'EIP-SNN-26_baseline-no-reg-r19-c10',
        'model': 'ResNet19',
        'dataset': 'CIFAR10',
        'gpu': 1,
    },
    {
        'name': 'ResNet19 CIFAR100',
        'dir': '_baseline_no_reg_r19_c100',
        'exp_name': 'EIP-SNN-26_baseline-no-reg-r19-c100',
        'model': 'ResNet19',
        'dataset': 'CIFAR100',
        'gpu': 2,
    },
]


def generate_config(bl, run_dir):
    config_path = os.path.join(run_dir, 'config_sweep.py')
    orig_config = os.path.join(PROJECT_ROOT, 'config_snn_training.py')

    with open(orig_config, 'r') as f:
        content = f.read()

    # GPU
    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{bl["gpu"]}"'
    )

    # Model
    if bl['model'] == 'VGG16':
        content = content.replace("conf.model='ResNet19'", "conf.model='VGG16'")

    # Dataset
    if bl['dataset'] == 'CIFAR10':
        content = content.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")

    # Disable spike regularization
    content = content.replace("conf.reg_spike_out=True", "conf.reg_spike_out=False")

    # Enable detail logging
    content = content.replace(
        '#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
        'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'
    )

    # exp name
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        f"conf.exp_set_name='{bl['exp_name']}'"
    )

    with open(config_path, 'w') as f:
        f.write(content)


def generate_main(run_dir):
    main_path = os.path.join(run_dir, 'main_sweep.py')
    orig_main = os.path.join(PROJECT_ROOT, 'main_snn_training.py')

    with open(orig_main, 'r') as f:
        content = f.read()

    content = content.replace(
        'from config_snn_training import config',
        'from config_sweep import config'
    )

    with open(main_path, 'w') as f:
        f.write(content)


def main():
    processes = []

    for bl in BASELINES:
        run_dir = os.path.join(PROJECT_ROOT, bl['dir'])
        os.makedirs(run_dir, exist_ok=True)

        generate_config(bl, run_dir)
        generate_main(run_dir)

        log_file = os.path.join(run_dir, 'train.log')
        print(f'[GPU {bl["gpu"]}] {bl["name"]} baseline -> {log_file}')

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
        processes.append((bl['name'], bl['gpu'], p, log_file))

    print(f'\n--- {len(processes)} baseline experiments launched ---')
    print('Monitor:')
    for bl in BASELINES:
        print(f'  tail -f {bl["dir"]}/train.log')

    try:
        for name, gpu, p, log_file in processes:
            p.wait()
            status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'
            print(f'[GPU {gpu}] {name} finished: {status}')
    except KeyboardInterrupt:
        print('\nInterrupted - terminating all...')
        for _, _, p, _ in processes:
            p.terminate()
        for _, _, p, _ in processes:
            p.wait()
        print('All terminated.')


if __name__ == '__main__':
    main()
