"""
Early Regularization Experiment: apply reg from epoch 0, release at end_ep.
- Fixed lambda, no adaptive
- VGG-C10: lambda sweep × end_ep sweep
- GPU 0, 1, 2, 5
"""

import subprocess
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_early_reg')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

EXPERIMENTS = [
    # (model, dataset, lambda, end_ep, gpu)
    ('VGG16', 'CIFAR10', 1e-6, 50,  0),
    ('VGG16', 'CIFAR10', 1e-6, 100, 1),
    ('VGG16', 'CIFAR10', 5e-7, 50,  2),
    ('VGG16', 'CIFAR10', 5e-7, 100, 5),
]


def generate_config(model, dataset, lmb, end_ep, gpu_id, run_dir, exp_name):
    orig_config = os.path.join(PROJECT_ROOT, 'config_snn_training.py')
    with open(orig_config, 'r') as f:
        content = f.read()

    # GPU
    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'
    )

    # Experiment name
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        f"conf.exp_set_name='{exp_name}'"
    )

    # Enable reg with wta_rev
    content = content.replace(
        'conf.reg_spike_out_alpha=3  # temperature',
        'conf.reg_spike_out_alpha=4  # temperature'
    )
    content = content.replace(
        f"conf.reg_spike_out_const=1E-8",
        f"conf.reg_spike_out_const={lmb}"
    )
    content = content.replace(
        '#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
        'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'
    )
    content = content.replace(
        '#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
        'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'
    )

    # Add end_ep setting after sc_loss_scd line
    content = content.replace(
        'conf.sc_loss_scd = False',
        f'conf.sc_loss_scd = False\n        conf.reg_spike_end_ep = {end_ep}'
    )

    config_path = os.path.join(run_dir, 'config_sweep.py')
    with open(config_path, 'w') as f:
        f.write(content)


def generate_main(run_dir):
    orig_main = os.path.join(PROJECT_ROOT, 'main_snn_training.py')
    with open(orig_main, 'r') as f:
        content = f.read()
    content = content.replace(
        'from config_snn_training import config',
        'from config_sweep import config'
    )
    main_path = os.path.join(run_dir, 'main_sweep.py')
    with open(main_path, 'w') as f:
        f.write(content)


def main():
    os.makedirs(SWEEP_DIR, exist_ok=True)
    processes = []

    for model, dataset, lmb, end_ep, gpu_id in EXPERIMENTS:
        exp_name = f'early-reg-lmb-{lmb}-end-{end_ep}'
        run_name = f'vgg_c10_lmb{lmb}_end{end_ep}'
        run_dir = os.path.join(SWEEP_DIR, run_name)
        os.makedirs(run_dir, exist_ok=True)

        generate_config(model, dataset, lmb, end_ep, gpu_id, run_dir, exp_name)
        generate_main(run_dir)

        log_file = os.path.join(run_dir, 'train.log')
        print(f'[GPU {gpu_id}] {model}-{dataset} lmb={lmb} end_ep={end_ep} -> {run_dir}')

        env = os.environ.copy()
        env['PYTHONPATH'] = run_dir + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
        env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
        env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'

        with open(log_file, 'w') as lf:
            p = subprocess.Popen(
                [PYTHON, os.path.join(run_dir, 'main_sweep.py')],
                cwd=PROJECT_ROOT,
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=env,
            )
        processes.append((run_name, gpu_id, p, log_file))

    print(f'\n--- {len(processes)} experiments launched ---')
    print('Monitor:')
    print(f'  tail -f {SWEEP_DIR}/*/train.log')
    print(f'\nKill all:')
    print(f'  kill {" ".join(str(p.pid) for _, _, p, _ in processes)}')

    try:
        for name, gpu_id, p, log_file in processes:
            p.wait()
            status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'
            print(f'[GPU {gpu_id}] {name} finished: {status}')
    except KeyboardInterrupt:
        print('\nInterrupted - terminating...')
        for _, _, p, _ in processes:
            p.terminate()
        for _, _, p, _ in processes:
            p.wait()
        print('All terminated.')


if __name__ == '__main__':
    main()
