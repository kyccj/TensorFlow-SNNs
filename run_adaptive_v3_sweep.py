'''
    Adaptive Lambda v3: "start strong, back off if needed"
    - lambda_init = lambda_max (start at maximum strength from epoch 50)
    - lambda_min = 1e-8 (allow backing off to very low)
    - step_down = 0.3, step_up = 0.5, margin = 0.5%
    - VGG-C10 (GPU 0-2), R19-C10 (GPU 3-4)
    - lmax sweep: 1e-6, 3e-6, 1e-5
'''

import subprocess
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_adaptive_lambda_v3')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

EXPERIMENTS = [
    # (model, dataset, lmax, gpu)
    ('VGG16',    'CIFAR10', 1e-6, 0),
    ('VGG16',    'CIFAR10', 3e-6, 1),
    ('VGG16',    'CIFAR10', 1e-5, 2),
    ('ResNet19', 'CIFAR10', 1e-6, 3),
    ('ResNet19', 'CIFAR10', 1e-5, 4),
]


def generate_config(model, dataset, lmax, gpu_id, run_dir):
    orig_config = os.path.join(PROJECT_ROOT, 'config_snn_training.py')
    with open(orig_config, 'r') as f:
        content = f.read()

    # GPU
    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'
    )

    # Model
    content = content.replace("conf.model='ResNet19'", f"conf.model='{model}'")
    content = content.replace("conf.dataset='CIFAR100'", f"conf.dataset='{dataset}'")

    # Experiment name
    model_short = 'vgg' if model == 'VGG16' else 'r19'
    ds_short = 'c10' if dataset == 'CIFAR10' else 'c100'
    exp_name = f'adv3-{model_short}-{ds_short}-lmax-{lmax}'
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        f"conf.exp_set_name='{exp_name}'"
    )

    # Enable reg with wta_rev + adaptive
    content = content.replace(
        'conf.reg_spike_out_alpha=3  # temperature',
        'conf.reg_spike_out_alpha=4  # temperature'
    )
    content = content.replace(
        '#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
        'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'
    )
    content = content.replace(
        '#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
        'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'
    )

    # Adaptive lambda v3: start strong
    adaptive_block = f'''
        # Adaptive Lambda v3: start strong, back off if needed
        conf.reg_spike_adaptive = True
        conf.reg_spike_adaptive_start_ep = 50
        conf.reg_spike_adaptive_lambda_init = {lmax}
        conf.reg_spike_adaptive_lambda_max = {lmax}
        conf.reg_spike_adaptive_lambda_min = 1e-8
        conf.reg_spike_adaptive_step_up = 0.5
        conf.reg_spike_adaptive_step_down = 0.3
        conf.reg_spike_adaptive_margin = 0.005
'''
    content = content.replace(
        'conf.sc_loss_scd = False',
        'conf.sc_loss_scd = False' + adaptive_block
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

    for model, dataset, lmax, gpu_id in EXPERIMENTS:
        model_short = 'vgg' if model == 'VGG16' else 'r19'
        ds_short = 'c10' if dataset == 'CIFAR10' else 'c100'
        run_name = f'{model_short}_{ds_short}_lmax{lmax}'
        run_dir = os.path.join(SWEEP_DIR, run_name)
        os.makedirs(run_dir, exist_ok=True)

        generate_config(model, dataset, lmax, gpu_id, run_dir)
        generate_main(run_dir)

        log_file = os.path.join(run_dir, 'train.log')
        print(f'[GPU {gpu_id}] {model}-{dataset} lmax={lmax} -> {run_dir}')

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
