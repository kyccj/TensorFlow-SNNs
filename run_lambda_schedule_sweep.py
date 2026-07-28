"""
Lambda scheduling experiments:
  Method 1 (LR-linked): GPUs 0,1,2 — lambda inversely follows cosine LR
  Method 2 (Improved adaptive): GPUs 3,4,5 — moving window accuracy comparison
VGG-C10, 310 epochs
"""

import subprocess
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_lambda_schedule')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

# Method 1: LR-linked — vary lambda_max
LR_LINKED_EXPS = [
    # (lambda_max, gpu)
    (1e-7, 0),
    (5e-7, 1),
    (1e-6, 2),
]

# Method 2: Improved adaptive — vary window/margin
ADAPTIVE_EXPS = [
    # (window, margin, lambda_min, gpu)
    (10, 0.01, 5e-8, 3),
    (20, 0.01, 5e-8, 4),
    (10, 0.02, 5e-8, 5),
]


def generate_config_lr_linked(lambda_max, gpu_id, run_dir, exp_name):
    orig = os.path.join(PROJECT_ROOT, 'config_snn_training.py')
    with open(orig, 'r') as f:
        content = f.read()

    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'
    )
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        f"conf.exp_set_name='{exp_name}'"
    )
    content = content.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    content = content.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")
    content = content.replace(
        'conf.reg_spike_out_alpha=3  # temperature',
        'conf.reg_spike_out_alpha=4  # temperature'
    )
    content = content.replace(
        'conf.reg_spike_out_const=1E-8',
        f'conf.reg_spike_out_const={lambda_max}'
    )
    content = content.replace(
        '#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
        'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'
    )
    content = content.replace(
        '#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
        'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'
    )
    # Add LR-linked flag
    content = content.replace(
        'conf.sc_loss_scd = False',
        'conf.sc_loss_scd = False\n        conf.reg_spike_lr_linked = True'
    )

    with open(os.path.join(run_dir, 'config_sweep.py'), 'w') as f:
        f.write(content)


def generate_config_adaptive(window, margin, lambda_min, gpu_id, run_dir, exp_name):
    orig = os.path.join(PROJECT_ROOT, 'config_snn_training.py')
    with open(orig, 'r') as f:
        content = f.read()

    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'
    )
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        f"conf.exp_set_name='{exp_name}'"
    )
    content = content.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    content = content.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")
    content = content.replace(
        'conf.reg_spike_out_alpha=3  # temperature',
        'conf.reg_spike_out_alpha=4  # temperature'
    )
    content = content.replace(
        'conf.reg_spike_out_const=1E-8',
        f'conf.reg_spike_out_const=5e-8'
    )
    content = content.replace(
        '#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
        'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'
    )
    content = content.replace(
        '#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
        'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'
    )
    # Add adaptive flags
    adaptive_block = f"""conf.sc_loss_scd = False
        conf.reg_spike_adaptive = True
        conf.reg_spike_adaptive_start_ep = 50
        conf.reg_spike_adaptive_lambda_init = 1e-8
        conf.reg_spike_adaptive_margin = {margin}
        conf.reg_spike_adaptive_step_up = 0.3
        conf.reg_spike_adaptive_step_down = 0.5
        conf.reg_spike_adaptive_lambda_max = 1e-5
        conf.reg_spike_adaptive_lambda_min = {lambda_min}
        conf.reg_spike_adaptive_window = {window}"""
    content = content.replace('conf.sc_loss_scd = False', adaptive_block)

    with open(os.path.join(run_dir, 'config_sweep.py'), 'w') as f:
        f.write(content)


def generate_main(run_dir):
    orig_main = os.path.join(PROJECT_ROOT, 'main_snn_training.py')
    with open(orig_main, 'r') as f:
        content = f.read()
    content = content.replace(
        'from config_snn_training import config',
        'from config_sweep import config'
    )
    with open(os.path.join(run_dir, 'main_sweep.py'), 'w') as f:
        f.write(content)


def main():
    os.makedirs(SWEEP_DIR, exist_ok=True)
    processes = []

    # Method 1: LR-linked
    for lambda_max, gpu_id in LR_LINKED_EXPS:
        exp_name = f'lr-linked-lmax-{lambda_max}'
        run_name = f'lr_linked_lmax{lambda_max}'
        run_dir = os.path.join(SWEEP_DIR, run_name)
        os.makedirs(run_dir, exist_ok=True)

        generate_config_lr_linked(lambda_max, gpu_id, run_dir, exp_name)
        generate_main(run_dir)

        log_file = os.path.join(run_dir, 'train.log')
        print(f'[GPU {gpu_id}] M1-LR-linked lmax={lambda_max} -> {run_dir}')

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

    # Method 2: Improved adaptive
    for window, margin, lambda_min, gpu_id in ADAPTIVE_EXPS:
        exp_name = f'adp-w{window}-m{margin}-lmin{lambda_min}'
        run_name = f'adp_w{window}_m{margin}_lmin{lambda_min}'
        run_dir = os.path.join(SWEEP_DIR, run_name)
        os.makedirs(run_dir, exist_ok=True)

        generate_config_adaptive(window, margin, lambda_min, gpu_id, run_dir, exp_name)
        generate_main(run_dir)

        log_file = os.path.join(run_dir, 'train.log')
        print(f'[GPU {gpu_id}] M2-Adaptive w={window} m={margin} lmin={lambda_min} -> {run_dir}')

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
