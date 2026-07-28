"""
Grow-until-interference adaptive lambda experiments + baseline re-runs.
  GPU 0,1: baseline (no reg) x2 — variance check
  GPU 2: grow default (rate=1.2, decay=0.5, k=2)
  GPU 3: grow slow (rate=1.1)
  GPU 4: grow fast (rate=1.3)
  GPU 5: grow tolerant (k=3)
VGG-C10, 310 epochs
"""

import subprocess
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_grow')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'


def base_config():
    orig = os.path.join(PROJECT_ROOT, 'config_snn_training.py')
    with open(orig, 'r') as f:
        return f.read()


def apply_common(content, gpu_id, exp_name):
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
    return content


def config_baseline(gpu_id, exp_name):
    content = apply_common(base_config(), gpu_id, exp_name)
    # disable reg entirely
    content = content.replace('conf.reg_spike_out=True', 'conf.reg_spike_out=False')
    return content


def config_grow(gpu_id, exp_name, rate, decay, sigma_k):
    content = apply_common(base_config(), gpu_id, exp_name)
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
    block = f"""conf.sc_loss_scd = False
        conf.reg_spike_grow = True
        conf.reg_spike_grow_rate = {rate}
        conf.reg_spike_grow_decay = {decay}
        conf.reg_spike_grow_sigma_k = {sigma_k}"""
    content = content.replace('conf.sc_loss_scd = False', block)
    return content


EXPERIMENTS = [
    # (name, gpu, kind, params)
    ('baseline_run1',  0, 'baseline', None),
    ('baseline_run2',  1, 'baseline', None),
    ('grow_default',   2, 'grow', (1.2, 0.5, 2.0)),
    ('grow_slow',      3, 'grow', (1.1, 0.5, 2.0)),
    ('grow_fast',      4, 'grow', (1.3, 0.5, 2.0)),
    ('grow_tolerant',  5, 'grow', (1.2, 0.5, 3.0)),
]


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

    for name, gpu, kind, params in EXPERIMENTS:
        run_dir = os.path.join(SWEEP_DIR, name)
        os.makedirs(run_dir, exist_ok=True)

        if kind == 'baseline':
            content = config_baseline(gpu, f'grow-sweep-{name}')
        else:
            rate, decay, sigma_k = params
            content = config_grow(gpu, f'grow-sweep-{name}', rate, decay, sigma_k)

        with open(os.path.join(run_dir, 'config_sweep.py'), 'w') as f:
            f.write(content)
        generate_main(run_dir)

        log_file = os.path.join(run_dir, 'train.log')
        print(f'[GPU {gpu}] {name} ({kind}) -> {run_dir}')

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
        processes.append((name, gpu, p, log_file))

    print(f'\n--- {len(processes)} experiments launched ---')
    print('Monitor:')
    print(f'  tail -f {SWEEP_DIR}/*/train.log')
    print(f'\nKill all:')
    print(f'  kill {" ".join(str(p.pid) for _, _, p, _ in processes)}')

    try:
        for name, gpu, p, log_file in processes:
            p.wait()
            status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'
            print(f'[GPU {gpu}] {name} finished: {status}')
    except KeyboardInterrupt:
        print('\nInterrupted - terminating...')
        for _, _, p, _ in processes:
            p.terminate()
        for _, _, p, _ in processes:
            p.wait()
        print('All terminated.')


if __name__ == '__main__':
    main()
