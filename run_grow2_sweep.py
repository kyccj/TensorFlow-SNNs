"""
Grow-until-interference v2 (fixed): correct reg subtraction + anchored min-EMA reference.
  GPU 0-3: VGG-C10 variants (default / slow / fast / tolerant)
  GPU 4-5: ResNet19-C10 (default / tolerant) — generality check, same fixed constants
310 epochs
"""

import subprocess
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_grow2')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'


def base_config():
    orig = os.path.join(PROJECT_ROOT, 'config_snn_training.py')
    with open(orig, 'r') as f:
        return f.read()


def make_config(gpu_id, exp_name, model, rate, decay, sigma_k):
    content = base_config()
    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'
    )
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        f"conf.exp_set_name='{exp_name}'"
    )
    if model == 'VGG16':
        content = content.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    content = content.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")
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
    # (name, gpu, model, rate, decay, sigma_k)
    ('vgg_default',   0, 'VGG16',    1.2, 0.5, 2.0),
    ('vgg_slow',      1, 'VGG16',    1.1, 0.5, 2.0),
    ('vgg_fast',      2, 'VGG16',    1.3, 0.5, 2.0),
    ('vgg_tolerant',  3, 'VGG16',    1.2, 0.5, 3.0),
    ('r19_default',   4, 'ResNet19', 1.2, 0.5, 2.0),
    ('r19_tolerant',  5, 'ResNet19', 1.2, 0.5, 3.0),
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

    for name, gpu, model, rate, decay, sigma_k in EXPERIMENTS:
        run_dir = os.path.join(SWEEP_DIR, name)
        os.makedirs(run_dir, exist_ok=True)

        content = make_config(gpu, f'grow2-{name}', model, rate, decay, sigma_k)
        with open(os.path.join(run_dir, 'config_sweep.py'), 'w') as f:
            f.write(content)
        generate_main(run_dir)

        log_file = os.path.join(run_dir, 'train.log')
        print(f'[GPU {gpu}] {name} ({model}, rate={rate}, k={sigma_k}) -> {run_dir}')

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
