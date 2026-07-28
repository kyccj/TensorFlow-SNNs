"""
LR-linked lambda v2 experiments:
  1. Power schedule (GPU 0,1,2): concentrate reg at end
  2. Delayed start (GPU 3,4): start ramp at ep 150
  3. Hybrid safety (GPU 5): LR-linked + adaptive safety valve
VGG-C10, 310 epochs
"""

import subprocess
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_lr_linked_v2')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

# (name, gpu, lmax, power, start_ep, safety)
EXPERIMENTS = [
    ('power2_lmax5e-7',  0, 5e-7, 2.0, 20,  False),
    ('power3_lmax5e-7',  1, 5e-7, 3.0, 20,  False),
    ('power2_lmax1e-6',  2, 1e-6, 2.0, 20,  False),
    ('delay150_lmax5e-7', 3, 5e-7, 1.0, 150, False),
    ('delay150_lmax1e-6', 4, 1e-6, 1.0, 150, False),
    ('hybrid_lmax5e-7',  5, 5e-7, 1.0, 20,  True),
]


def generate_config(name, gpu_id, lmax, power, start_ep, safety, run_dir):
    orig = os.path.join(PROJECT_ROOT, 'config_snn_training.py')
    with open(orig, 'r') as f:
        content = f.read()

    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'
    )
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        f"conf.exp_set_name='lr-linked-v2-{name}'"
    )
    content = content.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    content = content.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")
    content = content.replace(
        'conf.reg_spike_out_alpha=3  # temperature',
        'conf.reg_spike_out_alpha=4  # temperature'
    )
    content = content.replace(
        'conf.reg_spike_out_const=1E-8',
        f'conf.reg_spike_out_const={lmax}'
    )
    content = content.replace(
        '#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
        'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'
    )
    content = content.replace(
        '#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
        'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'
    )

    lr_linked_block = f"""conf.sc_loss_scd = False
        conf.reg_spike_lr_linked = True
        conf.reg_spike_lr_linked_power = {power}
        conf.reg_spike_lr_linked_start_ep = {start_ep}"""
    if safety:
        lr_linked_block += '\n        conf.reg_spike_lr_linked_safety = True'
    content = content.replace('conf.sc_loss_scd = False', lr_linked_block)

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

    for name, gpu_id, lmax, power, start_ep, safety in EXPERIMENTS:
        run_dir = os.path.join(SWEEP_DIR, name)
        os.makedirs(run_dir, exist_ok=True)

        generate_config(name, gpu_id, lmax, power, start_ep, safety, run_dir)
        generate_main(run_dir)

        log_file = os.path.join(run_dir, 'train.log')
        desc = f'p={power} start={start_ep}'
        if safety:
            desc += ' +safety'
        print(f'[GPU {gpu_id}] {name} lmax={lmax} {desc} -> {run_dir}')

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
        processes.append((name, gpu_id, p, log_file))

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
