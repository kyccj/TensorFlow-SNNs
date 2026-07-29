"""
Gradient-ratio probe (measure-only): log rho = ||grad(reg)|| / ||grad(task)|| per epoch
while lambda is held at each setting's KNOWN-OPTIMAL fixed value.

Purpose: test whether rho at the optimum is transferable across settings whose optimal
lambda spans 500x (VGG-C10 5e-8, VGG-C100 1e-9, R19-C10 1e-10). If rho* clusters, lambda
can be solved from a dimensionless target instead of swept per model/dataset.

Lambda is NOT modified here -- this run only measures.
"""

import subprocess
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_grad_probe')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'


def base_config():
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py'), 'r') as f:
        return f.read()


def make_config(gpu_id, exp_name, model, dataset, lmbda, epochs):
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
    if dataset == 'CIFAR10':
        content = content.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")
    content = content.replace('conf.train_epoch = 310', f'conf.train_epoch = {epochs}')
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
        conf.reg_spike_out_const = {lmbda}
        conf.reg_spike_grad_ratio_measure = True"""
    content = content.replace('conf.sc_loss_scd = False', block)
    return content


# (name, gpu, model, dataset, lambda, epochs)
#   *_opt  : each setting's known-best lambda (from the 2026-06 sweeps)
#   others : off-optimal points, to map rho -> final accuracy
EXPERIMENTS = [
    ('vgg_c10_opt',   3, 'VGG16',    'CIFAR10',  '5e-8',  310),   # 95.03%  best
    ('r19_c10_opt',   4, 'ResNet19', 'CIFAR10',  '1e-10', 310),   # 96.95%  best
    ('vgg_c100_opt',  5, 'VGG16',    'CIFAR100', '1e-9',  310),   # 74.64%  best
    ('vgg_c10_weak',  0, 'VGG16',    'CIFAR10',  '1e-10', 310),   # 94.93%  ~baseline
    ('vgg_c10_over',  1, 'VGG16',    'CIFAR10',  '1e-6',  310),   # 93.33%  over-regularized
    ('r19_c10_edge',  2, 'ResNet19', 'CIFAR10',  '5e-7',  310),   # 95.74%  near collapse edge
]


def generate_main(run_dir):
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py'), 'r') as f:
        content = f.read()
    content = content.replace(
        'from config_snn_training import config',
        'from config_sweep import config'
    )
    with open(os.path.join(run_dir, 'main_sweep.py'), 'w') as f:
        f.write(content)


def main():
    # optional: restrict to a subset of GPUs, e.g. `python run_grad_ratio_probe.py 3,4,5`
    only_gpus = None
    if len(sys.argv) > 1:
        only_gpus = {int(g) for g in sys.argv[1].split(',')}
    override_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else None

    os.makedirs(SWEEP_DIR, exist_ok=True)
    processes = []

    for name, gpu, model, dataset, lmbda, epochs in EXPERIMENTS:
        if only_gpus is not None and gpu not in only_gpus:
            continue
        if override_epochs is not None:
            epochs = override_epochs
        run_dir = os.path.join(SWEEP_DIR, name)
        os.makedirs(run_dir, exist_ok=True)

        content = make_config(gpu, f'gradprobe-{name}', model, dataset, lmbda, epochs)
        with open(os.path.join(run_dir, 'config_sweep.py'), 'w') as f:
            f.write(content)
        generate_main(run_dir)

        log_file = os.path.join(run_dir, 'train.log')
        print(f'[GPU {gpu}] {name} ({model}/{dataset}, lambda={lmbda}, {epochs}ep) -> {run_dir}')

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

    print(f'\n--- {len(processes)} probes launched ---')

    try:
        for name, gpu, p, log_file in processes:
            p.wait()
            status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'
            print(f'[GPU {gpu}] {name} finished: {status}')
    except KeyboardInterrupt:
        for _, _, p, _ in processes:
            p.terminate()
        for _, _, p, _ in processes:
            p.wait()


if __name__ == '__main__':
    main()
