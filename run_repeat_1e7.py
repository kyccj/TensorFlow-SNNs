"""
Replicate runs for the fixed-lambda transfer claim (lambda = 1e-7).

Each of the 4 settings currently has n=1 at lambda=1e-7, and the reported deltas
(-0.06 .. -0.51 %p) are inside baseline run-to-run spread. This adds replicates so
the claim table has error bars instead of single samples.

Config matches the June sweeps exactly: alpha=7, wta_rev=True, log_detail=True, 310ep.
There is no seed flag in this codebase -- run-to-run variance comes from cuDNN/shuffle
nondeterminism, which is exactly the variance being quantified.

Lanes run sequentially within a GPU, in parallel across GPUs.
"""

import subprocess
import os
import sys
import threading

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_repeat_1e7')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

LAMBDA = '1e-07'
ALPHA = 7


def base_config():
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py'), 'r') as f:
        return f.read()


def make_config(gpu_id, exp_name, model, dataset, reg_on):
    c = base_config()
    c = c.replace('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
                  f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"')
    c = c.replace("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='{exp_name}'")
    if model == 'VGG16':
        c = c.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    if dataset == 'CIFAR10':
        c = c.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")

    c = c.replace('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
                  'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging')

    if reg_on:
        c = c.replace('conf.reg_spike_out_const=1E-8', f'conf.reg_spike_out_const={LAMBDA}')
        c = c.replace('conf.reg_spike_out_alpha=3  # temperature',
                      f'conf.reg_spike_out_alpha={ALPHA}  # temperature')
        c = c.replace('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
                      'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)')
    else:
        # baseline: no spike regularization at all
        c = c.replace('conf.reg_spike_out=True', 'conf.reg_spike_out=False')
    return c


# lane -> ordered list of (name, model, dataset, reg_on)
LANES = {
    0: [('vgg_c10_1e7_run2',   'VGG16',    'CIFAR10',  True),
        ('vgg_c10_1e7_run3',   'VGG16',    'CIFAR10',  True)],
    1: [('vgg_c100_1e7_run2',  'VGG16',    'CIFAR100', True),
        ('vgg_c100_1e7_run3',  'VGG16',    'CIFAR100', True)],
    2: [('r19_c10_1e7_run2',   'ResNet19', 'CIFAR10',  True)],
    5: [('r19_c100_1e7_run2',  'ResNet19', 'CIFAR100', True),
        ('r19_c100_base_run2', 'ResNet19', 'CIFAR100', False)],
}


def generate_main(run_dir):
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py'), 'r') as f:
        content = f.read()
    content = content.replace('from config_snn_training import config',
                              'from config_sweep import config')
    with open(os.path.join(run_dir, 'main_sweep.py'), 'w') as f:
        f.write(content)


def run_one(gpu, name, model, dataset, reg_on):
    run_dir = os.path.join(SWEEP_DIR, name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config_sweep.py'), 'w') as f:
        f.write(make_config(gpu, f'repeat-{name}', model, dataset, reg_on))
    generate_main(run_dir)

    env = os.environ.copy()
    env['PYTHONPATH'] = run_dir + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'

    tag = 'reg1e-7' if reg_on else 'baseline'
    print(f'[GPU {gpu}] START {name} ({model}/{dataset}, {tag})', flush=True)
    with open(os.path.join(run_dir, 'train.log'), 'w') as lf:
        p = subprocess.Popen([PYTHON, os.path.join(run_dir, 'main_sweep.py')],
                             cwd=PROJECT_ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env)
        rc = p.wait()
    print(f'[GPU {gpu}] DONE  {name} rc={rc}', flush=True)
    return rc


def lane_worker(gpu, jobs):
    for name, model, dataset, reg_on in jobs:
        run_one(gpu, name, model, dataset, reg_on)


def main():
    only = None
    if len(sys.argv) > 1:
        only = {int(g) for g in sys.argv[1].split(',')}
    os.makedirs(SWEEP_DIR, exist_ok=True)

    threads = []
    for gpu, jobs in LANES.items():
        if only is not None and gpu not in only:
            continue
        t = threading.Thread(target=lane_worker, args=(gpu, jobs), daemon=False)
        t.start()
        threads.append(t)
    print(f'--- {len(threads)} lanes started ---', flush=True)
    for t in threads:
        t.join()
    print('--- all lanes finished ---', flush=True)


if __name__ == '__main__':
    main()
