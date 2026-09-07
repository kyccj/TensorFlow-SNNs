"""
Baseline replicates — to put error bars on the settings that have too few.

Current counts:  VGG-C10 n=6 (spread 0.24%p) | VGG-C100 n=3 (0.76%p)
                 R19-C10 n=2 (0.10%p)        | R19-C100 n=1 (no spread at all)

Every Delta-accuracy claim for R19-C100 currently rests on a single number, and
VGG-C100 has the widest spread of any setting, so those two come first.

reg_spike_log_detail is OFF here: these runs only need best_val_acc and best_s_count,
both of which are collected outside that block, and it costs 24.5% of step time
(measured: 244 -> 184 ms/step). Saves ~8h on each ResNet19 run.
"""

import subprocess
import os
import sys
import threading

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_baseline_more')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'


def make_config(gpu_id, exp_name, model, dataset):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    c = c.replace('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
                  f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"')
    c = c.replace("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='{exp_name}'")
    if model == 'VGG16':
        c = c.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    if dataset == 'CIFAR10':
        c = c.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")
    # no spike regularization at all
    c = c.replace('conf.reg_spike_out=True', 'conf.reg_spike_out=False')
    return c


# gpu -> ordered list of (name, model, dataset)
LANES = {
    5: [('base_r19_c100_run2', 'ResNet19', 'CIFAR100')],                 # n=1 -> 2
    3: [('base_vgg_c100_run4', 'VGG16',    'CIFAR100'),                  # n=3 -> 5
        ('base_vgg_c100_run5', 'VGG16',    'CIFAR100')],
    4: [('base_r19_c10_run3',  'ResNet19', 'CIFAR10')],                  # n=2 -> 3
    1: [('base_r19_c100_run3', 'ResNet19', 'CIFAR100')],                 # n=2 -> 3
}


def run_one(gpu, name, model, dataset):
    d = os.path.join(SWEEP_DIR, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'config_sweep.py'), 'w') as f:
        f.write(make_config(gpu, f'basemore-{name}', model, dataset))
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')) as f:
        m = f.read().replace('from config_snn_training import config', 'from config_sweep import config')
    with open(os.path.join(d, 'main_sweep.py'), 'w') as f:
        f.write(m)

    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'

    print(f'[GPU {gpu}] START {name} ({model}/{dataset}, no reg)', flush=True)
    with open(os.path.join(d, 'train.log'), 'w') as lf:
        rc = subprocess.Popen([PYTHON, os.path.join(d, 'main_sweep.py')], cwd=PROJECT_ROOT,
                              stdout=lf, stderr=subprocess.STDOUT, env=env).wait()
    print(f'[GPU {gpu}] DONE  {name} rc={rc}', flush=True)


def main():
    only = {int(g) for g in sys.argv[1].split(',')} if len(sys.argv) > 1 else None
    os.makedirs(SWEEP_DIR, exist_ok=True)
    ts = []
    for gpu, jobs in LANES.items():
        if only is not None and gpu not in only:
            continue
        t = threading.Thread(target=lambda g=gpu, js=jobs: [run_one(g, *j) for j in js])
        t.start(); ts.append(t)
    print(f'--- {len(ts)} lanes started ---', flush=True)
    for t in ts:
        t.join()
    print('--- all lanes finished ---', flush=True)


if __name__ == '__main__':
    main()
