"""T=3: is T=2 a special case, or does the miss grow smoothly as T falls?

The old accounting (rho=5.8e-4, R reported as one time step) gives, on V16-CIFAR10 and
its three sibling settings:

    T=2   47.3%          <- the only real miss
    T=4   66.3 ~ 70.5%   (five points: four settings, one repeated)
    T=8   63.7%

Excluding T=2 the whole set spans 6.8%p, about 1.6x the 4.2%p replicate noise -- T=8 is
already inside the band. So the open question is not "does T transfer" but "where does it
stop", and T=3 is the one point that separates the two readings:

    lands near 65%   ->  T=2 is degenerate on its own; the rule holds for T>=3
    lands near 55%   ->  the miss grows smoothly below T=4; T=2 is just the far end

Run under the OLD accounting (reg_spike_R_per_step=True), because that is the curve the
five reference points sit on. Mixing it with the corrected accounting would put T=3 on a
different curve and answer nothing. The corrected series (T=2/4/8 at rho=2.32e-3) is a
separate set and still missing its T=8 point.

base_t3 has to be run too -- "% of baseline" needs this T's own reference, and no T=3
baseline exists at any setting.

Expected per-epoch cost sits between the measured T=2 and T=4 runs (55 and ~112 s/epoch
for baselines, 75 and 118 for regularized), so roughly 6-8 h each.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_t3')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

RHO = float(os.environ.get('T3_RHO', '5.8e-4'))   # old-accounting value
ALPHA = 7
T = 3

JOBS = [('base_t3', False), ('lr_t3', True)]

RESERVED = {6, 7}
RESERVED |= {int(x) for x in os.environ.get('T3_SKIP_GPUS', '').split(',') if x.strip()}


def make_config(gpu_id, exp_name, reg_on):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    c = c.replace('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
                  f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"')
    c = c.replace("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='{exp_name}'")
    c = c.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    c = c.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")
    c = c.replace('conf.train_epoch = 310', f'conf.time_step = {T}\nconf.train_epoch = 310')

    if reg_on:
        c = c.replace('conf.reg_spike_out_alpha=3  # temperature',
                      f'conf.reg_spike_out_alpha={ALPHA}  # temperature')
        c = c.replace('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
                      'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)')
        c = c.replace('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
                      'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging')
        block = f"""conf.sc_loss_scd = False
        conf.reg_spike_loss_ratio = True
        conf.reg_spike_loss_ratio_target = {RHO}
        conf.reg_spike_loss_ratio_start_ep = 0
        conf.reg_spike_R_per_step = True"""
        c = c.replace('conf.sc_loss_scd = False', block)
    else:
        c = c.replace('conf.reg_spike_out=True', 'conf.reg_spike_out=False')
    return c


def run_one(gpu, name, reg_on):
    d = os.path.join(SWEEP_DIR, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'config_sweep.py'), 'w') as f:
        f.write(make_config(gpu, f't3-{name}', reg_on))
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')) as f:
        m = f.read().replace('from config_snn_training import config', 'from config_sweep import config')
    with open(os.path.join(d, 'main_sweep.py'), 'w') as f:
        f.write(m)

    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'

    tag = f'rho={RHO:.1e} (old accounting)' if reg_on else 'no reg'
    print(f'[GPU {gpu}] START {name} (V16/CIFAR10, T={T}, {tag})', flush=True)
    with open(os.path.join(d, 'train.log'), 'w') as lf:
        rc = subprocess.Popen([PYTHON, os.path.join(d, 'main_sweep.py')], cwd=PROJECT_ROOT,
                              stdout=lf, stderr=subprocess.STDOUT, env=env).wait()
    print(f'[GPU {gpu}] DONE  {name} rc={rc}', flush=True)


def free_gpus(exclude, max_mib=200):
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            timeout=60).decode()
    except Exception:
        return []
    res = []
    for line in out.strip().split('\n'):
        idx, mem = [x.strip() for x in line.split(',')]
        idx = int(idx)
        if idx in RESERVED or idx in exclude:
            continue
        if int(mem) <= max_mib:
            res.append(idx)
    return res


def main():
    MAX_TAKE = int(os.environ.get('T3_MAX_GPUS', '2'))
    POLL = 300
    os.makedirs(SWEEP_DIR, exist_ok=True)
    pending, busy, threads, prev = list(JOBS), set(), [], set()
    print(f'--- queued {len(pending)} T=3 jobs, rho={RHO:.1e} old accounting ---', flush=True)
    while pending:
        now = set(free_gpus(busy))
        stable = now & prev
        prev = now
        while pending and stable and len(busy) < MAX_TAKE:
            g = min(stable); stable.discard(g); busy.add(g)
            args = pending.pop(0)

            def lane(g=g, a=args):
                try:
                    run_one(g, *a)
                finally:
                    busy.discard(g)
            t = threading.Thread(target=lane); t.start(); threads.append(t)
        if pending:
            print(f'[queue] {len(pending)} waiting, {len(busy)} running, free={sorted(now)}', flush=True)
            time.sleep(POLL)
    for t in threads:
        t.join()
    print('--- T=3 jobs finished ---', flush=True)


if __name__ == '__main__':
    main()
