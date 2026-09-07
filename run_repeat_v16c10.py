"""Two more V16-CIFAR10 loss-ratio runs, to put a number on replicate noise.

The claim that carries the method is "with rho fixed, the landing spread across settings
(4.2%p) is the same size as the spread you get from re-running one setting". The second
half of that sentence currently rests on a single pair -- V16-C100 at 66.3% and 70.5%.
An n=2 spread is a weak estimate of anything, and it is doing load-bearing work.

V16-C10 is the cheapest setting (~10 h against ResNet19's ~32 h) and already has one
loss-ratio run at 69.5%, so two more give a second n=3 estimate for 20 GPU-hours.

For contrast, fixed-lambda replicate noise was measured at 5.6%p on this same setting
(69.3 / 72.4 / 74.9, experiment 1). If loss-ratio comes out tighter than that, it is worth
saying; if it comes out similar, the honest claim is just that both are noisy at this scale
and the across-setting spread is not distinguishable from either.

Old accounting (reg_spike_R_per_step=True, rho=5.8e-4) -- these have to be replicates of
lr_vgg_c10_rho58, which predates the R fix. Anything else is a different experiment.

Seeds are not set anywhere in this repo, so re-running the same config is already an
independent draw; the existing baseline repeats (identical configs, 94.97 vs 95.11) confirm
that.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_repeat_v16c10')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

RHO = 5.8e-4
ALPHA = 7

JOBS = ['lr_v16c10_run2', 'lr_v16c10_run3']

RESERVED = {6, 7}
RESERVED |= {int(x) for x in os.environ.get('RPT_SKIP_GPUS', '').split(',') if x.strip()}


def make_config(gpu_id, exp_name):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    c = c.replace('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
                  f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"')
    c = c.replace("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='{exp_name}'")
    c = c.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    c = c.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")
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
    return c


def run_one(gpu, name):
    d = os.path.join(SWEEP_DIR, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'config_sweep.py'), 'w') as f:
        f.write(make_config(gpu, f'rpt-{name}'))
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')) as f:
        m = f.read().replace('from config_snn_training import config', 'from config_sweep import config')
    with open(os.path.join(d, 'main_sweep.py'), 'w') as f:
        f.write(m)

    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'

    print(f'[GPU {gpu}] START {name} (V16/CIFAR10, rho={RHO:.1e}, old accounting)', flush=True)
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
    MAX_TAKE = int(os.environ.get('RPT_MAX_GPUS', '2'))
    POLL = 300
    os.makedirs(SWEEP_DIR, exist_ok=True)
    pending, busy, threads, prev = list(JOBS), set(), [], set()
    print(f'--- queued {len(pending)} repeat runs, max {MAX_TAKE} GPUs ---', flush=True)
    while pending:
        now = set(free_gpus(busy))
        stable = now & prev
        prev = now
        while pending and stable and len(busy) < MAX_TAKE:
            g = min(stable); stable.discard(g); busy.add(g)
            name = pending.pop(0)

            def lane(g=g, n=name):
                try:
                    run_one(g, n)
                finally:
                    busy.discard(g)
            t = threading.Thread(target=lane); t.start(); threads.append(t)
        if pending:
            print(f'[queue] {len(pending)} waiting, {len(busy)} running, free={sorted(now)}', flush=True)
            time.sleep(POLL)
    for t in threads:
        t.join()
    print('--- repeat runs finished ---', flush=True)


if __name__ == '__main__':
    main()
