"""T axis, re-measured after the R accounting fix.

The first pass at this (run_timestep.py) found that rho=5.8e-4 did not survive a change in
T: 47.3% at T=2, 69.5% at T=4, 63.7% at T=8 against a 70% target, and the miss was not
monotone. The cause turned out to be a measurement bug rather than the rule -- add_loss
fires once per time step, so the loss received sum_t ||x_t||, while sc_loss_snap kept only
the last step's term. R was under-reported by roughly T, making the effective ratio T*rho
while the controller believed it was holding rho.

neurons.py now accumulates the snapshot across time steps. Measured on a 2-epoch V16-C10
T=4 run: reg_R 9,140 -> 33,594, a factor of 3.68 (not exactly 4 because the first time step
fires less, so the per-step norms are not equal).

Under the corrected definition R tracks spike count instead of moving against it:

    R_measured/spikes across T=2/4/8   varied 4.3x   (old)
    R_total/spikes across T=2/4/8      varies 1.3x   (fixed)

which is the reason to expect T to transfer now.

rho is restated on the corrected R: 4 x 5.8e-4 = 2.32e-3, T=4 being where it was
calibrated. T=4 is re-run rather than reused -- the old number came from the old code, and
the empirical factor was 3.68 rather than 4, so the reference has to be measured again
under the definition the other two points use.

Baselines are reused; they carry no regularization and the fix cannot touch them.

    T=2   base_t2                          46,215 spikes
    T=4   _baseline_no_reg* (n=4 mean)     78,189
    T=8   base_t8                         101,068
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_timestep_fix')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

RHO = float(os.environ.get('TSF_RHO', '2.32e-3'))
ALPHA = 7

JOBS = [
    ('lr_t2_fix', 2),
    ('lr_t4_fix', 4),
    ('lr_t8_fix', 8),
]

RESERVED = {6, 7}
RESERVED |= {int(x) for x in os.environ.get('TSF_SKIP_GPUS', '').split(',') if x.strip()}


def make_config(gpu_id, exp_name, T):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    c = c.replace('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
                  f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"')
    c = c.replace("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='{exp_name}'")
    c = c.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    c = c.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")
    c = c.replace('conf.train_epoch = 310', f'conf.time_step = {T}\nconf.train_epoch = 310')
    c = c.replace('conf.reg_spike_out_alpha=3  # temperature',
                  f'conf.reg_spike_out_alpha={ALPHA}  # temperature')
    c = c.replace('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
                  'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)')
    c = c.replace('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
                  'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging')
    block = f"""conf.sc_loss_scd = False
        conf.reg_spike_loss_ratio = True
        conf.reg_spike_loss_ratio_target = {RHO}
        conf.reg_spike_loss_ratio_start_ep = 0"""
    c = c.replace('conf.sc_loss_scd = False', block)
    return c


def run_one(gpu, name, T):
    d = os.path.join(SWEEP_DIR, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'config_sweep.py'), 'w') as f:
        f.write(make_config(gpu, f'tsf-{name}', T))
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')) as f:
        m = f.read().replace('from config_snn_training import config', 'from config_sweep import config')
    with open(os.path.join(d, 'main_sweep.py'), 'w') as f:
        f.write(m)

    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'

    print(f'[GPU {gpu}] START {name} (V16/CIFAR10, T={T}, rho={RHO:.3e} on corrected R)', flush=True)
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
    MAX_TAKE = int(os.environ.get('TSF_MAX_GPUS', '2'))
    POLL = 300
    os.makedirs(SWEEP_DIR, exist_ok=True)
    pending, busy, threads, prev = list(JOBS), set(), [], set()
    print(f'--- queued {len(pending)} T jobs at rho={RHO:.3e}, max {MAX_TAKE} GPUs ---', flush=True)
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
    print('--- T jobs finished ---', flush=True)


if __name__ == '__main__':
    main()
