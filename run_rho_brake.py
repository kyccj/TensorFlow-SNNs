"""Braked loss-ratio at rho=3e-3: does the early-phase speed limit fix the aggressive-regime death?

The 26-08-11 aggressive run (rho=3e-3, no brake) killed V16-C100 outright -- spikes to zero by
ep15, 8.58% final -- and V16-C10 landed at ~26% with -1.46%p after spending 200 epochs damaged.
The 26-08-12 retrospective over 189 completed runs found the one trajectory law that survives
leave-one-setting-out validation: cutting train-mode spikes below 0.19x of their epoch-1 level
inside the first 30 epochs predicts damage with precision 0.94, threshold stable across every
family. Both rho=3e-3 deaths violated it (S30/S1 = 0.000 / 0.071), and so did the only
standard-rho failure (V16-C100 at 5.8e-4: 0.159). Every passing run sat at 0.28+.

The brake (flags reg_spike_lr_brake*) enforces that envelope: while S(e)/S(1) < 0.22 inside the
first 30 reg epochs, lambda decays 0.5x instead of following the loss-ratio formula, and lambda
growth is capped at 1.5x/epoch everywhere to bound the R->0 -> lambda->inf feedback.

Pre-registered readings (added to 사전 등록 노트 §브레이크 26-08-12):
    B1  both runs keep S(e)/S(1) >= ~0.19 through ep30 (brake does its mechanical job)
    B2  V16-C100 does not die: final acc within 4%p of baseline 74.08 (vs -65.5 unbraked)
    B3  V16-C10 beats its unbraked run: Delta-acc > -1.46 at a comparable landing
    B4  landings stay aggressive (<55% of baseline) -- if the brake just suppresses lambda for
        310 epochs and lands ~70%, it "survived" by not regularizing, which is failure
    kill: V16-C100 still loses >5%p, or landings retreat above 70%  -> the brake is not the fix,
          and flat-rho loss-ratio has no aggressive-regime story.

Same protocol as run_rho_aggressive.py otherwise (alpha=7, R_per_step accounting, 310 ep).
SMOKE=1 runs a 4-epoch config on one GPU to verify the brake fields appear in logs.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get('SMOKE', '0') == '1'
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_rho_brake_smoke' if SMOKE else '_rho_brake')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

RHO = 3.0e-3
ALPHA = 7

JOBS = [
    dict(name='brk_v16_c100', model='VGG16', data='CIFAR100'),
    dict(name='brk_v16_c10',  model='VGG16', data='CIFAR10'),
]
if SMOKE:
    JOBS = [dict(name='smoke_v16_c100', model='VGG16', data='CIFAR100')]

RESERVED = {6, 7}     # juyun -- never touch


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    subs = [
        ('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
         f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'),
        ("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='rhobrk-{job['name']}'"),
        ("conf.dataset='CIFAR100'", f"conf.dataset='{job['data']}'"),
        ('conf.reg_spike_out_alpha=3  # temperature',
         f'conf.reg_spike_out_alpha={ALPHA}  # temperature'),
        ('#conf.reg_spike_out_wta_rev=True    # revised WTA',
         'conf.reg_spike_out_wta_rev=True    # revised WTA'),
        ('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
         'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'),
        ('conf.sc_loss_scd = False',
         f"""conf.sc_loss_scd = False
        conf.reg_spike_loss_ratio = True
        conf.reg_spike_loss_ratio_target = {RHO}
        conf.reg_spike_loss_ratio_start_ep = 0
        conf.reg_spike_R_per_step = True
        conf.reg_spike_lr_brake = True"""),
    ]
    if job['model'] != 'ResNet19':
        subs.insert(2, ("conf.model='ResNet19'", f"conf.model='{job['model']}'"))
    if SMOKE:
        subs.append(('conf.train_epoch = 310', 'conf.train_epoch = 4'))
    for old, new in subs:
        if old not in c:
            raise RuntimeError(f"config substitution target missing: {old!r}")
        c = c.replace(old, new, 1)
    return c


def verify(path, job):
    import re
    src = open(path).read()
    want = [
        (rf"^conf\.model='{job['model']}'$", 'model'),
        (rf"^conf\.dataset='{job['data']}'$", 'dataset'),
        (r"^\s*conf\.reg_spike_loss_ratio = True$", 'loss_ratio on'),
        (rf"^\s*conf\.reg_spike_loss_ratio_target = {RHO}$", 'rho'),
        (r"^\s*conf\.reg_spike_lr_brake = True$", 'brake on'),
        (r"^\s*conf\.reg_spike_R_per_step = True$", 'R_per_step'),
        (r"^\s*conf\.reg_spike_out_wta_rev=True", 'wta_rev'),
    ]
    bad = [name for pat, name in want if not __import__('re').search(pat, src, __import__('re').M)]
    if bad:
        raise RuntimeError(f"{job['name']}: config verification failed on {bad}")


def run_one(gpu, job):
    d = os.path.join(SWEEP_DIR, job['name'])
    os.makedirs(d, exist_ok=True)
    cfg = os.path.join(d, 'config_sweep.py')
    with open(cfg, 'w') as f:
        f.write(make_config(gpu, job))
    verify(cfg, job)
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')) as f:
        m = f.read().replace('from config_snn_training import config', 'from config_sweep import config')
    with open(os.path.join(d, 'main_sweep.py'), 'w') as f:
        f.write(m)

    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'

    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {job['name']} "
          f"({job['model']}/{job['data']}, rho={RHO:.1e}, brake ON)", flush=True)
    with open(os.path.join(d, 'train.log'), 'w') as lf:
        rc = subprocess.Popen([PYTHON, os.path.join(d, 'main_sweep.py')], cwd=PROJECT_ROOT,
                              stdout=lf, stderr=subprocess.STDOUT, env=env).wait()
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] DONE  {job['name']} rc={rc}", flush=True)


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
    MAX_TAKE = 1 if SMOKE else 2
    pending, busy, threads = list(JOBS), set(), []
    os.makedirs(SWEEP_DIR, exist_ok=True)
    print(f'--- braked rho={RHO:.1e}: {len(pending)} jobs ---', flush=True)
    while pending:
        for g in free_gpus(busy):
            if not pending or len(busy) >= MAX_TAKE:
                break
            busy.add(g)
            job = pending.pop(0)

            def lane(g=g, j=job):
                try:
                    run_one(g, j)
                finally:
                    busy.discard(g)
            t = threading.Thread(target=lane); t.start(); threads.append(t)
            time.sleep(45)
        if pending:
            time.sleep(120)
    for t in threads:
        t.join()
    print('--- braked rho drained ---', flush=True)


if __name__ == '__main__':
    main()
