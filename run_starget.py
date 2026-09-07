"""Direct spike-target control vs loss-ratio: the "why loss share?" head-to-head.

Pre-registered: 사전 등록 — 직접 제어 vs 손실비 26-08-14.md (Obsidian, before launch).

lambda <- lambda * clip((S/S_target)^1, 0.5, 1.5), lambda0=1e-9, same brake as loss-ratio
(floor 0.22 / 30 ep window / 1.5x growth cap), so the ONLY difference vs the braked rho=3e-3
runs is the knob: loss share vs spike target. Targets are handed the right answer -- the
braked loss-ratio runs' own measured final S/S1 -- so the test is maximally generous to
direct control: "given the correct target, does it match loss-ratio's accuracy?"

    setting    frac    compare against (braked loss-ratio dacc)
    V16-C10    0.096   -1.08
    V16-C100   0.134   -2.33
    R19-C10    0.073   -0.51 (unbraked aggressive; no braked R19 exists)

SMOKE=1 runs V16-C100 for 4 epochs first (new controller code path).
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get('SMOKE', '0') == '1'
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_starget_smoke' if SMOKE else '_starget')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

ALPHA = 7

JOBS = [
    dict(name='st_r19_c10',  model='ResNet19', data='CIFAR10',  frac=0.073),
    dict(name='st_v16_c100', model='VGG16',    data='CIFAR100', frac=0.134),
    dict(name='st_v16_c10',  model='VGG16',    data='CIFAR10',  frac=0.096),
]
if SMOKE:
    JOBS = [dict(name='smoke_st_v16_c100', model='VGG16', data='CIFAR100', frac=0.134)]

RESERVED = {6, 7}


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    subs = [
        ('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
         f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'),
        ("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='starget-{job['name']}'"),
        ("conf.dataset='CIFAR100'", f"conf.dataset='{job['data']}'"),
        ('conf.reg_spike_out_alpha=3  # temperature',
         f'conf.reg_spike_out_alpha={ALPHA}  # temperature'),
        ('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
         'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'),
        ('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
         'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'),
        ('conf.sc_loss_scd = False',
         f"""conf.sc_loss_scd = False
        conf.reg_spike_starget = True
        conf.reg_spike_starget_frac = {job['frac']}
        conf.reg_spike_lr_brake = True"""),
    ]
    if job['model'] != 'ResNet19':
        subs.insert(2, ("conf.model='ResNet19'", f"conf.model='{job['model']}'"))
    if SMOKE:
        subs.append(('conf.train_epoch = 310', 'conf.train_epoch = 4'))
    for old, new in subs:
        if old not in c:
            raise RuntimeError(f"substitution target missing: {old!r}")
        c = c.replace(old, new, 1)
    return c


def verify(path, job):
    import re
    src = open(path).read()
    want = [
        (rf"^conf\.model='{job['model']}'$", 'model'),
        (rf"^conf\.dataset='{job['data']}'$", 'dataset'),
        (r"^\s*conf\.reg_spike_starget = True$", 'starget on'),
        (rf"^\s*conf\.reg_spike_starget_frac = {job['frac']}$", 'frac'),
        (r"^\s*conf\.reg_spike_lr_brake = True$", 'brake'),
        (r"^\s*conf\.reg_spike_out_wta_rev=True", 'wta_rev'),
        (rf"^\s*conf\.reg_spike_out_alpha={ALPHA}\b", 'alpha'),
    ]
    if re.search(r"^\s*conf\.reg_spike_loss_ratio = True$", src, re.M):
        raise RuntimeError('loss_ratio unexpectedly on')
    bad = [n for p, n in want if not re.search(p, src, re.M)]
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

    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {job['name']} (frac={job['frac']})", flush=True)
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
    pending, busy, threads = list(JOBS), set(), []
    os.makedirs(SWEEP_DIR, exist_ok=True)
    print(f'--- spike-target control: {len(pending)} jobs ---', flush=True)
    while pending:
        for g in free_gpus(busy):
            if not pending:
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
    print('--- spike-target drained ---', flush=True)


if __name__ == '__main__':
    main()
