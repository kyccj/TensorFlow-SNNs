"""Aggressive rho: does loss-ratio still hold together where a single fixed lambda cannot?

Everything measured so far sits in the safe regime. At the two rho values tried (2.6e-4 -> 80%,
5.8e-4 -> 67%) nothing distinguishes loss-ratio from just picking one lambda and using it
everywhere -- and on accuracy the single fixed lambda is actually BETTER:

    single lambda=1e-7   Δacc worst −0.51   landing 42.1 ~ 87.9%  (45.8%p spread)
    loss-ratio rho=5.8e-4  Δacc worst −1.07   landing 66.9 ~ 67.3%  (0.4%p spread)

So the only thing loss-ratio uniquely buys is a predictable landing. That is worth nothing if a
fixed lambda can reach the same sparsity anyway -- which, in the safe regime, it can.

Push harder and the fixed lambda stops working. From the June sweeps, the lambda needed for a
~40% landing spans 10x across settings (1e-7 for R19-C10, ~1e-6 for V16-C100), and at 1e-6
R19-C10 collapses to 85.36% (−11.29%p) while V16-C100 has only reached 43%. No single value
survives that. This is where loss-ratio should earn its keep, and it is exactly where it has
never been run.

rho = 3.0e-3 from a log-linear extrapolation of the two measured points (80.6% and 67.1%,
slope −16.8 %p per ln unit). That is a 5x extrapolation from n=2, so the landing may well miss;
the pre-registration separates "landing missed" (P1, not fatal) from "spread blew up" (D1, fatal).

Predictions are registered in the Obsidian note before launch:
    01.Projects/EIP/138Server/사전 등록 — 공격적 rho 26-08-11.md

    P1  all four land 32~48%
    P2  landing spread < 20%p           (single lambda=1e-7 gives 45.8%p)
    P3  worst Δacc >= −4.0%p            (fixed lambda at matched landing gives −3.85%p on V16-C100)
    P4  R19-C10 does not collapse (Δacc > −1.5%p)
    P5  any failing setting shows final/min > 1.05; passing settings stay < 1.05
        -- first test of the baseline-free self-diagnosis signal

    D1  spread > 20%p           -> the one remaining claim dies
    D2  two or more settings lose > 5%p -> less safe than a single fixed lambda
    D3  P5 refuted             -> drop the trajectory signal

reg_spike_R_per_step=True reproduces the accounting every rho=5.8e-4 result was measured under
(sc_loss_snap holding one time step rather than the sum over T). It is wrong on its face and is
kept only so these points land on the same curve -- see flags.py:853.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_rho_agg')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

RHO = 3.0e-3
ALPHA = 7

# longest first -- the R19 runs are ~33.6 h each and set the critical path, the VGGs are ~10 h
# and can share whatever slot frees up
JOBS = [
    dict(name='agg_r19_c10',  model='ResNet19', data='CIFAR10'),
    dict(name='agg_r19_c100', model='ResNet19', data='CIFAR100'),
    dict(name='agg_v16_c100', model='VGG16',    data='CIFAR100'),
    dict(name='agg_v16_c10',  model='VGG16',    data='CIFAR10'),
]

RESERVED = {6, 7}     # juyun (LLaDA/opencompass) -- never touch
RESERVED |= {int(x) for x in os.environ.get('RA_SKIP_GPUS', '').split(',') if x.strip()}


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    subs = [
        ('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
         f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'),
        ("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='rhoagg-{job['name']}'"),
        ("conf.dataset='CIFAR100'", f"conf.dataset='{job['data']}'"),
        ('conf.reg_spike_out_alpha=3  # temperature',
         f'conf.reg_spike_out_alpha={ALPHA}  # temperature'),
        # wta_rev is commented out in the base config but ON in every loss-ratio run so far
        ('#conf.reg_spike_out_wta_rev=True    # revised WTA',
         'conf.reg_spike_out_wta_rev=True    # revised WTA'),
        ('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
         'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'),
        ('conf.sc_loss_scd = False',
         f"""conf.sc_loss_scd = False
        conf.reg_spike_loss_ratio = True
        conf.reg_spike_loss_ratio_target = {RHO}
        conf.reg_spike_loss_ratio_start_ep = 0
        conf.reg_spike_R_per_step = True"""),
    ]
    if job['model'] != 'ResNet19':
        subs.insert(2, ("conf.model='ResNet19'", f"conf.model='{job['model']}'"))

    for old, new in subs:
        if old not in c:
            raise RuntimeError(f"config substitution target missing: {old!r}")
        c = c.replace(old, new, 1)
    return c


def verify(path, job):
    """Catch a silently mangled config before burning 34 h on it."""
    import re
    src = open(path).read()
    want = [
        (rf"^conf\.model='{job['model']}'$", 'model'),
        (rf"^conf\.dataset='{job['data']}'$", 'dataset'),
        (r"^\s*conf\.reg_spike_loss_ratio = True$", 'loss_ratio on'),
        (rf"^\s*conf\.reg_spike_loss_ratio_target = {RHO}$", 'rho'),
        (r"^\s*conf\.reg_spike_R_per_step = True$", 'R_per_step'),
        (r"^\s*conf\.reg_spike_out_wta_rev=True", 'wta_rev'),
        (rf"^\s*conf\.reg_spike_out_alpha={ALPHA}\b", 'alpha'),
        (r"^conf\.train_epoch = 310$", '310 epochs'),
    ]
    bad = [name for pat, name in want if not re.search(pat, src, re.M)]
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
          f"({job['model']}/{job['data']}, rho={RHO:.1e})", flush=True)
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
    MAX_TAKE = int(os.environ.get('RA_MAX_GPUS', '4'))
    POLL = int(os.environ.get('RA_POLL', '180'))
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    jobs = [j for j in JOBS if only is None or j['name'] in only]

    os.makedirs(SWEEP_DIR, exist_ok=True)
    pending, busy, threads = list(jobs), set(), []
    print(f'--- rho={RHO:.1e} aggressive: {len(pending)} jobs, max {MAX_TAKE} concurrent ---',
          flush=True)
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
            time.sleep(45)      # let the process claim its memory before polling again
        if pending:
            time.sleep(POLL)
    for t in threads:
        t.join()
    print('--- rho aggressive drained ---', flush=True)


if __name__ == '__main__':
    main()
