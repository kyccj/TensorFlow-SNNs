"""Replicates to settle whether the brake does anything at all (V16-C100, rho=5.8e-4).

The 08-18 ablation turned out to be invalid: brake_on fired 0 times in ALL three arms
(floor never reached, growth cap set to 1e9 in two of them), so floor_only and neither were
both plain loss-ratio -- functionally identical configs that nonetheless landed +0.27 and
-2.08. Pooling them with the three original loss-ratio runs gives

    plain loss-ratio, n=5:  -0.91, -0.89, -1.41, +0.27, -2.08   (mean -1.00, sd 0.77)

The braked headline (+0.13, n=1) sits inside that range, so the brake's effect is NOT
established. Only rl_c100_58_br actually exercised the brake (11 activations).

This runs 2 braked + 2 plain replicates. With sd ~0.8 and n=3 vs n=7, a real 1%p effect is
detectable; if the braked arm's mean lands inside the plain arm's spread, the brake claim --
and the "-1.07 -> +0.13 fix" that anchors the current narrative -- has to be withdrawn.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_brake_rep')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
ALPHA = 7
RESERVED = {0, 1, 2, 3, 4, 6, 7}   # only GPU 5   # 1: tight cap, 4: SF transfer, 5: brake_rep running, 6/7: juyun

JOBS = [
    dict(name='br_on_r4',  rho=5.8e-4, floor=0.22, cap=1.5),
]


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    subs = [
        ('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
         f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'),
        ("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='brp-{job['name']}'"),
        ("conf.model='ResNet19'", "conf.model='VGG16'"),
        ("conf.dataset='CIFAR100'", "conf.dataset='CIFAR100'"),
        ('conf.reg_spike_out_alpha=3  # temperature',
         f'conf.reg_spike_out_alpha={ALPHA}  # temperature'),
        ('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
         'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'),
        ('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
         'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'),
        ('conf.sc_loss_scd = False',
         f"""conf.sc_loss_scd = False
        conf.reg_spike_loss_ratio = True
        conf.reg_spike_loss_ratio_target = {job['rho']}
        conf.reg_spike_loss_ratio_start_ep = 0
        conf.reg_spike_R_per_step = True
        conf.reg_spike_lr_brake = True
        conf.reg_spike_lr_brake_floor = {job['floor']}
        conf.reg_spike_lr_growth_cap = {job['cap']}"""),
    ]
    for old, new in subs:
        if old not in c:
            raise RuntimeError(f"substitution target missing: {old!r}")
        c = c.replace(old, new, 1)
    return c


def verify(path, job):
    import re
    src = open(path).read()
    want = [
        (r"^conf\.model='VGG16'$", 'model'),
        (r"^conf\.dataset='CIFAR100'$", 'dataset'),
        (r"^\s*conf\.reg_spike_loss_ratio = True$", 'loss_ratio'),
        (rf"^\s*conf\.reg_spike_loss_ratio_target = {job['rho']}$", 'rho'),
        (r"^\s*conf\.reg_spike_lr_brake = True$", 'brake on'),
        (rf"^\s*conf\.reg_spike_lr_brake_floor = {job['floor']}$", 'floor'),
        (rf"^\s*conf\.reg_spike_lr_growth_cap = {job['cap']}$", 'cap'),
        (r"^conf\.train_epoch = 310$", '310ep'),
    ]
    if re.search(r"^\s*conf\.reg_spike_starget = True$", src, re.M):
        raise RuntimeError('starget unexpectedly on')
    bad = [n for p, n in want if not re.search(p, src, re.M)]
    if bad:
        raise RuntimeError(f"{job['name']}: verification failed on {bad}")


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
          f"(floor={job['floor']}, cap={job['cap']})", flush=True)
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
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    jobs = [j for j in JOBS if only is None or j['name'] in only]
    pending, busy, threads = list(jobs), set(), []
    os.makedirs(SWEEP_DIR, exist_ok=True)
    print(f'--- brake replicates: {len(pending)} runs ---', flush=True)
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
            time.sleep(240)
    for t in threads:
        t.join()
    print('--- brake replicates drained ---', flush=True)


if __name__ == '__main__':
    main()
