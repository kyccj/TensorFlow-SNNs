"""Which half of the brake actually works? (ablation on the L3 headline setting)

The ramp intervention (26-08-17) refuted the "cutting early is what hurts" causal story:
early ramp -0.64 vs late ramp -0.78 at matched final sparsity, i.e. early was if anything
better. So the brake's benefit -- V16-C100 rho=5.8e-4 went from -1.07 (3/3 failures) to +0.13
-- probably does NOT come from the 30-epoch floor. The other half of the brake is the lambda
growth cap (<=1.5x per epoch, active at ALL epochs).

Three runs isolate it, all on that headline setting (V16-C100, rho=5.8e-4, brake ON baseline
= +0.13):

    floor_only   growth cap disabled (cap=1e9), 30ep floor kept
    cap_only     floor disabled (floor=0.0), growth cap kept
    neither      both off  == plain loss-ratio, should reproduce the old -1.07

Reading: whichever ablation loses the +0.13 identifies the working part. If cap_only holds
+0.13 and floor_only collapses, the brake is a growth limiter and the "early speed limit"
framing must be rewritten. Registered before launch in
사전 등록 — 경로 통합 + 브레이크 인과 26-08-16.md (see §브레이크 절제 addendum).
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_brake_ablate')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
RHO = 5.8e-4
ALPHA = 7
RESERVED = {0, 1, 4, 6, 7}   # 0/4: rho-transfer, 1: dcap R19, 6/7: juyun

JOBS = [
    dict(name='ba_floor_only', floor=0.22, cap=1e9),
    dict(name='ba_cap_only',   floor=0.0,  cap=1.5),
    dict(name='ba_neither',    floor=0.0,  cap=1e9),
]


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    subs = [
        ('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
         f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'),
        ("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='bab-{job['name']}'"),
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
        conf.reg_spike_loss_ratio_target = {RHO}
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
        (rf"^\s*conf\.reg_spike_loss_ratio_target = {RHO}$", 'rho'),
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
    print(f'--- brake ablation: {len(pending)} runs ---', flush=True)
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
    print('--- brake ablation drained ---', flush=True)


if __name__ == '__main__':
    main()
