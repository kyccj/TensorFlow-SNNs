"""Fill the Pareto measurement gaps (26-08-14 figure, gray bands) with fixed-lambda points.

    gap                         point(s) queued here
    R19-C10   16.5 ~ 42.1%      lambda = 2e-7, 3e-7   (largest gap: no-loss limit AND cliff
                                onset both hide inside it; also feeds the death-order story)
    V16-C100  43.4 ~ 61.7%      lambda = 7e-7
    V16-C10   37.5 ~ 49.0%      lambda = 4e-7
    R19-C100  29.2 ~ 57.1%      lambda = 2e-7

Fixed-lambda protocol identical to the June sweeps (alpha=7, wta_rev, 310 ep) so points drop
straight onto the existing curves. Runs only on GPUs 3-5 (0-2 hold the spike-target runs,
6-7 are juyun's). Five jobs on three GPUs: the two 34 h R19-C10 points and the 11 h V16-C100
point start first; the remaining two queue behind whichever GPU frees up.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_gapfill')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

ALPHA = 7

JOBS = [
    dict(name='gap_r19c10_2e-7',  model='ResNet19', data='CIFAR10',  lmb='2E-7'),
    dict(name='gap_r19c10_3e-7',  model='ResNet19', data='CIFAR10',  lmb='3E-7'),
    dict(name='gap_v16c100_7e-7', model='VGG16',    data='CIFAR100', lmb='7E-7'),
    dict(name='gap_v16c10_4e-7',  model='VGG16',    data='CIFAR10',  lmb='4E-7'),
    dict(name='gap_r19c100_2e-7', model='ResNet19', data='CIFAR100', lmb='2E-7'),
]

RESERVED = {0, 6, 7}    # 0: spike-target R19 진행 중, 6-7: juyun


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    subs = [
        ('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
         f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'),
        ("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='gapfill-{job['name']}'"),
        ("conf.dataset='CIFAR100'", f"conf.dataset='{job['data']}'"),
        ('conf.reg_spike_out_alpha=3  # temperature',
         f'conf.reg_spike_out_alpha={ALPHA}  # temperature'),
        ('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
         'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'),
        ('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
         'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'),
        ('conf.reg_spike_out_const=1E-8', f"conf.reg_spike_out_const={job['lmb']}"),
    ]
    if job['model'] != 'ResNet19':
        subs.insert(2, ("conf.model='ResNet19'", f"conf.model='{job['model']}'"))
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
        (rf"^\s*conf\.reg_spike_out_const={job['lmb']}$", 'lambda'),
        (r"^\s*conf\.reg_spike_out_wta_rev=True", 'wta_rev'),
        (rf"^\s*conf\.reg_spike_out_alpha={ALPHA}\b", 'alpha'),
        (r"^conf\.train_epoch = 310$", '310ep'),
    ]
    for bad_flag in ['reg_spike_loss_ratio = True', 'reg_spike_starget = True', 'reg_spike_lr_brake = True']:
        if re.search(rf"^\s*conf\.{re.escape(bad_flag)}$", src, re.M):
            raise RuntimeError(f"{job['name']}: {bad_flag} unexpectedly on")
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

    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {job['name']}", flush=True)
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
    import sys
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    jobs = [j for j in JOBS if only is None or j['name'] in only]
    pending, busy, threads = list(jobs), set(), []
    os.makedirs(SWEEP_DIR, exist_ok=True)
    print(f'--- gap-fill: {len(pending)} lambda points, GPUs 3-5 ---', flush=True)
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
            time.sleep(180)
    for t in threads:
        t.join()
    print('--- gap-fill drained ---', flush=True)


if __name__ == '__main__':
    main()
