"""Plain-softmax weighting (sc_rate = softmax, no 1- inversion) at the SAME lambdas as the
existing 1-softmax vs maxnorm comparison (_compare_maxnorm, VGG16-C10, 310ep).

The `else: sc_rate = sc_norm` branch was unreachable in every run in this repo
(reg_spike_out_sc_wta defaults True, flags.py), so "softmax weighting without the inversion"
has never actually been trained. New flag reg_spike_out_sm_plain reaches it while KEEPING the
wta_rev backward (l2_norm_wta_rev), so the weighting is the only difference vs the wta_rev runs.

Reference points at these lambdas (VGG16-C10, baseline 95.00 / 78,189):
    lambda   1-softmax        maxnorm
    1e-8     -0.09 @105.6%    -0.35 @ 92.7%
    1e-7     -0.14 @ 73.8%    -0.44 @ 72.8%
    3e-7     -0.66 @ 49.0%    -0.81 @ 47.6%
    1e-6     -1.74 @ 25.9%    -11.59 @ 19.3%

Registered expectation: mean(sc_rate) = 1/N under plain softmax (~1.5e-5) vs ~0.9996 under
1-softmax, so equal lambda carries ~65,000x less pressure -- all four runs should land at
~100% of baseline with dacc within baseline noise (+-0.2%p). If any run reduces spikes
meaningfully, the "softmax weighting is inert / only the mean scale matters" mechanism story
is wrong and needs rework.

Runs on the four GPUs freed from _fill3 (user call 08-13: softmax comparison takes priority
over the R19 brake pair and the brake replicates; those requeue later). DVS controls on
GPUs 4-5 keep running.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_softmax_cmp')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

ALPHA = 7
LAMBDAS = ['1E-8', '1E-7', '3E-7', '1E-6']

RESERVED = {6, 7}     # juyun -- never touch


def make_config(gpu_id, lmb):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    subs = [
        ('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
         f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'),
        ("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='smplain-lmb_{lmb.lower()}'"),
        ("conf.model='ResNet19'", "conf.model='VGG16'"),
        ("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'"),
        ('conf.reg_spike_out_alpha=3  # temperature',
         f'conf.reg_spike_out_alpha={ALPHA}  # temperature'),
        # wta_rev ON: keeps the modified backward; sm_plain overrides only the weighting
        ('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
         'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)\n'
         '        conf.reg_spike_out_sm_plain=True   # plain softmax weighting (no 1- inversion)'),
        ('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
         'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'),
        ('conf.reg_spike_out_const=1E-8', f'conf.reg_spike_out_const={lmb}'),
    ]
    for old, new in subs:
        if old not in c:
            raise RuntimeError(f"substitution target missing: {old!r}")
        c = c.replace(old, new, 1)
    return c


def verify(path, lmb):
    import re
    src = open(path).read()
    want = [
        (r"^conf\.model='VGG16'$", 'model'),
        (r"^conf\.dataset='CIFAR10'$", 'dataset'),
        (r"^\s*conf\.reg_spike_out_sm_plain=True", 'sm_plain'),
        (r"^\s*conf\.reg_spike_out_wta_rev=True", 'wta_rev backward'),
        (rf"^\s*conf\.reg_spike_out_const={lmb}$", 'lambda'),
        (rf"^\s*conf\.reg_spike_out_alpha={ALPHA}\b", 'alpha'),
        (r"^conf\.train_epoch = 310$", '310ep'),
    ]
    if re.search(r"^\s*conf\.reg_spike_loss_ratio = True$", src, re.M):
        raise RuntimeError('loss_ratio unexpectedly on')
    bad = [n for p, n in want if not re.search(p, src, re.M)]
    if bad:
        raise RuntimeError(f"lmb {lmb}: config verification failed on {bad}")


def run_one(gpu, lmb):
    name = f'smplain_lmb_{lmb.lower()}'
    d = os.path.join(SWEEP_DIR, name)
    os.makedirs(d, exist_ok=True)
    cfg = os.path.join(d, 'config_sweep.py')
    with open(cfg, 'w') as f:
        f.write(make_config(gpu, lmb))
    verify(cfg, lmb)
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')) as f:
        m = f.read().replace('from config_snn_training import config', 'from config_sweep import config')
    with open(os.path.join(d, 'main_sweep.py'), 'w') as f:
        f.write(m)

    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'

    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {name}", flush=True)
    with open(os.path.join(d, 'train.log'), 'w') as lf:
        rc = subprocess.Popen([PYTHON, os.path.join(d, 'main_sweep.py')], cwd=PROJECT_ROOT,
                              stdout=lf, stderr=subprocess.STDOUT, env=env).wait()
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] DONE  {name} rc={rc}", flush=True)


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
    pending, busy, threads = list(LAMBDAS), set(), []
    os.makedirs(SWEEP_DIR, exist_ok=True)
    print(f'--- plain-softmax comparison: {len(pending)} lambdas ---', flush=True)
    while pending:
        for g in free_gpus(busy):
            if not pending:
                break
            busy.add(g)
            lmb = pending.pop(0)

            def lane(g=g, l=lmb):
                try:
                    run_one(g, l)
                finally:
                    busy.discard(g)
            t = threading.Thread(target=lane); t.start(); threads.append(t)
            time.sleep(45)
        if pending:
            time.sleep(120)
    for t in threads:
        t.join()
    print('--- plain-softmax drained ---', flush=True)


if __name__ == '__main__':
    main()
