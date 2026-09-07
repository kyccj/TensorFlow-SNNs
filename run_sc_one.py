"""sc_rate = 1 (constant) at the standard lambda ladder: is softmax vestigial?

1-softmax measures sc_rate mean 0.9999 with ~0 spread (26-08-14 table), so replacing it with
the constant 1 -- keeping the wta_rev backward -- should reproduce the 1-softmax results
exactly:

    lambda   1-softmax reference (V16-C10)
    1e-8     -0.09 @ 105.6%
    1e-7     -0.14 @  73.8%
    3e-7     -0.66 @  49.0%
    1e-6     -1.74 @  25.9%

Registered prediction: all four sc_one runs land within replicate noise (dacc +-0.3%p,
remnant +-5%p) of those references. If they do, the method's effective content is
"uniform-weight L2 with gradient to silent neurons" and softmax can be dropped from the
paper's method description. If any lambda diverges materially, the 0.0001-scale sc_rate
structure somehow matters and the simplification is wrong.

GPU etiquette: all six of our GPUs are busy (0-2 spike-target, 3-5 gap-fill with 2 queued
jobs). This launcher WAITS until the gap-fill launcher has started all 5 of its jobs before
claiming anything, so the two pollers never race for a freed GPU.
"""

import subprocess
import os
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_sc_one')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

ALPHA = 7
LAMBDAS = ['1E-8', '1E-7', '3E-7', '1E-6']
RESERVED = {6, 7}
GAPFILL_LOG = os.path.join(PROJECT_ROOT, '_gapfill_launcher.log')


def gapfill_done_claiming():
    try:
        return open(GAPFILL_LOG).read().count('START') >= 4
    except Exception:
        return True


def make_config(gpu_id, lmb):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    subs = [
        ('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
         f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'),
        ("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='scone-lmb_{lmb.lower()}'"),
        ("conf.model='ResNet19'", "conf.model='VGG16'"),
        ("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'"),
        ('conf.reg_spike_out_alpha=3  # temperature',
         f'conf.reg_spike_out_alpha={ALPHA}  # temperature'),
        ('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
         'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)\n'
         '        conf.reg_spike_out_sc_one=True     # constant coefficient 1 (softmax-vestigial test)'),
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
        (r"^\s*conf\.reg_spike_out_sc_one=True", 'sc_one'),
        (r"^\s*conf\.reg_spike_out_wta_rev=True", 'wta_rev backward'),
        (rf"^\s*conf\.reg_spike_out_const={lmb}$", 'lambda'),
        (r"^conf\.train_epoch = 310$", '310ep'),
    ]
    for flag in ['reg_spike_loss_ratio = True', 'reg_spike_starget = True', 'reg_spike_out_sm_plain=True']:
        if re.search(rf"^\s*conf\.{re.escape(flag)}", src, re.M):
            raise RuntimeError(f'{flag} unexpectedly on')
    bad = [n for p, n in want if not re.search(p, src, re.M)]
    if bad:
        raise RuntimeError(f"lmb {lmb}: config verification failed on {bad}")


def run_one(gpu, lmb):
    name = f'scone_lmb_{lmb.lower()}'
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
    print(f'--- sc_rate=1: {len(pending)} lambdas, waiting behind gap-fill queue ---', flush=True)
    while pending:
        if gapfill_done_claiming():
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
            time.sleep(300)
    for t in threads:
        t.join()
    print('--- sc_rate=1 drained ---', flush=True)


if __name__ == '__main__':
    main()
