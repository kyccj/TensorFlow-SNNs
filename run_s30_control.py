"""S30 threshold control: does crossing the early threshold cause the damage?

Pre-registered: 사전 등록 — S30 임계 통제 26-08-18.md (before launch).

The 08-17 ramp pair failed as a causal test because BOTH stayed above the 0.19 threshold
(early ep0->30 landed at S30=0.195, late ep50->80 at 0.343). This pair fixes that: both ramps
start at ep0 and differ only in speed, placing S30 on either side of the threshold while the
final lambda (5e-7) -- hence final sparsity -- is identical.

    deep     ep0 -> ep12   expect S30 ~ 0.12  (below threshold)
    shallow  ep0 -> ep60   expect S30 ~ 0.35  (above)

S3: deep worse by >=0.5%p => the early speed limit is causal, not just correlational.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_s30ctl')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
ALPHA = 7
RESERVED = {4, 6, 7}   # 4: rho-transfer Spikformer, 6/7: juyun

JOBS = [
    dict(name='s30_deep',    kind='ramp', model='VGG16', data='CIFAR10', s=0, e=12),
    dict(name='s30_shallow', kind='ramp', model='VGG16', data='CIFAR10', s=0, e=60),
]


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    subs = [
        ('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
         f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'),
        ("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='s30-{job['name']}'"),
        ("conf.dataset='CIFAR100'", f"conf.dataset='{job['data']}'"),
        ('conf.reg_spike_out_alpha=3  # temperature',
         f'conf.reg_spike_out_alpha={ALPHA}  # temperature'),
        ('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
         'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'),
        ('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
         'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'),
    ]
    if job['model'] != 'ResNet19':
        subs.insert(2, ("conf.model='ResNet19'", f"conf.model='{job['model']}'"))
    if job['kind'] == 'dcap':
        subs.append(('conf.sc_loss_scd = False',
                     f"""conf.sc_loss_scd = False
        conf.reg_spike_starget = True
        conf.reg_spike_starget_frac = {job['frac']}
        conf.reg_spike_lr_brake = True
        conf.reg_spike_descent_cap = 0.05"""))
    else:
        subs.append(('conf.reg_spike_out_const=1E-8', 'conf.reg_spike_out_const=5E-7'))
        subs.append(('conf.sc_loss_scd = False',
                     f"""conf.sc_loss_scd = False
        conf.reg_spike_ramp_start_ep = {job['s']}
        conf.reg_spike_ramp_end_ep = {job['e']}"""))
    for old, new in subs:
        if old not in c:
            raise RuntimeError(f"substitution target missing: {old!r}")
        c = c.replace(old, new, 1)
    return c


def verify(path, job):
    import re
    src = open(path).read()
    want = [(rf"^conf\.model='{job['model']}'$", 'model'),
            (rf"^conf\.dataset='{job['data']}'$", 'dataset'),
            (r"^\s*conf\.reg_spike_out_wta_rev=True", 'wta_rev'),
            (r"^conf\.train_epoch = 310$", '310ep')]
    if job['kind'] == 'dcap':
        want += [(r"^\s*conf\.reg_spike_starget = True$", 'starget'),
                 (rf"^\s*conf\.reg_spike_starget_frac = {job['frac']}$", 'frac'),
                 (r"^\s*conf\.reg_spike_descent_cap = 0.05$", 'descent cap'),
                 (r"^\s*conf\.reg_spike_lr_brake = True$", 'brake')]
        if re.search(r"^\s*conf\.reg_spike_ramp_start_ep", src, re.M):
            raise RuntimeError('ramp unexpectedly set')
    else:
        want += [(rf"^\s*conf\.reg_spike_ramp_start_ep = {job['s']}$", 'ramp start'),
                 (rf"^\s*conf\.reg_spike_ramp_end_ep = {job['e']}$", 'ramp end'),
                 (r"^\s*conf\.reg_spike_out_const=5E-7$", 'lambda 5e-7')]
        for bad in ['reg_spike_starget = True', 'reg_spike_loss_ratio = True']:
            if re.search(rf"^\s*conf\.{re.escape(bad)}$", src, re.M):
                raise RuntimeError(f'{bad} unexpectedly on')
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
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    jobs = [j for j in JOBS if only is None or j['name'] in only]
    pending, busy, threads = list(jobs), set(), []
    os.makedirs(SWEEP_DIR, exist_ok=True)
    print(f'--- S30 control: {len(pending)} runs ---', flush=True)
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
    print('--- S30 control drained ---', flush=True)


if __name__ == '__main__':
    main()
