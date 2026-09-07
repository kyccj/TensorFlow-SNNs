"""rho ladder: braked vs unbraked loss-ratio across the full dial, V16-C10 and V16-C100.

Pre-registered: 사전 등록 — rho 사다리 26-08-15.md (written before launch).

Nine new runs complete the matrix over rho in {5.8e-4, 1.2e-3, 6e-3} (2.6e-4 and 3e-3 rows
already exist). Three claims tested at once:
  1. dial      -- braked arm: landing monotone in rho, zero deaths (L1)
  2. brake     -- unbraked arm at 6e-3 should die/lose >5%p on both settings (L2); the
                  braked 5.8e-4 C100 run (job ①) attacks the one standard-rho failure (L3)
  3. transfer  -- same rho on C10 vs C100: how far does landing agreement survive up the
                  ladder (L4)

Queue order puts the headline (①) and the braked pairs first; unbraked arms last since two
of them are expected deaths. Claims GPUs as tonight's five runs finish. GPU 6-7 reserved.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_rho_ladder')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

ALPHA = 7

JOBS = [
    dict(name='rl_c100_58_br',  data='CIFAR100', rho=5.8e-4, brake=True),   # ①
    dict(name='rl_c10_12_br',   data='CIFAR10',  rho=1.2e-3, brake=True),   # ②
    dict(name='rl_c100_12_br',  data='CIFAR100', rho=1.2e-3, brake=True),   # ③
    dict(name='rl_c10_60_br',   data='CIFAR10',  rho=6.0e-3, brake=True),   # ⑥
    dict(name='rl_c100_60_br',  data='CIFAR100', rho=6.0e-3, brake=True),   # ⑦
    dict(name='rl_c10_12_nb',   data='CIFAR10',  rho=1.2e-3, brake=False),  # ④
    dict(name='rl_c100_12_nb',  data='CIFAR100', rho=1.2e-3, brake=False),  # ⑤
    dict(name='rl_c10_60_nb',   data='CIFAR10',  rho=6.0e-3, brake=False),  # ⑧
    dict(name='rl_c100_60_nb',  data='CIFAR100', rho=6.0e-3, brake=False),  # ⑨
]

RESERVED = {6, 7}


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    brake_line = '\n        conf.reg_spike_lr_brake = True' if job['brake'] else ''
    subs = [
        ('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
         f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'),
        ("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='rhold-{job['name']}'"),
        ("conf.model='ResNet19'", "conf.model='VGG16'"),
        ("conf.dataset='CIFAR100'", f"conf.dataset='{job['data']}'"),
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
        conf.reg_spike_R_per_step = True{brake_line}"""),
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
        (rf"^conf\.dataset='{job['data']}'$", 'dataset'),
        (r"^\s*conf\.reg_spike_loss_ratio = True$", 'loss_ratio'),
        (rf"^\s*conf\.reg_spike_loss_ratio_target = {job['rho']}$", 'rho'),
        (r"^\s*conf\.reg_spike_R_per_step = True$", 'R_per_step'),
        (r"^\s*conf\.reg_spike_out_wta_rev=True", 'wta_rev'),
        (rf"^\s*conf\.reg_spike_out_alpha={ALPHA}\b", 'alpha'),
        (r"^conf\.train_epoch = 310$", '310ep'),
    ]
    has_brake = bool(re.search(r"^\s*conf\.reg_spike_lr_brake = True$", src, re.M))
    if has_brake != job['brake']:
        raise RuntimeError(f"{job['name']}: brake flag mismatch (want {job['brake']})")
    if re.search(r"^\s*conf\.reg_spike_starget = True$", src, re.M):
        raise RuntimeError('starget unexpectedly on')
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

    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {job['name']} "
          f"(rho={job['rho']:.1e}, brake={'ON' if job['brake'] else 'OFF'})", flush=True)
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
    print(f'--- rho ladder: {len(pending)} runs ---', flush=True)
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
            time.sleep(300)
    for t in threads:
        t.join()
    print('--- rho ladder drained ---', flush=True)


if __name__ == '__main__':
    main()
