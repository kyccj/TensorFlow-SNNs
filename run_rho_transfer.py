"""Does rho transfer to new domains at the TOP of the practical range (1.2e-3)?

Pre-registered: 사전 등록 — 실용 상단 rho 이전성 26-08-17.md (before launch).

Transfer evidence so far sits entirely at rho=5.8e-4 (DVS 0.00, Spikformer -0.22 n=3) -- the
safe regime where anything works. The ladder showed landings diverge between settings as rho
rises (C10/C100: identical at 5.8e-4, 0.155 vs 0.212 at 1.2e-3, 0.055 vs 0.082 at 6e-3), so
the open question is whether one rho still covers new domains at the practical ceiling.

    reference   CIFAR at 1.2e-3:  C10 -0.51 @ 0.155,  C100 -0.60 @ 0.212
    prediction  DVS / Spikformer land inside 0.155~0.212 +-0.08 with dacc >= -1.5%p

DVS needs its own LR arm injected (config asserts on unknown datasets) -- same block as
run_dvs.py, CIFAR recipe verbatim so only the input modality differs.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_rho_transfer')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
RHO = 1.2e-3
ALPHA = 7
RESERVED = {1, 2, 3, 5, 6, 7}   # 1/2/3/5: path-causal runs, 6/7: juyun

JOBS = [
    dict(name='tr_dvs_12',  model='VGG16',      data='CIFAR10_DVS'),
    dict(name='tr_sf_12',   model='Spikformer', data='CIFAR10'),
]

DVS_ARM_OLD = """    elif conf.dataset=='CIFAR100':
        # VGG-C100
        conf.learning_rate_init = 1E-5
        conf.learning_rate = 6E-3
        conf.weight_decay_AdamW = 2E-2
    else:
        assert False"""
DVS_ARM_NEW = """    elif conf.dataset=='CIFAR100':
        # VGG-C100
        conf.learning_rate_init = 1E-5
        conf.learning_rate = 6E-3
        conf.weight_decay_AdamW = 2E-2
    elif conf.dataset=='CIFAR10_DVS':
        # VGG-DVS: same recipe as the CIFAR runs, on purpose
        conf.learning_rate_init = 1E-5
        conf.learning_rate = 6E-3
        conf.weight_decay_AdamW = 2E-2
    else:
        assert False"""


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    subs = [
        ('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
         f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'),
        ("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='rhotr-{job['name']}'"),
        ("conf.model='ResNet19'", f"conf.model='{job['model']}'"),
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
        conf.reg_spike_loss_ratio_target = {RHO}
        conf.reg_spike_loss_ratio_start_ep = 0
        conf.reg_spike_R_per_step = True
        conf.reg_spike_lr_brake = True"""),
    ]
    for old, new in subs:
        if old not in c:
            raise RuntimeError(f"substitution target missing: {old!r}")
        c = c.replace(old, new, 1)
    if job['data'] == 'CIFAR10_DVS':
        c = c.replace('conf.train_epoch = 310', 'conf.time_step = 4\nconf.train_epoch = 310', 1)
        if DVS_ARM_OLD not in c:
            raise RuntimeError('DVS LR arm anchor missing')
        c = c.replace(DVS_ARM_OLD, DVS_ARM_NEW, 1)
    return c


def verify(path, job):
    import re
    src = open(path).read()
    want = [
        (rf"^conf\.model='{job['model']}'$", 'model'),
        (rf"^conf\.dataset='{job['data']}'$", 'dataset'),
        (r"^\s*conf\.reg_spike_loss_ratio = True$", 'loss_ratio'),
        (rf"^\s*conf\.reg_spike_loss_ratio_target = {RHO}$", 'rho'),
        (r"^\s*conf\.reg_spike_lr_brake = True$", 'brake'),
        (r"^\s*conf\.reg_spike_R_per_step = True$", 'R_per_step'),
        (r"^\s*conf\.reg_spike_out_wta_rev=True", 'wta_rev'),
        (r"^conf\.train_epoch = 310$", '310ep'),
    ]
    if job['data'] == 'CIFAR10_DVS':
        want += [(r"^conf\.time_step = 4$", 'T=4'), (r"CIFAR10_DVS':\n\s+# VGG-DVS", 'dvs LR arm')]
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
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {job['name']} (rho={RHO:.1e})", flush=True)
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
    print(f'--- rho transfer @ {RHO:.1e}: {len(pending)} runs ---', flush=True)
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
    print('--- rho transfer drained ---', flush=True)


if __name__ == '__main__':
    main()
