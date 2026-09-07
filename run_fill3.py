"""Fill GPUs 0-5 after the 08-13 braked-rho win. Six jobs, three purposes:

1) brk_r19_c10 / brk_r19_c100  -- braked rho=3e-3 on ResNet19, completing the 4-setting
   headline table. The unbraked runs finished at -0.51 / -1.75; both violated the early
   speed limit (S30/S1 0.145 / 0.090), so the brake should hold them at ~0.22 and, if the
   V16 pattern repeats, improve the C100 number. R19-C10 at -0.51 is also the one run that
   violated the limit and (barely) survived -- the braked replicate tells us whether that
   was the cosine tail rescuing a damaged run or a real counterexample.

2) brk_v16_c100_run2 / brk_v16_c10_run2 -- replicates of yesterday's braked wins (-2.33 /
   -1.08, both n=1). V16-C10 replicate spread has been as wide as 7.1%p on this machine;
   the claim "beats fixed lambda at matched landing" needs at least n=2.

3) dvs_lmb_1e-7 / dvs_lmb_1e-8 -- the missing fixed-lambda control on CIFAR10-DVS. The
   transfer claim currently rests on Spikformer alone (rho -0.22 n=3 vs lambda=1e-7 -0.82).
   DVS has a loss-ratio result (57.1% spikes, dacc 0.00) but NO fixed-lambda comparison.
   1e-7 is the CIFAR winner; 1e-8 is the Spikformer winner -- if either matches loss-ratio
   on DVS, the transfer story weakens; if both miss, it strengthens to n=2 domains.
   (The DVS loss-ratio run used rho=2.32e-3 under accumulated-R accounting == 5.8e-4 under
   the per-step accounting used everywhere else; T=4. Controls use the same T and recipe.)

Queue order puts the two 34 h R19 jobs first so they claim dedicated GPUs; the four ~11-15 h
jobs fill the rest and free slots tomorrow morning.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_fill3')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

RHO = 3.0e-3
ALPHA = 7

JOBS = [
    dict(name='brk_r19_c10',       kind='brake', model='ResNet19', data='CIFAR10'),
    dict(name='brk_r19_c100',      kind='brake', model='ResNet19', data='CIFAR100'),
    dict(name='brk_v16_c100_run2', kind='brake', model='VGG16',    data='CIFAR100'),
    dict(name='brk_v16_c10_run2',  kind='brake', model='VGG16',    data='CIFAR10'),
    dict(name='dvs_lmb_1e-7',      kind='dvs_fixed', lmb='1E-7'),
    dict(name='dvs_lmb_1e-8',      kind='dvs_fixed', lmb='1E-8'),
]

RESERVED = {6, 7}     # juyun -- never touch


def base_config(gpu_id, exp_name):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    c = c.replace('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
                  f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"')
    c = c.replace("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='{exp_name}'")
    c = c.replace('conf.reg_spike_out_alpha=3  # temperature',
                  f'conf.reg_spike_out_alpha={ALPHA}  # temperature')
    c = c.replace('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
                  'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)')
    c = c.replace('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
                  'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging')
    return c


def make_config(gpu_id, job):
    c = base_config(gpu_id, f"f3-{job['name']}")
    if job['kind'] == 'brake':
        if job['model'] != 'ResNet19':
            c = c.replace("conf.model='ResNet19'", f"conf.model='{job['model']}'")
        c = c.replace("conf.dataset='CIFAR100'", f"conf.dataset='{job['data']}'")
        c = c.replace('conf.sc_loss_scd = False',
                      f"""conf.sc_loss_scd = False
        conf.reg_spike_loss_ratio = True
        conf.reg_spike_loss_ratio_target = {RHO}
        conf.reg_spike_loss_ratio_start_ep = 0
        conf.reg_spike_R_per_step = True
        conf.reg_spike_lr_brake = True""")
    else:  # dvs_fixed
        c = c.replace("conf.model='ResNet19'", "conf.model='VGG16'")
        c = c.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10_DVS'")
        c = c.replace('conf.train_epoch = 310', 'conf.time_step = 4\nconf.train_epoch = 310')
        # DVS arm for the LR recipe (config asserts on unknown datasets) -- same injection
        # as run_dvs.py, CIFAR recipe verbatim
        c = c.replace("""    elif conf.dataset=='CIFAR100':
        # VGG-C100
        conf.learning_rate_init = 1E-5
        conf.learning_rate = 6E-3
        conf.weight_decay_AdamW = 2E-2
    else:
        assert False""",
                      """    elif conf.dataset=='CIFAR100':
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
        assert False""")
        # fixed lambda: reg_spike_out_const is the lambda when no adaptive flag is set
        c = c.replace('conf.reg_spike_out_const=1E-8',
                      f"conf.reg_spike_out_const={job['lmb']}")
    return c


def verify(path, job):
    import re
    src = open(path).read()
    if job['kind'] == 'brake':
        want = [
            (rf"^conf\.model='{job['model']}'$", 'model'),
            (rf"^conf\.dataset='{job['data']}'$", 'dataset'),
            (r"^\s*conf\.reg_spike_loss_ratio = True$", 'loss_ratio'),
            (rf"^\s*conf\.reg_spike_loss_ratio_target = {RHO}$", 'rho'),
            (r"^\s*conf\.reg_spike_lr_brake = True$", 'brake'),
            (r"^\s*conf\.reg_spike_R_per_step = True$", 'R_per_step'),
        ]
    else:
        want = [
            (r"^conf\.model='VGG16'$", 'model'),
            (r"^conf\.dataset='CIFAR10_DVS'$", 'dataset'),
            (r"^conf\.time_step = 4$", 'T=4'),
            (rf"^\s*conf\.reg_spike_out_const={job['lmb']}$", 'lambda'),
            (r"CIFAR10_DVS':\n", 'dvs LR arm'),
        ]
        # fixed lambda must NOT have the adaptive controller on
        if re.search(r"^\s*conf\.reg_spike_loss_ratio = True$", src, re.M):
            raise RuntimeError(f"{job['name']}: loss_ratio unexpectedly on")
    want.append((r"^\s*conf\.reg_spike_out_wta_rev=True", 'wta_rev'))
    want.append((rf"^\s*conf\.reg_spike_out_alpha={ALPHA}\b", 'alpha'))
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
    MAX_TAKE = int(os.environ.get('F3_MAX_GPUS', '6'))
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    jobs = [j for j in JOBS if only is None or j['name'] in only]
    os.makedirs(SWEEP_DIR, exist_ok=True)
    pending, busy, threads = list(jobs), set(), []
    print(f'--- fill3: {len(pending)} jobs, max {MAX_TAKE} ---', flush=True)
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
            time.sleep(45)
        if pending:
            time.sleep(120)
    for t in threads:
        t.join()
    print('--- fill3 drained ---', flush=True)


if __name__ == '____main__'.replace('____', '__'):
    main()
