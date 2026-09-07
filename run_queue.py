"""Standing queue over GPU 0-5. Takes a card the moment it frees, keeps going.

GPU 6-7 are someone else's and are never touched. 0-5 include cards currently running
other work of the user's -- the 200 MiB threshold means those are only taken once they are
genuinely idle, so nothing gets pushed off a card.

Jobs are ordered by what they buy, cheapest-decisive first:

  1. sf_lr          Spikformer -- the fifth architecture, and the first one whose landing
                    was PREDICTED before running. Its baseline and a four-point lambda
                    sweep already exist (alpha=7, wta_rev, same as the CIFAR settings), so
                    the sweep gives rho -> landing directly:

                        lambda 1e-9  rho 3.31e-5  -> 100.4%
                        lambda 1e-8  rho 2.93e-4  ->  81.5%
                        lambda 1e-7  rho 1.86e-3  ->  36.1%
                        lambda 5e-7  rho 4.35e-3  ->  11.7%

                    interpolating to rho=5.8e-4 predicts **64.7% and ~94.48%** (baseline
                    94.65%). Landing near that is the cleanest evidence yet: an untouched
                    architecture, predicted in advance from independent data.

  2. base_t3 x2     T=3's accuracy cost came out -0.49%p, larger than T=4's -0.03%p despite
                    sparsifying LESS. Its baseline is n=1 so that number cannot be compared
                    against a spread. Two more baselines settle it for ~6 h each.

  3. R19 repeats    R19-C10 and R19-C100 loss-ratio are both n=1, and the claim that the
                    across-setting spread equals replicate noise leans on a single V16-C100
                    pair. 32 h each, so these go last.

Old accounting throughout (reg_spike_R_per_step=True): the T sweep settled it at 9.6%p
against the corrected accounting's 67.0%p, and every reference point is on that curve.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_queue')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

RHO = 5.8e-4
ALPHA = 7

JOBS = [
    dict(name='sf_lr',            model='Spikformer', data='CIFAR10',  mode='lr'),
    dict(name='base_t3_run2',     model='VGG16',      data='CIFAR10',  mode='none', T=3),
    dict(name='base_t3_run3',     model='VGG16',      data='CIFAR10',  mode='none', T=3),
    dict(name='lr_r19_c10_run2',  model='ResNet19',   data='CIFAR10',  mode='lr'),
    dict(name='lr_r19_c10_run3',  model='ResNet19',   data='CIFAR10',  mode='lr'),
    dict(name='lr_r19_c100_run2', model='ResNet19',   data='CIFAR100', mode='lr'),
    dict(name='lr_r19_c100_run3', model='ResNet19',   data='CIFAR100', mode='lr'),
    # added 08-09. V16-C10 came back at 7.1%p over three runs, well above the 4.2%p that
    # every "spread equals replicate noise" statement has been leaning on -- and that 4.2%p
    # is a two-run estimate from V16-C100 alone. A third V16-C100 run decides whether 4.2
    # or 7.1 is the number to quote. lr_t3 is n=1 and now carries the T>=3 claim.
    dict(name='lr_v16_c100_run3', model='VGG16',      data='CIFAR100', mode='lr'),
    dict(name='lr_t3_run2',       model='VGG16',      data='CIFAR10',  mode='lr', T=3),
    # added 08-10. sf_lr came in at 60.6%, about 6%p under the four CNN settings whose n=3
    # means sit at 66.9-67.3%. At n=1 that gap is not separable from run-to-run spread, which
    # has measured anywhere from 1.5%p (R19) to 7.1%p (V16-C10) depending on the setting. Two
    # more draws decide whether a transformer sparsifies further than the CNNs under the same
    # rho, or whether the first run just landed low. ~11 h each.
    dict(name='sf_lr_run2',       model='Spikformer', data='CIFAR10',  mode='lr'),
    dict(name='sf_lr_run3',       model='Spikformer', data='CIFAR10',  mode='lr'),
]

RESERVED = {6, 7}
RESERVED |= {int(x) for x in os.environ.get('Q_SKIP_GPUS', '').split(',') if x.strip()}


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    c = c.replace('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
                  f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"')
    c = c.replace("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='q-{job['name']}'")
    if job['model'] != 'ResNet19':
        c = c.replace("conf.model='ResNet19'", f"conf.model='{job['model']}'")
    if job['data'] != 'CIFAR100':
        c = c.replace("conf.dataset='CIFAR100'", f"conf.dataset='{job['data']}'")
    if job.get('T'):
        c = c.replace('conf.train_epoch = 310', f"conf.time_step = {job['T']}\nconf.train_epoch = 310")

    if job['mode'] == 'lr':
        c = c.replace('conf.reg_spike_out_alpha=3  # temperature',
                      f'conf.reg_spike_out_alpha={ALPHA}  # temperature')
        c = c.replace('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
                      'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)')
        c = c.replace('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
                      'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging')
        block = f"""conf.sc_loss_scd = False
        conf.reg_spike_loss_ratio = True
        conf.reg_spike_loss_ratio_target = {RHO}
        conf.reg_spike_loss_ratio_start_ep = 0
        conf.reg_spike_R_per_step = True"""
        c = c.replace('conf.sc_loss_scd = False', block)
    else:
        c = c.replace('conf.reg_spike_out=True', 'conf.reg_spike_out=False')
    return c


def run_one(gpu, job):
    d = os.path.join(SWEEP_DIR, job['name'])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'config_sweep.py'), 'w') as f:
        f.write(make_config(gpu, job))
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')) as f:
        m = f.read().replace('from config_snn_training import config', 'from config_sweep import config')
    with open(os.path.join(d, 'main_sweep.py'), 'w') as f:
        f.write(m)

    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'

    tag = f"rho={RHO:.1e}" if job['mode'] == 'lr' else 'no reg'
    tt = f", T={job['T']}" if job.get('T') else ''
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {job['name']} "
          f"({job['model']}/{job['data']}{tt}, {tag})", flush=True)
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
    """One consecutive idle reading is enough here -- a card that just finished one of these
    is genuinely free, and the two-poll rule cost 5 minutes of idle GPU every handover."""
    MAX_TAKE = int(os.environ.get('Q_MAX_GPUS', '4'))
    POLL = int(os.environ.get('Q_POLL', '120'))
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    jobs = [j for j in JOBS if only is None or j['name'] in only]

    os.makedirs(SWEEP_DIR, exist_ok=True)
    pending, busy, threads = list(jobs), set(), []
    print(f'--- standing queue: {len(pending)} jobs, GPU 0-5, max {MAX_TAKE} at once ---', flush=True)
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
            time.sleep(45)      # let TF claim the card before the next poll reads memory
        if pending:
            time.sleep(POLL)
    for t in threads:
        t.join()
    print('--- standing queue drained ---', flush=True)


if __name__ == '__main__':
    main()
