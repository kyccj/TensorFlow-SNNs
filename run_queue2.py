"""Standing queue, round 2. GPU 0-5; 6-7 belong to someone else and are never touched.

Two things this buys, in order.

**A. Does the lambda rule survive a change of regularizer?** (plain_* jobs)

Every result to date sits on wta_rev. The rule lambda = rho*L_task/L_eip does not
syntactically reference the regularizer, which is exactly why it was easy to describe it as
a separable contribution -- but rho's *value* is set by whichever R you divide by, and
plain L2 uses sc_rate = softmax ~ 1/65536 against wta_rev's ~1, so R differs by roughly
that factor. Whether the rule still lands somewhere stable under a different R is untested
and currently unanswerable.

Two fixed-lambda plain-L2 runs come first because they do double duty: they fill the gap in
the plain-L2 sparsity curve (it jumps 46% -> 9% between lambda 1e-6 and 5e-6, so the
deep-sparsity comparison against wta_rev is interpolated across that hole), AND they log R,
which is what a plain-L2 rho has to be calibrated against. log_detail is on for that reason.

**B. Is T=2 really a lone exception, or is its baseline just n=1?** (base_t2_run2/3)

T=2 is the one point that breaks the T>=3 story, and both its baseline and its regularized
run are single draws. Every setting measured at n=3 so far has shown 1.5-7.1%p of replicate
spread, so a 47.3% single point is not yet distinguishable from a 55% one.

Not queued: plain L2 + loss-ratio itself. It needs a rho, and rho needs R from the runs
above. Launch it once those land.

Old accounting throughout (reg_spike_R_per_step=True) -- the T sweep settled on it, and
every reference point is on that curve.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_queue2')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

RHO = 5.8e-4
ALPHA = 7

# mode: 'lr' loss-ratio | 'fixed' fixed lambda | 'none' no regularization
# reg:  'wta_rev' | 'plain'   (plain = sc_rate = sc_norm, standard L2 gradient)
JOBS = [
    dict(name='plain_lmb_2e-6', model='VGG16', data='CIFAR10', mode='fixed', reg='plain', lmb=2e-6),
    dict(name='plain_lmb_3e-6', model='VGG16', data='CIFAR10', mode='fixed', reg='plain', lmb=3e-6),
    dict(name='base_t2_run2',   model='VGG16', data='CIFAR10', mode='none',  T=2),
    dict(name='base_t2_run3',   model='VGG16', data='CIFAR10', mode='none',  T=2),
    dict(name='lr_t2_run2',     model='VGG16', data='CIFAR10', mode='lr',    reg='wta_rev', T=2),
    dict(name='lr_t8_run2',     model='VGG16', data='CIFAR10', mode='lr',    reg='wta_rev', T=8),
]

RESERVED = {6, 7}
RESERVED |= {int(x) for x in os.environ.get('Q_SKIP_GPUS', '').split(',') if x.strip()}


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    c = c.replace('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
                  f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"')
    c = c.replace("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='q2-{job['name']}'")
    if job['model'] != 'ResNet19':
        c = c.replace("conf.model='ResNet19'", f"conf.model='{job['model']}'")
    if job['data'] != 'CIFAR100':
        c = c.replace("conf.dataset='CIFAR100'", f"conf.dataset='{job['data']}'")
    if job.get('T'):
        c = c.replace('conf.train_epoch = 310', f"conf.time_step = {job['T']}\nconf.train_epoch = 310")

    if job['mode'] == 'none':
        return c.replace('conf.reg_spike_out=True', 'conf.reg_spike_out=False')

    c = c.replace('conf.reg_spike_out_alpha=3  # temperature',
                  f'conf.reg_spike_out_alpha={ALPHA}  # temperature')
    c = c.replace('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
                  'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging')
    # wta_rev flips sc_rate to 1-sc_norm and swaps in the modified backward; leaving the flag
    # off is what "plain" means here -- sc_rate stays as sc_norm and the L2 gradient is standard.
    if job.get('reg') == 'wta_rev':
        c = c.replace('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
                      'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)')

    if job['mode'] == 'fixed':
        c = c.replace('conf.reg_spike_out_const=1E-8', f"conf.reg_spike_out_const={job['lmb']}")
    else:
        block = f"""conf.sc_loss_scd = False
        conf.reg_spike_loss_ratio = True
        conf.reg_spike_loss_ratio_target = {job.get('rho', RHO)}
        conf.reg_spike_loss_ratio_start_ep = 0
        conf.reg_spike_R_per_step = True"""
        c = c.replace('conf.sc_loss_scd = False', block)
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

    tag = {'lr': f"rho={RHO:.1e}", 'none': 'no reg'}.get(job['mode'], f"lambda={job.get('lmb')}")
    extra = f" {job.get('reg', '')}" + (f", T={job['T']}" if job.get('T') else '')
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {job['name']} "
          f"({job['model']}/{job['data']}{extra}, {tag})", flush=True)
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
    MAX_TAKE = int(os.environ.get('Q_MAX_GPUS', '5'))
    POLL = int(os.environ.get('Q_POLL', '120'))
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    jobs = [j for j in JOBS if only is None or j['name'] in only]

    os.makedirs(SWEEP_DIR, exist_ok=True)
    pending, busy, threads = list(jobs), set(), []
    print(f'--- queue2: {len(pending)} jobs, GPU 0-5, max {MAX_TAKE} at once ---', flush=True)
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
            time.sleep(45)      # let TF claim the card before the next memory read
        if pending:
            time.sleep(POLL)
    for t in threads:
        t.join()
    print('--- queue2 drained ---', flush=True)


if __name__ == '__main__':
    main()
