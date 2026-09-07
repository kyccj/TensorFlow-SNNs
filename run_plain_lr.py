"""plain L2 + loss-ratio: is the lambda rule tied to wta_rev, or does it transfer?

Every loss-ratio result to date runs on wta_rev, so "rho = 5.8e-4" has only ever been
measured for that one regularizer. The argument written on 08-09 -- that rho cannot be
separated from the regularizer because sc_rate is ~1 under wta_rev and ~1/65536 otherwise
-- turned out to be wrong. `reg_spike_out_sc_wta` defaults to True (flags.py:811) and the
configs leave it commented out, so

    elif conf.reg_spike_out_wta_rev or conf.reg_spike_out_sc_wta:
        sc_rate = 1.0 - sc_norm

takes the same branch either way. The `else: sc_rate = sc_norm` arm is unreachable in every
run in this repo. Measured: sc_rate 0.9996 and R 33,629 with wta_rev OFF, against 0.9996 and
33,620 with it ON.

So the two differ in the backward pass alone:

    l2_norm           grad = x/||x||        -> zero for a neuron that did not fire
    l2_norm_wta_rev   grad = sc_rate/||x||  -> non-zero for all of them

Forward, sc_rate, and therefore R and rho are on the same scale. That makes this experiment
possible at rho=5.8e-4 with no recalibration, and it makes the question sharp: the two runs
below differ from their wta_rev counterparts in the gradient and nothing else.

Reference (wta_rev, rho=5.8e-4, n=3 means):  V16-C10 66.9%,  R19-C10 67.2%

Reading the outcome:

    both land near 67%      the rule does not depend on this regularizer; the earlier
                            "inseparable" claim dies entirely and the lambda rule can be
                            described on its own
    both land together but  the rule transfers, but its constant is regularizer-specific --
    away from 67%           report rho per regularizer
    they land apart         the pairing is real; rho belongs to (rule + regularizer)

R19-C10 is the transfer test and is ~32 h; V16-C10 is ~10 h and goes first.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_plain_lr')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

RHO = 5.8e-4
ALPHA = 7

JOBS = [
    dict(name='plainlr_v16_c10', model='VGG16',    data='CIFAR10'),
    dict(name='plainlr_r19_c10', model='ResNet19', data='CIFAR10'),
    # added 08-10, after the first V16-C10 run came in at 79.2% against wta_rev's 66.9%
    # (n=3 mean). That 12.3%p gap is the whole basis for saying rho belongs to the
    # (rule + regularizer) pair, and it currently rests on one draw against three. V16-C10 is
    # also the noisiest setting measured so far -- its wta_rev replicates spanned 7.1%p -- so
    # a single 79.2% could be a high draw from a distribution centred nearer 73%.
    dict(name='plainlr_v16_c10_run2', model='VGG16', data='CIFAR10'),
]

RESERVED = {6, 7}
RESERVED |= {int(x) for x in os.environ.get('PL_SKIP_GPUS', '').split(',') if x.strip()}


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    c = c.replace('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
                  f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"')
    c = c.replace("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='pl-{job['name']}'")
    if job['model'] != 'ResNet19':
        c = c.replace("conf.model='ResNet19'", f"conf.model='{job['model']}'")
    c = c.replace("conf.dataset='CIFAR100'", f"conf.dataset='{job['data']}'")
    c = c.replace('conf.reg_spike_out_alpha=3  # temperature',
                  f'conf.reg_spike_out_alpha={ALPHA}  # temperature')
    c = c.replace('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
                  'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging')
    # wta_rev stays OFF -- that is the whole point. reg_spike_out_norm is already True in the
    # base config, so sc_loss goes through l2_norm (standard backward) instead.
    block = f"""conf.sc_loss_scd = False
        conf.reg_spike_loss_ratio = True
        conf.reg_spike_loss_ratio_target = {RHO}
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

    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {job['name']} "
          f"({job['model']}/{job['data']}, plain L2 + loss-ratio, rho={RHO:.1e})", flush=True)
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
    MAX_TAKE = int(os.environ.get('PL_MAX_GPUS', '2'))
    POLL = int(os.environ.get('PL_POLL', '120'))
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    jobs = [j for j in JOBS if only is None or j['name'] in only]

    os.makedirs(SWEEP_DIR, exist_ok=True)
    pending, busy, threads = list(jobs), set(), []
    print(f'--- plain+loss-ratio: {len(pending)} jobs, max {MAX_TAKE} ---', flush=True)
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
            time.sleep(POLL)
    for t in threads:
        t.join()
    print('--- plain+loss-ratio drained ---', flush=True)


if __name__ == '__main__':
    main()
