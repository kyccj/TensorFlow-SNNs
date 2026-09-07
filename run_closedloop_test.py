"""
Experiment (1): is the closed loop doing something a fixed lambda cannot?

Loss-ratio holds  rho = lambda*L_eip/L_task  constant for the whole run, so lambda moves
every epoch and settles at some final value. A fixed-lambda run set to that SAME final
value ends at the same place in lambda but travels a different path -- rho drifts upward
as L_eip and L_task fall.

If the two land at the same sparsity, the controller is only a way of finding a good
lambda and could be replaced by a sweep. If they land apart, the *path* is what matters
and the closed loop is doing real work. No interpolation is involved either way, which is
the weakness of the sweep-based comparison this replaces (its lambda grid skips 5-50x).

Reference -- loss-ratio at rho=5.8e-4, its converged lambda, and where it landed:

    V16-C10     lambda 7.646e-8   69.5% of baseline spikes
    V16-C100    lambda 1.424e-7   66.3% / 70.5%  (two runs)
    R19-C10     lambda 2.454e-8   68.1%
    R19-C100    lambda 4.398e-8   67.9%

Sweep interpolation predicts the fixed-lambda runs should land at 71.6 / 77.0 / 62.9 /
71.0% -- a 14.1%p spread against the closed loop's 4.2%p. That prediction is what this
experiment tests directly.

v16_c10 is repeated three times: the closed loop's spread is only known against one
replicate pair (V16-C100, 4.2%p apart), so fixed-lambda seed noise at this lambda needs
its own estimate before the two spreads can be compared.

reg_spike_log_detail stays ON -- R must be logged to compare rho trajectories, which is
the whole point. Costs 24.5% of step time.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_closedloop_test')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

ALPHA = 7

# name -> (model, dataset, lambda, approx hours)
JOBS = [
    ('fx_v16_c10',      'VGG16',    'CIFAR10',  7.646e-8, 10),
    ('fx_v16_c100',     'VGG16',    'CIFAR100', 1.424e-7, 10),
    ('fx_v16_c10_run2', 'VGG16',    'CIFAR10',  7.646e-8, 10),
    ('fx_v16_c10_run3', 'VGG16',    'CIFAR10',  7.646e-8, 10),
    ('fx_r19_c10',      'ResNet19', 'CIFAR10',  2.454e-8, 32),
    ('fx_r19_c100',     'ResNet19', 'CIFAR100', 4.398e-8, 32),
]


def make_config(gpu_id, exp_name, model, dataset, lmb):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    c = c.replace('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
                  f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"')
    c = c.replace("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='{exp_name}'")
    if model == 'VGG16':
        c = c.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    if dataset == 'CIFAR10':
        c = c.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")
    c = c.replace('conf.reg_spike_out_const=1E-8', f'conf.reg_spike_out_const={lmb}')
    c = c.replace('conf.reg_spike_out_alpha=3  # temperature',
                  f'conf.reg_spike_out_alpha={ALPHA}  # temperature')
    c = c.replace('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
                  'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)')
    c = c.replace('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
                  'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging')
    return c


def run_one(gpu, name, model, dataset, lmb):
    d = os.path.join(SWEEP_DIR, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'config_sweep.py'), 'w') as f:
        f.write(make_config(gpu, f'cl-{name}', model, dataset, lmb))
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')) as f:
        m = f.read().replace('from config_snn_training import config', 'from config_sweep import config')
    with open(os.path.join(d, 'main_sweep.py'), 'w') as f:
        f.write(m)

    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'

    print(f'[GPU {gpu}] START {name} ({model}/{dataset}, fixed lambda={lmb:.3e})', flush=True)
    with open(os.path.join(d, 'train.log'), 'w') as lf:
        rc = subprocess.Popen([PYTHON, os.path.join(d, 'main_sweep.py')], cwd=PROJECT_ROOT,
                              stdout=lf, stderr=subprocess.STDOUT, env=env).wait()
    print(f'[GPU {gpu}] DONE  {name} rc={rc}', flush=True)


RESERVED = {6, 7}                                        # another user's
RESERVED |= {int(x) for x in os.environ.get('CL_SKIP_GPUS', '').split(',') if x.strip()}


def free_gpus(exclude, max_mib=200):
    """GPUs idle enough to take.

    max_mib is deliberately tight: a TF run here grabs the whole card, so a GPU holding
    even a few hundred MB of someone else's work must not be taken -- that work would OOM.
    """
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            timeout=60).decode()
    except Exception:
        return []
    out_list = []
    for line in out.strip().split('\n'):
        idx, mem = [x.strip() for x in line.split(',')]
        idx = int(idx)
        if idx in RESERVED or idx in exclude:
            continue
        if int(mem) <= max_mib:
            out_list.append(idx)
    return out_list


def main():
    """Queue the jobs, taking GPUs only once they have been idle on two consecutive polls.

    MAX_TAKE leaves headroom so this never occupies the whole machine.
    """
    only = sys.argv[1] if len(sys.argv) > 1 else None
    jobs = [j for j in JOBS if only is None or j[0] in only.split(',')]
    MAX_TAKE = int(os.environ.get('CL_MAX_GPUS', '3'))
    POLL = 300

    os.makedirs(SWEEP_DIR, exist_ok=True)
    pending = list(jobs)
    busy, threads, prev_free = set(), [], set()
    print(f'--- queued {len(pending)} jobs, max {MAX_TAKE} GPUs at once ---', flush=True)
    while pending:
        now_free = set(free_gpus(busy))
        stable = now_free & prev_free          # idle on two consecutive polls
        prev_free = now_free
        while pending and stable and len(busy) < MAX_TAKE:
            g = min(stable); stable.discard(g); busy.add(g)
            name, model, ds, lmb, _h = pending.pop(0)

            def lane(g=g, args=(name, model, ds, lmb)):
                try:
                    run_one(g, *args)
                finally:
                    busy.discard(g)
            t = threading.Thread(target=lane); t.start(); threads.append(t)
        if pending:
            print(f'[queue] {len(pending)} waiting, {len(busy)} running, free={sorted(now_free)}', flush=True)
            time.sleep(POLL)
    for t in threads:
        t.join()
    print('--- all jobs finished ---', flush=True)


if __name__ == '__main__':
    main()
