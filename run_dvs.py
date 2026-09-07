"""
CIFAR10-DVS: does rho=5.8e-4 carry to a different input modality?

Every result so far is frame-based CIFAR. DVS is event data, so this is the first test
outside the 2x2 (V16/R19 x C10/C100) the constant was fitted on -- the axis the exponent
form died on when it was tried on alpha.

Run at two time steps, because the two questions are entangled otherwise:

  T=4   matches every existing result, so a miss here is the modality's doing
  T=16  the repo's default for DVS (config_snn_training_H_direct.py:405), so this is the
        configuration anyone would actually use -- and it adds a fourth point to the T
        axis, which currently has 2/4/8

Both need their own no-reg baseline: "70% of baseline" needs that configuration's own
reference, and no DVS baseline exists in this repo at any T.

Two cautions on reading the output.

  Accuracy is not resolvable here. The dataset is 10,000 samples with no test split;
  datasets/cifar10_dvs.py holds out 10% for validation, so accuracy is measured on 1,000
  samples -- roughly +-1.5%p at 1 sigma. The CIFAR conclusions turn on 0.1-0.9%p, which
  this cannot see. Spike count is the quantity to read; treat accuracy as a guard against
  outright collapse, nothing finer.

  9,000 training samples at batch 100 is 90 steps/epoch against CIFAR's 500, so these are
  cheap -- but the per-step cost at T=16 is 4x the T=4 cost.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_dvs')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

# rho is stated on the corrected R (neurons.py now sums sc_loss_snap over time steps), so
# it is 4x the old nominal 5.8e-4 -- same regularization, different bookkeeping.
RHO = float(os.environ.get('DVS_RHO', '2.32e-3'))
ALPHA = 7

# name -> (T, reg_on)
#
# T=16 is left out of the default set. Measured cost is ~175 s/epoch at T=4 (2 s/step
# against CIFAR's 185 ms -- the event-to-frame conversion dominates, not the model), so
# T=4 is already ~15 h and T=16 would be ~60 h per run. Add it back once the T=4 pair
# says whether the modality transfers at all.
JOBS = [
    ('dvs_base_t4',  4,  False),
    ('dvs_lr_t4',    4,  True),
]

RESERVED = {6, 7}
RESERVED |= {int(x) for x in os.environ.get('DVS_SKIP_GPUS', '').split(',') if x.strip()}


def make_config(gpu_id, exp_name, T, reg_on):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    c = c.replace('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
                  f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"')
    c = c.replace("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='{exp_name}'")
    c = c.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    c = c.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10_DVS'")
    c = c.replace('conf.train_epoch = 310', f'conf.time_step = {T}\nconf.train_epoch = 310')

    # config_snn_training.py only branches on CIFAR10/CIFAR100 and asserts otherwise, so
    # DVS needs its own arm. It gets the CIFAR recipe verbatim -- the point of this run is
    # to change the input modality and nothing else, so a DVS-tuned LR would confound it.
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

    if reg_on:
        c = c.replace('conf.reg_spike_out_alpha=3  # temperature',
                      f'conf.reg_spike_out_alpha={ALPHA}  # temperature')
        c = c.replace('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
                      'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)')
        c = c.replace('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
                      'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging')
        block = f"""conf.sc_loss_scd = False
        conf.reg_spike_loss_ratio = True
        conf.reg_spike_loss_ratio_target = {RHO}
        conf.reg_spike_loss_ratio_start_ep = 0"""
        c = c.replace('conf.sc_loss_scd = False', block)
    else:
        c = c.replace('conf.reg_spike_out=True', 'conf.reg_spike_out=False')
    return c


def run_one(gpu, name, T, reg_on, epochs=None):
    d = os.path.join(SWEEP_DIR, name)
    os.makedirs(d, exist_ok=True)
    cfg = make_config(gpu, f'dvs-{name}', T, reg_on)
    if epochs:                                    # smoke test
        cfg = cfg.replace('conf.train_epoch = 310', f'conf.train_epoch = {epochs}')
    with open(os.path.join(d, 'config_sweep.py'), 'w') as f:
        f.write(cfg)
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')) as f:
        m = f.read().replace('from config_snn_training import config', 'from config_sweep import config')
    with open(os.path.join(d, 'main_sweep.py'), 'w') as f:
        f.write(m)

    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'

    tag = f'rho={RHO:.1e}' if reg_on else 'no reg'
    print(f'[GPU {gpu}] START {name} (V16/CIFAR10_DVS, T={T}, {tag})', flush=True)
    with open(os.path.join(d, 'train.log'), 'w') as lf:
        rc = subprocess.Popen([PYTHON, os.path.join(d, 'main_sweep.py')], cwd=PROJECT_ROOT,
                              stdout=lf, stderr=subprocess.STDOUT, env=env).wait()
    print(f'[GPU {gpu}] DONE  {name} rc={rc}', flush=True)


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
    """--smoke runs the T=4 pair for a few epochs first; DVS has never trained in this repo."""
    smoke = '--smoke' in sys.argv
    jobs = [(n, T, r, 3) for n, T, r in JOBS if T == 4] if smoke else [(n, T, r, None) for n, T, r in JOBS]
    MAX_TAKE = int(os.environ.get('DVS_MAX_GPUS', '2'))
    POLL = 300

    os.makedirs(SWEEP_DIR, exist_ok=True)
    pending, busy, threads, prev = list(jobs), set(), [], set()
    print(f'--- queued {len(pending)} DVS jobs, max {MAX_TAKE} GPUs ---', flush=True)
    while pending:
        now = set(free_gpus(busy))
        stable = now & prev
        prev = now
        while pending and stable and len(busy) < MAX_TAKE:
            g = min(stable); stable.discard(g); busy.add(g)
            args = pending.pop(0)

            def lane(g=g, a=args):
                try:
                    run_one(g, *a)
                finally:
                    busy.discard(g)
            t = threading.Thread(target=lane); t.start(); threads.append(t)
        if pending:
            print(f'[queue] {len(pending)} waiting, {len(busy)} running, free={sorted(now)}', flush=True)
            time.sleep(POLL)
    for t in threads:
        t.join()
    print('--- DVS jobs finished ---', flush=True)


if __name__ == '__main__':
    main()
