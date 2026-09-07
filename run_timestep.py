"""
Time-step axis: does the single rho survive a change in T?

This is the sharpest test of the rule's *form*, not just its constant.  Spike count is
roughly proportional to T, so R should scale with T while the task loss barely moves.
If  lambda = rho * L / R  is the right combination, then

    T=2  ->  R about half   ->  lambda about double,  sparsity still ~70%
    T=8  ->  R about double ->  lambda about half,    sparsity still ~70%

Landing near 70% at both means L/R is carrying the T dependence for free. Missing means
a term is absent from the rule, and the direction of the miss says which.

Baselines are re-run at each T because "70% of baseline" needs that T's own reference.
reg_spike_log_detail is OFF (24.5% of step time, feeds only reg_detail.csv; s_count and
the loss-ratio control do not use it).

Reference point (T=4, rho=5.8e-4): V16-C10 landed at 69.6%, lambda 7.65e-8, acc -0.02%p.
"""

import subprocess
import os
import sys
import threading

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_timestep')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

RHO = 5.8e-4
ALPHA = 7


def make_config(gpu_id, exp_name, T, reg_on):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    c = c.replace('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
                  f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"')
    c = c.replace("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='{exp_name}'")
    c = c.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    c = c.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")
    # time step
    c = c.replace('conf.train_epoch = 310', f'conf.time_step = {T}\nconf.train_epoch = 310')

    if reg_on:
        c = c.replace('conf.reg_spike_out_alpha=3  # temperature',
                      f'conf.reg_spike_out_alpha={ALPHA}  # temperature')
        c = c.replace('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
                      'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)')
        block = f"""conf.sc_loss_scd = False
        conf.reg_spike_loss_ratio = True
        conf.reg_spike_loss_ratio_target = {RHO}
        conf.reg_spike_loss_ratio_start_ep = 0"""
        c = c.replace('conf.sc_loss_scd = False', block)
    else:
        c = c.replace('conf.reg_spike_out=True', 'conf.reg_spike_out=False')
    return c


# gpu -> ordered list of (name, T, reg_on)
LANES = {
    2: [('base_t2', 2, False), ('lr_t8', 8, True)],
    4: [('lr_t2',   2, True)],
    5: [('base_t8', 8, False)],
}


def run_one(gpu, name, T, reg_on):
    d = os.path.join(SWEEP_DIR, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'config_sweep.py'), 'w') as f:
        f.write(make_config(gpu, f'ts-{name}', T, reg_on))
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')) as f:
        m = f.read().replace('from config_snn_training import config', 'from config_sweep import config')
    with open(os.path.join(d, 'main_sweep.py'), 'w') as f:
        f.write(m)

    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'

    tag = f'rho={RHO:.1e}' if reg_on else 'no reg'
    print(f'[GPU {gpu}] START {name} (V16/CIFAR10, T={T}, {tag})', flush=True)
    with open(os.path.join(d, 'train.log'), 'w') as lf:
        rc = subprocess.Popen([PYTHON, os.path.join(d, 'main_sweep.py')], cwd=PROJECT_ROOT,
                              stdout=lf, stderr=subprocess.STDOUT, env=env).wait()
    print(f'[GPU {gpu}] DONE  {name} rc={rc}', flush=True)


def main():
    only = {int(g) for g in sys.argv[1].split(',')} if len(sys.argv) > 1 else None
    os.makedirs(SWEEP_DIR, exist_ok=True)
    ts = []
    for gpu, jobs in LANES.items():
        if only is not None and gpu not in only:
            continue
        t = threading.Thread(target=lambda g=gpu, js=jobs: [run_one(g, *j) for j in js])
        t.start(); ts.append(t)
    print(f'--- {len(ts)} lanes started ---', flush=True)
    for t in ts:
        t.join()
    print('--- all lanes finished ---', flush=True)


if __name__ == '__main__':
    main()
