"""
Spike-normalized lambda (auto-k) — first test.

Idea: lambda is dimensional. Its effect scales with how many spikes the network emits,
which is why the "optimal" lambda looked model-dependent (6x between VGG-C10 and R19-C10).
Divide it out and a single constant K = lambda * S_final should transfer.

S_final is not known at the start, but the spike-decay trajectory has nearly the same
shape in every setting (S_final / S_ep5 = 0.33 +- 7% across all 4 settings), so it is
predicted from epoch 5. Reg stays off until then -- which is required anyway, since
early regularization was shown to be destructive (early-reg: -22%p).

K = 6.75e-3 is the CIFAR-10 value fitted to a 70% spike target:
    VGG-C10  lambda 8.58e-8 x S 78.2K = 6.71e-3
    R19-C10  lambda 1.39e-8 x S 486K  = 6.77e-3
Prediction under test: both settings land near 70% spike from ONE constant, despite
their lambdas differing 6x. CIFAR-100 needs K ~2.5-3x larger (separate test).
"""

import subprocess
import os
import sys
import threading

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_auto_k')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

MEAS_EP = 5
DECAY = 0.33
ALPHA = 7          # matches the June sweeps that K was fitted to


def base_config():
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py'), 'r') as f:
        return f.read()


def make_config(gpu_id, exp_name, model, dataset, K):
    c = base_config()
    c = c.replace('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
                  f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"')
    c = c.replace("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='{exp_name}'")
    if model == 'VGG16':
        c = c.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    if dataset == 'CIFAR10':
        c = c.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")
    c = c.replace('conf.reg_spike_out_alpha=3  # temperature',
                  f'conf.reg_spike_out_alpha={ALPHA}  # temperature')
    c = c.replace('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
                  'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)')
    # NOTE: reg_spike_log_detail costs 24.5% of step time (244 -> 184 ms/step measured) --
    # its tf.sort/top_k run per layer, per time step, per batch. It feeds only
    # reg_detail.csv; s_count and the auto-k measurement do not depend on it.
    # Keep it ON while per-layer analysis is still needed; turn it OFF for final runs.
    c = c.replace('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
                  'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging')
    block = f"""conf.sc_loss_scd = False
        conf.reg_spike_auto_k = True
        conf.reg_spike_auto_k_const = {K}
        conf.reg_spike_auto_k_meas_ep = {MEAS_EP}
        conf.reg_spike_auto_k_decay = {DECAY}"""
    c = c.replace('conf.sc_loss_scd = False', block)
    return c


# gpu -> [(name, model, dataset, K)]
#
# Matrix tests two things at once:
#   (a) transfer  -- one K per dataset, across two architectures whose lambdas differ 6x
#   (b) control   -- a second K on CIFAR-10, to show K is a usable dial and not one lucky
#                    constant. K=1.43e-2 is the fitted value for a 60% spike target.
# Not run (already answerable from the June sweeps): K=6.75e-3 applied to CIFAR-100 gives
# lambda=7.8e-8 -> ~84% spike, i.e. undershoots the 70% target, as predicted.
LANES = {
    # lane 3 already produced autok_vgg_c10 (94.79%, spike 70.4%) -- the only run that fell
    # below every baseline, and contradicted by autok_vgg_c10_k60 (95.07% at MORE reg).
    # replicating it to see whether -0.21%p is real or a bad draw.
    3: [('autok_vgg_c10_run2', 'VGG16',    'CIFAR10',  6.75e-3)],   # target 70%, replicate
    4: [('autok_r19_c10',      'ResNet19', 'CIFAR10',  6.75e-3)],   # target 70%
    0: [('autok_vgg_c100',     'VGG16',    'CIFAR100', 2.04e-2)],   # target 70%
    1: [('autok_r19_c100',     'ResNet19', 'CIFAR100', 2.04e-2)],   # target 70%
    2: [('autok_vgg_c10_k60',  'VGG16',    'CIFAR10',  1.43e-2)],   # target 60%
    5: [('autok_r19_c10_k60',  'ResNet19', 'CIFAR10',  1.43e-2)],   # target 60%
}


def generate_main(run_dir):
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py'), 'r') as f:
        content = f.read()
    content = content.replace('from config_snn_training import config',
                              'from config_sweep import config')
    with open(os.path.join(run_dir, 'main_sweep.py'), 'w') as f:
        f.write(content)


def run_one(gpu, name, model, dataset, K):
    run_dir = os.path.join(SWEEP_DIR, name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config_sweep.py'), 'w') as f:
        f.write(make_config(gpu, f'autok-{name}', model, dataset, K))
    generate_main(run_dir)

    env = os.environ.copy()
    env['PYTHONPATH'] = run_dir + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'

    print(f'[GPU {gpu}] START {name} ({model}/{dataset}, K={K:.3e})', flush=True)
    with open(os.path.join(run_dir, 'train.log'), 'w') as lf:
        rc = subprocess.Popen([PYTHON, os.path.join(run_dir, 'main_sweep.py')],
                              cwd=PROJECT_ROOT, stdout=lf, stderr=subprocess.STDOUT,
                              env=env).wait()
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
