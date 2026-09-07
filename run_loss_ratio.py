"""
Loss-ratio control — the only method so far that needs nothing known in advance.

Rule:   lambda = rho * L_task / R      (solved exactly each epoch, no gain, no iteration)

R is the raw pre-lambda regularization value and L_task the task loss; both are already
computed inside the run. No baseline, no spike target, no calibration sweep. rho is a
single published constant, like Adam's beta1.

rho = 2.6e-4 is the geometric mean of the value measured at matched sparsity (70%) across
the four June settings, which spanned 1.5e-4 .. 3.5e-4. Fitting rho on the same four
settings we test here makes this in-sample -- a real test needs a fifth setting.

Predicted fixed points at rho=2.6e-4 (from the measured R(lambda) power law):
    VGG-C10   7.4e-8  (ideal for 70%: 8.6e-8, x0.86)
    VGG-C100  1.5e-7  (1.98e-7, x0.77)
    R19-C10   1.7e-8  (1.40e-8, x1.23)
    R19-C100  4.4e-8  (5.05e-8, x0.87)
so sparsity should land in roughly a 66-76% band rather than exactly 70%.
"""

import subprocess
import os
import sys
import threading

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_loss_ratio')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

RHO = float(os.environ.get('LR_RHO', '5.8e-4'))
START_EP = 0
ALPHA = 7          # matches the June sweeps rho was fitted to


def make_config(gpu_id, exp_name, model, dataset, rho):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
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
    # keep per-layer logging on while the analysis is still needed
    c = c.replace('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
                  'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging')
    block = f"""conf.sc_loss_scd = False
        conf.reg_spike_loss_ratio = True
        conf.reg_spike_loss_ratio_target = {rho}
        conf.reg_spike_loss_ratio_start_ep = {START_EP}"""
    c = c.replace('conf.sc_loss_scd = False', block)
    return c


# gpu -> (name, model, dataset, rho).  GPU 0 and 2 belong to other work; these are the
# lanes the auto-k runs vacate.
LANES = {
    3: (os.environ.get('LR_NAME3', 'lr_vgg_c10'), 'VGG16', 'CIFAR10', RHO),
    4: ('lr_r19_c10',   'ResNet19', 'CIFAR10',  RHO),
    5: ('lr_vgg_c100',  'VGG16',    'CIFAR100', RHO),
    1: ('lr_r19_c100',  'ResNet19', 'CIFAR100', RHO),
    # transfer test: the SAME rho that calibrated VGG-C10 (5.8e-4), applied to VGG-C100.
    # its own calibrated value would be ~7.2e-4, so landing near 70% here means the
    # constant carries across datasets; landing high means it does not.
    0: ('lr_vgg_c100_rho58', 'VGG16', 'CIFAR100', RHO),
}
# architecture-transfer test: the rho calibrated on VGG-C10 (5.8e-4), applied to ResNet19.
# auto-k failed exactly here (R19-C10 missed 70% by 15.7pp). VGG-C10 landed at 69.6% and
# VGG-C100 at 66.3% with this rho, so 66-74% here means rho carries across architectures.
LANES_R19 = {
    0: ('lr_r19_c10_rho58',  'ResNet19', 'CIFAR10',  RHO),
    3: ('lr_r19_c100_rho58', 'ResNet19', 'CIFAR100', RHO),
    # VGG-C100 at rho=5.8e-4 lost 0.91%p, the only run outside its baseline spread
    # (0.76%p). n=1, so a replicate decides whether that is real.
    5: ('lr_vgg_c100_rho58_run2', 'VGG16', 'CIFAR100', RHO),
}
if os.environ.get('LR_R19'):
    LANES = LANES_R19


def run_one(gpu, name, model, dataset, rho):
    d = os.path.join(SWEEP_DIR, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'config_sweep.py'), 'w') as f:
        f.write(make_config(gpu, f'lossratio-{name}', model, dataset, rho))
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')) as f:
        m = f.read().replace('from config_snn_training import config', 'from config_sweep import config')
    with open(os.path.join(d, 'main_sweep.py'), 'w') as f:
        f.write(m)

    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'

    print(f'[GPU {gpu}] START {name} ({model}/{dataset}, rho={rho:.2e})', flush=True)
    with open(os.path.join(d, 'train.log'), 'w') as lf:
        rc = subprocess.Popen([PYTHON, os.path.join(d, 'main_sweep.py')], cwd=PROJECT_ROOT,
                              stdout=lf, stderr=subprocess.STDOUT, env=env).wait()
    print(f'[GPU {gpu}] DONE  {name} rc={rc}', flush=True)


def main():
    only = {int(g) for g in sys.argv[1].split(',')} if len(sys.argv) > 1 else None
    os.makedirs(SWEEP_DIR, exist_ok=True)
    ts = []
    for gpu, job in LANES.items():
        if only is not None and gpu not in only:
            continue
        t = threading.Thread(target=run_one, args=(gpu,) + job)
        t.start(); ts.append(t)
    print(f'--- {len(ts)} lanes started ---', flush=True)
    for t in ts:
        t.join()
    print('--- all lanes finished ---', flush=True)


if __name__ == '__main__':
    main()
