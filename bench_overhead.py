"""
Measure where the per-step time goes. Runs a few epochs of VGG-C10 under different
instrumentation settings on one GPU, sequentially, and reports ms/step.

reg_spike_log_detail runs tf.sort + tf.math.top_k per layer, per time step, per batch
(~36k sorts/epoch). Those feed reg_detail.csv only -- they do not affect training math.
This measures what turning them off actually buys.
"""

import subprocess
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(PROJECT_ROOT, '_bench')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
EPOCHS = 4


def make_config(gpu_id, name, log_detail, reg_on):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    c = c.replace('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
                  f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"')
    c = c.replace("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='bench-{name}'")
    c = c.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    c = c.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")
    c = c.replace('conf.train_epoch = 310', f'conf.train_epoch = {EPOCHS}')
    c = c.replace('conf.reg_spike_out_alpha=3  # temperature',
                  'conf.reg_spike_out_alpha=7  # temperature')
    if reg_on:
        c = c.replace('conf.reg_spike_out_const=1E-8', 'conf.reg_spike_out_const=1e-07')
        c = c.replace('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
                      'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)')
    else:
        c = c.replace('conf.reg_spike_out=True', 'conf.reg_spike_out=False')
    if log_detail:
        c = c.replace('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
                      'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging')
    return c


# (name, log_detail, reg_on)
CASES = [
    ('reg_logdetail', True,  True),    # what every experiment so far has used
    ('reg_nolog',     False, True),    # same training, instrumentation off
    ('noreg_nolog',   False, False),   # no spike regularization at all
]


def run(name, log_detail, reg_on, gpu):
    d = os.path.join(BENCH_DIR, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'config_sweep.py'), 'w') as f:
        f.write(make_config(gpu, name, log_detail, reg_on))
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')) as f:
        m = f.read().replace('from config_snn_training import config', 'from config_sweep import config')
    with open(os.path.join(d, 'main_sweep.py'), 'w') as f:
        f.write(m)

    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'

    log = os.path.join(d, 'train.log')
    print(f'[bench] {name}: log_detail={log_detail} reg={reg_on} ...', flush=True)
    with open(log, 'w') as lf:
        subprocess.Popen([PYTHON, os.path.join(d, 'main_sweep.py')], cwd=PROJECT_ROOT,
                         stdout=lf, stderr=subprocess.STDOUT, env=env).wait()

    steps = re.findall(r'- (\d+)s (\d+)ms/step', open(log).read())
    # drop the first epoch (graph tracing / warmup)
    ms = [int(s[1]) for s in steps][1:]
    return ms


def main():
    gpu = sys.argv[1] if len(sys.argv) > 1 else '2'
    os.makedirs(BENCH_DIR, exist_ok=True)
    res = {}
    for name, ld, ro in CASES:
        ms = run(name, ld, ro, gpu)
        res[name] = ms
        print(f'[bench] {name}: {ms} ms/step', flush=True)

    print('\n=== RESULT (warmup epoch excluded) ===', flush=True)
    ref = None
    for name, _, _ in CASES:
        ms = res.get(name) or []
        if not ms:
            print(f'{name:16s} (no timing captured)'); continue
        avg = sum(ms) / len(ms)
        if ref is None:
            ref = avg
        print(f'{name:16s} {avg:7.1f} ms/step   x{avg/ref:5.2f} vs reg_logdetail   '
              f'epoch {avg*500/1000:6.1f}s   310ep {avg*500*310/3.6e6:5.1f}h', flush=True)


if __name__ == '__main__':
    main()
