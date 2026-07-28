#!/usr/bin/env python3
"""Launch channel-wise WTA (queued) + accumulated loss experiments in parallel on free GPUs."""

import os
import shutil
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, '_vgg_c10_lambda_sweep/lmb_1e-07')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'

EXPERIMENTS = [
    # (gpu, output_root, exp_name_prefix, lambda, extra_flags)
    # Channel-wise WTA (queued experiments)
    ('0', '_channel_wise_wta', 'vgg_c10_lmb_3e-07', 3e-7,
     'conf.reg_spike_channel_wise = True'),
    ('1', '_channel_wise_wta', 'vgg_c10_lmb_1e-06', 1e-6,
     'conf.reg_spike_channel_wise = True'),
    # Accumulated loss (global softmax)
    ('2', '_accum_loss', 'vgg_c10_lmb_1e-07', 1e-7,
     'conf.reg_spike_accum_loss = True'),
    ('3', '_accum_loss', 'vgg_c10_lmb_3e-07', 3e-7,
     'conf.reg_spike_accum_loss = True'),
    ('5', '_accum_loss', 'vgg_c10_lmb_1e-06', 1e-6,
     'conf.reg_spike_accum_loss = True'),
]

procs = []

for gpu, out_root, exp_name, lmb, extra_flags in EXPERIMENTS:
    out_dir = os.path.join(BASE_DIR, out_root)
    exp_dir = os.path.join(out_dir, exp_name)

    if os.path.exists(exp_dir):
        print(f'[SKIP] {out_root}/{exp_name} already exists')
        continue

    os.makedirs(exp_dir, exist_ok=True)

    # Copy main_sweep.py
    shutil.copy2(os.path.join(TEMPLATE_DIR, 'main_sweep.py'), exp_dir)

    # Read template config and modify
    with open(os.path.join(TEMPLATE_DIR, 'config_sweep.py'), 'r') as f:
        content = f.read()

    # Set GPU
    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="4"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu}"'
    )

    lmb_str = f'{lmb:.0e}'.replace('+', '')
    method = 'ch-wta' if 'channel_wise' in extra_flags else 'accum'

    # Set experiment name
    content = content.replace(
        "conf.exp_set_name='vgg-c10-lmb-1e-07'",
        f"conf.exp_set_name='{method}-vgg-c10-lmb-{lmb_str}'"
    )

    # Set lambda
    content = content.replace(
        'conf.reg_spike_out_const=1e-07',
        f'conf.reg_spike_out_const={lmb}'
    )

    # Add extra flags before config.set()
    content = content.replace(
        'config.set()',
        f'{extra_flags}\n\nconfig.set()'
    )

    with open(os.path.join(exp_dir, 'config_sweep.py'), 'w') as f:
        f.write(content)

    print(f'[LAUNCH] {out_root}/{exp_name} (lambda={lmb}) on GPU {gpu}')

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu
    env['LD_LIBRARY_PATH'] = '/home/kyccj/anaconda3/envs/venv_1/lib:' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'
    env['PYTHONPATH'] = BASE_DIR + ':' + env.get('PYTHONPATH', '')

    log_file = os.path.join(BASE_DIR, f'{out_root.strip("_")}_{exp_name}.log')
    with open(log_file, 'w') as log_f:
        proc = subprocess.Popen(
            [PYTHON, 'main_sweep.py'],
            env=env,
            cwd=exp_dir,
            stdout=log_f,
            stderr=subprocess.STDOUT
        )
    procs.append((f'{out_root}/{exp_name}', gpu, proc))
    print(f'  PID={proc.pid}, log={log_file}')

print(f'\n[SUMMARY] Launched {len(procs)} experiments:')
for name, gpu, proc in procs:
    print(f'  GPU {gpu}: {name} (PID {proc.pid})')

print('\nAll running in background. Check with: nvidia-smi')
