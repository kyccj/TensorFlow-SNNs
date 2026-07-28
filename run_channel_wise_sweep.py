#!/usr/bin/env python3
"""Channel-wise WTA sweep on GPU 4 (VGG16-C10, sequential)"""

import os
import shutil
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, '_vgg_c10_lambda_sweep/lmb_1e-07')
OUT_ROOT = os.path.join(BASE_DIR, '_channel_wise_wta')
GPU = '4'

LAMBDAS = [1e-7, 3e-7, 1e-6]

os.makedirs(OUT_ROOT, exist_ok=True)

for lmb in LAMBDAS:
    lmb_str = f'{lmb:.0e}'.replace('+', '')
    exp_name = f'vgg_c10_lmb_{lmb_str}'
    exp_dir = os.path.join(OUT_ROOT, exp_name)

    if os.path.exists(exp_dir):
        print(f'[SKIP] {exp_name} already exists')
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
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{GPU}"'
    )

    # Set experiment name
    content = content.replace(
        "conf.exp_set_name='vgg-c10-lmb-1e-07'",
        f"conf.exp_set_name='ch-wta-vgg-c10-lmb-{lmb_str}'"
    )

    # Set lambda
    content = content.replace(
        'conf.reg_spike_out_const=1e-07',
        f'conf.reg_spike_out_const={lmb}'
    )

    # Add channel-wise flag before config.set()
    content = content.replace(
        'config.set()',
        'conf.reg_spike_channel_wise = True\n\nconfig.set()'
    )

    with open(os.path.join(exp_dir, 'config_sweep.py'), 'w') as f:
        f.write(content)

    print(f'[RUN] {exp_name} (lambda={lmb}) on GPU {GPU}')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = GPU
    env['LD_LIBRARY_PATH'] = '/home/kyccj/anaconda3/envs/venv_1/lib:' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'
    env['PYTHONPATH'] = BASE_DIR + ':' + env.get('PYTHONPATH', '')

    proc = subprocess.run(
        ['/home/kyccj/anaconda3/envs/venv_1/bin/python', 'main_sweep.py'],
        env=env,
        cwd=exp_dir
    )

    if proc.returncode != 0:
        print(f'[ERROR] {exp_name} failed with code {proc.returncode}')
        break
    else:
        print(f'[DONE] {exp_name}')

    time.sleep(5)

print('[ALL DONE]')
