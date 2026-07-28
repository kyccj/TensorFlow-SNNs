'''
    VGG16 CIFAR10 - MaxNorm vs MaxNorm+Encourage sweep
    alpha=4, 310 epochs

    GPU 0-3: MaxNorm (lambdas: 1e-8, 1e-7, 3e-7, 1e-6)
    GPU 4-7: MaxNorm+Enc (lambdas: 1e-8, 1e-7, 3e-7, 1e-6)
'''
import subprocess
import os
import sys
import threading
import re

PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/miniconda3/envs/venv_1/lib'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

FIXED_ALPHA = 4
LAMBDAS = [1e-8, 1e-7, 3e-7, 1e-6]

GPU_EXPERIMENTS = {
    0: {'lmb': 1e-8,  'maxnorm': True,  'encourage': False, 'dir': '_vgg_c10_maxnorm/maxnorm_lmb_1e-08'},
    1: {'lmb': 1e-7,  'maxnorm': True,  'encourage': False, 'dir': '_vgg_c10_maxnorm/maxnorm_lmb_1e-07'},
    2: {'lmb': 3e-7,  'maxnorm': True,  'encourage': False, 'dir': '_vgg_c10_maxnorm/maxnorm_lmb_3e-07'},
    3: {'lmb': 1e-6,  'maxnorm': True,  'encourage': False, 'dir': '_vgg_c10_maxnorm/maxnorm_lmb_1e-06'},
    4: {'lmb': 1e-8,  'maxnorm': True,  'encourage': True,  'dir': '_vgg_c10_maxnorm/maxnorm_enc_lmb_1e-08'},
    5: {'lmb': 1e-7,  'maxnorm': True,  'encourage': True,  'dir': '_vgg_c10_maxnorm/maxnorm_enc_lmb_1e-07'},
    6: {'lmb': 3e-7,  'maxnorm': True,  'encourage': True,  'dir': '_vgg_c10_maxnorm/maxnorm_enc_lmb_3e-07'},
    7: {'lmb': 1e-6,  'maxnorm': True,  'encourage': True,  'dir': '_vgg_c10_maxnorm/maxnorm_enc_lmb_1e-06'},
}


def generate_config(gpu, exp):
    run_dir = os.path.join(PROJECT_ROOT, exp['dir'])
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, 'config_sweep.py')

    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        content = f.read()

    # GPU
    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu}"'
    )
    # Model: VGG16
    content = content.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    # Dataset: CIFAR10
    content = content.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")
    # Alpha
    content = content.replace(
        'conf.reg_spike_out_alpha=3  # temperature',
        f'conf.reg_spike_out_alpha={FIXED_ALPHA}  # temperature'
    )
    # Lambda
    content = content.replace(
        'conf.reg_spike_out_const=1E-8',
        f'conf.reg_spike_out_const={exp["lmb"]}'
    )
    # Enable WTA-Rev
    content = content.replace(
        '#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
        'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'
    )
    # MaxNorm
    if exp['maxnorm']:
        content = content.replace(
            '#conf.reg_spike_out_sc_maxnorm=True  # max-normalization: spike_count/max(spike_count), strong WTA',
            'conf.reg_spike_out_sc_maxnorm=True  # max-normalization: spike_count/max(spike_count), strong WTA'
        )
    # Encourage
    if exp['encourage']:
        content = content.replace(
            '#conf.reg_spike_out_encourage=True   # encourage winners to fire: loss += l2_norm((1-spike)*(1-sc_rate))',
            'conf.reg_spike_out_encourage=True   # encourage winners to fire: loss += l2_norm((1-spike)*(1-sc_rate))'
        )
    # Detail logging
    content = content.replace(
        '#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
        'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'
    )
    # Experiment name
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        f"conf.exp_set_name='{exp['dir']}'"
    )

    with open(config_path, 'w') as f:
        f.write(content)


def generate_main(exp):
    run_dir = os.path.join(PROJECT_ROOT, exp['dir'])
    main_path = os.path.join(run_dir, 'main_sweep.py')
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')) as f:
        content = f.read()
    content = content.replace(
        'from config_snn_training import config',
        'from config_sweep import config'
    )
    with open(main_path, 'w') as f:
        f.write(content)


def gpu_pipeline(gpu, exp):
    tag = 'MaxNorm+Enc' if exp['encourage'] else 'MaxNorm'
    thread_name = f'GPU{gpu}-{tag}-lmb{exp["lmb"]:.0e}'

    print(f'[{thread_name}] Starting experiment', flush=True)

    generate_config(gpu, exp)
    generate_main(exp)

    run_dir = os.path.join(PROJECT_ROOT, exp['dir'])
    log_file = os.path.join(run_dir, 'train.log')

    env = os.environ.copy()
    env['PYTHONPATH'] = run_dir + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/miniconda3/envs/venv_1'

    with open(log_file, 'w') as lf:
        p = subprocess.Popen(
            [PYTHON, os.path.join(run_dir, 'main_sweep.py')],
            cwd=PROJECT_ROOT,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
        )
    p.wait()
    status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'

    best_acc = '?'
    if os.path.exists(log_file):
        with open(log_file) as f:
            accs = re.findall(r'best_val_acc: ([0-9.]+)', f.read())
            if accs:
                best_acc = accs[-1]

    print(f'[{thread_name}] DONE: {status} (best_acc={best_acc})', flush=True)


def main():
    print(f'VGG16-C10 MaxNorm sweep (alpha={FIXED_ALPHA})')
    print('GPU mapping:')
    for gpu, exp in GPU_EXPERIMENTS.items():
        tag = 'MaxNorm+Enc' if exp['encourage'] else 'MaxNorm'
        print(f'  GPU {gpu}: {tag} lmb={exp["lmb"]}')

    threads = []
    for gpu, exp in GPU_EXPERIMENTS.items():
        t = threading.Thread(target=gpu_pipeline, args=(gpu, exp), name=f'gpu{gpu}')
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print('\n' + '=' * 60)
    print('  ALL EXPERIMENTS COMPLETED')
    print('=' * 60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted.')
        sys.exit(1)
