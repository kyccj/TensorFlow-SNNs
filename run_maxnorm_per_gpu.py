'''
    Max-normalization vs Softmax WTA comparison - per-GPU pipeline
    Each GPU starts as soon as run4 finishes on that GPU.
    VGG16 CIFAR10, alpha=4, reg_spike_log_detail=True, 310 epochs

    GPU 0-3: Softmax WTA (current)
    GPU 4-7: MaxNorm WTA (new)
'''
import subprocess
import os
import sys
import time
import threading
import re

PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/miniconda3/envs/venv_1/lib'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

FIXED_ALPHA = 4
LAMBDAS = [1e-8, 1e-7, 3e-7, 1e-6]

# GPU -> experiment mapping
GPU_EXPERIMENTS = {
    0: {'lmb': 1e-8, 'maxnorm': False, 'dir': '_compare_maxnorm/softmax_lmb_1e-08'},
    1: {'lmb': 1e-7, 'maxnorm': False, 'dir': '_compare_maxnorm/softmax_lmb_1e-07'},
    2: {'lmb': 3e-7, 'maxnorm': False, 'dir': '_compare_maxnorm/softmax_lmb_3e-07'},
    3: {'lmb': 1e-6, 'maxnorm': False, 'dir': '_compare_maxnorm/softmax_lmb_1e-06'},
    4: {'lmb': 1e-8, 'maxnorm': True,  'dir': '_compare_maxnorm/maxnorm_lmb_1e-08'},
    5: {'lmb': 1e-7, 'maxnorm': True,  'dir': '_compare_maxnorm/maxnorm_lmb_1e-07'},
    6: {'lmb': 3e-7, 'maxnorm': True,  'dir': '_compare_maxnorm/maxnorm_lmb_3e-07'},
    7: {'lmb': 1e-6, 'maxnorm': True,  'dir': '_compare_maxnorm/maxnorm_lmb_1e-06'},
}

# Run4 GPU mapping (to check if run4 is done on each GPU)
RUN4_LMB_BY_GPU = {
    0: '1e-09', 1: '5e-09', 2: '1e-08', 3: '3e-08',
    4: '1e-07', 5: '3e-07', 6: '5e-07', 7: '1e-06',
}


def generate_config(gpu, exp):
    run_dir = os.path.join(PROJECT_ROOT, exp['dir'])
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, 'config_sweep.py')

    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        content = f.read()

    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu}"'
    )
    content = content.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    content = content.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")
    content = content.replace(
        'conf.reg_spike_out_alpha=3  # temperature',
        f'conf.reg_spike_out_alpha={FIXED_ALPHA}  # temperature'
    )
    content = content.replace(
        'conf.reg_spike_out_const=1E-8',
        f'conf.reg_spike_out_const={exp["lmb"]}'
    )
    content = content.replace(
        '#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
        'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'
    )
    if exp['maxnorm']:
        content = content.replace(
            '#conf.reg_spike_out_sc_maxnorm=True  # max-normalization: spike_count/max(spike_count), strong WTA',
            'conf.reg_spike_out_sc_maxnorm=True  # max-normalization: spike_count/max(spike_count), strong WTA'
        )
    content = content.replace(
        '#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
        'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'
    )
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


def is_run4_done(gpu):
    lmb_tag = RUN4_LMB_BY_GPU[gpu]
    result = subprocess.run(
        ['pgrep', '-f', f'_r19_c10_lambda_sweep_run4/lmb_{lmb_tag}/main_sweep'],
        capture_output=True, text=True
    )
    return result.returncode != 0


def gpu_pipeline(gpu, exp):
    tag = 'MaxNorm' if exp['maxnorm'] else 'Softmax'
    thread_name = f'GPU{gpu}-{tag}-lmb{exp["lmb"]:.0e}'

    # Wait for run4 to finish on this GPU
    if not is_run4_done(gpu):
        print(f'[{thread_name}] waiting for run4 on GPU {gpu}...', flush=True)
        while not is_run4_done(gpu):
            time.sleep(15)
    print(f'[{thread_name}] GPU {gpu} free, starting experiment', flush=True)

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
    print(f'MaxNorm vs Softmax comparison (alpha={FIXED_ALPHA})')
    print('GPU mapping:')
    for gpu, exp in GPU_EXPERIMENTS.items():
        tag = 'MaxNorm' if exp['maxnorm'] else 'Softmax'
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
