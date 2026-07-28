'''
    Entropy-based WTA sweep: R19-C10 (GPU 0-3) + VGG-C10 (GPU 4-7)
    Lambda values: 1e-8, 1e-7, 3e-7, 1e-6
    Fixed: alpha=4, entropy + wta_rev
'''
import subprocess
import os

PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/miniconda3/envs/venv_1/lib'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    # R19-C10
    {'model': 'ResNet19', 'dataset': 'CIFAR10', 'lmb': 1e-8, 'gpu': 0, 'dir': '_entropy_sweep/r19_c10_lmb_1e-08'},
    {'model': 'ResNet19', 'dataset': 'CIFAR10', 'lmb': 1e-7, 'gpu': 1, 'dir': '_entropy_sweep/r19_c10_lmb_1e-07'},
    {'model': 'ResNet19', 'dataset': 'CIFAR10', 'lmb': 3e-7, 'gpu': 2, 'dir': '_entropy_sweep/r19_c10_lmb_3e-07'},
    {'model': 'ResNet19', 'dataset': 'CIFAR10', 'lmb': 1e-6, 'gpu': 3, 'dir': '_entropy_sweep/r19_c10_lmb_1e-06'},
    # VGG-C10
    {'model': 'VGG16', 'dataset': 'CIFAR10', 'lmb': 1e-8, 'gpu': 4, 'dir': '_entropy_sweep/vgg_c10_lmb_1e-08'},
    {'model': 'VGG16', 'dataset': 'CIFAR10', 'lmb': 1e-7, 'gpu': 5, 'dir': '_entropy_sweep/vgg_c10_lmb_1e-07'},
    {'model': 'VGG16', 'dataset': 'CIFAR10', 'lmb': 3e-7, 'gpu': 6, 'dir': '_entropy_sweep/vgg_c10_lmb_3e-07'},
    {'model': 'VGG16', 'dataset': 'CIFAR10', 'lmb': 1e-6, 'gpu': 7, 'dir': '_entropy_sweep/vgg_c10_lmb_1e-06'},
]

FIXED_ALPHA = 4


def generate_config(exp):
    run_dir = os.path.join(PROJECT_ROOT, exp['dir'])
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, 'config_sweep.py')

    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        content = f.read()

    # GPU
    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{exp["gpu"]}"'
    )

    # Model
    if exp['model'] == 'VGG16':
        content = content.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    # Dataset
    content = content.replace("conf.dataset='CIFAR100'", f"conf.dataset='{exp['dataset']}'")

    # alpha (fixed)
    content = content.replace(
        'conf.reg_spike_out_alpha=3  # temperature',
        f'conf.reg_spike_out_alpha={FIXED_ALPHA}  # temperature'
    )

    # lambda
    content = content.replace(
        'conf.reg_spike_out_const=1E-8',
        f'conf.reg_spike_out_const={exp["lmb"]}'
    )

    # Enable entropy + wta_rev
    content = content.replace(
        '#conf.reg_spike_out_entropy=True    # entropy-based WTA: weight = -(1+log(p)), moderate differentiation',
        'conf.reg_spike_out_entropy=True    # entropy-based WTA: weight = -(1+log(p)), moderate differentiation'
    )
    content = content.replace(
        '#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
        'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'
    )

    # reg detail logging
    content = content.replace(
        '#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
        'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'
    )

    # exp name
    model_short = 'r19' if exp['model'] == 'ResNet19' else 'vgg'
    ds_short = 'c10' if exp['dataset'] == 'CIFAR10' else 'c100'
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        f"conf.exp_set_name='{model_short}-{ds_short}-entropy-lmb-{exp['lmb']:.0e}'"
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


def main():
    processes = []

    for exp in EXPERIMENTS:
        generate_config(exp)
        generate_main(exp)

        run_dir = os.path.join(PROJECT_ROOT, exp['dir'])
        log_file = os.path.join(run_dir, 'train.log')
        model_short = 'R19' if exp['model'] == 'ResNet19' else 'VGG'
        print(f'[GPU {exp["gpu"]}] {model_short} C10 Entropy+WTA-Rev alpha={FIXED_ALPHA} lambda={exp["lmb"]} -> {log_file}')

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
        processes.append((exp, p, log_file))

    print(f'\n--- {len(processes)} experiments launched ---')
    print('Monitor:')
    for exp, _, _ in processes:
        print(f'  tail -f {exp["dir"]}/train.log')

    try:
        for exp, p, log_file in processes:
            p.wait()
            status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'
            model_short = 'R19' if exp['model'] == 'ResNet19' else 'VGG'
            print(f'[GPU {exp["gpu"]}] {model_short} lambda={exp["lmb"]} finished: {status}')
    except KeyboardInterrupt:
        print('\nInterrupted - terminating...')
        for _, p, _ in processes:
            p.terminate()
        for _, p, _ in processes:
            p.wait()


if __name__ == '__main__':
    main()
