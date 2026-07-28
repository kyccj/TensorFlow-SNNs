'''
    Max-normalization vs Softmax WTA comparison
    VGG16 CIFAR10, alpha=4 fixed, reg_spike_log_detail=True
    310 epochs full training

    GPU 0-3: Softmax WTA (current, wta_rev=True)
    GPU 4-7: Max-norm WTA (new, sc_maxnorm=True + wta_rev=True)
    Lambda: 1e-8, 1e-7, 1e-6, 3e-7
'''
import subprocess
import os

PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/miniconda3/envs/venv_1/lib'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

FIXED_ALPHA = 4
LAMBDAS = [1e-8, 1e-7, 3e-7, 1e-6]

EXPERIMENTS = []
for i, lmb in enumerate(LAMBDAS):
    # Softmax WTA (current method)
    EXPERIMENTS.append({
        'lmb': lmb, 'gpu': i, 'maxnorm': False,
        'dir': f'_compare_maxnorm/softmax_lmb_{lmb:.0e}',
    })
    # Max-norm WTA (new method)
    EXPERIMENTS.append({
        'lmb': lmb, 'gpu': i + 4, 'maxnorm': True,
        'dir': f'_compare_maxnorm/maxnorm_lmb_{lmb:.0e}',
    })


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

    # VGG16 + CIFAR10
    content = content.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    content = content.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")

    # alpha
    content = content.replace(
        'conf.reg_spike_out_alpha=3  # temperature',
        f'conf.reg_spike_out_alpha={FIXED_ALPHA}  # temperature'
    )

    # lambda
    content = content.replace(
        'conf.reg_spike_out_const=1E-8',
        f'conf.reg_spike_out_const={exp["lmb"]}'
    )

    # wta_rev: enable for both (gradient flow to non-firing neurons)
    content = content.replace(
        '#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
        'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'
    )

    # max-norm: enable only for maxnorm experiments
    if exp['maxnorm']:
        content = content.replace(
            '#conf.reg_spike_out_sc_maxnorm=True  # max-normalization: spike_count/max(spike_count), strong WTA',
            'conf.reg_spike_out_sc_maxnorm=True  # max-normalization: spike_count/max(spike_count), strong WTA'
        )

    # reg detail logging - always enable
    content = content.replace(
        '#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
        'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'
    )

    # exp name
    tag = 'maxnorm' if exp['maxnorm'] else 'softmax'
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


def main():
    processes = []

    for exp in EXPERIMENTS:
        generate_config(exp)
        generate_main(exp)

        run_dir = os.path.join(PROJECT_ROOT, exp['dir'])
        log_file = os.path.join(run_dir, 'train.log')
        tag = 'MaxNorm' if exp['maxnorm'] else 'Softmax'
        print(f'[GPU {exp["gpu"]}] {tag} a={FIXED_ALPHA} lmb={exp["lmb"]} -> {log_file}')

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

    print(f'\n--- {len(processes)} experiments launched (4 Softmax + 4 MaxNorm) ---')
    print('Monitor:')
    for exp, _, _ in processes:
        print(f'  tail -f {exp["dir"]}/train.log')

    try:
        for exp, p, log_file in processes:
            p.wait()
            tag = 'MaxNorm' if exp['maxnorm'] else 'Softmax'
            status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'
            print(f'[GPU {exp["gpu"]}] {tag} lmb={exp["lmb"]} finished: {status}')
    except KeyboardInterrupt:
        print('\nInterrupted - terminating...')
        for _, p, _ in processes:
            p.terminate()
        for _, p, _ in processes:
            p.wait()


if __name__ == '__main__':
    main()
