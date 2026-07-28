'''
    Adaptive lambda experiment: R19-C10 (GPU 3) + VGG16-C10 (GPU 5)
    Late-start (ep200) + accuracy-gated adaptive lambda
    Uses original softmax WTA method with adaptive lambda scaling
'''
import subprocess
import os

PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/miniconda3/envs/venv_1/lib'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    {'model': 'ResNet19', 'dataset': 'CIFAR10', 'gpu': 3,
     'dir': '_adaptive_lambda/r19_c10'},
    {'model': 'VGG16', 'dataset': 'CIFAR10', 'gpu': 5,
     'dir': '_adaptive_lambda/vgg_c10'},
]


def generate_config(exp):
    run_dir = os.path.join(PROJECT_ROOT, exp['dir'])
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, 'config_sweep.py')

    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        content = f.read()

    # GPU
    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]=\"{exp["gpu"]}\"'
    )

    # Model
    if exp['model'] == 'VGG16':
        content = content.replace("conf.model='ResNet19'", "conf.model='VGG16'")

    # Dataset
    content = content.replace("conf.dataset='CIFAR100'", f"conf.dataset='{exp['dataset']}'")

    # Enable adaptive lambda + spike reg with original softmax method
    # Keep reg enabled (if True stays), add adaptive flags after reg block
    content = content.replace(
        "#\nconfig.set()",
        f"# adaptive lambda\n"
        f"conf.reg_spike_adaptive = True\n"
        f"conf.reg_spike_adaptive_start_ep = 200\n"
        f"conf.reg_spike_adaptive_lambda_init = 1E-9\n"
        f"conf.reg_spike_adaptive_margin = 0.005\n"
        f"conf.reg_spike_adaptive_step_up = 0.3\n"
        f"conf.reg_spike_adaptive_step_down = 0.5\n"
        f"conf.reg_spike_adaptive_lambda_max = 1E-5\n"
        f"\n"
        f"conf.reg_spike_log_detail = True\n"
        f"\n#\nconfig.set()"
    )

    # exp name
    model_short = 'r19' if exp['model'] == 'ResNet19' else 'vgg'
    ds_short = 'c10' if exp['dataset'] == 'CIFAR10' else 'c100'
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        f"conf.exp_set_name='adaptive-lambda-{model_short}-{ds_short}'"
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
        print(f'[GPU {exp["gpu"]}] {model_short} C10 Adaptive Lambda -> {log_file}')

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
    for exp, _, lf in processes:
        print(f'  tail -f {lf}')

    try:
        for exp, p, log_file in processes:
            p.wait()
            status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'
            model_short = 'R19' if exp['model'] == 'ResNet19' else 'VGG'
            print(f'[GPU {exp["gpu"]}] {model_short} finished: {status}')
    except KeyboardInterrupt:
        print('\nInterrupted - terminating...')
        for _, p, _ in processes:
            p.terminate()
        for _, p, _ in processes:
            p.wait()


if __name__ == '__main__':
    main()
