'''
    Adaptive Lambda v2: Early start + aggressive steps

    Key changes from v1:
    - start_ep=1 (was 200) - apply reg from the beginning
    - step_up=1.0 or 2.0 (was 0.3) - λ doubles or triples each epoch
    - step_down=0.3 (was 0.5) - gentler decay on accuracy drop

    Motivation: Fixed λ=3e-8 from ep1 achieves 30% spike reduction with no acc loss.
    Adaptive v1 failed because it started too late (ep200) and grew λ too slowly (×1.3).

    Experiments:
    GPU 0: R19-C10, step_up=1.0 (λ×2.0/ep)
    GPU 1: R19-C10, step_up=2.0 (λ×3.0/ep)
    GPU 2: VGG-C10, step_up=1.0 (λ×2.0/ep)
    GPU 5: VGG-C10, step_up=2.0 (λ×3.0/ep)
'''
import subprocess
import os

PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/miniconda3/envs/venv_1/lib'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    {'model': 'ResNet19', 'dataset': 'CIFAR10', 'gpu': 0,
     'dir': '_adaptive_lambda_v2/r19_c10_su1.0',
     'start_ep': 1, 'step_up': 1.0, 'step_down': 0.3},
    {'model': 'ResNet19', 'dataset': 'CIFAR10', 'gpu': 1,
     'dir': '_adaptive_lambda_v2/r19_c10_su2.0',
     'start_ep': 1, 'step_up': 2.0, 'step_down': 0.3},
    {'model': 'VGG16', 'dataset': 'CIFAR10', 'gpu': 2,
     'dir': '_adaptive_lambda_v2/vgg_c10_su1.0',
     'start_ep': 1, 'step_up': 1.0, 'step_down': 0.3},
    {'model': 'VGG16', 'dataset': 'CIFAR10', 'gpu': 5,
     'dir': '_adaptive_lambda_v2/vgg_c10_su2.0',
     'start_ep': 1, 'step_up': 2.0, 'step_down': 0.3},
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
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{exp["gpu"]}"'
    )

    # Model
    if exp['model'] == 'VGG16':
        content = content.replace("conf.model='ResNet19'", "conf.model='VGG16'")

    # Dataset
    content = content.replace("conf.dataset='CIFAR100'", f"conf.dataset='{exp['dataset']}'")

    # Enable adaptive lambda v2
    content = content.replace(
        "#\nconfig.set()",
        f"# adaptive lambda v2: early start + aggressive steps\n"
        f"conf.reg_spike_adaptive = True\n"
        f"conf.reg_spike_adaptive_start_ep = {exp['start_ep']}\n"
        f"conf.reg_spike_adaptive_lambda_init = 1E-9\n"
        f"conf.reg_spike_adaptive_margin = 0.005\n"
        f"conf.reg_spike_adaptive_step_up = {exp['step_up']}\n"
        f"conf.reg_spike_adaptive_step_down = {exp['step_down']}\n"
        f"conf.reg_spike_adaptive_lambda_max = 1E-5\n"
        f"\n"
        f"conf.reg_spike_log_detail = True\n"
        f"\n#\nconfig.set()"
    )

    # exp name
    model_short = 'r19' if exp['model'] == 'ResNet19' else 'vgg'
    ds_short = 'c10' if exp['dataset'] == 'CIFAR10' else 'c100'
    su = exp['step_up']
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        f"conf.exp_set_name='adp-v2-{model_short}-{ds_short}-su{su}'"
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
        su = exp['step_up']
        sd = exp['step_down']
        print(f'[GPU {exp["gpu"]}] {model_short}-C10 Adaptive v2 '
              f'(st{exp["start_ep"]}, su{su}, sd{sd}) -> {log_file}')

        env = os.environ.copy()
        env['PYTHONPATH'] = run_dir + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
        env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
        env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/miniconda3/envs/venv_1'
        env['PYTHONUNBUFFERED'] = '1'

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
