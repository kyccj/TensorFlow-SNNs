'''
    Adaptive Lambda v2 - lambda_max sweep

    Key findings so far:
    - R19 v2 su1.0: λ hit max 1e-5 → acc 95.77% (-0.93%), spike 97K (-80%)
    - R19 v1 st200: λ reached 2.7e-6 naturally → acc 96.65% (-0.05%), spike 261K (-46%)
    - VGG su0.5: acc 94.77% (-0.31%), spike 39K (-47%) ← best VGG so far
    - VGG lmax=1e-6: acc 94.72% (-0.36%), spike 53K (-28%)

    Goal: Find optimal lambda_max for R19 (hasn't been tested) + explore VGG further

    GPU 0: R19-C10, su1.0, sd0.3, lmax=3e-6 (near v1's natural λ=2.7e-6)
    GPU 1: R19-C10, su1.0, sd0.3, lmax=1e-6
    GPU 2: R19-C10, su1.0, sd0.3, lmax=5e-7
    GPU 3: VGG-C10, su0.5, sd0.3, lmax=1e-6 (combine best su with lmax cap)
    GPU 4: VGG-C10, su0.5, sd0.3, lmax=3e-7
    GPU 5: VGG-C10, su1.0, sd0.3, lmax=1e-7 (very tight cap)
'''
import subprocess
import os

PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/miniconda3/envs/venv_1/lib'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    {'model': 'ResNet19', 'dataset': 'CIFAR10', 'gpu': 0,
     'dir': '_adaptive_lambda_v2/r19_c10_lmax3e-6',
     'start_ep': 1, 'step_up': 1.0, 'step_down': 0.3,
     'lambda_max': 3E-6, 'margin': 0.005, 'label': 'lmax3e-6'},
    {'model': 'ResNet19', 'dataset': 'CIFAR10', 'gpu': 1,
     'dir': '_adaptive_lambda_v2/r19_c10_lmax1e-6',
     'start_ep': 1, 'step_up': 1.0, 'step_down': 0.3,
     'lambda_max': 1E-6, 'margin': 0.005, 'label': 'lmax1e-6'},
    {'model': 'ResNet19', 'dataset': 'CIFAR10', 'gpu': 2,
     'dir': '_adaptive_lambda_v2/r19_c10_lmax5e-7',
     'start_ep': 1, 'step_up': 1.0, 'step_down': 0.3,
     'lambda_max': 5E-7, 'margin': 0.005, 'label': 'lmax5e-7'},
    {'model': 'VGG16', 'dataset': 'CIFAR10', 'gpu': 3,
     'dir': '_adaptive_lambda_v2/vgg_c10_su0.5_lmax1e-6',
     'start_ep': 1, 'step_up': 0.5, 'step_down': 0.3,
     'lambda_max': 1E-6, 'margin': 0.005, 'label': 'su0.5_lmax1e-6'},
    {'model': 'VGG16', 'dataset': 'CIFAR10', 'gpu': 4,
     'dir': '_adaptive_lambda_v2/vgg_c10_su0.5_lmax3e-7',
     'start_ep': 1, 'step_up': 0.5, 'step_down': 0.3,
     'lambda_max': 3E-7, 'margin': 0.005, 'label': 'su0.5_lmax3e-7'},
    {'model': 'VGG16', 'dataset': 'CIFAR10', 'gpu': 5,
     'dir': '_adaptive_lambda_v2/vgg_c10_lmax1e-7',
     'start_ep': 1, 'step_up': 1.0, 'step_down': 0.3,
     'lambda_max': 1E-7, 'margin': 0.005, 'label': 'lmax1e-7'},
]


def generate_config(exp):
    run_dir = os.path.join(PROJECT_ROOT, exp['dir'])
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, 'config_sweep.py')

    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        content = f.read()

    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{exp["gpu"]}"'
    )

    if exp['model'] == 'VGG16':
        content = content.replace("conf.model='ResNet19'", "conf.model='VGG16'")

    content = content.replace("conf.dataset='CIFAR100'", f"conf.dataset='{exp['dataset']}'")

    label = exp['label']
    model_short = 'r19' if exp['model'] == 'ResNet19' else 'vgg'
    ds_short = 'c10' if exp['dataset'] == 'CIFAR10' else 'c100'

    content = content.replace(
        "#\nconfig.set()",
        f"# adaptive lambda v2 lmax sweep: {label}\n"
        f"conf.reg_spike_adaptive = True\n"
        f"conf.reg_spike_adaptive_start_ep = {exp['start_ep']}\n"
        f"conf.reg_spike_adaptive_lambda_init = 1E-9\n"
        f"conf.reg_spike_adaptive_margin = {exp['margin']}\n"
        f"conf.reg_spike_adaptive_step_up = {exp['step_up']}\n"
        f"conf.reg_spike_adaptive_step_down = {exp['step_down']}\n"
        f"conf.reg_spike_adaptive_lambda_max = {exp['lambda_max']}\n"
        f"\n"
        f"conf.reg_spike_log_detail = True\n"
        f"\n#\nconfig.set()"
    )

    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        f"conf.exp_set_name='adp-v2s-{model_short}-{ds_short}-{label}'"
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
        label = exp['label']
        print(f'[GPU {exp["gpu"]}] {model_short}-C10 {label} '
              f'(su{exp["step_up"]}, sd{exp["step_down"]}, '
              f'lmax={exp["lambda_max"]}, margin={exp["margin"]}) -> {log_file}')

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
            label = exp['label']
            status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'
            print(f'[GPU {exp["gpu"]}] {label} finished: {status}')
    except KeyboardInterrupt:
        print('\nInterrupted - terminating...')
        for _, p, _ in processes:
            p.terminate()
        for _, p, _ in processes:
            p.wait()


if __name__ == '__main__':
    main()
