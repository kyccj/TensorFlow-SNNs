'''
    VGG16-C10 Adaptive Lambda — start_ep=50 + lambda_max sweep

    Hypothesis: Natural spike reduction mostly happens in ep1-50 (critical period).
    Starting regulation at ep50 avoids interfering with this natural process.
    Model is more mature at ep50, should tolerate larger lambda_max.

    Baseline: 95.02%, 74K spikes
    Best so far (start_ep=1): su0.5, lmax=1e-6 → 94.98%, 58K (-22%)

    GPU 0: lmax=1e-6  (direct comparison with start_ep=1 best)
    GPU 1: lmax=3e-6  (was bad with start_ep=1, retest)
    GPU 2: lmax=5e-6  (aggressive)
    GPU 3: lmax=1e-5  (upper bound search)
    GPU 5: lmax=1e-4  (extreme, find collapse point)
'''
import subprocess
import os
import time

PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = [
    {'gpu': 0, 'lambda_max': 1E-6,  'label': 'vgg_c10_ep50_lmax1e-6'},
    {'gpu': 1, 'lambda_max': 3E-6,  'label': 'vgg_c10_ep50_lmax3e-6'},
    {'gpu': 2, 'lambda_max': 5E-6,  'label': 'vgg_c10_ep50_lmax5e-6'},
    {'gpu': 3, 'lambda_max': 1E-5,  'label': 'vgg_c10_ep50_lmax1e-5'},
    {'gpu': 5, 'lambda_max': 1E-4,  'label': 'vgg_c10_ep50_lmax1e-4'},
]

# Common settings
COMMON = {
    'model': 'VGG16',
    'dataset': 'CIFAR10',
    'start_ep': 50,
    'step_up': 0.5,
    'step_down': 0.3,
    'lambda_init': 1E-7,
    'margin': 0.005,
}


def generate_config(exp):
    run_dir = os.path.join(PROJECT_ROOT, '_adaptive_lambda_v2', exp['label'])
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
    content = content.replace("conf.nn_model_name='VGG16_AP'", "conf.nn_model_name='VGG16_AP'")

    # Dataset
    content = content.replace("conf.dataset_name='CIFAR100'", "conf.dataset_name='CIFAR10'")

    # Ensure spike regularization is enabled
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        new_lines.append(line)
    content = '\n'.join(new_lines)

    # Set model and dataset
    content = content.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    content = content.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")

    # Insert adaptive lambda config BEFORE config.set()
    insert_block = f"""
# === Adaptive Lambda v2 — start_ep=50 sweep ===
conf.reg_spike_adaptive = True
conf.reg_spike_adaptive_start_ep = {COMMON['start_ep']}
conf.reg_spike_adaptive_step_up = {COMMON['step_up']}
conf.reg_spike_adaptive_step_down = {COMMON['step_down']}
conf.reg_spike_adaptive_lambda_max = {exp['lambda_max']:.1E}
conf.reg_spike_adaptive_lambda_init = {COMMON['lambda_init']:.1E}
conf.reg_spike_adaptive_margin = {COMMON['margin']}

conf.reg_spike_log_detail = True
"""
    content = content.replace('config.set()', insert_block + '\nconfig.set()')

    with open(config_path, 'w') as f:
        f.write(content)

    return run_dir, config_path


def launch(exp):
    run_dir, config_path = generate_config(exp)

    # Copy main_sweep.py if not exists
    main_src = os.path.join(PROJECT_ROOT, '_adaptive_lambda_v2', 'vgg_c10_wta_rev_lmax1e-6', 'main_sweep.py')
    main_dst = os.path.join(run_dir, 'main_sweep.py')
    if not os.path.exists(main_dst):
        import shutil
        shutil.copy2(main_src, main_dst)

    log_path = os.path.join(run_dir, 'train.log')

    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'
    env['PYTHONPATH'] = PROJECT_ROOT
    env['CUDA_VISIBLE_DEVICES'] = str(exp['gpu'])

    cmd = f'{PYTHON} {main_dst}'

    with open(log_path, 'w') as log_f:
        proc = subprocess.Popen(
            cmd, shell=True, cwd=run_dir,
            stdout=log_f, stderr=subprocess.STDOUT,
            env=env
        )

    print(f'[GPU {exp["gpu"]}] {exp["label"]} — PID {proc.pid}, lmax={exp["lambda_max"]:.0E}')
    return proc


def main():
    print(f'=== VGG16-C10 start_ep=50 lambda_max sweep ===')
    print(f'Common: start_ep={COMMON["start_ep"]}, su={COMMON["step_up"]}, '
          f'sd={COMMON["step_down"]}, lambda_init={COMMON["lambda_init"]:.0E}, '
          f'margin={COMMON["margin"]}')
    print()

    procs = []
    for exp in EXPERIMENTS:
        proc = launch(exp)
        procs.append(proc)
        time.sleep(15)

    print(f'\nAll {len(procs)} experiments launched.')
    print('Monitor: tail -f _adaptive_lambda_v2/vgg_c10_ep50_*/train.log')


if __name__ == '__main__':
    main()
