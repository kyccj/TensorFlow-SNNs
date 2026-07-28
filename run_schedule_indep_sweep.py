"""
Schedule-independent lambda experiments:
  A. Spike-count feedback (GPU 0,1,2): lambda adjusts to hit target spike count
  B. Loss-ratio feedback (GPU 3,4): lambda adjusts to maintain reg/task loss ratio
  C. Epoch ramp (GPU 5): lambda = lmax * (epoch/total)^p, simple baseline
VGG-C10, 310 epochs
"""

import subprocess
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_schedule_indep')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'


def base_config():
    orig = os.path.join(PROJECT_ROOT, 'config_snn_training.py')
    with open(orig, 'r') as f:
        return f.read()


def apply_common(content, gpu_id, exp_name):
    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'
    )
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        f"conf.exp_set_name='{exp_name}'"
    )
    content = content.replace("conf.model='ResNet19'", "conf.model='VGG16'")
    content = content.replace("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'")
    content = content.replace(
        'conf.reg_spike_out_alpha=3  # temperature',
        'conf.reg_spike_out_alpha=4  # temperature'
    )
    content = content.replace(
        '#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
        'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'
    )
    content = content.replace(
        '#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
        'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'
    )
    return content


# A: Spike-count feedback
def config_sc_feedback(gpu_id, target, exp_name):
    content = apply_common(base_config(), gpu_id, exp_name)
    content = content.replace(
        'conf.reg_spike_out_const=1E-8',
        'conf.reg_spike_out_const=1e-7'
    )
    block = f"""conf.sc_loss_scd = False
        conf.reg_spike_sc_feedback = True
        conf.reg_spike_sc_target = {target}"""
    content = content.replace('conf.sc_loss_scd = False', block)
    return content


# B: Loss-ratio feedback
def config_loss_ratio(gpu_id, target_ratio, exp_name):
    content = apply_common(base_config(), gpu_id, exp_name)
    content = content.replace(
        'conf.reg_spike_out_const=1E-8',
        'conf.reg_spike_out_const=1e-7'
    )
    block = f"""conf.sc_loss_scd = False
        conf.reg_spike_loss_ratio = True
        conf.reg_spike_loss_ratio_target = {target_ratio}"""
    content = content.replace('conf.sc_loss_scd = False', block)
    return content


# C: Epoch ramp
def config_epoch_ramp(gpu_id, lmax, power, exp_name):
    content = apply_common(base_config(), gpu_id, exp_name)
    content = content.replace(
        'conf.reg_spike_out_const=1E-8',
        f'conf.reg_spike_out_const={lmax}'
    )
    block = f"""conf.sc_loss_scd = False
        conf.reg_spike_epoch_ramp = True
        conf.reg_spike_epoch_ramp_power = {power}"""
    content = content.replace('conf.sc_loss_scd = False', block)
    return content


EXPERIMENTS = [
    # (name, gpu, config_func, args)
    ('sc_fb_target55k', 0, config_sc_feedback, (0, 55000, 'sc-fb-55k')),
    ('sc_fb_target40k', 1, config_sc_feedback, (1, 40000, 'sc-fb-40k')),
    ('sc_fb_target25k', 2, config_sc_feedback, (2, 25000, 'sc-fb-25k')),
    ('loss_ratio_0.001', 3, config_loss_ratio, (3, 0.001, 'loss-ratio-0.001')),
    ('loss_ratio_0.005', 4, config_loss_ratio, (4, 0.005, 'loss-ratio-0.005')),
    ('epoch_ramp_p3', 5, config_epoch_ramp, (5, 5e-7, 3.0, 'epoch-ramp-p3')),
]


def generate_main(run_dir):
    orig_main = os.path.join(PROJECT_ROOT, 'main_snn_training.py')
    with open(orig_main, 'r') as f:
        content = f.read()
    content = content.replace(
        'from config_snn_training import config',
        'from config_sweep import config'
    )
    with open(os.path.join(run_dir, 'main_sweep.py'), 'w') as f:
        f.write(content)


def main():
    os.makedirs(SWEEP_DIR, exist_ok=True)
    processes = []

    for name, gpu, config_func, args in EXPERIMENTS:
        run_dir = os.path.join(SWEEP_DIR, name)
        os.makedirs(run_dir, exist_ok=True)

        content = config_func(*args)
        with open(os.path.join(run_dir, 'config_sweep.py'), 'w') as f:
            f.write(content)
        generate_main(run_dir)

        log_file = os.path.join(run_dir, 'train.log')
        print(f'[GPU {gpu}] {name} -> {run_dir}')

        env = os.environ.copy()
        env['PYTHONPATH'] = run_dir + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
        env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
        env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'

        with open(log_file, 'w') as lf:
            p = subprocess.Popen(
                [PYTHON, os.path.join(run_dir, 'main_sweep.py')],
                cwd=PROJECT_ROOT,
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=env,
            )
        processes.append((name, gpu, p, log_file))

    print(f'\n--- {len(processes)} experiments launched ---')
    print('Monitor:')
    print(f'  tail -f {SWEEP_DIR}/*/train.log')
    print(f'\nKill all:')
    print(f'  kill {" ".join(str(p.pid) for _, _, p, _ in processes)}')

    try:
        for name, gpu, p, log_file in processes:
            p.wait()
            status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'
            print(f'[GPU {gpu}] {name} finished: {status}')
    except KeyboardInterrupt:
        print('\nInterrupted - terminating...')
        for _, _, p, _ in processes:
            p.terminate()
        for _, _, p, _ in processes:
            p.wait()
        print('All terminated.')


if __name__ == '__main__':
    main()
