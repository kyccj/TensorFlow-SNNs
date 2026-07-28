'''
    VGG16 CIFAR100 Baseline (no spike regularization) on GPU 0
    Previous run failed with RandAugment height_factor error - retry
'''
import subprocess
import os

PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/miniconda3/envs/venv_1/lib'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    run_dir = os.path.join(PROJECT_ROOT, '_baseline_no_reg_vgg_c100_v2')
    os.makedirs(run_dir, exist_ok=True)

    # generate config
    config_path = os.path.join(run_dir, 'config_sweep.py')
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        content = f.read()

    # GPU 0
    content = content.replace(
        'os.environ["CUDA_VISIBLE_DEVICES"]="9"',
        'os.environ["CUDA_VISIBLE_DEVICES"]="0"'
    )

    # VGG16 (already VGG16 after replace, but original is ResNet19)
    content = content.replace("conf.model='ResNet19'", "conf.model='VGG16'")

    # Keep CIFAR100 (default)

    # Disable spike regularization
    content = content.replace("conf.reg_spike_out=True", "conf.reg_spike_out=False")

    # Enable detail logging
    content = content.replace(
        '#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
        'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'
    )

    # exp name
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        "conf.exp_set_name='EIP-SNN-26_baseline-no-reg-vgg-c100'"
    )

    with open(config_path, 'w') as f:
        f.write(content)

    # generate main
    main_path = os.path.join(run_dir, 'main_sweep.py')
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')) as f:
        content = f.read()
    content = content.replace(
        'from config_snn_training import config',
        'from config_sweep import config'
    )
    with open(main_path, 'w') as f:
        f.write(content)

    # launch
    log_file = os.path.join(run_dir, 'train.log')
    print(f'[GPU 0] VGG16 C100 Baseline -> {log_file}')

    env = os.environ.copy()
    env['PYTHONPATH'] = run_dir + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/miniconda3/envs/venv_1'

    with open(log_file, 'w') as lf:
        p = subprocess.Popen(
            [PYTHON, main_path],
            cwd=PROJECT_ROOT,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
        )

    print('Launched. Monitor: tail -f', log_file)

    try:
        p.wait()
        status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'
        print(f'VGG16 C100 Baseline finished: {status}')
    except KeyboardInterrupt:
        print('Interrupted - terminating...')
        p.terminate()
        p.wait()


if __name__ == '__main__':
    main()
