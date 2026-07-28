'''
    Adaptive lambda experiments - round 2
    Phase 1: R19-C10 Adaptive + wta_rev (GPU 4) — immediate
    Phase 2: VGG-C10 Adaptive with different start_ep (GPU 0,1,2) — after entropy finishes
'''
import subprocess
import os
import time

PYTHON = '/home/kyccj/miniconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/miniconda3/envs/venv_1/lib'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PHASE1 = [
    {'model': 'ResNet19', 'dataset': 'CIFAR10', 'gpu': 4,
     'dir': '_adaptive_lambda/r19_c10_wta_rev',
     'wta_rev': True, 'start_ep': 200},
]

PHASE2 = [
    {'model': 'VGG16', 'dataset': 'CIFAR10', 'gpu': 0,
     'dir': '_adaptive_lambda/vgg_c10_start_ep50',
     'wta_rev': False, 'start_ep': 50},
    {'model': 'VGG16', 'dataset': 'CIFAR10', 'gpu': 1,
     'dir': '_adaptive_lambda/vgg_c10_start_ep100',
     'wta_rev': False, 'start_ep': 100},
    {'model': 'VGG16', 'dataset': 'CIFAR10', 'gpu': 2,
     'dir': '_adaptive_lambda/vgg_c10_start_ep150',
     'wta_rev': False, 'start_ep': 150},
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

    # Enable wta_rev if specified
    if exp.get('wta_rev'):
        content = content.replace(
            "#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)",
            "conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)"
        )

    # exp name
    model_short = 'r19' if exp['model'] == 'ResNet19' else 'vgg'
    ds_short = 'c10' if exp['dataset'] == 'CIFAR10' else 'c100'
    suffix = '-wta-rev' if exp.get('wta_rev') else ''
    content = content.replace(
        "conf.exp_set_name='EIP-SNN-26'",
        f"conf.exp_set_name='adp-{model_short}-{ds_short}-st{exp['start_ep']}{suffix}'"
    )

    # Adaptive lambda config
    content = content.replace(
        "#\nconfig.set()",
        f"# adaptive lambda\n"
        f"conf.reg_spike_adaptive = True\n"
        f"conf.reg_spike_adaptive_start_ep = {exp['start_ep']}\n"
        f"conf.reg_spike_adaptive_lambda_init = 1E-9\n"
        f"conf.reg_spike_adaptive_margin = 0.005\n"
        f"conf.reg_spike_adaptive_step_up = 0.3\n"
        f"conf.reg_spike_adaptive_step_down = 0.5\n"
        f"conf.reg_spike_adaptive_lambda_max = 1E-5\n"
        f"\n"
        f"conf.reg_spike_log_detail = True\n"
        f"\n#\nconfig.set()"
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


def launch(exp):
    generate_config(exp)
    generate_main(exp)

    run_dir = os.path.join(PROJECT_ROOT, exp['dir'])
    log_file = os.path.join(run_dir, 'train.log')
    model_short = 'R19' if exp['model'] == 'ResNet19' else 'VGG'
    suffix = ' +wta_rev' if exp.get('wta_rev') else ''
    print(f'[GPU {exp["gpu"]}] {model_short} C10 Adaptive start_ep={exp["start_ep"]}{suffix} -> {log_file}')

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
    return p, log_file


def wait_for_gpu(gpus, check_interval=60):
    """Wait until specified GPUs are free (< 1GB memory used)"""
    import subprocess as sp
    while True:
        result = sp.run(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        gpu_mem = {}
        for line in result.stdout.strip().split('\n'):
            idx, mem = line.split(',')
            gpu_mem[int(idx.strip())] = int(mem.strip())

        all_free = all(gpu_mem.get(g, 99999) < 1000 for g in gpus)
        if all_free:
            return
        print(f'Waiting for GPUs {gpus}... (mem: {[gpu_mem.get(g,0) for g in gpus]} MiB)')
        time.sleep(check_interval)


def main():
    # Phase 1: Launch immediately on GPU 4
    print('=== Phase 1: R19-C10 Adaptive + wta_rev (GPU 4) ===')
    phase1_procs = []
    for exp in PHASE1:
        p, lf = launch(exp)
        phase1_procs.append((exp, p, lf))

    # Phase 2: Wait for GPUs 0,1,2 to be free, then launch
    print(f'\n=== Phase 2: Waiting for GPUs 0,1,2 to be free ===')
    wait_for_gpu([0, 1, 2])
    print('GPUs 0,1,2 are free! Launching Phase 2...\n')

    phase2_procs = []
    for exp in PHASE2:
        p, lf = launch(exp)
        phase2_procs.append((exp, p, lf))

    all_procs = phase1_procs + phase2_procs
    print(f'\n--- {len(all_procs)} experiments launched ---')
    print('Monitor:')
    for exp, _, lf in all_procs:
        print(f'  tail -f {lf}')

    try:
        for exp, p, log_file in all_procs:
            p.wait()
            status = 'OK' if p.returncode == 0 else f'FAIL(code={p.returncode})'
            model_short = 'R19' if exp['model'] == 'ResNet19' else 'VGG'
            print(f'[GPU {exp["gpu"]}] {model_short} start_ep={exp["start_ep"]} finished: {status}')
    except KeyboardInterrupt:
        print('\nInterrupted - terminating...')
        for _, p, _ in all_procs:
            p.terminate()
        for _, p, _ in all_procs:
            p.wait()


if __name__ == '__main__':
    main()
