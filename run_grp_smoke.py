"""softmax 묶음 축 비교 — 스모크 (26-08-20)

본실험 전 2에폭 검증. 확인 대상은 로그의 sc_rate_mean이 이론값과 맞는지 하나뿐이다.
mean(1-softmax) = 1 - 1/묶음크기 이며 alpha와 무관하게 정확히 성립한다.

VGG16-C10 층별 이론값:
    층        H×W    C  | 전역      채널내    채널간    행
    conv1     32×32  64 | 0.99998  0.99902  0.98438  0.96875
    conv3     8×8   256 | 0.99994  0.98438  0.99609  0.87500
    conv5     2×2   512 | 0.99951  0.75000  0.99805  0.50000

7월 버그(채널 축을 잘못 잡아 실제로는 행 단위였음)가 한 달간 안 걸린 이유가
이 대조를 한 번도 안 해봐서다. 같은 실수를 반복하지 않는다.

동시에 드러난 문제 하나 — 묶음이 공간축을 쓰면 **깊이별로 실효 lambda가 달라진다.**
'행'은 conv5에서 mean 0.50, 즉 깊은 층의 벌점이 절반이다. '채널내'도 0.75다.
이건 WTA 효과가 아니라 depth-dependent lambda 프로파일이므로, 7월의 -0.28%p가
이것만으로 설명될 수 있다. 그래서 flat 대조군(같은 프로파일, 차등화 0)을 같이 넣는다.

스모크 4조건 x 2에폭. 통과 기준: 층별 sc_rate_mean이 위 표와 소수 셋째 자리까지 일치.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_grp_smoke')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

ALPHA = 4

JOBS = [
    dict(name='sm_within', data='CIFAR10', lmb='3E-7', alpha=0.5, grp='within_channel', flat='False'),
    dict(name='sm_chan',   data='CIFAR10', lmb='3E-7', alpha=7,   grp='channel',        flat='False'),
    dict(name='sm_row',    data='CIFAR10', lmb='3E-7', alpha=4,   grp='row',            flat='False'),
    dict(name='sm_rowflat',data='CIFAR10', lmb='3E-7', alpha=4,   grp='row',            flat='True'),
]

RESERVED = {6, 7}


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    subs = [
        ('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
         f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'),
        ("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='grpsm-{job['name']}'"),
        ("conf.model='ResNet19'", "conf.model='VGG16'"),
        ("conf.dataset='CIFAR100'", f"conf.dataset='{job['data']}'"),
        ('conf.reg_spike_out_alpha=3  # temperature',
         f"conf.reg_spike_out_alpha={job['alpha']}  # temperature"),
        ('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
         'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'),
        ('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
         'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'),
        ('conf.reg_spike_out_const=1E-8', f"conf.reg_spike_out_const={job['lmb']}"),
        ('conf.sc_loss_scd = False',
         f"conf.sc_loss_scd = False\n        conf.reg_spike_sm_group = '{job['grp']}'\n        conf.reg_spike_sm_group_flat = {job['flat']}"),
        ('conf.train_epoch = 310', 'conf.train_epoch = 2'),
    ]
    for old, new in subs:
        if old not in c:
            raise RuntimeError(f"substitution target missing: {old!r}")
        c = c.replace(old, new, 1)
    return c


def verify(path, job):
    import re
    src = open(path).read()
    want = [
        (r"^conf\.model='VGG16'$", 'model'),
        (rf"^conf\.dataset='{job['data']}'$", 'dataset'),
        (rf"^\s*conf\.reg_spike_sm_group = '{job['grp']}'$", 'sm_group'),
        (rf"^\s*conf\.reg_spike_sm_group_flat = {job['flat']}$", 'flat'),
        (rf"^\s*conf\.reg_spike_out_const={job['lmb']}$", 'lambda'),
        (r"^\s*conf\.reg_spike_out_wta_rev=True", 'wta_rev'),
        (rf"^\s*conf\.reg_spike_out_alpha={job['alpha']}\b", 'alpha'),
        (r"^\s*conf\.reg_spike_log_detail=True", 'log_detail'),
        (r"^conf\.train_epoch = 2$", '2ep smoke'),
    ]
    off = ['reg_spike_loss_ratio = True', 'reg_spike_starget = True',
           'reg_spike_lr_brake = True', 'reg_spike_out_sc_maxnorm=True',
           'reg_spike_out_sc_one=True', 'reg_spike_out_sm_plain=True',
           'reg_spike_out_entropy=True', 'reg_spike_channel_wise = True']
    for flag in off:
        if re.search(rf'^\s*conf\.{re.escape(flag)}', src, re.M):
            raise RuntimeError(f'{flag} unexpectedly on')
    bad = [n for p, n in want if not re.search(p, src, re.M)]
    if bad:
        raise RuntimeError(f"{job['name']}: config verification failed on {bad}")


def run_one(gpu, job):
    d = os.path.join(SWEEP_DIR, job['name'])
    os.makedirs(d, exist_ok=True)
    cfg = os.path.join(d, 'config_sweep.py')
    with open(cfg, 'w') as f:
        f.write(make_config(gpu, job))
    verify(cfg, job)
    with open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')) as f:
        m = f.read().replace('from config_snn_training import config', 'from config_sweep import config')
    with open(os.path.join(d, 'main_sweep.py'), 'w') as f:
        f.write(m)

    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'

    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {job['name']} "
          f"({job['grp']}, alpha={job['alpha']}, flat={job['flat']})", flush=True)
    with open(os.path.join(d, 'train.log'), 'w') as lf:
        rc = subprocess.Popen([PYTHON, os.path.join(d, 'main_sweep.py')], cwd=PROJECT_ROOT,
                              stdout=lf, stderr=subprocess.STDOUT, env=env).wait()
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] DONE  {job['name']} rc={rc}", flush=True)


def free_gpus(exclude, max_mib=200):
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            timeout=60).decode()
    except Exception:
        return []
    res = []
    for line in out.strip().split('\n'):
        idx, mem = [x.strip() for x in line.split(',')]
        idx = int(idx)
        if idx in RESERVED or idx in exclude:
            continue
        if int(mem) <= max_mib:
            res.append(idx)
    return res


def main():
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    jobs = [j for j in JOBS if only is None or j['name'] in only]
    pending, busy, threads = list(jobs), set(), []
    os.makedirs(SWEEP_DIR, exist_ok=True)
    print(f'--- softmax group smoke (2ep): {len(pending)} runs ---', flush=True)
    while pending:
        for g in free_gpus(busy):
            if not pending:
                break
            busy.add(g)
            job = pending.pop(0)

            def lane(g=g, j=job):
                try:
                    run_one(g, j)
                finally:
                    busy.discard(g)
            t = threading.Thread(target=lane); t.start(); threads.append(t)
            time.sleep(45)
        if pending:
            time.sleep(300)
    for t in threads:
        t.join()
    print('--- rho ladder drained ---', flush=True)


if __name__ == '__main__':
    main()
