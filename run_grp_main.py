"""softmax 묶음 축 본실험 — lambda=3e-7 고정, 묶음/온도만 변경 (26-08-20)

스모크(2에폭) 결과:
  * 묶음 축 4/4 정확 — 13개 층 전부 sc_rate_mean이 이론값 1-1/묶음과 일치
  * within_channel(alpha=0.5): sc_rate_std 층평균 0.123, 최대 0.403  <- 실제 차등화
  * channel(alpha=7):          sc_rate_std 층평균 0.033, 최대 0.120
  * row(alpha=4):              sc_rate_std 0.00000                   <- 균일. 7월(0.29)과 재현 안 됨
    => row는 본실험에서 제외. 7월 이후 코드 변경으로 재현 불가 추정, 별도 조사 필요.

교란 하나 — 묶음이 공간축을 쓰면 깊이별로 실효 lambda가 달라진다:
    층      within_channel mean   channel mean
    conv1        0.99902            0.98438
    conv3        0.98438            0.99609
    conv5        0.75000            0.99805
  within_channel은 conv5에서 벌점이 25% 깎인다. 이건 WTA가 아니라 깊이별 lambda
  프로파일이므로, 차등화의 효과와 구별하려면 flat 대조군이 필요하다.
  flat = 같은 층별 평균, 퍼짐 0. (전역에서 sc_rate=1이 했던 역할과 동일)

대조군 (V16-C10, 잔량 41~46% 구간, base 95.01/78,189):
    전역 1-softmax  -0.66 | 균일(sc_rate=1) -0.66 | maxnorm -0.45/-0.81/-0.84/-0.89
합격선: 잔량 42~46%에서 Δacc >= -0.30.

예측 (등록, 결과 후 수정 금지):
  G1  within alpha 스윕에서 최소 하나가 Δacc >= -0.30  -> 곡선을 움직인 첫 변형
  G2  within이 좋은데 within-flat도 같이 좋다 -> 원인은 차등화가 아니라 깊이별 lambda
      프로파일. "가중치"가 아니라 "층별 lambda"로 서술해야 함
  G3  within 전부 -0.45 ~ -0.89 -> 채널 내 경쟁도 무효. 가중치 계보 종료
  G4  channel이 within보다 나음 -> 묶음 크기(64 vs 1024)가 지배. 경쟁 단위 수는 무관

6 run x 310ep, 각 약 11시간.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_grp')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

ALPHA = 4

JOBS = [
    dict(name='wc_a0.5',   data='CIFAR10', lmb='3E-7', alpha=0.5, grp='within_channel', flat='False'),
    dict(name='wc_a0.2',   data='CIFAR10', lmb='3E-7', alpha=0.2, grp='within_channel', flat='False'),
    dict(name='wc_a1',     data='CIFAR10', lmb='3E-7', alpha=1,   grp='within_channel', flat='False'),
    dict(name='wc_flat',   data='CIFAR10', lmb='3E-7', alpha=0.5, grp='within_channel', flat='True'),
    dict(name='ch_a7',     data='CIFAR10', lmb='3E-7', alpha=7,   grp='channel',        flat='False'),
    dict(name='ch_flat',   data='CIFAR10', lmb='3E-7', alpha=7,   grp='channel',        flat='True'),
]

RESERVED = {6, 7}


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    subs = [
        ('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
         f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'),
        ("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='grp-{job['name']}'"),
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
        (r"^conf\.train_epoch = 310$", '310ep'),
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
    print(f'--- softmax group main (310ep): {len(pending)} runs ---', flush=True)
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
