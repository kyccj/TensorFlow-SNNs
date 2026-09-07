"""채널 단위 WTA 재검증 (26-08-20 등록) — 7월 "동등" 판정 재검토

사용자 착상 두 가지에 대한 답:
 (1) softmax 분모를 작게(온도 alpha를 낮춰) -> **구조적으로 막힘.**
     spike_count가 정수 0~4뿐이라 최댓값 4를 351개 뉴런이 공유한다(V16-C10 conv1 실측
     발화율 13.1% 기준). alpha->0이면 softmax는 그 351개에 균등 분배되므로 승자값
     상한이 1/351 = 2.85e-3이고, 1-softmax는 0.9972 아래로 못 내려간다.
     실측 alpha 스윕(0.5~10, 일부 n=3)도 잔량 79~106%로 전부 노이즈 안이었다.
 (2) 채널 단위 -> **작동한다.** 공간 합산으로 값이 0~1639로 커지고 N이 65,536->64로
     줄어, alpha=7에서도 승자 채널의 1-softmax가 0.044까지 내려간다(표준편차 0.119).
     실측 sc_rate_std 0.17~0.49 (전역은 0.00001).

재검토 이유 — 7월 판정은 대조군이 성길 때 내려졌다. 지금 조밀한 곡선과 맞춰보면:

    잔량 42~46% 구간의 전역 방식들
        1-softmax(alpha=4) 42.4% -> -0.66
        sc_rate=1          43.6% -> -0.66
        maxnorm            43.9% -> -0.45
        1-softmax          46.0% -> -0.62
        (고정 lambda 4e-7) 37.7% -> -0.40
        => 평균 약 -0.56

    채널 단위 lambda=3e-7   44.6% -> **-0.28**    <- 0.28%p 우위

    한편 얕은 쪽 lambda=1e-7은 72.9% -> -0.04 로 전역과 동일(전역 보간 -0.04).
    즉 **깊은 구간에서만 우위**로 보이는데, 거기가 바로 무료 예산 경계가 있는 곳이다.
    (V16-C10 무료 예산 경계 = 잔량 57.5%)

단 n=1이고, lambda=1e-6은 발산했다(-27.95). 반복 없이는 못 쓴다.

설계: 7월 조건 그대로 복제 — VGG16-C10, alpha=4, channel_wise, wta_rev backward, 310ep.
  lambda=3e-7 x2 (그 점의 반복)
  lambda=5e-7 x1 (3e-7과 발산한 1e-6 사이. 경계를 얼마나 더 미는지 확인)

예측 (등록, 결과 후 수정 금지):
  C1  3e-7 반복 2개의 평균이 -0.45 이상(즉 전역 평균 -0.56보다 나음)
      -> 채널 단위 우위 실재. **가중치 계보에서 처음으로 곡선을 움직인 변형**
  C2  평균이 -0.56 근처(-0.45 ~ -0.70) -> 7월 판정대로 동등. 계보 종료
  C3  5e-7이 잔량 30%대에서 -0.45 이상 -> 무료 예산 경계가 57.5%에서 크게 밀림. 최상
  C4  5e-7도 발산 -> 채널 단위는 깊은 쪽 여유가 좁다. 실용성 제한

alpha=4를 쓰는 이유: 7월 run과 동일 조건 유지. 전역에서는 alpha가 무력하지만
채널 단위에서는 채널 합이 커서 alpha가 실제로 날카로움을 지배한다(위 (1) 참조).
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_chwise')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

ALPHA = 4

JOBS = [
    dict(name='cw_3e-7_r1', data='CIFAR10', lmb='3E-7'),
    dict(name='cw_3e-7_r2', data='CIFAR10', lmb='3E-7'),
    dict(name='cw_5e-7',    data='CIFAR10', lmb='5E-7'),
]

RESERVED = {6, 7}


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    subs = [
        ('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
         f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'),
        ("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='chw-{job['name']}'"),
        ("conf.model='ResNet19'", "conf.model='VGG16'"),
        ("conf.dataset='CIFAR100'", f"conf.dataset='{job['data']}'"),
        ('conf.reg_spike_out_alpha=3  # temperature',
         f'conf.reg_spike_out_alpha={ALPHA}  # temperature'),
        ('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
         'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'),
        ('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
         'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'),
        ('conf.reg_spike_out_const=1E-8', f"conf.reg_spike_out_const={job['lmb']}"),
        ('conf.sc_loss_scd = False',
         'conf.sc_loss_scd = False\nconf.reg_spike_channel_wise = True'),
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
        (r"^conf\.reg_spike_channel_wise = True$", 'channel_wise'),
        (rf"^\s*conf\.reg_spike_out_const={job['lmb']}$", 'lambda'),
        (r"^\s*conf\.reg_spike_out_wta_rev=True", 'wta_rev'),
        (rf"^\s*conf\.reg_spike_out_alpha={ALPHA}\b", 'alpha=4'),
        (r"^conf\.train_epoch = 310$", '310ep'),
    ]
    off = ['reg_spike_loss_ratio = True', 'reg_spike_starget = True',
           'reg_spike_lr_brake = True', 'reg_spike_out_sc_maxnorm=True',
           'reg_spike_out_sc_one=True', 'reg_spike_out_sm_plain=True',
           'reg_spike_out_entropy=True']
    for flag in off:
        if re.search(rf'^\s*conf\.{re.escape(flag)}', src, re.M):
            raise RuntimeError(f'{flag} unexpectedly on')
    if re.search(r'^\s*conf\.reg_spike_out_pnorm = [1-9]', src, re.M):
        raise RuntimeError('pnorm unexpectedly on')
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
          f"(lambda={job['lmb']}, alpha=4, channel-wise)", flush=True)
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
    print(f'--- channel-wise re-test: {len(pending)} runs ---', flush=True)
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
