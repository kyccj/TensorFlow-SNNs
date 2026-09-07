"""R19-C10에서 loss-ratio가 무료 예산까지 내려갈 수 있는가 (26-08-19 등록)

동기 — [[무료 희소화 예산]] 재분석:
  R19-C10의 무료 예산은 잔량 39.6% (Δacc -0.04 ~ -0.24, 독립 6 run 재현).
  그런데 상수 rho=5.8e-4는 이 설정을 잔량 67.2%에 착지시킨다.
  => 27.6%p의 무손실 감소분을 안 쓰고 멈춘다. 네 설정 중 미사용분이 가장 크다.

질문: rho를 올려 40%를 겨냥하면, 그 지점에서 고정 lambda만큼 무손실인가?
  같은 잔량의 고정 lambda=1e-7 대조군: Δacc -0.06/-0.10/-0.10/-0.10/-0.24 (n=5, 평균 -0.12)

rho 선택: R19-C10의 rho->잔량 3점(2.6e-4->82.3%, 5.8e-4->67.2%, 3.0e-3->23.6%).
  깊은 구간 기울기 -26.5 %p/ln -> 40% 겨냥 rho = 1.62e-3. 1.6e-3으로 반올림.
  단 사다리에서 이 외삽이 VGG에서 크게 빗나간 전력이 있다(예측 40% vs 실측 26/35%).

예측 (등록, 결과 후 수정 금지):
  F1  착지 잔량 30~50% (외삽 오차 허용). 벗어나면 rho->잔량 관계 재추정 필요
  F2  Δacc >= -0.30%p  -> 폐루프도 예산 경계까지 무손실로 내려간다.
      => "rho를 설정별 예산에 맞춰야 한다"가 다음 과제로 확정
  F3  Δacc <= -0.50%p  -> 같은 잔량에서 고정 lambda(-0.12)보다 유의하게 나쁘다.
      => 폐루프가 깊은 목표에서 손해라는 실질적 약점. loss-ratio의 적용 범위를 얕은
         구간으로 한정해 서술해야 함
  F2/F3 사이(-0.30 ~ -0.50)면 판정 보류, 반복 필요

브레이크 ON (깊은 목표이므로 사다리 결과에 따라 필수).
1 run, R19-C10은 약 34시간.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_rho_budget')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

ALPHA = 7

JOBS = [
    dict(name='rb_r19_c10_16', data='CIFAR10', rho=1.6e-3, brake=True),
]

RESERVED = {6, 7}


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    brake_line = '\n        conf.reg_spike_lr_brake = True' if job['brake'] else ''
    subs = [
        ('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
         f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'),
        ("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='rhobud-{job['name']}'"),
        ("conf.dataset='CIFAR100'", f"conf.dataset='{job['data']}'"),
        ('conf.reg_spike_out_alpha=3  # temperature',
         f'conf.reg_spike_out_alpha={ALPHA}  # temperature'),
        ('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
         'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'),
        ('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
         'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'),
        ('conf.sc_loss_scd = False',
         f"""conf.sc_loss_scd = False
        conf.reg_spike_loss_ratio = True
        conf.reg_spike_loss_ratio_target = {job['rho']}
        conf.reg_spike_loss_ratio_start_ep = 0
        conf.reg_spike_R_per_step = True{brake_line}"""),
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
        (r"^conf\.model='ResNet19'$", 'model'),
        (rf"^conf\.dataset='{job['data']}'$", 'dataset'),
        (r"^\s*conf\.reg_spike_loss_ratio = True$", 'loss_ratio'),
        (rf"^\s*conf\.reg_spike_loss_ratio_target = {job['rho']}$", 'rho'),
        (r"^\s*conf\.reg_spike_R_per_step = True$", 'R_per_step'),
        (r"^\s*conf\.reg_spike_out_wta_rev=True", 'wta_rev'),
        (rf"^\s*conf\.reg_spike_out_alpha={ALPHA}\b", 'alpha'),
        (r"^conf\.train_epoch = 310$", '310ep'),
    ]
    has_brake = bool(re.search(r"^\s*conf\.reg_spike_lr_brake = True$", src, re.M))
    if has_brake != job['brake']:
        raise RuntimeError(f"{job['name']}: brake flag mismatch (want {job['brake']})")
    if re.search(r"^\s*conf\.reg_spike_starget = True$", src, re.M):
        raise RuntimeError('starget unexpectedly on')
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
          f"(rho={job['rho']:.1e}, brake={'ON' if job['brake'] else 'OFF'})", flush=True)
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
    print(f'--- rho ladder: {len(pending)} runs ---', flush=True)
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
