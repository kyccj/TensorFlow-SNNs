"""브레이크 판정의 잔량 교란 제거 (26-08-20 등록)

08-20 01:00에 브레이크 반복이 n=7 vs 7로 찼다. 원시 비교는:
    ON  n=7  평균 -0.20%p (sd 0.41)   잔량 평균 71.7%
    OFF n=7  평균 -0.95%p (sd 0.72)   잔량 평균 68.2%
    차이 +0.75%p, 순열 p = 0.0169   <- 0.05를 넘김

그러나 **두 군의 잔량이 3.5%p 다르다.** OFF 군 안에서 잔량 vs Δacc를 회귀하면
기울기가 +0.129 %p/%p라, 잔량 3.5%p 차이만으로 0.45%p가 설명된다.
공통 기울기로 보정하면 차이 +0.28%p, 순열 p = 0.130 으로 무너진다.
잔량 겹치는 구간(69~77%)만 봐도 ON -0.20 (n=7) vs OFF -0.42 (n=3), 차이 +0.22%p.

즉 브레이크의 이점 상당 부분이 "정확도를 지킨다"가 아니라 **"덜 규제한다"**일 수 있다.
브레이크는 lambda 성장을 1.5배/epoch로 막으므로 당연히 덜 깎인다.

08-18의 nb_matched가 이 교란을 겨냥했으나 목표 83.6%를 놓치고 89.5%에 착지해 실패했다.
이번엔 **ON 군의 실제 착지(71.7%)를 겨냥**한다.

rho 산정: 무브레이크 C100 2점 (2.6e-4 -> 79.8%, 5.8e-4 -> 68.2%), 기울기 -14.47 %p/ln.
    68.2% -> 71.7% 는 +3.5%p 이므로 ln 비 -0.242, rho = 5.8e-4 * 0.785 = 4.55e-4 -> 4.6e-4 채택.

예측 (등록, 결과 후 수정 금지):
  M1  착지 잔량 69~75% (ON 군 범위 69.6~75.1과 겹침). 벗어나면 rho 재산정
  M2  Δacc <= -0.60%p  -> 같은 잔량에서도 무브레이크가 진다. **브레이크 효과 실재**
  M3  Δacc >= -0.30%p  -> 잔량만 맞추면 차이가 사라진다.
      => 브레이크의 이점은 "덜 규제함"의 부산물. 헤드라인 주장 폐기하고
         브레이크는 "공격 rho에서의 생존 장치"로만 서술 (사다리 결과는 그대로 유효)
  M3 이면 [[Method 정립 — wta-rev + loss-ratio]] 재작성 필요

무브레이크, V16-C100, rho=4.6e-4, 310ep, 3 run. p-norm 3 run 뒤 자리가 나는 대로.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_brake_matched')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

ALPHA = 7

JOBS = [
    dict(name='bm_r1', data='CIFAR100', rho=4.6e-4, brake=False),
    dict(name='bm_r2', data='CIFAR100', rho=4.6e-4, brake=False),
    dict(name='bm_r3', data='CIFAR100', rho=4.6e-4, brake=False),
]

RESERVED = {6, 7}


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    brake_line = '\n        conf.reg_spike_lr_brake = True' if job['brake'] else ''
    subs = [
        ('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
         f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'),
        ("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='bmch-{job['name']}'"),
        ("conf.model='ResNet19'", "conf.model='VGG16'"),
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
        (r"^conf\.model='VGG16'$", 'model'),
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
    print(f'--- brake remnant-matched control: {len(pending)} runs ---', flush=True)
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
