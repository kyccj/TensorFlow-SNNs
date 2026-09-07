"""p-norm 가중치: 1-softmax와 maxnorm을 잇는 단일 노브 (26-08-19 등록)

착상 — 두 방법의 차이는 **분모 하나**뿐이다.
    1-softmax : sc_rate = 1 - softmax(sc/alpha)   분모가 합(sum)   -> 값이 O(1/N) -> 균일
    maxnorm   : sc_rate = 1 - sc/max(sc)          분모가 최대(max) -> 값이 O(1)   -> 극단
따라서 분모를 p-노름으로 두면 둘이 한 족보로 이어진다:
    sc_rate = 1 - sc / ||sc||_p ,   ||sc||_p = (sum sc^p)^(1/p)
    p=1   -> 합   = 1-softmax 영역
    p=inf -> 최대 = maxnorm 그 자체

수치 확인 (VGG16-C10 conv1, N=65,536, 발화 13.1% 실측 기준. TF 구현으로 검산 완료):
    p        평균      승자 가중   침묵 가중
    1      0.999985   0.999701   1.000000    <- 1-softmax와 구별 불가
    2      0.998752   0.975508   1.000000
    4      0.990298   0.809682   1.000000
    8      0.976189   0.532941   1.000000
    16     0.964708   0.307714   1.000000
    maxnorm 0.949020  0.000000   1.000000

핵심: **평균이 0.949~1.000 안에 머문다.** 즉 p를 바꿔도 실효 lambda가 3.5% 이내로만
변하므로, 같은 lambda로 p만 바꾼 비교가 정당하다. (plain softmax가 평균을 1/N로
무너뜨려 6.5만 배 약해졌던 것과 대조. 그때의 실패를 반복하지 않기 위한 설계.)

가치 가설 — "정확도를 올린다"가 아니다:
  maxnorm의 max는 표본 하나짜리 통계라 한 뉴런이 전체 스케일을 정한다. 깊은 lambda에서
  maxnorm이 -10.73/-12.59로 튀는 것(R19-C10)이 이 잡음 탓이라면, 중간 p는 상위
  k^(1/p)개를 부드럽게 평균내므로 **양 끝보다 안정할 수 있다.**
  => 노리는 것은 [[무료 희소화 예산]]의 **경계를 더 깊게 미는 것**.

시험 지점: V16-C10, lambda=3e-7 고정. 이 lambda는 잔량 ~42%에 착지하는데,
V16-C10의 무료 예산 경계가 57.5%이므로 경계 **바깥**이다. 여기서 무손실이 되면 경계가 밀린다.
  같은 구간 기존 대조군 (잔량 41~44%):
    1-softmax  -0.66            (n=1)
    maxnorm    -0.45/-0.81/-0.84/-0.89  (n=4)

예측 (등록, 결과 후 수정 금지):
  W1  착지 잔량 38~48%. 벗어나면 p가 실효 lambda를 크게 바꾼 것이므로 재설계
  W2  Δacc >= -0.20  -> **성공.** 무료 예산 경계가 57.5% -> ~42%로 밀린다
  W3  Δacc가 -0.45 ~ -0.89 (기존 두 방법의 범위) 안 -> p는 무효.
      균일/중간/극단이 전부 같은 곡선 => **가중치 계보 최종 종료**
  W4  중간 p가 양 끝보다 나쁘면(< -0.89) 분모 보간 자체가 해로움. 예상 밖

mean-1 재정규화는 **끄고** 돌린다 (기존 대조군이 재정규화 없이 돌아갔으므로 비교 가능성 유지).
p = 4 / 8 / 16, V16-C10, alpha=7, wta_rev backward 유지, 310ep, 각 약 11시간.
"""

import subprocess
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_pnorm')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'

ALPHA = 7
LMB = '3E-7'

JOBS = [
    dict(name='pn_p8',  data='CIFAR10', p=8.0),
    dict(name='pn_p4',  data='CIFAR10', p=4.0),
    dict(name='pn_p16', data='CIFAR10', p=16.0),
]

RESERVED = {6, 7}


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    subs = [
        ('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
         f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'),
        ("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='pnorm-{job['name']}'"),
        ("conf.model='ResNet19'", "conf.model='VGG16'"),
        ("conf.dataset='CIFAR100'", f"conf.dataset='{job['data']}'"),
        ('conf.reg_spike_out_alpha=3  # temperature',
         f'conf.reg_spike_out_alpha={ALPHA}  # temperature'),
        ('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
         'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'),
        ('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
         'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'),
        ('conf.reg_spike_out_const=1E-8', f"conf.reg_spike_out_const={LMB}"),
        ('conf.sc_loss_scd = False',
         f"""conf.sc_loss_scd = False
        conf.reg_spike_out_pnorm = {job['p']}
        conf.reg_spike_out_pnorm_mean1 = False"""),
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
        (rf"^\s*conf\.reg_spike_out_pnorm = {job['p']}$", 'pnorm p'),
        (r"^\s*conf\.reg_spike_out_pnorm_mean1 = False$", 'mean1 off'),
        (rf"^\s*conf\.reg_spike_out_const={LMB}$", 'lambda'),
        (r"^\s*conf\.reg_spike_out_wta_rev=True", 'wta_rev'),
        (rf"^\s*conf\.reg_spike_out_alpha={ALPHA}\b", 'alpha'),
        (r"^conf\.train_epoch = 310$", '310ep'),
    ]
    off = ['reg_spike_loss_ratio = True', 'reg_spike_starget = True',
           'reg_spike_lr_brake = True', 'reg_spike_out_sc_maxnorm=True',
           'reg_spike_out_sc_one=True', 'reg_spike_out_sm_plain=True',
           'reg_spike_out_entropy=True']
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
          f"(p={job['p']:.0f})", flush=True)
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
    print(f'--- p-norm sweep: {len(pending)} runs ---', flush=True)
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
