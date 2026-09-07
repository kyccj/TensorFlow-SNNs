"""미팅 후속 3건 (26-08-20 등록) — 하나의 런처로 묶어 GPU 경합 방지

① cutmix 유지 (2 run)
   현재 config는 mix_off_iter=500*200이라 **200에폭에서 cutmix가 꺼진다**
   (datasets/augmentation_cifar.py:213 `train_counter < mix_off_iter` 조건).
   그 순간 train loss가 2.69 -> 2.03으로 계단 하락하고, loss-ratio는 lambda = rho*L_task/R
   이므로 분모 L_task가 한 번에 25% 줄면서 lambda가 급변한다.
   => cutmix를 끝까지 유지(mix_off_iter=500*400)하면 그 급변이 사라진다. 영향 확인.
   **baseline도 같이 돌린다** — cutmix를 유지하면 baseline 정확도 자체가 달라지므로,
   baseline 없이는 "lambda 급변의 효과"와 "cutmix 스케줄의 효과"를 구별할 수 없다.

③ loss-ratio 분모를 전체 loss로 (1 run)
   현재: lambda*R / L_task = rho          (상대비)
   대안: lambda*R / (L_task + lambda*R) = rho   (비중)
   후자를 풀면 lambda = rho*L_task / ((1-rho)*R) 이라 앞의 것과 **1/(1-rho) = 1.00058배**
   차이다. 즉 rho=5.8e-4에서 두 정의는 0.06% 차이이고 결과가 같을 것으로 예상된다.
   대수적으로 자명하지만 실측으로 한 번 닫아둔다.

④ maxnorm (1- 없음) (3 run)
   명칭: sc_rate = 1 - sc/max 는 **1-maxnorm**, sc_rate = sc/max 는 **maxnorm**.
   maxnorm은 방향이 정반대다 — 많이 발화한 뉴런이 벌점을 다 받고, 침묵 뉴런은 0을 받는다.
   침묵 뉴런의 sc_rate가 0이므로 wta_rev backward가 그들에게 아무것도 전달하지 못한다.
   또 mean(sc_rate) = mean(sc)/max ~ 0.04 이라 같은 lambda에서 실효 압력이 ~25배 약하다.
   그래서 lambda를 3e-7(1-maxnorm 대조점) / 3e-6 / 1e-5 세 점으로 훑는다.

전부 VGG16-CIFAR10, alpha=7, wta_rev backward, 310ep.
대조군 (baseline 95.00% / 78,189):
    1-maxnorm lambda=3e-7 -> 94.55% / 34,340
    loss-ratio rho=5.8e-4 -> 94.87% / 52,313 (n=3)
"""

import subprocess
import os
import threading
import time
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_meeting')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
ALPHA = 7
RESERVED = {6, 7}

LR_BLOCK = """conf.sc_loss_scd = False
        conf.reg_spike_loss_ratio = True
        conf.reg_spike_loss_ratio_target = 0.00058
        conf.reg_spike_loss_ratio_start_ep = 0
        conf.reg_spike_R_per_step = True"""

JOBS = [
    # ① cutmix 끝까지 유지
    dict(name='cm_base',   lmb='1E-8', cutmix_all=True,  extra='conf.sc_loss_scd = False',
         reg=False, want=[]),
    dict(name='cm_lr',     lmb='1E-8', cutmix_all=True,  extra=LR_BLOCK,
         reg=True,  want=[r"^\s*conf\.reg_spike_loss_ratio = True$"]),
    # ③ 분모 = 전체 loss
    dict(name='lr_full',   lmb='1E-8', cutmix_all=False,
         extra=LR_BLOCK + "\n        conf.reg_spike_loss_ratio_full = True",
         reg=True,  want=[r"^\s*conf\.reg_spike_loss_ratio_full = True$"]),
    # ④ maxnorm (1- 없음)
    dict(name='mx_3e-7',   lmb='3E-7', cutmix_all=False,
         extra="conf.sc_loss_scd = False\n        conf.reg_spike_out_sc_maxnorm_plain = True",
         reg=True,  want=[r"^\s*conf\.reg_spike_out_sc_maxnorm_plain = True$"]),
    dict(name='mx_3e-6',   lmb='3E-6', cutmix_all=False,
         extra="conf.sc_loss_scd = False\n        conf.reg_spike_out_sc_maxnorm_plain = True",
         reg=True,  want=[r"^\s*conf\.reg_spike_out_sc_maxnorm_plain = True$"]),
    dict(name='mx_1e-5',   lmb='1E-5', cutmix_all=False,
         extra="conf.sc_loss_scd = False\n        conf.reg_spike_out_sc_maxnorm_plain = True",
         reg=True,  want=[r"^\s*conf\.reg_spike_out_sc_maxnorm_plain = True$"]),
]


def make_config(gpu_id, job):
    with open(os.path.join(PROJECT_ROOT, 'config_snn_training.py')) as f:
        c = f.read()
    subs = [
        ('os.environ["CUDA_VISIBLE_DEVICES"]="9"',
         f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"'),
        ("conf.exp_set_name='EIP-SNN-26'", f"conf.exp_set_name='mtg-{job['name']}'"),
        ("conf.model='ResNet19'", "conf.model='VGG16'"),
        ("conf.dataset='CIFAR100'", "conf.dataset='CIFAR10'"),
        ('conf.reg_spike_out_alpha=3  # temperature',
         f'conf.reg_spike_out_alpha={ALPHA}  # temperature'),
        ('#conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)',
         'conf.reg_spike_out_wta_rev=True    # revised WTA: reduce_mean (gradient to non-firing neurons)'),
        ('#conf.reg_spike_log_detail=True   # per-layer regularization metrics logging',
         'conf.reg_spike_log_detail=True   # per-layer regularization metrics logging'),
        ('conf.reg_spike_out_const=1E-8', f"conf.reg_spike_out_const={job['lmb']}"),
        ('conf.sc_loss_scd = False', job['extra']),
    ]
    if job['cutmix_all']:
        # 500 step/epoch x 400 > 310ep 이므로 cutmix가 끝까지 켜진 채로 남는다
        subs.append(('conf.mix_off_iter = 500*200', 'conf.mix_off_iter = 500*400'))
    if not job['reg']:
        subs.append(('conf.reg_spike_out=True', 'conf.reg_spike_out=False'))
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
        (r"^conf\.dataset='CIFAR10'$", 'dataset'),
        (rf"^\s*conf\.reg_spike_out_const={job['lmb']}$", 'lambda'),
        (rf"^\s*conf\.reg_spike_out_alpha={ALPHA}\b", 'alpha'),
        (r"^conf\.train_epoch = 310$", '310ep'),
        (r"^conf\.mix_off_iter = 500\*" + ('400' if job['cutmix_all'] else '200') + r"$", 'mix_off_iter'),
    ] + [(p, 'job-specific') for p in job['want']]
    off = ['reg_spike_starget = True', 'reg_spike_lr_brake = True',
           'reg_spike_out_sc_one=True', 'reg_spike_out_sm_plain=True',
           'reg_spike_out_entropy=True', 'reg_spike_channel_wise = True']
    if job['name'] != 'lr_full':
        off.append('reg_spike_loss_ratio_full = True')
    if not job['name'].startswith('mx_'):
        off.append('reg_spike_out_sc_maxnorm_plain = True')
    if not job['reg']:
        off.append('reg_spike_loss_ratio = True')
    for flag in off:
        if re.search(rf'^\s*conf\.{re.escape(flag)}', src, re.M):
            raise RuntimeError(f"{job['name']}: {flag} unexpectedly on")
    if re.search(r"^\s*conf\.reg_spike_sm_group = '(?!none)", src, re.M):
        raise RuntimeError('sm_group unexpectedly on')
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
          f"(λ={job['lmb']}, cutmix_all={job['cutmix_all']})", flush=True)
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
    sel = sys.argv[1:]
    pending = [j for j in JOBS if not sel or j['name'] in sel]
    busy, threads = set(), []
    os.makedirs(SWEEP_DIR, exist_ok=True)
    print(f'--- meeting follow-ups: {len(pending)} runs ---', flush=True)
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
    print('--- meeting follow-ups drained ---', flush=True)


if __name__ == '__main__':
    main()
