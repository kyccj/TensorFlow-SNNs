"""accum 켬/끔 × loss-ratio 짝비교 (26-08-29 등록)

**왜 loss-ratio 인가** — 고정 lambda 로는 accum 켬/끔을 공정하게 비교할 수가 없다.
accum 을 켜면 R 의 정의가 sum_t ||x_t|| 에서 ||sum_t x_t|| 로 바뀌어 크기가 달라지는데,
그 배율을 세 번 다 크게 틀렸다 (예상 10.8배 -> 실제 반대 방향, 20배 오차).
그래서 1차는 전부 붕괴 구간에, 2차는 대조군보다 훨씬 얕은 쪽에 착지해
**두 곡선이 겹치는 구간이 826 스파이크뿐**이었다 (49,137~49,963).

loss-ratio 는 lambda = rho * L_task / R 을 매 에폭 다시 푼다. 그러면 규제 기여분
lambda*R 이 rho*L_task 로 **고정**되므로, R 의 크기가 어떻게 바뀌든 같은 세기가 된다.
lambda 보정이 아예 필요 없어진다.

**설계** — 채널 내 1-maxnorm 고정, rho 2점 x accum 켬/끔 = 4 run. 짝지어 비교한다.
    rho=1.2e-3  ->  ~39,000 예상   (_aclr/ 에서 진행 중)
    rho=5.8e-4  ->  ~52,000 예상   (이 묶음)
처음에 rho=3e-3 (~18,000) 을 넣었다가 뺐다 — 거기선 93.5% 로 무손실(94.78%) 한참 아래라
"정확도 유지 + 스파이크 감소" 라는 목표에 못 쓰는 점이다.
(전역 1-softmax 기준: rho 5.8e-4 -> 52,313 / 1e-3 -> 42,113 / 3e-3 -> 18,207)
이 두 점이 지금 비어 있는 구간(20,000~50,000)을 정확히 메운다.

**지금까지의 유일한 직접 비교** (고정 lambda, 겹치는 한 점, 각 n=1):
    ~49,500 에서  accum 끔 94.72%  vs  accum 켬 94.94%   **+0.22%p**
부호는 긍정적이지만 반복 편차 sd 0.31 의 2/3 라 혼자서는 판정 불가.

**무엇을 묻나** — 지금은 규제 손실이 매 타임스텝 붙는데 sc_rate 가 그 시각까지의 누적
spike_count 로 계산되므로 t=1 은 sc_rate {1,0} 인 이진 마스크다. 손실 4개 중 앞쪽이
저해상도다. accum 은 t=T 에서 한 번, 항상 5단계로 계산한다.

코드 확인 — neurons.py:1092 가 accum 과 loss_ratio 조합을 명시적으로 처리한다
(accum 켬이면 sc_loss 가 이미 합의 norm 이라 plain assign 이 맞다).

대조군 (accum 끔, 고정 lambda):
    lambda=1e-7  94.72% @ 49,963 (n=1) / 3e-7  94.46% @ 29,646 (n=1) / 5e-7  94.19% @ 20,258 (n=3)
규제 없음 95.00% (sd 0.11, n=5) @ 78,789.
"""
import subprocess, os, threading, time, sys, re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_aclr2')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
RESERVED = {6, 7}          # juyun 님 GPU — 절대 쓰지 않는다

SRC = '_mxg/mx_wc_5e-7'    # 채널 내 1-maxnorm, wta_rev, log_detail 켜짐, λ=5e-7

#        이름        gain    lambda
#        이름          accum   rho
JOBS = [
    ('on_5.8e-4',  'True',  '0.00058'),
    ('off_5.8e-4', 'False', '0.00058'),
]

RE_NAME = re.compile(r"^conf\.exp_set_name\s*=\s*'[^']*'", re.M)
RE_GPU  = re.compile(r'^os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=\s*"\d+"', re.M)
RE_LMB  = re.compile(r'^(\s*)conf\.reg_spike_out_const\s*=\s*(\S+)', re.M)
RE_MXG  = re.compile(r"^(\s*)conf\.reg_spike_maxnorm_group\s*=\s*'within_channel'", re.M)
RE_GAIN = re.compile(r"^\s*conf\.reg_spike_shape_beta\s*=\s*(\S+)", re.M)


def active_name(txt):
    """주석 처리된 exp_set_name 이 아니라 실제로 마지막에 대입되는 값을 읽는다.
    (08-21 에 주석 줄을 치환해서 원본 체크포인트를 날린 사고 재발 방지)"""
    i = txt.find('conf.root_model_save=conf.exp_set_name')
    if i < 0:
        raise RuntimeError('root_model_save 줄을 못 찾음')
    ms = RE_NAME.findall(txt[:i])
    if not ms:
        raise RuntimeError('활성 exp_set_name 대입이 없음')
    return ms[-1].split("'")[1]


def lambdas(txt):
    return [m.group(2) for m in RE_LMB.finditer(txt)]


def make_config(gpu_id, name, gain, lmb):
    c = open(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')).read()
    c, n = RE_GPU.subn(f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"', c)
    if n != 1:
        raise RuntimeError(f'{name}: GPU 줄이 {n}개')
    c, n = RE_NAME.subn(f"conf.exp_set_name='l2-{name}'", c)
    if n < 1:
        raise RuntimeError(f'{name}: 활성 exp_set_name 없음')
    # lambda 는 loss-ratio 가 정하므로 건드리지 않는다
    c, n = RE_MXG.subn(lambda m: (f"{m.group(1)}conf.reg_spike_maxnorm_group = 'within_channel'\n"
                                  f"{m.group(1)}conf.reg_spike_loss_ratio = True\n"
                                  f"{m.group(1)}conf.reg_spike_loss_ratio_target = {lmb}\n"
                                  f"{m.group(1)}conf.reg_spike_loss_ratio_start_ep = 0\n"
                                  f"{m.group(1)}conf.reg_spike_accum_loss = {gain}"), c, count=1)
    if n != 1:
        raise RuntimeError(f'{name}: loss-ratio 삽입 실패')
    return c


def verify(path, name, gpu, gain, lmb):
    a = open(path).read()
    b = open(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')).read()
    norm = lambda s: RE_LMB.sub('', RE_NAME.sub('', RE_GPU.sub('', s)))
    a_cmp = re.sub(r"^\s*conf\.reg_spike_(accum_loss|loss_ratio|loss_ratio_target|loss_ratio_start_ep) = \S+$\n?", '', a, flags=re.M)             # 새로 넣은 줄은 '비교용 사본'에서만 지운다
    if norm(a_cmp).replace('\n\n', '\n') != norm(b).replace('\n\n', '\n'):
        raise RuntimeError(f'{name}: GPU/이름/lambda/gain 외의 차이가 있음')
    if f'"CUDA_VISIBLE_DEVICES"]="{gpu}"' not in a:
        raise RuntimeError(f'{name}: GPU 지정 안 됨')
    eff = active_name(a)
    if eff != f'l2-{name}':
        raise RuntimeError(f'{name}: 유효 exp_set_name 이 {eff!r} — 원본 덮어쓸 위험')
    if active_name(b) == eff:
        raise RuntimeError(f'{name}: 원본과 저장 경로가 같음')
    if not re.search(rf"^\s*conf\.reg_spike_accum_loss = {gain}$", a, re.M):
        raise RuntimeError(f'{name}: accum_loss={gain} 설정 안 됨')
    if not re.search(rf"^\s*conf\.reg_spike_loss_ratio_target = {lmb}$", a, re.M):
        raise RuntimeError(f'{name}: rho={lmb} 설정 안 됨')
    if not re.search(r"^\s*conf\.reg_spike_loss_ratio = True$", a, re.M):
        raise RuntimeError(f'{name}: loss_ratio 안 켜짐')

    # 주석이 붙은 줄도 있으므로 $ 로 닫지 않는다
    for pat, why in [(r"^\s*conf\.reg_spike_out_wta_rev\s*=\s*True\b", 'wta_rev'),
                     (r"^\s*conf\.reg_spike_out_sc_maxnorm\s*=\s*True\b", '1-maxnorm'),
                     (r"^\s*conf\.reg_spike_maxnorm_group\s*=\s*'within_channel'", '채널 내'),
                     (r"^\s*conf\.reg_spike_log_detail\s*=\s*True\b", 'log_detail')]:
        if not re.search(pat, a, re.M):
            raise RuntimeError(f'{name}: {why} 설정 안 됨')
    for bad, why in [(r"^\s*conf\.reg_spike_out_sc_maxnorm_plain\s*=\s*True\b", 'maxnorm_plain'),
                     (r"^\s*conf\.reg_spike_starget\s*=\s*True\b", 'starget'),
                     (r"^\s*conf\.reg_spike_wta_rev_floor\s*=", 'floor'),
                     (r"^\s*conf\.reg_spike_vmem_gain\s*=", 'vmem gain'),
                     (r"^\s*conf\.reg_spike_layer_cost\s*=", 'layer_cost (이번 실험과 섞이면 안 됨)')]:
        if re.search(bad, a, re.M):
            raise RuntimeError(f'{name}: {why} 가 켜져 있음')
    if lambdas(a) != lambdas(b):
        raise RuntimeError(f'{name}: lambda 줄이 바뀜 (loss-ratio 가 정해야 함)')


def preflight(jobs):
    bad = []
    if not os.path.exists(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')):
        bad.append(f'{SRC} (원본 config 없음)')
    for name, gain, lmb in jobs:
        tgt = os.path.join(PROJECT_ROOT, f'l2-{name}')
        if os.path.exists(tgt) and any(f.endswith('.weights.h5')
                                       for _, _, fs in os.walk(tgt) for f in fs):
            bad.append(tgt)
    # 플래그가 실제로 존재하는지 (반영 안 됐으면 조용히 무시되고 gain=0 이 된다)
    if not re.search(r"reg_spike_shape_beta", open(os.path.join(PROJECT_ROOT, 'flags.py')).read()):
        bad.append('flags.py 에 reg_spike_shape_beta 가 없음')
    if not re.search(r"reg_spike_shape_beta",
                     open(os.path.join(PROJECT_ROOT, 'lib_snn', 'neurons.py')).read()):
        bad.append('lib_snn/neurons.py 에 beta 반영 안 됨')
    if bad:
        raise SystemExit('중단:\n  ' + '\n  '.join(bad))
    print('사전 점검 통과: 저장 경로 충돌 없음, beta 플래그 반영됨', flush=True)


def run_one(gpu, name, gain, lmb):
    d = os.path.join(SWEEP_DIR, name)
    os.makedirs(d, exist_ok=True)
    cfg = os.path.join(d, 'config_sweep.py')
    open(cfg, 'w').write(make_config(gpu, name, gain, lmb))
    verify(cfg, name, gpu, gain, lmb)
    m = open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')).read()
    open(os.path.join(d, 'main_sweep.py'), 'w').write(
        m.replace('from config_snn_training import config', 'from config_sweep import config'))
    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {name}  (accum={gain}, ρ={lmb})",
          flush=True)
    with open(os.path.join(d, 'train.log'), 'w') as lf:
        rc = subprocess.Popen([PYTHON, os.path.join(d, 'main_sweep.py')], cwd=PROJECT_ROOT,
                              stdout=lf, stderr=subprocess.STDOUT, env=env).wait()
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] DONE  {name} rc={rc}", flush=True)


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
    pending = [j for j in JOBS if not sel or j[0] in sel]
    busy, threads = set(), []
    os.makedirs(SWEEP_DIR, exist_ok=True)
    preflight(pending)
    print(f'--- accum × loss-ratio 짝비교: {len(pending)} runs ---', flush=True)
    while pending:
        for g in free_gpus(busy):
            if not pending:
                break
            job = pending.pop(0)
            busy.add(g)

            def worker(g=g, job=job):
                try:
                    run_one(g, *job)
                finally:
                    busy.discard(g)

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)
            time.sleep(90)      # 같은 GPU 를 두 job 이 잡는 것 방지
        if pending:
            time.sleep(60)
    for t in threads:
        t.join()
    print('--- 전부 완료 ---', flush=True)


if __name__ == '__main__':
    main()
