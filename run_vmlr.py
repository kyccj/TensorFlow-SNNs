"""vmem silent_only + loss-ratio 스윕 (26-08-31 등록)

**왜 loss-ratio 인가** — 고정 lambda 로는 vmem 의 착지점을 맞추기가 어려웠다.
vmem 은 침묵 뉴런의 sc_rate 를 1.0 -> 0.775~0.975 로 낮춰 그들이 받는 gradient 를 줄인다.
그래서 같은 lambda 에서 규제가 약해지고 더 얕은 곳에 착지한다
(lambda=5e-7 에서 vmem 36,567 vs 1-s 24,364, 1.5배). 이 배율을 세 번 크게 틀렸다.
loss-ratio 는 lambda = rho * L_task / R 을 매 에폭 다시 풀어 lambda 를 자동으로 정한다.

**주의 — loss-ratio 가 착지점을 맞춰주지는 않는다.**
침묵 뉴런은 spike=0 이라 x = spike*sc_rate 에 안 들어간다. 그래서 **R 은 vmem 을 켜도
그대로**이고, 같은 rho 면 lambda 도 같다. 즉 vmem 은 여전히 1.5배 얕게 착지한다.
(accum x loss-ratio 짝비교에서도 같은 rho 에 스파이크가 27~43% 벌어졌다.)
그래도 스윕으로는 유효하다 — rho 를 그만큼 크게 잡으면 같은 구간을 덮는다.

**rho 선택** (1-s loss-ratio 실측: 1e-3 -> 42,113 / 3e-3 -> 18,207, vmem 은 ~1.5배 얕음):
    rho=1e-3 -> ~63,000    2e-3 -> ~40,000    3e-3 -> ~27,000    5e-3 -> ~20,000

**비교 대상** — 같은 rho 의 1-s loss-ratio 점이 두 개 있다 (직접 짝비교):
    rho=1e-3  1-s 94.43% @ 42,113
    rho=3e-3  1-s 93.54% @ 18,207
그리고 고정 lambda 로 그린 vmem 7점 곡선(평균 +0.03%p, 양수 4/음수 3)과도 겹쳐 본다.

설정 — 채널 내 1-maxnorm + reg_spike_vmem_silent_only, gain=1.0, loss_ratio.
규제 없음 95.00% (sd 0.11, n=5) @ 78,789.  무손실 기준 94.78%.
"""
import subprocess, os, threading, time, sys, re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_vmlr')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
# 이 런처만 GPU 0~3 을 쓴다 (런처 병행 시 같은 GPU 를 잡는 사고 방지)
RESERVED = {4, 5, 6, 7}          # juyun 님 GPU — 절대 쓰지 않는다

SRC = '_mxg/mx_wc_5e-7'    # 채널 내 1-maxnorm, wta_rev, log_detail 켜짐, λ=5e-7

#        이름        gain    lambda
#        이름        gain    rho
JOBS = [
    ('r1e-3', '1.0', '0.001'),    # ~63,000 예상  (1-s 같은 ρ: 94.43% @ 42,113)
    ('r2e-3', '1.0', '0.002'),    # ~40,000
    ('r3e-3', '1.0', '0.003'),    # ~27,000  (1-s 같은 ρ: 93.54% @ 18,207)
    ('r5e-3', '1.0', '0.005'),    # ~20,000
]

RE_NAME = re.compile(r"^conf\.exp_set_name\s*=\s*'[^']*'", re.M)
RE_GPU  = re.compile(r'^os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=\s*"\d+"', re.M)
RE_LMB  = re.compile(r'^(\s*)conf\.reg_spike_out_const\s*=\s*(\S+)', re.M)
RE_MXG  = re.compile(r"^(\s*)conf\.reg_spike_maxnorm_group\s*=\s*'within_channel'", re.M)
RE_GAIN = re.compile(r"^\s*conf\.reg_spike_vmem_gain\s*=\s*(\S+)", re.M)


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
    c, n = RE_NAME.subn(f"conf.exp_set_name='vl-{name}'", c)
    if n < 1:
        raise RuntimeError(f'{name}: 활성 exp_set_name 없음')
    # lambda 는 loss-ratio 가 정한다
    c, n = RE_MXG.subn(lambda m: (f"{m.group(1)}conf.reg_spike_maxnorm_group = 'within_channel'\n"
                                  f"{m.group(1)}conf.reg_spike_vmem_gain = {gain}\n"
                                  f"{m.group(1)}conf.reg_spike_vmem_silent_only = True\n"
                                  f"{m.group(1)}conf.reg_spike_loss_ratio = True\n"
                                  f"{m.group(1)}conf.reg_spike_loss_ratio_target = {lmb}\n"
                                  f"{m.group(1)}conf.reg_spike_loss_ratio_start_ep = 0"), c, count=1)
    if n != 1:
        raise RuntimeError(f'{name}: gain 삽입 실패')
    return c


def verify(path, name, gpu, gain, lmb):
    a = open(path).read()
    b = open(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')).read()
    norm = lambda s: RE_LMB.sub('', RE_NAME.sub('', RE_GPU.sub('', s)))
    a_cmp = re.sub(r"^\s*conf\.reg_spike_(vmem_silent_only|loss_ratio|loss_ratio_target|loss_ratio_start_ep) = \S+$\n?", '', RE_GAIN.sub('', a), flags=re.M)             # 새로 넣은 줄은 '비교용 사본'에서만 지운다
    if norm(a_cmp).replace('\n\n', '\n') != norm(b).replace('\n\n', '\n'):
        raise RuntimeError(f'{name}: GPU/이름/lambda/gain 외의 차이가 있음')
    if f'"CUDA_VISIBLE_DEVICES"]="{gpu}"' not in a:
        raise RuntimeError(f'{name}: GPU 지정 안 됨')
    eff = active_name(a)
    if eff != f'vl-{name}':
        raise RuntimeError(f'{name}: 유효 exp_set_name 이 {eff!r} — 원본 덮어쓸 위험')
    if active_name(b) == eff:
        raise RuntimeError(f'{name}: 원본과 저장 경로가 같음')
    if not re.search(r"^\s*conf\.reg_spike_vmem_silent_only = True", a, re.M):
        raise RuntimeError(f'{name}: silent_only 설정 안 됨')
    gs = RE_GAIN.findall(a)
    if gs != [gain]:
        raise RuntimeError(f'{name}: gain 이 {gs} (기대 [{gain!r}])')
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
                     (r"^\s*conf\.reg_spike_layer_cost\s*=", 'layer_cost'),
                     (r"^\s*conf\.reg_spike_shape_beta\s*=", 'shape_beta (이번 실험과 섞이면 안 됨)')]:
        if re.search(bad, a, re.M):
            raise RuntimeError(f'{name}: {why} 가 켜져 있음')
    if lambdas(a) != lambdas(b):
        raise RuntimeError(f'{name}: lambda 줄이 바뀜 (loss-ratio 가 정해야 함)')
    if not re.search(rf"^\s*conf\.reg_spike_loss_ratio_target = {lmb}$", a, re.M):
        raise RuntimeError(f'{name}: rho={lmb} 설정 안 됨')


def preflight(jobs):
    bad = []
    if not os.path.exists(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')):
        bad.append(f'{SRC} (원본 config 없음)')
    for name, gain, lmb in jobs:
        tgt = os.path.join(PROJECT_ROOT, f'vl-{name}')
        if os.path.exists(tgt) and any(f.endswith('.weights.h5')
                                       for _, _, fs in os.walk(tgt) for f in fs):
            bad.append(tgt)
    # 플래그가 실제로 존재하는지 (반영 안 됐으면 조용히 무시되고 gain=0 이 된다)
    if not re.search(r"reg_spike_vmem_silent_only", open(os.path.join(PROJECT_ROOT, 'flags.py')).read()):
        bad.append('flags.py 에 reg_spike_vmem_silent_only 가 없음')
    if not re.search(r"reg_spike_vmem_silent_only",
                     open(os.path.join(PROJECT_ROOT, 'lib_snn', 'neurons.py')).read()):
        bad.append('lib_snn/neurons.py 에 silent_only 반영 안 됨')
    if bad:
        raise SystemExit('중단:\n  ' + '\n  '.join(bad))
    print('사전 점검 통과: 저장 경로 충돌 없음, silent_only 플래그 반영됨', flush=True)


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
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {name}  (gain={gain}, ρ={lmb})",
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
    print(f'--- vmem silent_only × loss-ratio: {len(pending)} runs ---', flush=True)
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
