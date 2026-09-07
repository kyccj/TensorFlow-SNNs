"""vmem 재구현 (silent_only) — gain 2점, lambda=5e-7 고정 (26-08-27 등록)

    sc_soft = spike_count            (터진 적 있는 뉴런 — 정수 그대로)
            = gain * readiness       (한 번도 안 터진 뉴런만)
    readiness = clip(vmem/vth, 0, 1)

옛 구현(`scp = scp + gain*readiness`, 모든 뉴런에 더함)의 결함 두 가지를 고친다:

  (1) **리셋 오염** — 리셋은 vmem = vmem - spike 이고 vth=1.0, binary_spike=True 이므로
      터진 뉴런의 vmem 은 "넘치고 남은 것"이지 근접도가 아니다. 옛 구현은 그걸 더했다.
      재구현은 spike_count==0 인 뉴런만 건드리는데, **그들은 한 번도 리셋된 적이 없어서**
      vmem 이 순수한 누적 전하다. 오염 경로가 원천적으로 없다.

  (2) **순위 역전 + max 부풀림** — gain=2 에서 침묵(문턱 0.9) 이 sc_soft 1.8 로
      1회 발화(1.2) 를 이겼고, 승자가 4+1.8=5.8 이 되어 max 가 4->5.8 로 부풀었다.
      재구현은 침묵이 [0, gain) 에 갇히고 발화는 정수 그대로라 둘 다 안 생긴다.

gain 의 뜻이 옛것과 다르다 — **침묵 뉴런을 얼마나 넓게 펼칠까**이고 상한이 1 이다
(max=4 기준 침묵의 sc_rate 가 gain=1 에서 0.750~1.000, 1회 발화가 정확히 0.750).

부수 효과 — 지금은 t=1 에서 sc_rate 가 사실상 이진값인데, 재구현은 t=1 에서도
침묵 뉴런이 readiness 로 갈린다. "손실 4개 중 앞쪽이 저해상도" 문제가 같이 완화된다.

대조군 (gain=0, 같은 lambda, 같은 묶음):
    채널 내 1-maxnorm lambda=5e-7  **94.19% @ 20,258  (n=3, sd 0.31)**
규제 없음 95.00% (sd 0.11, n=5) @ 78,789.  기준선 = 전역 1-softmax 곡선.

lambda 오염 — mean(sc_rate) 가 0.958 -> 약 0.93 (3% 변화) 로 옛 구현(32~114%) 보다
훨씬 작다. 착지점이 20,000 부근으로 거의 같게 와서 비교가 깨끗할 것이다.

판정 — 같은 스파이크에서 94.19% 를 뚜렷이 넘으면 (반복 편차 sd 0.31) 침묵 뉴런을
가른 것이 처음으로 효과를 낸 것이다. 겹치면 그 축은 닫는다.
n=1 씩이라 선별용이다.
"""
import subprocess, os, threading, time, sys, re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_vmsil')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
RESERVED = {6, 7}          # juyun 님 GPU — 절대 쓰지 않는다

SRC = '_mxg/mx_wc_5e-7'    # 채널 내 1-maxnorm, wta_rev, log_detail 켜짐, λ=5e-7

#        이름        gain    lambda
JOBS = [
    ('g100', '1.0', '5E-7'),   # 침묵을 최대로 펼침 (상한)
    ('g050', '0.5', '5E-7'),   # 절반
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
    c, n = RE_NAME.subn(f"conf.exp_set_name='vs-{name}'", c)
    if n < 1:
        raise RuntimeError(f'{name}: 활성 exp_set_name 없음')
    c, n = RE_LMB.subn(lambda m: f'{m.group(1)}conf.reg_spike_out_const={lmb}', c, count=1)
    if n != 1:
        raise RuntimeError(f'{name}: lambda 치환 실패')
    c, n = RE_MXG.subn(lambda m: (f"{m.group(1)}conf.reg_spike_maxnorm_group = 'within_channel'\n"
                                  f"{m.group(1)}conf.reg_spike_vmem_gain = {gain}\n"
                                  f"{m.group(1)}conf.reg_spike_vmem_silent_only = True"), c, count=1)
    if n != 1:
        raise RuntimeError(f'{name}: gain 삽입 실패')
    return c


def verify(path, name, gpu, gain, lmb):
    a = open(path).read()
    b = open(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')).read()
    norm = lambda s: RE_LMB.sub('', RE_NAME.sub('', RE_GPU.sub('', s)))
    a_cmp = re.sub(r"^\s*conf\.reg_spike_vmem_silent_only = True$\n?", '', RE_GAIN.sub('', a), flags=re.M)             # 새로 넣은 줄은 '비교용 사본'에서만 지운다
    if norm(a_cmp).replace('\n\n', '\n') != norm(b).replace('\n\n', '\n'):
        raise RuntimeError(f'{name}: GPU/이름/lambda/gain 외의 차이가 있음')
    if f'"CUDA_VISIBLE_DEVICES"]="{gpu}"' not in a:
        raise RuntimeError(f'{name}: GPU 지정 안 됨')
    eff = active_name(a)
    if eff != f'vs-{name}':
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
                     (r"^\s*conf\.reg_spike_loss_ratio\s*=\s*True\b", 'loss_ratio'),
                     (r"^\s*conf\.reg_spike_starget\s*=\s*True\b", 'starget'),
                     (r"^\s*conf\.reg_spike_wta_rev_floor\s*=", 'floor'),
                     (r"^\s*conf\.reg_spike_layer_cost\s*=", 'layer_cost'),
                     (r"^\s*conf\.reg_spike_shape_beta\s*=", 'shape_beta (이번 실험과 섞이면 안 됨)')]:
        if re.search(bad, a, re.M):
            raise RuntimeError(f'{name}: {why} 가 켜져 있음')
    la, lb = lambdas(a), lambdas(b)
    if len(la) != len(lb):
        raise RuntimeError(f'{name}: lambda 줄 개수가 바뀜 ({len(lb)}→{len(la)})')
    if la[0] != lmb:
        raise RuntimeError(f'{name}: 첫 lambda 가 {la[0]} (기대 {lmb})')
    if la[1:] != lb[1:]:
        raise RuntimeError(f'{name}: 두 번째 이후 lambda 가 바뀜 {lb[1:]}→{la[1:]}')


def preflight(jobs):
    bad = []
    if not os.path.exists(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')):
        bad.append(f'{SRC} (원본 config 없음)')
    for name, gain, lmb in jobs:
        tgt = os.path.join(PROJECT_ROOT, f'vs-{name}')
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
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {name}  (gain={gain}, λ={lmb})",
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
    print(f'--- vmem silent_only gain sweep: {len(pending)} runs ---', flush=True)
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
