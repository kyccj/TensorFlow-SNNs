"""vmem 작은 lambda 쪽 스윕 — 곡선 상단 (26-09-01 등록)

**왜 작은 lambda 쪽인가** — vmem 곡선의 얕은 끝이 비어 있고, 하나 있는 점이 이상하다.
    lambda=2e-7  94.59% @ 50,296  (n=1)   <- 이상치
    lambda=3e-7  94.876% @ 45,396 (n=5)
규제를 **약하게** 걸었는데 정확도가 **더 낮다.** 단조성이 깨져 있으니 둘 중 하나다 --
2e-7 이 n=1 잡음이거나, 곡선 상단이 실제로 꺾여 있거나.

**맞대결 상대가 여기에 있다.** 1-s 의 가장 잘 측정된 점이 이 구간이다:
    1-s lambda=1e-7   94.89% @ 53,160  (n=3, sd **0.026** -- 전체에서 가장 조밀)
    1-s lambda=5e-8   95.03% @ 60,215  (n=1)
`_shoulder` 에서 45~47K 맞대결은 두 방법 다 sd 가 커서 판정이 안 됐다 (합쳐 +1.14 SE).
53K 는 1-s 쪽 기준이 sd 0.026 이라 **같은 n 으로 훨씬 예민한 검정**이 된다.

**착지점 외삽** (vmem 상단 실측 지수: 2e-7 -> 50,296 과 3e-7 -> 45,396 에서 0.253):
    lambda=2e-7 -> 50,300      1.5e-7 -> 54,100      1e-7 -> 60,000

**배치 구성** (6런, GPU 6장, 약 13시간)
    1.5e-7 x3  핵심 맞대결 (~54,100 vs 1-s 53,160 n=3)
    2e-7   x2  이상치 해소 (기존 n=1 -> n=3)
    1e-7   x1  정찰 (~60,000, 1-s 5e-8 의 60,215 자리) -- n=1 이므로 착지점 확인용으로만 읽는다

시드는 고정하지 않는다 (기존 vmem 5런과 같은 조건으로 비교하기 위해).
설정 -- 채널 내 1-maxnorm + vmem silent_only, gain=1.0.
규제 없음 95.00% (sd 0.08~0.11, n=4~5) @ 78,789.  무손실 기준 94.78%.
"""
import subprocess, os, threading, time, sys, re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_vmlow')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
RESERVED = {6, 7}          # juyun 님 GPU — 절대 쓰지 않는다

SRC = '_vmsil3/g100_3e-7'  # vmem silent_only, gain=1.0, 채널 내 1-maxnorm, λ=3E-7

#        이름              lambda      예상 착지
JOBS = [
    ('l15_r1', '1.5E-7'),   # ~54,100  핵심
    ('l15_r2', '1.5E-7'),
    ('l20_r2', '2E-7'),     # ~50,300  이상치 해소
    ('l15_r3', '1.5E-7'),
    ('l20_r3', '2E-7'),
    ('l10_r1', '1E-7'),     # ~60,000  정찰 (n=1)
]

PREFIX = 'vlo'

RE_NAME = re.compile(r"^conf\.exp_set_name\s*=\s*'[^']*'", re.M)
RE_GPU  = re.compile(r'^os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=\s*"\d+"', re.M)
RE_LMB  = re.compile(r'^(\s*)conf\.reg_spike_out_const\s*=\s*(\S+)', re.M)


def src_text():
    return open(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')).read()


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


def make_config(gpu_id, name, lmb):
    c = src_text()
    c, n = RE_GPU.subn(f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"', c)
    if n != 1:
        raise RuntimeError(f'{name}: GPU 줄이 {n}개')
    c, n = RE_NAME.subn(f"conf.exp_set_name='{PREFIX}-{name}'", c)
    if n < 1:
        raise RuntimeError(f'{name}: 활성 exp_set_name 없음')
    c, n = RE_LMB.subn(lambda m: f'{m.group(1)}conf.reg_spike_out_const={lmb}', c, count=1)
    if n != 1:
        raise RuntimeError(f'{name}: lambda 치환 실패')
    return c


MUST = [(r"^\s*conf\.reg_spike_out_wta_rev\s*=\s*True\b", 'wta_rev'),
        (r"^\s*conf\.reg_spike_out_sc_maxnorm\s*=\s*True\b", '1-maxnorm'),
        (r"^\s*conf\.reg_spike_maxnorm_group\s*=\s*'within_channel'", '채널 내'),
        (r"^\s*conf\.reg_spike_vmem_gain\s*=\s*1\.0\b", 'vmem gain=1.0'),
        (r"^\s*conf\.reg_spike_vmem_silent_only\s*=\s*True\b", 'silent_only'),
        (r"^\s*conf\.reg_spike_log_detail\s*=\s*True\b", 'log_detail')]
NEVER = [(r"^\s*conf\.reg_spike_out_sc_maxnorm_plain\s*=\s*True\b", 'maxnorm_plain'),
         (r"^\s*conf\.reg_spike_loss_ratio\s*=\s*True\b", 'loss_ratio'),
         (r"^\s*conf\.reg_spike_accum_loss\s*=\s*True\b", 'accum'),
         (r"^\s*conf\.reg_spike_starget\s*=\s*True\b", 'starget'),
         (r"^\s*conf\.reg_spike_wta_rev_floor\s*=", 'floor'),
         (r"^\s*conf\.reg_spike_shape_beta\s*=", 'shape_beta'),
         (r"^\s*conf\.reg_spike_layer_cost\s*=\s*'synops'", 'layer_cost')]


def verify(path, name, gpu, lmb):
    a = open(path).read()
    b = src_text()
    norm = lambda s: RE_LMB.sub('', RE_NAME.sub('', RE_GPU.sub('', s)))
    if norm(a) != norm(b):
        raise RuntimeError(f'{name}: GPU/이름/lambda 외의 차이가 있음')
    if f'"CUDA_VISIBLE_DEVICES"]="{gpu}"' not in a:
        raise RuntimeError(f'{name}: GPU 지정 안 됨')
    eff = active_name(a)
    if eff != f'{PREFIX}-{name}':
        raise RuntimeError(f'{name}: 유효 exp_set_name 이 {eff!r} — 원본 덮어쓸 위험')
    if active_name(b) == eff:
        raise RuntimeError(f'{name}: 원본과 저장 경로가 같음')
    # 주석이 붙은 줄도 있으므로 $ 로 닫지 않는다
    for pat, why in MUST:
        if not re.search(pat, a, re.M):
            raise RuntimeError(f'{name}: {why} 설정 안 됨')
    for pat, why in NEVER:
        if re.search(pat, a, re.M):
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
    for name, lmb in jobs:
        tgt = os.path.join(PROJECT_ROOT, f'{PREFIX}-{name}')
        if os.path.exists(tgt) and any(f.endswith('.weights.h5')
                                       for _, _, fs in os.walk(tgt) for f in fs):
            bad.append(tgt)
        if os.path.exists(os.path.join(SWEEP_DIR, name, 'train.log')):
            bad.append(f'{SWEEP_DIR}/{name}/train.log (이미 돈 흔적)')
    fl = open(os.path.join(PROJECT_ROOT, 'flags.py')).read()
    nr = open(os.path.join(PROJECT_ROOT, 'lib_snn', 'neurons.py')).read()
    for tok in ('reg_spike_vmem_silent_only', 'reg_spike_vmem_gain'):
        if tok not in fl:
            bad.append(f'flags.py 에 {tok} 없음')
        if tok not in nr:
            bad.append(f'lib_snn/neurons.py 에 {tok} 반영 안 됨')
    if bad:
        raise SystemExit('중단:\n  ' + '\n  '.join(bad))
    # 만들어질 config 를 미리 전부 검증한다 (돌기 시작한 뒤 죽는 것 방지)
    os.makedirs(SWEEP_DIR, exist_ok=True)
    for name, lmb in jobs:
        tmp = os.path.join(SWEEP_DIR, f'.pre_{name}.py')
        open(tmp, 'w').write(make_config(0, name, lmb))
        try:
            verify(tmp, name, 0, lmb)
        finally:
            os.remove(tmp)
    print(f'사전 점검 통과: {len(jobs)}개 config 생성·검증 완료, 저장 경로 충돌 없음', flush=True)


def run_one(gpu, name, lmb):
    d = os.path.join(SWEEP_DIR, name)
    os.makedirs(d, exist_ok=True)
    cfg = os.path.join(d, 'config_sweep.py')
    open(cfg, 'w').write(make_config(gpu, name, lmb))
    verify(cfg, name, gpu, lmb)
    m = open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')).read()
    open(os.path.join(d, 'main_sweep.py'), 'w').write(
        m.replace('from config_snn_training import config', 'from config_sweep import config'))
    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {name}  (vmem silent_only λ={lmb})",
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
    print(f'--- vmem 작은 λ 스윕: {len(pending)} runs ---', flush=True)
    while pending:
        for g in free_gpus(busy):
            if not pending:
                break
            job = pending.pop(0)
            busy.add(g)

            def worker(g=g, job=job):
                try:
                    run_one(g, *job)
                except Exception as e:                 # 한 job 이 죽어도 나머지는 계속
                    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {g}] FAIL {job[0]}: {e}",
                          flush=True)
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
