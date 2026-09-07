"""무손실 어깨 판정 — vmem λ=3e-7 복제 + 1-s λ=1.5e-7 맞대결 (26-08-31 등록)

**왜 이 실험인가 — 지금까지 답한 질문이 목표와 달랐다.**
"세 방법이 같은 곡선"은 14K~60K 전체의 *기울기*에 대한 결론이고 그건 튼튼하다
(추세선 대비 vmem +0.017%p, sd 0.26, n=7). 그런데 실제 목표는
"정확도를 희생하지 않으면서 스파이크를 줄인다" = **어깨가 어디까지 뻗느냐는 문턱 판정**이다.
평균 잔차 검정은 전 구간을 평균 내므로 어깨 위치 차이에 구조적으로 둔감하다.

**문턱 근처는 잰 적이 없다.** 무손실 기준 94.78% (규제 없음 95.00, sd 0.11, n=5 − 2sd).
    53,160  32.5%  1-s   1e-7    94.89  n=3  O   <- 현재 무손실 최심점
    50,296  36.2%  vmem  2e-7    94.59  n=1  X
    49,963  36.6%  1-m   1e-7    94.72  n=1  X
    46,749  40.7%  vmem  3e-7    95.08  n=1  O   <- 전체에서 가장 깊은 무손실 점
    42,471  46.1%  vmem  3.5e-7  94.62  n=1  X
    40,832  48.2%  1-s   2e-7    94.66  n=1  X
    40,175  49.0%  vmem  4e-7    94.73  n=1  X
36K~50K 여덟 점 중 여섯이 n=1 이고 94.59~95.08 로 0.49%p 흩어져 있으며 λ 순서도 안 맞는다.

**기전도 이 구간을 가리킨다.** vmem 은 문턱에 가까운 침묵 뉴런의 sc_rate 를 낮춰
(scp = g*readiness) 벌점을 줄여 보호한다. 약한 규제에서 정확도를 깎는 원인이
"곧 발화할 뻔한 뉴런을 죽이는 것"이므로, 이점이 있다면 붕괴 구간이 아니라 어깨에서 나온다.

**표본 수** — vmem 5e-7 복제 sd 0.19. n=5 면 SE 0.085 이고, 참값이 94.95 면 문턱과 2 SE.
n=1 은 ±0.5%p 라 애초에 판정이 불가능했다 (이번 세션에서 n=1 고점이 4번 무너졌다).

**두 팔**
  vm3-r2~r5  : `_vmsil3/g100_3e-7` 그대로 복제. GPU/이름만 바꾼다 (λ=3E-7 유지) -> n=5
  sm15-r1~r3 : `_gap3/sm_2e-7` 에서 λ 만 1.5E-7 로. 예상 ~45,500 -> vmem 46,749 와 짝비교
               (`_gap3/sm_1.5e-7` 은 run_gap3.py 의 verify() NameError 로 죽어서 미실행)

**판정** — vm3 가 n=5 로 94.78 이상이면 무손실 최심점이 53,160 -> 46,749 (32.5% -> 40.7%).
sm15 도 94.78 이상이면 그건 1-s 도 되는 것이므로 vmem 고유 이득이 아니다. 둘 다 필요하다.
"""
import subprocess, os, threading, time, sys, re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_shoulder')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
# GPU 4 = 본인 DVS 런, 6·7 = juyun 님. 0~3 은 _vmlr 이 끝나면 풀린다.
RESERVED = {4, 6, 7}

SRC_VM = '_vmsil3/g100_3e-7'   # vmem silent_only, gain=1.0, 채널 내 1-maxnorm, λ=3E-7
SRC_SM = '_gap3/sm_2e-7'       # 전역 1−softmax + wta_rev, λ=2E-7

#        이름          팔      lambda      (None = 원본 그대로)
JOBS = [
    ('vm3-r2',   'vm', None),      # vmem λ=3e-7 복제 (기존 n=1 -> 목표 n=5)
    ('vm3-r3',   'vm', None),
    ('sm15-r1',  'sm', '1.5E-7'),  # 1-s λ=1.5e-7, 예상 ~45,500
    ('vm3-r4',   'vm', None),
    ('sm15-r2',  'sm', '1.5E-7'),
    ('vm3-r5',   'vm', None),
    ('sm15-r3',  'sm', '1.5E-7'),
]

PREFIX = 'sh'                  # 저장 경로 접두사 — 원본과 절대 겹치지 않게

RE_NAME = re.compile(r"^conf\.exp_set_name\s*=\s*'[^']*'", re.M)
RE_GPU  = re.compile(r'^os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=\s*"\d+"', re.M)
RE_LMB  = re.compile(r'^(\s*)conf\.reg_spike_out_const\s*=\s*(\S+)', re.M)

SRC = {'vm': SRC_VM, 'sm': SRC_SM}


def src_text(arm):
    return open(os.path.join(PROJECT_ROOT, SRC[arm], 'config_sweep.py')).read()


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


def make_config(gpu_id, name, arm, lmb):
    c = src_text(arm)
    c, n = RE_GPU.subn(f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"', c)
    if n != 1:
        raise RuntimeError(f'{name}: GPU 줄이 {n}개')
    c, n = RE_NAME.subn(f"conf.exp_set_name='{PREFIX}-{name}'", c)
    if n < 1:
        raise RuntimeError(f'{name}: 활성 exp_set_name 없음')
    if lmb is not None:
        c, n = RE_LMB.subn(lambda m: f'{m.group(1)}conf.reg_spike_out_const={lmb}', c, count=1)
        if n != 1:
            raise RuntimeError(f'{name}: lambda 치환 실패')
    # 플래그는 하나도 새로 넣지 않는다 — 원본을 그대로 복제하는 게 이 실험의 요점
    return c


# 팔마다 반드시 켜져 있어야 하는 것 / 절대 켜져 있으면 안 되는 것
MUST = {
    'vm': [(r"^\s*conf\.reg_spike_out_wta_rev\s*=\s*True\b", 'wta_rev'),
           (r"^\s*conf\.reg_spike_out_sc_maxnorm\s*=\s*True\b", '1-maxnorm'),
           (r"^\s*conf\.reg_spike_maxnorm_group\s*=\s*'within_channel'", '채널 내'),
           (r"^\s*conf\.reg_spike_vmem_gain\s*=\s*1\.0\b", 'vmem gain=1.0'),
           (r"^\s*conf\.reg_spike_vmem_silent_only\s*=\s*True\b", 'silent_only'),
           (r"^\s*conf\.reg_spike_log_detail\s*=\s*True\b", 'log_detail')],
    'sm': [(r"^\s*conf\.reg_spike_out_wta_rev\s*=\s*True\b", 'wta_rev'),
           (r"^\s*conf\.reg_spike_out_sc_sm\s*=\s*True\b", '1-softmax'),
           (r"^\s*conf\.reg_spike_log_detail\s*=\s*True\b", 'log_detail')],
}
NEVER = {
    'vm': [(r"^\s*conf\.reg_spike_out_sc_maxnorm_plain\s*=\s*True\b", 'maxnorm_plain'),
           (r"^\s*conf\.reg_spike_loss_ratio\s*=\s*True\b", 'loss_ratio'),
           (r"^\s*conf\.reg_spike_accum_loss\s*=\s*True\b", 'accum'),
           (r"^\s*conf\.reg_spike_starget\s*=\s*True\b", 'starget'),
           (r"^\s*conf\.reg_spike_wta_rev_floor\s*=", 'floor'),
           (r"^\s*conf\.reg_spike_shape_beta\s*=", 'shape_beta'),
           (r"^\s*conf\.reg_spike_layer_cost\s*=\s*'synops'", 'layer_cost')],
    'sm': [(r"^\s*conf\.reg_spike_out_sc_maxnorm\s*=\s*True\b", 'maxnorm (기준선이어야 함)'),
           (r"^\s*conf\.reg_spike_out_sc_maxnorm_plain\s*=\s*True\b", 'maxnorm_plain'),
           (r"^\s*conf\.reg_spike_loss_ratio\s*=\s*True\b", 'loss_ratio'),
           (r"^\s*conf\.reg_spike_accum_loss\s*=\s*True\b", 'accum'),
           (r"^\s*conf\.reg_spike_starget\s*=\s*True\b", 'starget'),
           (r"^\s*conf\.reg_spike_vmem_gain\s*=", 'vmem gain'),
           (r"^\s*conf\.reg_spike_wta_rev_floor\s*=", 'floor'),
           (r"^\s*conf\.reg_spike_shape_beta\s*=", 'shape_beta'),
           (r"^\s*conf\.reg_spike_layer_cost\s*=\s*'synops'", 'layer_cost')],
}


def verify(path, name, arm, gpu, lmb):
    a = open(path).read()
    b = src_text(arm)
    # GPU / 이름 / lambda 를 지운 나머지가 원본과 완전히 같아야 한다
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
    for pat, why in MUST[arm]:
        if not re.search(pat, a, re.M):
            raise RuntimeError(f'{name}: {why} 설정 안 됨')
    for pat, why in NEVER[arm]:
        if re.search(pat, a, re.M):
            raise RuntimeError(f'{name}: {why} 가 켜져 있음')
    la, lb = lambdas(a), lambdas(b)
    if len(la) != len(lb):
        raise RuntimeError(f'{name}: lambda 줄 개수가 바뀜 ({len(lb)}→{len(la)})')
    want = lb[0] if lmb is None else lmb
    if la[0] != want:
        raise RuntimeError(f'{name}: 첫 lambda 가 {la[0]} (기대 {want})')
    if la[1:] != lb[1:]:
        raise RuntimeError(f'{name}: 두 번째 이후 lambda 가 바뀜 {lb[1:]}→{la[1:]}')


def preflight(jobs):
    bad = []
    for arm in {j[1] for j in jobs}:
        if not os.path.exists(os.path.join(PROJECT_ROOT, SRC[arm], 'config_sweep.py')):
            bad.append(f'{SRC[arm]} (원본 config 없음)')
    for name, arm, lmb in jobs:
        tgt = os.path.join(PROJECT_ROOT, f'{PREFIX}-{name}')
        if os.path.exists(tgt) and any(f.endswith('.weights.h5')
                                       for _, _, fs in os.walk(tgt) for f in fs):
            bad.append(tgt)
        d = os.path.join(SWEEP_DIR, name)
        if os.path.exists(os.path.join(d, 'train.log')):
            bad.append(f'{d}/train.log (이미 돈 흔적)')
    # 플래그가 실제로 있는지 (없으면 조용히 무시되고 gain=0 이 된다)
    fl = open(os.path.join(PROJECT_ROOT, 'flags.py')).read()
    nr = open(os.path.join(PROJECT_ROOT, 'lib_snn', 'neurons.py')).read()
    for tok, why in [('reg_spike_vmem_silent_only', 'silent_only'),
                     ('reg_spike_out_sc_sm', '1-softmax')]:
        if tok not in fl:
            bad.append(f'flags.py 에 {why} 없음')
        if tok not in nr:
            bad.append(f'lib_snn/neurons.py 에 {why} 반영 안 됨')
    if bad:
        raise SystemExit('중단:\n  ' + '\n  '.join(bad))
    # 만들어질 config 를 미리 전부 검증한다 (돌기 시작한 뒤 죽는 것 방지)
    for name, arm, lmb in jobs:
        tmp = os.path.join(SWEEP_DIR, f'.pre_{name}.py')
        os.makedirs(SWEEP_DIR, exist_ok=True)
        open(tmp, 'w').write(make_config(0, name, arm, lmb))
        try:
            verify(tmp, name, arm, 0, lmb)
        finally:
            os.remove(tmp)
    print(f'사전 점검 통과: {len(jobs)}개 config 생성·검증 완료, 저장 경로 충돌 없음', flush=True)


def run_one(gpu, name, arm, lmb):
    d = os.path.join(SWEEP_DIR, name)
    os.makedirs(d, exist_ok=True)
    cfg = os.path.join(d, 'config_sweep.py')
    open(cfg, 'w').write(make_config(gpu, name, arm, lmb))
    verify(cfg, name, arm, gpu, lmb)
    m = open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')).read()
    open(os.path.join(d, 'main_sweep.py'), 'w').write(
        m.replace('from config_snn_training import config', 'from config_sweep import config'))
    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'
    tag = 'vmem silent_only λ=3E-7' if arm == 'vm' else f'전역 1−softmax λ={lmb}'
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {name}  ({tag})", flush=True)
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
    print(f'--- 무손실 어깨 판정: {len(pending)} runs '
          f'(vmem {sum(1 for j in pending if j[1] == "vm")} / '
          f'1-s {sum(1 for j in pending if j[1] == "sm")}) ---', flush=True)
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
