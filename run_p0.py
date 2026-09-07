"""Phase 0 — R19-C10 rho 정찰 (26-09-03 등록)

**왜 정찰이 먼저인가.** `reg_spike_final_step` 을 켜면 손실이 받는 양이
    Sum_t ||spike_t * sc_rate_t||   (T 개 항)      ->   ||Sum_t spike_t * sc_rate||   (1 개 항)
으로 바뀐다. 두 양의 비는 시간 상관에 달려 있다 -- 뉴런이 매 t 다 터지면 같고,
흩어져 터지면 sqrt(T) 배쯤 작아진다. R 이 작아지면 lambda = rho*L/R 이 커지므로
**같은 rho 라도 더 깊이 착지한다.** 배율이 1~T 사이 어디인지 모른다.

lambda 를 이번 세션에 세 번 크게 틀렸다 (vmem gain 32~114%, accum 은 방향까지 반대로
20배, silent_only 21%). 그래서 4점 스윕 전에 2점으로 사상을 먼저 잰다.

**기준점** — 같은 R19-C10 에서 1-softmax + loss-ratio (`_rho_agg/agg_r19_c10`):
    rho=3e-3, R_per_step=True  ->  96.14% @ 115,637   (baseline 96.65% @ 490K)
단 그 런은 `R_per_step=True` 라 R 을 T 배 과소보고했다 (flags.py:857-858). 새 실험은 False.
`final_step` 이 켜지면 sc_loss_snap 이 어차피 assign 이라 R_per_step 은 무효가 되지만,
설정 기록을 위해 명시적으로 끈다.

**제안법 설정** — 채널 내 1-maxnorm + vmem(silent_only, gain=1.0) + final_step + loss-ratio.

원본: `_rho_agg/agg_r19_c10` (R19-C10, loss-ratio 배선이 이미 되어 있음).
바꾸는 것: GPU / 이름 / rho / R_per_step(False) / maxnorm·vmem·final_step 3줄 추가.
"""
import subprocess, os, threading, time, sys, re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_p0')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
# 1·3·5 = _vmlow 진행 중, 4 = 다른 프로세스, 6·7 = juyun 님
RESERVED = {1, 3, 4, 5, 6, 7}

SRC = '_rho_agg/agg_r19_c10'
PREFIX = 'p0'

#          이름      rho
JOBS = [
    ('rho1e-3', '0.001'),
    ('rho3e-3', '0.003'),
]

RE_NAME = re.compile(r"^conf\.exp_set_name\s*=\s*'[^']*'", re.M)
RE_GPU  = re.compile(r'^os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=\s*"\d+"', re.M)
RE_LMB  = re.compile(r'^(\s*)conf\.reg_spike_out_const\s*=\s*(\S+)', re.M)
RE_RHO  = re.compile(r'^(\s*)conf\.reg_spike_loss_ratio_target\s*=\s*(\S+)', re.M)
RE_RPS  = re.compile(r'^(\s*)conf\.reg_spike_R_per_step\s*=\s*(\S+)', re.M)


def src_text():
    return open(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')).read()


def active_name(txt):
    """주석이 아니라 실제로 마지막에 대입되는 exp_set_name 을 읽는다.
    (08-21 에 주석 줄을 치환해 원본 체크포인트를 날린 사고 재발 방지)"""
    i = txt.find('conf.root_model_save=conf.exp_set_name')
    if i < 0:
        raise RuntimeError('root_model_save 줄을 못 찾음')
    ms = RE_NAME.findall(txt[:i])
    if not ms:
        raise RuntimeError('활성 exp_set_name 대입이 없음')
    return ms[-1].split("'")[1]


def lambdas(txt):
    return [m.group(2) for m in RE_LMB.finditer(txt)]


# R_per_step 줄을 제안법 설정 블록으로 통째로 갈아끼운다
def _method_block(indent):
    return (f"{indent}conf.reg_spike_R_per_step = False\n"
            f"{indent}conf.reg_spike_out_sc_maxnorm = True\n"
            f"{indent}conf.reg_spike_maxnorm_group = 'within_channel'\n"
            f"{indent}conf.reg_spike_vmem_gain = 1.0\n"
            f"{indent}conf.reg_spike_vmem_silent_only = True\n"
            f"{indent}conf.reg_spike_final_step = True")


def make_config(gpu_id, name, rho):
    c = src_text()
    c, n = RE_GPU.subn(f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"', c)
    if n != 1:
        raise RuntimeError(f'{name}: GPU 줄이 {n}개')
    c, n = RE_NAME.subn(f"conf.exp_set_name='{PREFIX}-{name}'", c)
    if n < 1:
        raise RuntimeError(f'{name}: 활성 exp_set_name 없음')
    c, n = RE_RHO.subn(lambda m: f'{m.group(1)}conf.reg_spike_loss_ratio_target = {rho}', c, count=1)
    if n != 1:
        raise RuntimeError(f'{name}: rho 치환 실패 ({n}개)')
    c, n = RE_RPS.subn(lambda m: _method_block(m.group(1)), c, count=1)
    if n != 1:
        raise RuntimeError(f'{name}: R_per_step 줄을 못 찾음 ({n}개)')
    return c


MUST = [(r"^\s*conf\.reg_spike_out\s*=\s*True\b", 'reg_spike_out'),
        (r"^\s*conf\.reg_spike_out_wta_rev\s*=\s*True\b", 'wta_rev'),
        (r"^\s*conf\.reg_spike_out_sc_maxnorm\s*=\s*True\b", '1-maxnorm'),
        (r"^\s*conf\.reg_spike_maxnorm_group\s*=\s*'within_channel'", '채널 내'),
        (r"^\s*conf\.reg_spike_vmem_gain\s*=\s*1\.0\b", 'vmem gain=1.0'),
        (r"^\s*conf\.reg_spike_vmem_silent_only\s*=\s*True\b", 'silent_only'),
        (r"^\s*conf\.reg_spike_final_step\s*=\s*True\b", 'final_step'),
        (r"^\s*conf\.reg_spike_loss_ratio\s*=\s*True\b", 'loss_ratio'),
        (r"^\s*conf\.reg_spike_R_per_step\s*=\s*False\b", 'R_per_step=False'),
        (r"^\s*conf\.reg_spike_log_detail\s*=\s*True\b", 'log_detail'),
        (r"^conf\.model\s*=\s*'ResNet19'", 'ResNet19'),
        (r"^conf\.dataset\s*=\s*'CIFAR10'", 'CIFAR10')]
NEVER = [(r"^\s*conf\.reg_spike_accum_loss\s*=\s*True\b", 'accum (final_step 과 배타)'),
         (r"^\s*conf\.reg_spike_out_sc_maxnorm_plain\s*=\s*True\b", 'maxnorm_plain'),
         (r"^\s*conf\.reg_spike_starget\s*=\s*True\b", 'starget'),
         (r"^\s*conf\.reg_spike_adaptive\s*=\s*True\b", 'adaptive'),
         (r"^\s*conf\.reg_spike_grow\s*=\s*True\b", 'grow'),
         (r"^\s*conf\.reg_spike_wta_rev_floor\s*=", 'floor'),
         (r"^\s*conf\.reg_spike_shape_beta\s*=", 'shape_beta'),
         (r"^\s*conf\.reg_spike_layer_cost\s*=\s*'synops'", 'layer_cost'),
         (r"^\s*conf\.reg_spike_lr_brake\s*=\s*True\b", 'lr_brake (정찰은 순수 loss-ratio)')]


def verify(path, name, gpu, rho):
    a = open(path).read()
    b = src_text()
    # GPU/이름/rho/제안법블록 을 지운 나머지가 원본과 같아야 한다
    strip = lambda s: RE_RPS.sub('', RE_RHO.sub('', RE_NAME.sub('', RE_GPU.sub('', s))))
    a_cmp = re.sub(r"^\s*conf\.reg_spike_(out_sc_maxnorm|maxnorm_group|vmem_gain|"
                   r"vmem_silent_only|final_step)\s*=.*$\n?", '', a, flags=re.M)
    if strip(a_cmp).replace('\n\n', '\n') != strip(b).replace('\n\n', '\n'):
        raise RuntimeError(f'{name}: GPU/이름/rho/제안법 블록 외의 차이가 있음')
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
    if not re.search(rf"^\s*conf\.reg_spike_loss_ratio_target = {re.escape(rho)}$", a, re.M):
        raise RuntimeError(f'{name}: rho={rho} 설정 안 됨')
    if lambdas(a) != lambdas(b):
        raise RuntimeError(f'{name}: lambda 줄이 바뀜 (loss-ratio 가 정해야 함)')


def preflight(jobs):
    bad = []
    if not os.path.exists(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')):
        bad.append(f'{SRC} (원본 config 없음)')
    for name, rho in jobs:
        tgt = os.path.join(PROJECT_ROOT, f'{PREFIX}-{name}')
        if os.path.exists(tgt) and any(f.endswith('.weights.h5')
                                       for _, _, fs in os.walk(tgt) for f in fs):
            bad.append(tgt)
        if os.path.exists(os.path.join(SWEEP_DIR, name, 'train.log')):
            bad.append(f'{SWEEP_DIR}/{name}/train.log (이미 돈 흔적)')
    # 플래그가 실제로 반영돼 있는지 (없으면 조용히 무시된다)
    fl = open(os.path.join(PROJECT_ROOT, 'flags.py')).read()
    nr = open(os.path.join(PROJECT_ROOT, 'lib_snn', 'neurons.py')).read()
    for tok in ('reg_spike_final_step', 'reg_spike_vmem_silent_only'):
        if tok not in fl:
            bad.append(f'flags.py 에 {tok} 없음')
        if tok not in nr:
            bad.append(f'lib_snn/neurons.py 에 {tok} 반영 안 됨')
    if bad:
        raise SystemExit('중단:\n  ' + '\n  '.join(bad))
    os.makedirs(SWEEP_DIR, exist_ok=True)
    for name, rho in jobs:
        tmp = os.path.join(SWEEP_DIR, f'.pre_{name}.py')
        open(tmp, 'w').write(make_config(0, name, rho))
        try:
            verify(tmp, name, 0, rho)
        finally:
            os.remove(tmp)
    print(f'사전 점검 통과: {len(jobs)}개 config 생성·검증 완료', flush=True)


def run_one(gpu, name, rho):
    d = os.path.join(SWEEP_DIR, name)
    os.makedirs(d, exist_ok=True)
    cfg = os.path.join(d, 'config_sweep.py')
    open(cfg, 'w').write(make_config(gpu, name, rho))
    verify(cfg, name, gpu, rho)
    m = open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')).read()
    open(os.path.join(d, 'main_sweep.py'), 'w').write(
        m.replace('from config_snn_training import config', 'from config_sweep import config'))
    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {name}  "
          f"(R19-C10, final_step + vmem + loss-ratio rho={rho})", flush=True)
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
    print(f'--- Phase 0 rho 정찰: {len(pending)} runs ---', flush=True)
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
