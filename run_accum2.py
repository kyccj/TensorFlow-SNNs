"""accum_loss 재시도 — lambda 다시 맞춤 (26-08-28 등록)

1차 시도(_accum/)는 **lambda 를 반대 방향으로 틀려서** 네 점 다 붕괴 구간에 착지했다:
    lambda=2e-6  67.04% @  6,636   침묵 97.67%
    lambda=4e-6  89.15% @  6,438   침묵 97.24%
    lambda=6e-6  85.35% @  3,156   침묵 98.13%
    lambda=1e-5  50.10% @ 142,760  best 가 8에폭째, 전 층 사망
목표는 20,000 부근이었다. 비교 대상이 없어 결론을 못 냈다.

무엇을 틀렸나 — "accum 은 손실이 t=T 에 한 번만 붙으니 gradient 가 1/T 로 약해진다"
고 봤는데, x = sum_t (spike_t * sc_rate_t) 이라 **역전파가 그 합을 타고 T 개 타임스텝
전부로 흘러간다.** 한 번이 아니라 T 번 적용된다. 그것만 고쳐도 10.8배가 2.7배가 되고,
실측은 그보다도 더 세다 (accum lambda=2e-6 이 accum-off lambda=5e-7 보다 훨씬 깊다).

이번엔 계산 대신 **실측 기준점으로 외삽**한다:
    accum 끔 곡선에서  스파이크 ~ lambda^-0.561   (1e-7 -> 49,963,  5e-7 -> 20,258)
    accum 켬 기준점    lambda=2e-6 -> 6,636
    -> 대조군 자리에 오려면 lambda = 5.5e-8 / 1.4e-7 / 2.7e-7

주의 — 기준점 6,636 이 붕괴 구간이라 외삽이 안 맞을 수 있다. 그래서 8배 범위로 넓게 잡는다.

무엇을 묻나 — 지금은 규제 손실이 **매 타임스텝** 붙는데 sc_rate 는 그 시각까지의 누적
spike_count 로 계산되므로 해상도가 시각마다 다르다:
    t=1  sc_rate {1, 0}          이진 마스크
    t=4  {1, .75, .5, .25, 0}    5단계
**손실 4개 중 앞쪽이 저해상도다.** accum 을 켜면 t=T 에서 한 번, 항상 5단계로 계산한다.

대조군 (accum 끔, 채널 내 1-maxnorm):
    lambda=1e-7  94.72% @ 49,963  (n=1)
    lambda=3e-7  94.46% @ 29,646  (n=1)
    lambda=5e-7  94.19% @ 20,258  (n=3)
규제 없음 95.00% (sd 0.11, n=5) @ 78,789.  기준선 = 전역 1-softmax 곡선.

판정 — 같은 스파이크에서 accum 켬이 위면 저해상도 타임스텝이 차등을 희석하고 있었던
것이다. 겹치면 해상도는 무관하고 차등화 무효 결론이 더 강해진다.
"""
import subprocess, os, threading, time, sys, re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_accum2')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
RESERVED = {6, 7}          # juyun 님 GPU — 절대 쓰지 않는다

SRC = '_mxg/mx_wc_5e-7'    # 채널 내 1-maxnorm, wta_rev, log_detail 켜짐, λ=5e-7

#        이름        gain    lambda
JOBS = [
    ('a2_6e-8',   '1.0', '6E-8'),     # 예상 ~47,000  (대조 1e-7 = 49,963)
    ('a2_1.2e-7', '1.0', '1.2E-7'),   # 예상 ~32,000  (대조 3e-7 = 29,646)
    ('a2_2.5e-7', '1.0', '2.5E-7'),   # 예상 ~21,000  (대조 5e-7 = 20,258)
    ('a2_3e-8',   '1.0', '3E-8'),     # 예상 ~70,000  얕은 쪽 여유
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
    c, n = RE_NAME.subn(f"conf.exp_set_name='a2-{name}'", c)
    if n < 1:
        raise RuntimeError(f'{name}: 활성 exp_set_name 없음')
    c, n = RE_LMB.subn(lambda m: f'{m.group(1)}conf.reg_spike_out_const={lmb}', c, count=1)
    if n != 1:
        raise RuntimeError(f'{name}: lambda 치환 실패')
    c, n = RE_MXG.subn(lambda m: (f"{m.group(1)}conf.reg_spike_maxnorm_group = 'within_channel'\n"
                                  f"{m.group(1)}conf.reg_spike_shape_beta = {gain}\n"
                                  f"{m.group(1)}conf.reg_spike_accum_loss = True"), c, count=1)
    if n != 1:
        raise RuntimeError(f'{name}: gain 삽입 실패')
    return c


def verify(path, name, gpu, gain, lmb):
    a = open(path).read()
    b = open(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')).read()
    norm = lambda s: RE_LMB.sub('', RE_NAME.sub('', RE_GPU.sub('', s)))
    a_cmp = re.sub(r"^\s*conf\.reg_spike_accum_loss = True$\n?", '', RE_GAIN.sub('', a), flags=re.M)             # 새로 넣은 줄은 '비교용 사본'에서만 지운다
    if norm(a_cmp).replace('\n\n', '\n') != norm(b).replace('\n\n', '\n'):
        raise RuntimeError(f'{name}: GPU/이름/lambda/gain 외의 차이가 있음')
    if f'"CUDA_VISIBLE_DEVICES"]="{gpu}"' not in a:
        raise RuntimeError(f'{name}: GPU 지정 안 됨')
    eff = active_name(a)
    if eff != f'a2-{name}':
        raise RuntimeError(f'{name}: 유효 exp_set_name 이 {eff!r} — 원본 덮어쓸 위험')
    if active_name(b) == eff:
        raise RuntimeError(f'{name}: 원본과 저장 경로가 같음')
    if not re.search(r"^\s*conf\.reg_spike_accum_loss = True", a, re.M):
        raise RuntimeError(f'{name}: accum_loss 설정 안 됨')
    gs = RE_GAIN.findall(a)
    if gs != [gain]:
        raise RuntimeError(f'{name}: beta 가 {gs} (기대 [{gain!r}])')
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
                     (r"^\s*conf\.reg_spike_vmem_gain\s*=", 'vmem gain'),
                     (r"^\s*conf\.reg_spike_layer_cost\s*=", 'layer_cost (이번 실험과 섞이면 안 됨)')]:
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
        tgt = os.path.join(PROJECT_ROOT, f'a2-{name}')
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
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {name}  (β={gain}, λ={lmb})",
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
    print(f'--- accum_loss 재시도 (λ 재보정): {len(pending)} runs ---', flush=True)
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
