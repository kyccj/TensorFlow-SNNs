"""논문용 통합 런처 (26-09-03).

기존 런처들은 원본 config 의 특정 줄을 정규식으로 갈아끼웠다. 방법이 하나뿐일 때는 됐지만
조합 x 방법 x 손잡이 x 시드로 늘어나면 치환 대상이 매번 달라져 못 버틴다.
(`run_gap3.py` 는 `verify()` 가 `a`/`b` 미정의인 채로 남아 job 을 조용히 죽였다.)

**대신 `config.set()` 바로 앞에 덮어쓰기 블록을 끼워 넣는다.** 파이썬은 나중 대입이 이기므로
앞쪽 설정이 무엇이든 이 블록이 최종값이 된다. 원본은 GPU 줄과 exp_set_name 만 건드린다.

    python run_paper.py r19c10 prop 1e-3 --seeds 1,2,3,4,5
    python run_paper.py r19c10 sm 1e-7 --seeds 1,2,3 --gpus 0,2
    python run_paper.py --list

**방법**
    prop        제안법 — 채널 내 1-maxnorm + vmem(silent_only, gain=1.0) + final_step + loss-ratio(rho)
    ours_w1     prop 와 전부 같고 wta_rev 의 spike 기울기만 w 한 번 (기본은 w^2). knob 은 rho
    ours_ch     prop 와 전부 같고 max 범위만 채널 합끼리 ('channel'). knob 은 rho
    sm          1-softmax + 고정 lambda        (내부 최강 기준선)
    l2          plain L2 + 고정 lambda         (문헌 기준선. reg_spike_out_sc=False 로 두면
                                                neurons.py:1238 의 `else: # old - previous work`
                                                가지로 가서 l2_norm(spike) 이 된다)
    abl_nofinal 제안법에서 final_step 만 끔     (시점의 기여)
    abl_novmem  제안법에서 vmem 만 끔 (gain=0)  (막전위의 기여)
    abl_noinv   제안법에서 1- 반전만 끔         (maxnorm_plain = sc/max. 이름의 근거)
    abl_nolr    제안법에서 loss-ratio 만 끔  (고정 lambda. knob 이 rho 가 아니라 lambda)

**선별** — 조건당 5런을 돌리고 4개를 쓴다. 배제 규칙은 제안법과 비교군에 똑같이 적용한다:
    (1) 310에폭 미완주  (2) S30/S1 < 0.19  (3) 그래도 5개 남으면 val_acc 상위 4개.
판정은 `collect_paper.py` 가 한다.
"""
import argparse, os, re, subprocess, sys, threading, time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# 서버마다 conda 환경 위치가 다르다. EIP_PYTHON 으로 덮어쓴다.
#   EIP_PYTHON=/srv2/kyccj/miniconda3/envs/venv_1/bin/python python run_paper.py ...
PYTHON = os.environ.get('EIP_PYTHON', '/home/kyccj/anaconda3/envs/venv_1/bin/python')
CUDA_LD_PATH = os.environ.get(
    'EIP_CUDA_LD', os.path.join(os.path.dirname(os.path.dirname(PYTHON)), 'lib'))

# 조합 -> 원본 config (모델·데이터셋·학습 하이퍼파라미터를 그대로 물려받는다)
SOURCES = {
    'r19c10':   ('_rho_agg/agg_r19_c10',                 'ResNet19', 'CIFAR10'),
    'vggc10':   ('_sweep_wta_rev/lambda_1e-07',          'VGG16',    'CIFAR10'),
    'r19c100':  ('_sweep_wta_rev_r19_c100/lambda_1e-07', 'ResNet19', 'CIFAR100'),
    'vggc100':  ('_sweep_wta_rev_c100/lambda_1e-07',     'VGG16',    'CIFAR100'),
}

# 모든 방법이 공유하는 바닥값. 켜고 끄는 것을 전부 명시해 원본 상태에 안 기댄다.
BASE = {
    'reg_spike_out': True,
    'reg_spike_log_detail': True,       # spike_count 는 reg 와 무관하게 찍히므로 L2 에서도 S30/S1 이 나온다
    'reg_spike_R_per_step': False,      # 26-08-07 이전 회계(R 을 T배 과소보고)와 섞지 않는다
    'reg_spike_accum_loss': False,
    'reg_spike_out_encourage': False,
    'reg_spike_starget': False,
    'reg_spike_adaptive': False,
    'reg_spike_grow': False,
    'reg_spike_sc_feedback': False,
    'reg_spike_lr_linked': False,
    'reg_spike_lr_brake': False,
    'sc_loss_scd': False,
    'reg_spike_wta_rev_w1': False,      # 기본 경로(실효 기울기 w^2). ours_w1 만 True 로 덮는다
}
WTA = {'reg_spike_out_sc': True, 'reg_spike_out_wta_rev': True}
MAXNORM = {'reg_spike_out_sc_maxnorm': True, 'reg_spike_maxnorm_group': "'within_channel'",
           'reg_spike_out_sc_maxnorm_plain': False, 'reg_spike_out_sm_plain': False,
           'reg_spike_out_sc_one': False}
VMEM = {'reg_spike_vmem_gain': 1.0, 'reg_spike_vmem_silent_only': True}
RATIO = lambda rho: {'reg_spike_loss_ratio': True, 'reg_spike_loss_ratio_target': rho,
                     'reg_spike_loss_ratio_start_ep': 0}
FIXED = lambda lmb: {'reg_spike_loss_ratio': False, 'reg_spike_out_const': lmb}


def METHODS(name, knob):
    """방법 이름과 손잡이 값 -> 덮어쓸 플래그 dict."""
    if name == 'base':
        # 규제 없음. knob 은 안 쓴다 (CLI 자리를 맞추려면 '-' 를 넘긴다).
        # reg 블록이 통째로 안 돌지만 list_spike_count 는 proc.py:1040 에서 따로 채워지므로
        # spike_count 와 S30/S1 은 그대로 나온다.
        return {**{k: v for k, v in BASE.items() if k != 'reg_spike_out'},
                'reg_spike_out': False, 'reg_spike_loss_ratio': False}
    if name == 'prop':
        return {**BASE, **WTA, **MAXNORM, **VMEM, 'reg_spike_final_step': True, **RATIO(knob)}
    if name == 'ours_w1':
        # prop 와 **모든 설정이 같고** reg_spike_wta_rev_w1 만 True.
        # wta_rev 의 backward 가 spike 에 sc_rate 를 두 번 곱해 실효 기울기가 w^2 였다
        # (곱셈이 custom_gradient 바깥에 있어서다. flags.py 의 해당 항목 참고).
        # 이 팔은 설계 의도대로 w 한 번일 때 무엇이 달라지는지를 묻는다.
        # forward(R)는 두 경로가 같으므로 loss-ratio 의 rho 눈금도 그대로다.
        return {**METHODS('prop', knob), 'reg_spike_wta_rev_w1': True}
    if name == 'ours_ch':
        # prop 와 전부 같고 max 를 잡는 범위만 다르다: 채널의 공간 합끼리 경쟁한다
        # (neurons.py 의 'channel' 분기). 채널 안 모든 위치가 같은 w 를 받으므로
        # 순수한 채널 선별기다 -- 약한 채널이 통째로 눌린다. within_channel 이 못 보는
        # 채널 전체 활동량을 본다. 구조적 프루닝 각도를 재는 팔이다.
        return {**METHODS('prop', knob), 'reg_spike_maxnorm_group': "'channel'"}
    if name == 'sm':
        return {**BASE, **WTA, 'reg_spike_out_sc_sm': True, 'reg_spike_out_sc_maxnorm': False,
                'reg_spike_final_step': False, **FIXED(knob)}
    if name == 'l2':
        # sc 경로를 끄면 plain L2 가지로 간다 (neurons.py:1238)
        return {**BASE, 'reg_spike_out_sc': False, 'reg_spike_out_norm': True,
                'reg_spike_out_norm_sq': False, 'reg_spike_final_step': False, **FIXED(knob)}
    if name == 'bpsr':
        # BPSR (Yan et al. 2022, Front. Neurosci.) eq.(4) 의 spiking sparsity 항 그대로.
        # (lambda/2) * sum_t sum_i s^2. 스파이크가 이진이라 사실상 lambda/2 x 총 스파이크 수이고
        # 발화 뉴런당 gradient 가 lambda 로 **상수**다. 우리 l2 가지(sqrt(sum s^2))는
        # 뉴런당 lambda/sqrt(N) 라 층마다 최대 30배, 학습 중 2배 달라져서 lambda 재조정으로
        # 맞출 수 없다. 문헌 표준 기준선을 인용하려면 이 팔이 따로 있어야 한다.
        return {**BASE, 'reg_spike_out_sc': False, 'reg_spike_out_norm': False,
                'reg_spike_out_norm_sq': False, 'reg_spike_out_bpsr': True,
                'reg_spike_final_step': False, **FIXED(knob)}
    if name == 'ours_layer':
        # 제안법에서 채널 내 max 를 **층 전체 max** 로 바꾼다 (maxnorm_group='none').
        # within_channel 은 채널마다 최고 뉴런이 sc_rate=0 으로 면제되어 어떤 채널도
        # 통째로 죽지 않는다 (측정: 채널 단위 완전침묵이 plain L2 대비 50개뿐).
        # 'none' 은 약한 채널을 통째로 지울 수 있어 구조적 프루닝에는 유리하지만
        # 정확도를 잃을 수 있다. 그 맞바꿈의 크기를 아직 모른다.
        return {**BASE, **WTA, **MAXNORM, **VMEM,
                'reg_spike_maxnorm_group': "'none'",
                'reg_spike_final_step': True, **RATIO(knob)}
    if name == 'abl_nofinal':
        return {**BASE, **WTA, **MAXNORM, **VMEM, 'reg_spike_final_step': False, **RATIO(knob)}
    if name == 'abl_novmem':
        return {**BASE, **WTA, **MAXNORM, 'reg_spike_vmem_gain': 0.0,
                'reg_spike_vmem_silent_only': False,
                'reg_spike_final_step': True, **RATIO(knob)}
    if name == 'abl_nolr':
        # loss-ratio 제거 — 제안법 그대로에 **고정 lambda**. knob 이 rho 가 아니라 lambda 다.
        # loss-ratio 는 lambda 를 매 에폭 rho*L_task/R 로 다시 푼다. 실측(prop rho=4e-3, n=5)
        # 궤적은 ep1 4.35e-7 -> ep50 4.9e-7 -> ep300 2.6e-7 로 올랐다가 반으로 떨어진다.
        # 시간평균 4.0e-7, 중반 고원 4.83e-7, 후반 2.58e-7.
        # 이 팔은 "그 궤적이 필요한가, 같은 착지점이면 그만인가" 를 묻는다.
        return {**BASE, **WTA, **MAXNORM, **VMEM, 'reg_spike_final_step': True, **FIXED(knob)}
    if name == 'abl_noinv':
        # 1- 반전 제거: sc_rate = sc/max. 침묵 뉴런이 sc_rate=0 이 되어 규제에서 빠진다
        return {**BASE, **WTA, **VMEM, 'reg_spike_out_sc_maxnorm': False,
                'reg_spike_out_sc_maxnorm_plain': True,
                'reg_spike_final_step': True, **RATIO(knob)}
    raise SystemExit(f'모르는 방법: {name}')


# 실험 산출물(체크포인트·CSV·train.log)은 로컬 SSD 가 아니라 HDD 에 쌓는다.
# 루트 디렉토리가 652개까지 늘고 SSD 가 84% 차서 26-09-14 에 옮겼다.
# paper/ 는 논문 후보, archive/ 는 과거 실험 라인.
# 서버마다 저장 위치가 다를 수 있어 환경변수로 덮어쓸 수 있게 둔다.
#   EIP_STORE=/data/kyccj/EIP/paper python run_paper.py ...
STORE = os.environ.get('EIP_STORE', '/media/hdd1/kyccj/EIP/paper')

RE_NAME = re.compile(r"^conf\.exp_set_name\s*=\s*'[^']*'", re.M)
RE_GPU = re.compile(r'^os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=\s*"\d+"', re.M)
RE_SET = re.compile(r'^config\.set\(\)$', re.M)   # \s*$ 로 두면 끝의 개행까지 먹는다
MARK_A, MARK_B = '# >>> run_paper overrides', '# <<< run_paper overrides'


def fmt(v):
    return v if isinstance(v, str) else repr(v)


def override_block(flags_d, seed, tag, epochs=0, fix_seed=False):
    lines = [MARK_A]
    # config_snn_training.py:35 의 `conf.root_model_save=conf.exp_set_name` 을 덮어쓴다.
    # 리터럴 tag 를 쓰는 이유: 같은 파일 L171 이 exp_set_name 에 '_asym' 을 덧붙이므로
    # conf.exp_set_name 을 참조하면 이 블록의 실행 순서에 결과가 달라진다.
    lines.append(f"conf.root_model_save = {STORE + '/' + tag!r}")
    for k in sorted(flags_d):
        lines.append(f'conf.{k} = {fmt(flags_d[k])}')
    # 시드는 기본적으로 **태그에만** 쓰고 RNG 는 안 건드린다 (run_seed=-1).
    #
    # `tf.keras.utils.set_random_seed` 로 전역 시드를 고정했더니 R19-C10 baseline 이
    # 96.65% -> 95.72% 로 0.93%p 떨어지고 런 간 sd 가 0.050 -> 0.429 로 8배가 됐다.
    # train-val 격차 부호가 뒤집힌 것이 원인을 말해 준다:
    #     기존(시드 없음)  train 95.05 / val 96.17  ->  -1.12  (증강이 세서 정상)
    #     시드 고정        train 95.85 / val 95.12  ->  +0.73  (증강이 쉬워짐)
    # 전역 시드를 박으면 tf.data 파이프라인의 난수 연산이 에폭마다 같은 시퀀스를
    # 재생해 증강 다양성이 사라진다. 후반부에 과적합한다.
    #
    # 게다가 시드를 고정해도 GPU 연산이 비결정적이라 재현도 안 된다 (같은 시드 2회:
    # val_acc 37.35% vs 41.31%). 얻는 것 없이 잃기만 하므로 쓰지 않는다.
    lines.append(f'conf.run_seed = {seed if fix_seed else -1}')
    if epochs:
        # 연기 시험용. 설정이 죽지 않는지 2에폭으로 먼저 본다 (계획 §7-1)
        lines.append(f'conf.train_epoch = {epochs}')
    lines.append(MARK_B)
    return '\n'.join(lines)


def src_text(combo):
    src = SOURCES[combo][0]
    return open(os.path.join(PROJECT_ROOT, src, 'config_sweep.py')).read()


def active_name(txt):
    """주석이 아니라 실제로 마지막에 대입되는 exp_set_name.
    (08-21 에 주석 줄을 치환해 원본 체크포인트를 날린 사고 재발 방지)"""
    i = txt.find('conf.root_model_save=conf.exp_set_name')
    if i < 0:
        raise RuntimeError('root_model_save 줄을 못 찾음')
    ms = RE_NAME.findall(txt[:i])
    if not ms:
        raise RuntimeError('활성 exp_set_name 대입이 없음')
    return ms[-1].split("'")[1]


def make_config(combo, method, knob, seed, gpu, tag, epochs=0, fix_seed=False):
    c = src_text(combo)
    c, n = RE_GPU.subn(f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu}"', c)
    if n != 1:
        raise RuntimeError(f'{tag}: GPU 줄이 {n}개')
    c, n = RE_NAME.subn(f"conf.exp_set_name='{tag}'", c)
    if n < 1:
        raise RuntimeError(f'{tag}: 활성 exp_set_name 없음')
    blk = override_block(METHODS(method, knob), seed, tag, epochs, fix_seed)
    c, n = RE_SET.subn(blk + '\nconfig.set()', c, count=1)
    if n != 1:
        raise RuntimeError(f'{tag}: config.set() 을 못 찾음 ({n}개)')
    return c


def verify(path, combo, method, knob, seed, gpu, tag, epochs=0, fix_seed=False):
    a = open(path).read()
    b = src_text(combo)
    # 덮어쓰기 블록 / GPU / 이름 을 지운 나머지가 원본과 같아야 한다
    stripped = re.sub(re.escape(MARK_A) + r'.*?' + re.escape(MARK_B) + r'\n', '', a, flags=re.S)
    norm = lambda s: RE_NAME.sub('', RE_GPU.sub('', s))
    if norm(stripped) != norm(b):
        raise RuntimeError(f'{tag}: 덮어쓰기 블록/GPU/이름 외의 차이가 있음')
    if f'"CUDA_VISIBLE_DEVICES"]="{gpu}"' not in a:
        raise RuntimeError(f'{tag}: GPU 지정 안 됨')
    if active_name(a) != tag:
        raise RuntimeError(f'{tag}: 유효 exp_set_name 이 {active_name(a)!r} — 원본 덮어쓸 위험')
    if active_name(b) == tag:
        raise RuntimeError(f'{tag}: 원본과 저장 경로가 같음')
    # 블록이 config.set() 앞에 있어야 효력이 있다
    if a.index(MARK_B) > a.rindex('config.set()'):
        raise RuntimeError(f'{tag}: 덮어쓰기 블록이 config.set() 뒤에 있음')
    # 요청한 플래그가 그대로 들어갔는지
    want = METHODS(method, knob)
    for k, v in want.items():
        if not re.search(rf'^conf\.{re.escape(k)} = {re.escape(fmt(v))}$', a, re.M):
            raise RuntimeError(f'{tag}: {k}={fmt(v)} 가 블록에 없음')
    want_seed = seed if fix_seed else -1
    if not re.search(rf'^conf\.run_seed = {want_seed}$', a, re.M):
        raise RuntimeError(f'{tag}: run_seed={want_seed} 설정 안 됨')
    # 모델·데이터셋이 요청한 조합인지
    _, model, dataset = SOURCES[combo]
    if not re.search(rf"^conf\.model\s*=\s*'{model}'", a, re.M):
        raise RuntimeError(f'{tag}: model 이 {model} 이 아님')
    if not re.search(rf"^conf\.dataset\s*=\s*'{dataset}'", a, re.M):
        raise RuntimeError(f'{tag}: dataset 이 {dataset} 이 아님')
    # 배타 조합
    if want.get('reg_spike_final_step') and want.get('reg_spike_accum_loss'):
        raise RuntimeError(f'{tag}: final_step 과 accum 을 같이 켤 수 없음')


def preflight(jobs, sweep_dir, epochs=0, fix_seed=False):
    bad = []
    for combo, method, knob, seed, tag in jobs:
        if combo not in SOURCES:
            bad.append(f'모르는 조합 {combo}')
            continue
        if not os.path.exists(os.path.join(PROJECT_ROOT, SOURCES[combo][0], 'config_sweep.py')):
            bad.append(f'{SOURCES[combo][0]} (원본 config 없음)')
        tgt = os.path.join(PROJECT_ROOT, tag)
        if os.path.exists(tgt) and any(f.endswith('.weights.h5')
                                       for _, _, fs in os.walk(tgt) for f in fs):
            bad.append(f'{tgt} (체크포인트가 이미 있음)')
        if os.path.exists(os.path.join(sweep_dir, tag, 'train.log')):
            bad.append(f'{sweep_dir}/{tag}/train.log (이미 돈 흔적)')
    fl = open(os.path.join(PROJECT_ROOT, 'flags.py')).read()
    nr = open(os.path.join(PROJECT_ROOT, 'lib_snn', 'neurons.py')).read()
    for tok in ('reg_spike_final_step', 'reg_spike_vmem_silent_only', 'run_seed'):
        if tok not in fl:
            bad.append(f'flags.py 에 {tok} 없음')
    for tok in ('reg_spike_final_step', 'reg_spike_vmem_silent_only'):
        if tok not in nr:
            bad.append(f'lib_snn/neurons.py 에 {tok} 반영 안 됨')
    if bad:
        raise SystemExit('중단:\n  ' + '\n  '.join(bad))
    os.makedirs(sweep_dir, exist_ok=True)
    for combo, method, knob, seed, tag in jobs:
        tmp = os.path.join(sweep_dir, f'.pre_{tag}.py')
        open(tmp, 'w').write(make_config(combo, method, knob, seed, 0, tag, epochs, fix_seed))
        try:
            verify(tmp, combo, method, knob, seed, 0, tag, epochs, fix_seed)
        finally:
            os.remove(tmp)
    print(f'사전 점검 통과: {len(jobs)}개 config 생성·검증 완료', flush=True)


def run_one(gpu, combo, method, knob, seed, tag, sweep_dir, epochs=0, fix_seed=False):
    d = os.path.join(sweep_dir, tag)
    os.makedirs(d, exist_ok=True)
    cfg = os.path.join(d, 'config_sweep.py')
    open(cfg, 'w').write(make_config(combo, method, knob, seed, gpu, tag, epochs, fix_seed))
    verify(cfg, combo, method, knob, seed, gpu, tag, epochs, fix_seed)
    m = open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')).read()
    open(os.path.join(d, 'main_sweep.py'), 'w').write(
        m.replace('from config_snn_training import config', 'from config_sweep import config'))
    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {tag}  "
          f"({combo} {method} knob={knob} seed={seed})", flush=True)
    # train.log 도 HDD 에 쌓는다 (런당 ~25MB, 누적 14GB 였다). 로컬에는 심볼릭 링크만
    # 남겨 collect_paper.py 등 _paper/<tag>/train.log 를 읽는 도구가 그대로 동작하게 한다.
    store_dir = os.path.join(STORE, tag)
    os.makedirs(store_dir, exist_ok=True)
    # 어느 서버에서 돌았는지 런 옆에 남긴다. canus(…138) 와 sejong(…23) 은 TF·numpy·
    # CUDA 버전이 달라서, 결과를 합칠 때 서버를 공변량으로 넣어야 한다.
    with open(os.path.join(store_dir, 'host.txt'), 'w') as hf:
        import socket as _s
        hf.write(f'{_s.gethostname()}\t{_s.gethostbyname(_s.gethostname())}\t'
                 f'gpu{gpu}\t{time.strftime("%Y-%m-%d %H:%M")}\n')
    log_path = os.path.join(store_dir, 'train.log')
    link = os.path.join(d, 'train.log')
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(log_path, link)
    with open(log_path, 'w') as lf:
        rc = subprocess.Popen([PYTHON, os.path.join(d, 'main_sweep.py')], cwd=PROJECT_ROOT,
                              stdout=lf, stderr=subprocess.STDOUT, env=env).wait()
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] DONE  {tag} rc={rc}", flush=True)
    return rc


def free_gpus(allowed, exclude, max_mib=200):
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
        if idx not in allowed or idx in exclude:
            continue
        if int(mem) <= max_mib:
            res.append(idx)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('combo', nargs='?', help=' / '.join(SOURCES))
    ap.add_argument('method', nargs='?', help='base prop ours_w1 ours_ch l2 sm bpsr ours_layer abl_nofinal abl_novmem abl_noinv abl_nolr')
    ap.add_argument('knob', nargs='?', help="prop/ours_w1/ours_ch/abl 은 rho, sm/l2 는 lambda, base 는 '-'")
    ap.add_argument('--seeds', default='1,2,3,4,5', help='복제 시드 (쉼표). 서로 달라야 한다')
    ap.add_argument('--gpus', default='0,1,2,3,4,5', help='쓸 GPU (쉼표). 6·7 은 기본 제외')
    ap.add_argument('--dir', default='_paper', help='스윕 디렉토리')
    ap.add_argument('--list', action='store_true', help='조합·방법 목록만 출력')
    ap.add_argument('--dry', action='store_true', help='config 만 만들어 검증하고 끝')
    ap.add_argument('--epochs', type=int, default=0, help='연기 시험용 에폭 수 (0 = 원본 그대로)')
    ap.add_argument('--fix-seed', action='store_true', default=False,
                    help='RNG 를 실제로 고정한다. 기본은 끔 — 고정하면 tf.data 증강이 '
                         '에폭마다 같은 시퀀스를 재생해 R19-C10 이 96.65%%->95.72%% 로 떨어졌다')
    a = ap.parse_args()

    if a.list or not (a.combo and a.method and a.knob):
        print('조합:')
        for k, (s, m, d) in SOURCES.items():
            print(f'  {k:9s} {m:9s} {d:9s}  <- {s}')
        print('\n방법: base prop ours_w1 ours_ch l2 sm bpsr ours_layer abl_nofinal abl_novmem abl_noinv abl_nolr')
        print('\n예) python run_paper.py r19c10 prop 1e-3 --seeds 1,2,3,4,5 --gpus 0,2')
        return

    seeds = [int(s) for s in a.seeds.split(',') if s.strip()]
    if len(set(seeds)) != len(seeds):
        raise SystemExit('시드가 중복이다 — 같은 시드면 같은 런이 나온다')
    allowed = {int(g) for g in a.gpus.split(',') if g.strip()}
    sweep_dir = os.path.join(PROJECT_ROOT, a.dir)

    knob_tag = str(a.knob).replace('.', 'p')
    jobs = [(a.combo, a.method, a.knob, s, f'{a.combo}-{a.method}-{knob_tag}-s{s}') for s in seeds]

    preflight(jobs, sweep_dir, a.epochs, a.fix_seed)
    if a.dry:
        print('--dry: 검증만 하고 종료')
        return

    pending, busy, threads = list(jobs), set(), []
    failed = []
    print(f'--- {a.combo} / {a.method} / knob={a.knob} : {len(pending)} runs '
          f'(GPU {sorted(allowed)}) ---', flush=True)
    while pending:
        for g in free_gpus(allowed, busy):
            if not pending:
                break
            combo, method, knob, seed, tag = pending.pop(0)
            busy.add(g)

            def worker(g=g, combo=combo, method=method, knob=knob, seed=seed, tag=tag):
                try:
                    rc = run_one(g, combo, method, knob, seed, tag, sweep_dir,
                                 a.epochs, a.fix_seed)
                    if rc:
                        failed.append((tag, rc))
                except Exception as e:              # 한 job 이 죽어도 나머지는 계속
                    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {g}] FAIL {tag}: {e}",
                          flush=True)
                    failed.append((tag, 'exception'))
                finally:
                    busy.discard(g)

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)
            time.sleep(90)          # 같은 GPU 를 두 job 이 잡는 것 방지
        if pending:
            time.sleep(60)
    for t in threads:
        t.join()
    if failed:
        # 학습이 죽었는데 0 을 반환하면 큐 실행기가 'done' 으로 표시한다.
        # sejong 에서 Keras 2.11 이 체크포인트 디렉토리를 안 만들어 6런이 9분 만에
        # done 으로 찍혔다 (26-09-14). 실패는 반드시 종료코드로 올린다.
        for tag, rc in failed:
            print(f'실패: {tag} rc={rc}', flush=True)
        sys.exit(1)
    print('--- 전부 완료 ---', flush=True)


if __name__ == '__main__':
    main()
