"""WTA가 실제로 일어나는가 — 학습된 모델에서 뉴런별 발화를 직접 기록해 검사한다.

성능 차이는 노이즈 안이므로(0.05%p vs 노이즈 0.2%p) 기전으로 판정한다.
핵심 질문: 각 방법이 **자기 경쟁 단위 안에서** 발화를 몰아주는가?

측정 4가지 (층별, conv1~conv5_2):
  A. Gini 분해 — 층 전체 불균등을 '채널 사이'와 '채널 안(공간)'으로 쪼갠다.
       채널 내 방식이 WTA면 -> 채널 안 불균등이 커져야 한다
       채널 간 방식이 WTA면 -> 채널 사이 불균등이 커져야 한다
       각자 자기 축에서만 커지면 경쟁이 실재. 아니면 가중치는 축과 무관하게 작동한 것.
  B. 승자 선택성 — 이미지마다 상위 1% 뉴런이 바뀌는가.
       진짜 경쟁이면 입력에 따라 승자가 바뀐다. 항상 같은 뉴런이 이기면
       그건 경쟁이 아니라 그냥 '산 뉴런 / 죽은 뉴런' 구조다.
  C. 단위 내 음의 상관 — 같은 경쟁 단위 안의 뉴런 쌍이 이미지에 걸쳐 서로 반대로 움직이는가.
       측면 억제의 직접 증거. 단위 밖 쌍을 대조군으로 같이 잰다.
  D. 죽은 뉴런 비율 (참고).

사용법:  python analyze_wta.py <출력디렉터리> [배치수]
  예)    python analyze_wta.py grp-ch_a7 5
설정은 해당 run의 config_sweep.py를 그대로 쓰되 train=False, load_model=True로 바꾼다.
"""
import os, sys, glob, json
import numpy as np

RUN = sys.argv[1]
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 5
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join('/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/'
                   '71038697-b6a5-495d-b315-c2838d5a976a/scratchpad/wta2', RUN + '.json')
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def gini(x):
    """x: 1-D 음이 아닌 배열"""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    s = x.sum()
    if n == 0 or s <= 0:
        return 0.0
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * s))


def analyze(sc):
    """sc: [이미지, H, W, C] 뉴런별 스파이크 수"""
    B, H, W, C = sc.shape
    mean_n = sc.mean(axis=0)                       # [H,W,C] 뉴런별 평균 발화
    flat = mean_n.reshape(-1)
    res = {}
    res['gini_total'] = gini(flat)
    # A. 분해
    ch_tot = mean_n.sum(axis=(0, 1))               # [C] 채널별 총 발화
    res['gini_between_ch'] = gini(ch_tot)
    wi = [gini(mean_n[:, :, c].reshape(-1)) for c in range(C) if mean_n[:, :, c].sum() > 0]
    res['gini_within_ch'] = float(np.mean(wi)) if wi else 0.0
    res['n_live_ch'] = len(wi)
    # 공간 축 대조: 같은 위치를 채널 넘어 묶었을 때
    sp_tot = mean_n.sum(axis=2)                    # [H,W]
    res['gini_between_pos'] = gini(sp_tot.reshape(-1))
    # B. 승자 선택성 — 이미지별 상위 1% 뉴런 집합의 겹침
    k = max(1, int(0.01 * H * W * C))
    f2 = sc.reshape(B, -1)
    tops = [set(np.argpartition(-f2[i], k)[:k].tolist()) for i in range(min(B, 40))]
    ov = [len(a & b) / k for i, a in enumerate(tops) for b in tops[i + 1:]]
    res['winner_overlap'] = float(np.mean(ov)) if ov else float('nan')
    # C. 단위 내 vs 단위 밖 뉴런 쌍 상관
    rng = np.random.default_rng(0)
    live = np.where(f2.std(axis=0) > 1e-6)[0]
    def pair_corr(pairs):
        v = []
        for a, b in pairs:
            x, y = f2[:, a], f2[:, b]
            sx, sy = x.std(), y.std()
            if sx > 1e-6 and sy > 1e-6:
                v.append(float(np.corrcoef(x, y)[0, 1]))
        return float(np.mean(v)) if v else float('nan')
    idx = lambda i: np.unravel_index(i, (H, W, C))
    same, diff = [], []
    tries = 0
    while (len(same) < 400 or len(diff) < 400) and tries < 40000:
        tries += 1
        a, b = rng.choice(live, 2, replace=False)
        ca, cb = idx(a)[2], idx(b)[2]
        if ca == cb and len(same) < 400: same.append((a, b))
        elif ca != cb and len(diff) < 400: diff.append((a, b))
    res['corr_same_ch'] = pair_corr(same)      # 채널 내 경쟁 단위 안
    res['corr_diff_ch'] = pair_corr(diff)      # 대조군
    # --- 추가 지표 (Gini 의 맹점 보완) ---
    # E. 동시 활성 개수: 한 채널 안에서 이미지당 몇 개 뉴런이 켜지는가.
    #    진짜 WTA 면 1 에 가까워야 한다. Gini 는 '얼마나 불균등한가'만 재고
    #    '몇 명이 이기는가'는 못 잰다.
    on = (sc > 0)                                   # [B,H,W,C]
    per = on.reshape(B, -1, C).sum(axis=1)          # [B,C] 채널별 켜진 개수
    liveq = per > 0
    res['coactive_per_ch'] = float(per[liveq].mean()) if liveq.any() else 0.0
    res['coactive_frac'] = float(res['coactive_per_ch'] / (H * W))
    # F. 조건부 억제: 채널 안 최대 발화가 큰 이미지에서 나머지가 실제로 약해지는가.
    #    lateral inhibition 이면 음의 상관이어야 한다. 뉴런쌍 상관보다 방향이 분명하다.
    f3 = sc.reshape(B, -1, C)
    cs = []
    for c in range(C):
        x = f3[:, :, c]
        if x.sum() == 0: continue
        mx = x.max(axis=1)                          # [B] 그 이미지의 승자 세기
        tot = x.sum(axis=1)
        rest = (tot - mx) / max(x.shape[1] - 1, 1)  # [B] 나머지 평균
        if mx.std() > 1e-6 and rest.std() > 1e-6:
            cs.append(float(np.corrcoef(mx, rest)[0, 1]))
    res['suppression_corr'] = float(np.mean(cs)) if cs else float('nan')
    # G. 참여 비율 (Σx)^2 / Σx^2 — 실질적으로 일하는 뉴런 수. 채널당 비율로 낸다.
    prs = []
    for c in range(C):
        v = mean_n[:, :, c].reshape(-1)
        if v.sum() <= 0: continue
        prs.append(float((v.sum() ** 2) / (np.square(v).sum() + 1e-12) / v.size))
    res['participation_ratio'] = float(np.mean(prs)) if prs else 0.0
    # D
    res['dead_ratio'] = float((mean_n.reshape(-1) == 0).mean())
    res['shape'] = [int(H), int(W), int(C)]
    return res


def main():
    cfg_dirs = glob.glob(os.path.join(PROJECT_ROOT, '_*', '*', 'config_sweep.py'))
    src = None
    import re as _re
    # 활성(주석 아닌) 대입 중 root_model_save 대입 직전의 마지막 값이 실제 이름이다.
    # (주석 줄을 잡으면 엉뚱한 이름이 나온다 — 26-08-21 사고와 같은 원인)
    RE_N = _re.compile(r"^conf\.exp_set_name\s*=\s*'([^']*)'", _re.M)
    def eff_name(txt):
        i = txt.find('conf.root_model_save=conf.exp_set_name')
        ms = RE_N.findall(txt[:i] if i > 0 else txt)
        return ms[-1] if ms else None
    for c in cfg_dirs:
        if eff_name(open(c, errors='ignore').read()) == RUN:
            src = c; break
    if src is None:
        raise SystemExit(f'{RUN}: config_sweep.py를 못 찾음')
    work = os.path.join('/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/'
                        '71038697-b6a5-495d-b315-c2838d5a976a/scratchpad/wta', RUN + '_cfg')
    os.makedirs(work, exist_ok=True)
    txt = open(src).read()
    # config.train / config.load_model 은 conf.mode 에서 파생된다 (config.py)
    # config.train / config.load_model 은 conf.mode 에서 파생된다 (config.py)
    # 가중치는 root_model_save(=exp_set_name) 밑에 저장되므로 load root 도 거기로 맞춘다
    txt = txt.replace('\nconfig.set()',
                      "\nconf.mode='inference'\nconf.verbose=False"
                      "\nconf.root_model_load=conf.root_model_save\nconfig.set()")
    txt = txt.replace('os.environ["CUDA_VISIBLE_DEVICES"]="0"', 'os.environ["CUDA_VISIBLE_DEVICES"]="-1"')
    for g in '012345':
        txt = txt.replace(f'os.environ["CUDA_VISIBLE_DEVICES"]="{g}"',
                          'os.environ["CUDA_VISIBLE_DEVICES"]="-1"')
    open(os.path.join(work, 'config_sweep.py'), 'w').write(txt)
    sys.path.insert(0, work); sys.path.insert(1, PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)

    import tensorflow as tf
    from config_sweep import config
    import lib_snn, datasets, callbacks   # noqa
    lib_snn.utils.set_gpu()
    _, _, test_ds, _, _, test_num, num_class, spe = datasets.datasets.load()
    model = lib_snn.model_builder.model_builder(num_class, spe, test_ds)
    # config 가 이미 최신 체크포인트를 해석해 뒀다
    wpath = config.load_weight
    if not os.path.exists(wpath):
        raise SystemExit(f'{RUN}: 가중치 없음 ({wpath})')
    model.load_weights(wpath)
    print(f'{RUN}: {os.path.basename(wpath)} 로드', flush=True)

    if not model.layers_w_neuron:
        model.init_snn()          # 보통 학습 시작 시 호출됨. inference 경로에선 직접 부른다
    print('layers_w_neuron:', len(model.layers_w_neuron), flush=True)
    def neuron_of(l):
        return l if isinstance(l, lib_snn.neurons.Neuron) else getattr(l, 'act', None)
    layers = []
    for l in model.layers_w_neuron:
        a = neuron_of(l)
        if a is None: continue
        if getattr(a, 'loc', None) == 'HID' and len(getattr(a, 'dim', [])) == 4:
            layers.append((l.name, a))
    print('선택된 층:', [n for n, _ in layers], flush=True)
    store = {n: [] for n, _ in layers}
    it = iter(test_ds)
    for bi in range(NB):
        x, y = next(it)
        model(x, training=False)
        for n, a in layers:
            store[n].append(np.array(a.spike_count.numpy(), dtype=np.float32))
        print(f'  batch {bi+1}/{NB}', flush=True)
    out = {}
    for n, chunks in store.items():
        sc = np.concatenate(chunks, axis=0)
        out[n] = analyze(sc)
        print(f'  {n}: gini_total {out[n]["gini_total"]:.3f} '
              f'동시활성 {out[n]["coactive_per_ch"]:.1f} 억제상관 {out[n]["suppression_corr"]:+.3f} '
              f'참여비율 {out[n]["participation_ratio"]:.4f}',
              flush=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    print('saved', OUT)


if __name__ == '__main__':
    main()
