"""inter_results_261009.json -> inter_results_261009.md (표는 전부 json 에서 생성, 해설만 손으로)."""
import json, math
from pathlib import Path

P = Path(__file__).with_name('inter_results_261009.json')
d = json.load(open(P))
L = []
w = L.append


def f(v, nd=2):
    return '' if v is None else f'{v:.{nd}f}'


def sp(v):
    return '' if v is None else f'{v:,.0f}'


def pm(m, k, nd=2):
    if m[k] is None:
        return ''
    return f(m[k], nd) + (f' ±{m[k + "_sd"]:.{nd}f}' if m.get(k + '_sd') is not None else '')


HDR = '| ρ | seed | 서버 | train loss | train acc | val loss | val acc | spikes | spikes vs base | Δacc vs base | S30/S1 |'
SEP = '|---|---|---|---|---|---|---|---|---|---|---|'


def run_line(r, knob=None, note=''):
    st = '' if r['status'] == 'done' and not r.get('exclude_reason') else f" ({r.get('exclude_reason') or r['status']})"
    return (f"| {knob or r['knob']} | s{r['seed']}{st} | {r['server']} | {f(r['train_loss'],4)} | {f(r['train_acc'])} | "
            f"{f(r['val_loss'],4)} | {f(r['val_acc'])} | {sp(r['spikes'])} |  |  | {f(r['s30'],3)} |{note}")


def mean_line(label, m, pct=None, da=None, se=None):
    return (f"| **{label}** | **n={m['n']}** |  | {pm(m,'train_loss',4)} | {pm(m,'train_acc')} | {pm(m,'val_loss',4)} | "
            f"**{pm(m,'val_acc')}** | **{sp(m['spikes'])}** | "
            f"{'' if pct is None else f'{pct:+.1f}%'} | {'' if da is None else f'{da:+.2f} ±{se:.2f}'} | {pm(m,'s30',3)} |")


def curve(c, title, base_label):
    b = c['baseline']
    w(HDR); w(SEP)
    for r in b['runs']:
        w(run_line(r, knob='baseline'))
    w(mean_line(f'{base_label} 평균', b['mean']))
    for g in c['groups']:
        if not g['mean']:
            continue
        for r in g['runs']:
            if r['status'] in ('done',) and not r.get('exclude_reason'):
                w(run_line(r))
        m = g['mean']
        w(mean_line(f"{g['knob']} 평균", m, g['spikes_pct_vs_base'], g['dacc_vs_base'], g['dacc_se']))


# ================================================================ 머리
C10, C100 = d['C10_baseline_inter'], d['C100_baseline_inter']
b10, b100 = C10['baseline']['mean'], C100['baseline']['mean']
g10 = {g['knob']: g for g in C10['groups']}
g100 = {g['knob']: g for g in C100['groups']}
bp = d['bpsr_C10']
dr = bp['direct']
qk = d['qkformer_C10']
a34, a24, a13 = bp['ancova_300_420'], bp['ancova_200_420'], bp['ancova_180_300']
vm = [b for b in d['ablation_C10'] if b['key'] == 'ch_novmem'][0]['pooled']

w('# ours_inter_ch 실험 결과 정리 (26-10-09)')
w('')
w(f"원장 집계 {d['generated_from']['inventory']} 기준 (TF R19 C10/C100 + VGGSNN). QKFormer 는 런 폴더를 13:25 에 직접 읽음, "
  "SDTv3 는 sdt.json (12:56). 생성: `analysis/inter_results_261009.py` → `.json`, 표는 `inter_results_261009_md.py` 가 json 에서 만든다.")
w('')
w('## 요약')
w('')
g3 = g10['3e-3']
w(f"1. **대표점.** C10 ρ 3e-3: val acc {g3['mean']['val_acc']:.2f} ±{g3['mean']['val_acc_sd']:.2f} @ {sp(g3['mean']['spikes'])} "
  f"(n=4), baseline {b10['val_acc']:.2f} ±{b10['val_acc_sd']:.2f} @ {sp(b10['spikes'])} (n=6) 대비 스파이크 {g3['spikes_pct_vs_base']:+.1f}%, "
  f"Δacc {g3['dacc_vs_base']:+.2f} ±{g3['dacc_se']:.2f}. C100 대표 ρ 2e-3 은 n=1 ({g100['2e-3']['mean']['val_acc']:.2f} @ "
  f"{sp(g100['2e-3']['mean']['spikes'])}, {g100['2e-3']['spikes_pct_vs_base']:+.1f}%, baseline 80.65 n=6) — 시드 2~4 진행 중 (38~39/310 에폭), **아직 판단 불가**.")
w(f"2. **BPSR 대비 (C10).** 정확도가 거의 같은 두 점끼리 (inter 3e-3 n=4 {dr['inter_3e-3']['val_acc']:.2f} vs BPSR 3e-11 n=3 "
  f"{dr['bpsr_3e-11']['val_acc']:.2f}) 스파이크 {100*(dr['spike_ratio']-1):+.0f}% ({sp(dr['inter_3e-3']['spikes'])} vs {sp(dr['bpsr_3e-11']['spikes'])}). "
  f"**단 불확실성이 크다:** 정확도 차 {dr['dacc']:+.2f} ±{dr['dacc_se']:.2f} 이고 BPSR 곡선이 평평해서 이 정도 정확도 오차가 스파이크 수십 % 에 해당한다. "
  f"BPSR 곡선 보간으로 같은 정확도를 맞추면 {100*(dr['spike_ratio_interp']-1):+.0f}%, 부트스트랩 95% 구간 "
  f"[{100*(dr['spike_ratio_interp_boot95_finite'][0]-1):+.0f}%, {100*(dr['spike_ratio_interp_boot95_finite'][1]-1):+.0f}%] "
  f"(그리고 {100*dr['interp_out_of_range_frac']:.0f}% 의 재표본에서 inter 정확도가 BPSR 최고점보다 높아 비교 자체가 안 됨). "
  f"같은 스파이크에서의 정확도 차 (ANCOVA, 300~420K) {a34['c']:+.3f} ±{a34['c_se']:.3f} (p={a34['c_p']:.2f}, inter n={a34['n_inter']}, BPSR n={a34['n_bpsr']}). "
  "→ **\"−17%\" 는 점추정으로만 쓸 수 있고, 유의한 우위로는 쓸 수 없다.**")
lo = [g10[k] for k in ('6e-3', '7e-3', '8e-3', '1.2e-2')]
w(f"3. **저스파이크 구간.** −40% 부근부터 평균이 baseline 아래로 내려가고 −47% 이하에서는 2σ 를 넘는다: 4.5e-3 (−40%) {g10['4.5e-3']['dacc_vs_base']:+.2f} ±{g10['4.5e-3']['dacc_se']:.2f} (n=3), "
  f"5.5e-3 (−47%) {g10['5.5e-3']['dacc_vs_base']:+.2f} ±{g10['5.5e-3']['dacc_se']:.2f} (n=3), 6e-3 (−51%) {g10['6e-3']['dacc_vs_base']:+.2f} ±{g10['6e-3']['dacc_se']:.2f} (n=3), "
  f"1.2e-2 (−70%) {g10['1.2e-2']['dacc_vs_base']:+.2f} (n=1). 같은 스파이크에서 BPSR 과는 구분 안 됨 (180~300K ANCOVA {a13['c']:+.3f} ±{a13['c_se']:.3f}, p={a13['c_p']:.2f}, n={a13['n_inter']}/{a13['n_bpsr']}).")
w(f"4. **어블레이션은 전부 inter 곡선의 시드 산포 안이다.** 같은 스파이크에서 Δacc: `- vmem` 6점 평균 {vm['delta']:+.3f} ±{vm['se']:.3f} (p={vm['p']:.2f}), "
  "`- final_step` +0.06, `- loss-ratio` +0.01, `- 1- inversion` +0.05 (각 n=1, 개별 SE ±0.10). "
  "n=1 이라 0.2%p 보다 작은 효과는 볼 수 없다 — \"요소가 필요 없다\" 가 아니라 \"정확도–스파이크 곡선 위치로는 차이가 안 보인다\" 까지다.")
w(f"5. **다른 모델.** QKFormer C10 ρ 7e-3: best {qk['inter_7e-3']['best_top1']:.2f} / fp32 재측정 {qk['inter_7e-3']['measure']['acc_remeasure']:.2f} @ "
  f"{sp(qk['inter_7e-3']['measure']['spikes'])} (n=1) vs baseline best {qk['baseline_best_mean']:.2f} (n=7) / 재측정 {qk['baseline_remeasure_mean']:.2f} (n=2) @ "
  f"{sp(qk['baseline_spikes_mean'])} (n=2) → 스파이크 {100*(qk['inter_7e-3']['measure']['spikes']/qk['baseline_spikes_mean']-1):+.0f}%. "
  f"ρ 1.2e-2 는 13:24 완주 (best {qk['inter_1.2e-2']['best_top1']:.2f}, 학습 중 평가 스파이크 {sp(qk['inter_1.2e-2']['best_eval_spikes'])} = {100*(qk['inter_1.2e-2']['best_eval_spikes']/qk['baseline_spikes_mean']-1):+.0f}% 잠정, 재측정 전). 둘 다 n=1 이고 정확도 기준(best/마지막/재측정)에 따라 부호가 바뀌어 판단 불가. "
  "VGGSNN DVS 는 68/200 에폭 진행 중. SDTv3 ImageNet 은 공식 체크포인트 파인튜닝 ρ 보정 단계로, ρ 3e-2 까지 스파이크 변화 −0.4% 이하 — 규제가 아직 먹지 않는다.")
w('')
w('표 규약: 열 순서 train loss → train acc → val loss → val acc → spikes. 반복 런은 개별값 뒤 **평균 ±SD (n)** 행. '
  '"spikes vs base" 와 "Δacc vs base" 는 평균 행에만. Δacc 의 ± 는 차이의 표준오차 (√(SD²/n + SD_base²/n_base)); n=1 점은 '
  f"합동 시드 SD ({C10['pooled_seed_sd']:.3f}, C10 inter+baseline 자유도 {C10['pooled_seed_sd_dof']}; C100 {C100['pooled_seed_sd']:.3f}, 자유도 {C100['pooled_seed_sd_dof']}) 로 대신한다. "
  'S30/S1 = 30에폭 스파이크 / 1에폭 스파이크 (0.19 미만은 배제). 스파이크는 원장 값 (샘플당, 원값).')
w('')

# ================================================================ 1. C10
w('## 1. R19 CIFAR-10 — ours_inter_ch ρ 곡선')
w('')
curve(C10, 'C10', 'baseline')
w('')
w('- 완주·비배제 런만 표에 넣었다. 배제·미완주 행은 맨 아래 7절.')
w('- 무손실 구간: ρ ≤ 3e-3 (−30%) 은 Δacc 가 SE 안. 4e-3 (−38%, n=3) 부터 평균이 baseline 아래로 내려가고 '
  '5.5e-3 (−47%, 2.6σ)·6e-3 (−51%, 4.3σ) 은 2σ 를 넘는다. 7e-3 은 −0.16 ±0.11 (1.5σ).')
w('- n=1 점: 5e-3, 8e-3, 1.2e-2. 8e-3 시드 2·3 은 hold.')
w('')

# ================================================================ 2. C100
w('## 2. R19 CIFAR-100 — ours_inter_ch')
w('')
curve(C100, 'C100', 'baseline')
w('')
w('**진행 중 (평균에 넣지 않음, 현재 에폭 값):**')
w('')
w('| ρ | seed | 서버 | 에폭 | train loss | train acc | val loss | val acc | spikes | S30/S1 |')
w('|---|---|---|---|---|---|---|---|---|---|')
for r in d['running']:
    if r['ds'] == 'C100':
        w(f"| {r['knob']} | s{r['seed']} | {r['server']} | {r['epochs']}/310 | {f(r['train_loss'],4)} | {f(r['train_acc'])} | {f(r['val_loss'],4)} | {f(r['val_acc'])} | {sp(r['spikes'])} | {f(r['s30'],3)} |")
w('')
w('- 대표 ρ 2e-3 (−23%) 은 지금 n=1 이라 **판단 불가**. 1e-3 (−13%, n=3) 은 baseline 과 같고, 3e-3 (−31%, n=3) 은 '
  f"{g100['3e-3']['dacc_vs_base']:+.2f} ±{g100['3e-3']['dacc_se']:.2f} (2.4σ) 로 낮다. C100 은 시드 SD 가 C10 의 약 2배 ({C100['pooled_seed_sd']:.2f}%p) 다.")
w('- 1.2e-3·1.5e-3·2.5e-3 은 n=1.')
w('')

# ================================================================ 3. ablation
f200 = d['inter_fit_C10_200_420']
f280 = d['inter_fit_C10_280_420']


def arm_table(blocks):
    w('| 팔 | ρ (λ) | seed | 서버 | train loss | train acc | val loss | val acc | spikes | S30/S1 | inter 곡선 예측 | **Δacc (같은 스파이크)** | z |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for b in blocks:
        for r in b['runs']:
            v = r.get('vs_inter')
            if r['status'] == 'hold':
                continue
            if v is None:
                w(f"| {b['name'].split(' (')[0]} | {r['knob']} | s{r['seed']} ({r.get('exclude_reason') or r['status']}) | {r['server']} | "
                  f"{f(r['train_loss'],4)} | {f(r['train_acc'])} | {f(r['val_loss'],4)} | {f(r['val_acc'])} | {sp(r['spikes'])} | {f(r['s30'],3)} | — | 배제 | |")
                continue
            flag = '' if v['in_range'] else ' (적합 범위 밖)'
            w(f"| {b['name'].split(' (')[0]} | {r['knob']} | s{r['seed']} | {r['server']} | {f(r['train_loss'],4)} | {f(r['train_acc'])} | "
              f"{f(r['val_loss'],4)} | {f(r['val_acc'])} | {sp(r['spikes'])} | {f(r['s30'],3)} | {v['pred']:.2f} | "
              f"{v['delta']:+.2f} ±{v['se']:.2f}{flag} | {v['z']:+.1f} |")
        for k, g in b['group_means'].items():
            v = g['vs_inter']
            w(f"| **{b['name'].split(' (')[0]} {k} 평균** | {k} | **n={g['n']}** | | {pm(g,'train_loss',4)} | {pm(g,'train_acc')} | {pm(g,'val_loss',4)} | "
              f"**{pm(g,'val_acc')}** | **{sp(g['spikes'])}** | {pm(g,'s30',3)} | | **{v['delta']:+.2f} ±{v['se']:.2f}** | {v['z']:+.1f} |")
        if b.get('pooled') and not (len(b['group_means']) == 1 and list(b['group_means'].values())[0]['n'] == b['pooled']['n']):
            p = b['pooled']
            w(f"| *{b['name'].split(' (')[0]} 전 점 합동* | | *n={p['n']}* | | | | | | | | | *{p['delta']:+.3f} ±{p['se']:.3f}* | {p['z']:+.1f} |")


w('## 3. inter 어블레이션 (C10) — 한 요소만 끈 팔')
w('')
w('각 팔은 `run_paper.py` 에서 `ours_ch` 설정을 그대로 두고 한 요소만 덮어쓴다 (`ch_novmem`: vmem readiness gain 0, '
  '`ch_nofinal`: 벌점을 T 합 대신 매 스텝, `ch_nolr`: loss-ratio 대신 고정 λ, `ch_noinv`: w = 1−sc/max 대신 sc/max, 채널 묶음·vmem 은 유지). '
  '그래서 intra 쪽 `- 1- inversion` (`abl_noinv`, 세 요소 동시 제거) 과 달리 **단일 요소 어블레이션이다.**')
w('')
w(f"**Δacc 정의.** inter 완주 런 중 스파이크 200~420K (n={f200['n']}) 에 `val acc = a + b·ln(spikes)` 를 맞춘 직선 "
  f"(b = {f200['beta'][1]:.3f} %p/ln, 잔차 SD {f200['s']:.3f}) 의 같은 스파이크 예측값과의 차. ± 는 곡선 예측 SE 와 잔차 SD 를 합친 값 (새 런 하나의 예측 구간 SE). "
  f"10-07 계산은 같은 대역 n=18 로 전달받았으나 그 스크립트는 찾지 못했다. 지금 완주분 기준 n={f200['n']} 이다. "
  f"민감도: 280~420K 만 (n={f280['n']}) 으로 맞추면 모든 Δ 가 ±0.01 안에서 같다 (json `vs_inter_hi`).")
w('')
arm_table(d['ablation_C10'])
w('')
w('- 비교용 inter 원점: 2e-3 평균 96.73 @ 385,705 (n=3), 3e-3 평균 96.72 @ 342,075 (n=4), 4.5e-3 평균 96.60 @ 291,930 (n=3), 7e-3 평균 96.57 @ 220,728 (n=3).')
w(f"- 9개 Δ 중 |z|>2 는 `- vmem` 3e-3 하나 (−0.23, z=−2.2, n=1). 9번 검정하면 하나쯤 나올 수 있는 크기다. `- vmem` 6점 합동은 {vm['delta']:+.3f} ±{vm['se']:.3f} (p={vm['p']:.2f}) 로 유의하지 않다.")
w('- 요소를 끄면 같은 ρ 에서 착지 스파이크가 달라진다 (예: `- final_step` 2.2e-3 → 288K, inter 2e-3 → 386K). 이 표는 그 차이를 곡선 위치로 보정한 것이다.')
w('- `- vmem` 2e-3 시드 2~4 는 세종에서 0에폭 사망(stale), 시드 5 는 hold.')
w('')

# ================================================================ 4. derived
w('## 4. inter 파생 팔 (C10)')
w('')
w('| 팔 | 바꾼 것 |')
w('|---|---|')
for b in d['derived_C10']:
    nm = b['name']
    w(f"| {nm.split(' (')[0]} | {nm.split(' (')[1].rstrip(')') if ' (' in nm else ''} |")
w('')
arm_table(d['derived_C10'])
w('')
w('- 전부 시드 SD 안 (|z| ≤ 1.4). st30 3e-3 n=3 이 +0.06 ±0.06 으로 가장 높지만 유의하지 않다.')
w('- β=0.7/0.4 의 2e-2 런, 그리고 본 팔 1.6e-2 이상은 S30/S1 < 0.19 로 배제.')
w('- st60 3e-3 시드 2·3 은 hold.')
w('')
w('**C100 파생 (참고)** — C100 inter 280~420K 완주 n=%d 직선 (잔차 SD %.3f) 대비:' % (d['inter_fit_C100_280_420']['n'], d['inter_fit_C100_280_420']['s']))
w('')
arm_table(d['derived_C100'])
w('')

# ================================================================ 5. BPSR
w('## 5. BPSR 대비 (C10)')
w('')
w('| BPSR λ | seed | 서버 | train loss | train acc | val loss | val acc | spikes | S30/S1 |')
w('|---|---|---|---|---|---|---|---|---|')
from collections import defaultdict
gb = defaultdict(list)
for r in bp['bpsr_runs']:
    gb[r['knob']].append(r)
for k, rs in gb.items():
    for r in rs:
        st = '' if not r.get('exclude_reason') else f" ({r['exclude_reason']})"
        w(f"| {k} | s{r['seed']}{st} | {r['server']} | {f(r['train_loss'],4)} | {f(r['train_acc'])} | {f(r['val_loss'],4)} | {f(r['val_acc'])} | {sp(r['spikes'])} | {f(r['s30'],3)} |")
    if k in bp['bpsr_groups']:
        m = bp['bpsr_groups'][k]
        w(f"| **{k} 평균** | **n={m['n']}** | | {pm(m,'train_loss',4)} | {pm(m,'train_acc')} | {pm(m,'val_loss',4)} | **{pm(m,'val_acc')}** | **{sp(m['spikes'])}** | {pm(m,'s30',3)} |")
w('')
w('| 비교 | 대역 | n (inter / BPSR) | 같은 스파이크 Δacc (inter − BPSR) | 같은 정확도 스파이크 비 (inter/BPSR) |')
w('|---|---|---|---|---|')
w(f"| 두 점 직접 (3e-3 vs 3e-11) | — | 4 / 3 | {dr['dacc']:+.2f} ±{dr['dacc_se']:.2f} (정확도 차) | {dr['spike_ratio']:.2f} (점추정, 정확도 차 무시) |")
w(f"| BPSR ρ 평균 곡선 보간 | — | 4 / 12 | — | {dr['spike_ratio_interp']:.2f}, 95% [{dr['spike_ratio_interp_boot95_finite'][0]:.2f}, {dr['spike_ratio_interp_boot95_finite'][1]:.2f}] (재표본 {100*dr['interp_out_of_range_frac']:.0f}% 는 범위 밖) |")
for a, lab in ((a34, '300~420K'), (a24, '200~420K'), (a13, '180~300K')):
    rb = a['spike_ratio_boot95']
    w(f"| ANCOVA `acc = a + b·ln(s) + c·[inter]` | {lab} | {a['n_inter']} / {a['n_bpsr']} | {a['c']:+.3f} ±{a['c_se']:.3f} (p={a['c_p']:.2f}) | "
      f"{a['spike_ratio_same_acc']:.2f}, 95% [{rb[0]:.2f}, {rb[1]:.2f}] (기울기 b={a['slope']:.2f} ±{a['slope_se']:.2f}, p={a['slope_p']:.2f}) |")
w('')
w('- 300~420K 에서는 기울기가 유의하지 않아 (p=0.52) 같은 정확도 스파이크 비의 구간이 사실상 무한하다. 200~420K 로 넓히면 기울기는 살지만 '
  'inter 와 BPSR 의 차가 0 이다 (스파이크 비 1.01).')
w('- 따라서 "같은 정확도에서 −17%" 는 정확도가 0.01%p 차이 나는 두 ρ 점을 고른 값이다. 같은 스파이크에서의 정확도 우위 (+0.04%p) 도 유의하지 않다. '
  '10-05 기록의 +0.076 (p=0.031) 은 다른 팔 합동 대비, 사후 선택 대역이었다.')
w('')

# ================================================================ 6. other models
w('## 6. 다른 모델의 inter')
w('')
w('### QKFormer CIFAR-10 (서버 138, 410 에폭)')
w('')
w('| 팔 | n | best top1 (best-of-410) | 마지막 에폭 top1 | fp32 재측정 (model_best) | spikes (재측정) | spikes vs base |')
w('|---|---|---|---|---|---|---|')
bm = qk['baseline_spike_measure']
w(f"| baseline | best n=7, 마지막 n=6, 재측정 n=2 | {qk['baseline_best_mean']:.2f} ±{qk['baseline_best_sd']:.2f} | {qk['baseline_last_mean']:.2f} ±{qk['baseline_last_sd']:.2f} | "
  f"{bm[0]['acc_remeasure']:.2f}, {bm[1]['acc_remeasure']:.2f} (평균 {qk['baseline_remeasure_mean']:.2f}) | {sp(bm[0]['spikes'])}, {sp(bm[1]['spikes'])} (평균 {sp(qk['baseline_spikes_mean'])}) | |")
i7, i12 = qk['inter_7e-3'], qk['inter_1.2e-2']
w(f"| ours_inter_ch 7e-3 | 1 | {i7['best_top1']:.2f} (ep {i7['best_epoch']}) | {i7['last_top1']:.2f} | {i7['measure']['acc_remeasure']:.2f} | {sp(i7['measure']['spikes'])} | "
  f"{100*(i7['measure']['spikes']/qk['baseline_spikes_mean']-1):+.1f}% |")
w(f"| ours_inter_ch 1.2e-2 | 1 | {i12['best_top1']:.2f} (ep {i12['best_epoch']}) | {i12['last_top1']:.2f} | 재측정 전 | {sp(i12['best_eval_spikes'])} (학습 중 평가값) | "
  f"{100*(i12['best_eval_spikes']/qk['baseline_spikes_mean']-1):+.1f}% (잠정) |")
w('')
w('- 개별 baseline best: ' + ', '.join(f'{v:.2f}' for v in qk['baseline_best_top1']) + '; 마지막 에폭: ' + ', '.join(f'{v:.2f}' for v in qk['baseline_last_top1'])
  + ' (base_s44 는 세종에만 있어 마지막 에폭 값 없음).')
w(f"- 정확도 기준을 무엇으로 잡느냐에 따라 부호가 바뀐다: 7e-3 은 best 기준 {i7['best_top1']-qk['baseline_best_mean']:+.2f}, 마지막 에폭 기준 "
  f"{i7['last_top1']-qk['baseline_last_mean']:+.2f}, 재측정 기준 {i7['measure']['acc_remeasure']-qk['baseline_remeasure_mean']:+.2f} (baseline 재측정 2개가 0.32 벌어져 있다). "
  "inter 쪽은 n=1 이라 **어느 기준으로도 판단 불가.** 재측정이 best 보다 0.3 낮게 나오는 원인 (AMP vs fp32 등) 은 확인 안 됨 — baseline 도 한 런은 0.25 낮고 한 런은 0.04 높다.")
w('- 1.2e-2 스파이크는 summary.csv 의 학습 중 평가값. 7e-3 에서 이 값과 재측정값 차는 0.04% 였다.')
w('- C100 inter (7e-3, 1.2e-2) 는 22·18 에폭에서 중단(stale).')
w('')
w('### VGGSNN CIFAR10-DVS (서버 138, 200 에폭, 진행 중)')
w('')
w('| 팔 | seed | 에폭 | train loss | train acc | val loss | val acc | spikes | S30/S1 |')
w('|---|---|---|---|---|---|---|---|---|')
for r in d['running']:
    if r['ds'] == 'DVS':
        w(f"| {r['arm']} {r['knob'] if r['knob']!='-' else ''} | s{r['seed']} | {r['epochs']}/200 | {f(r['train_loss'],4)} | {f(r['train_acc'])} | {f(r['val_loss'],4)} | {f(r['val_acc'])} | {sp(r['spikes'])} | {f(r['s30'],3)} |")
w('')
w('- 68 에폭, 각 n=1. 같은 에폭에서 스파이크 −25%, val acc 동일 — 중간값이라 **판단 불가**. baseline s2~4 와 inter 1e-3·1e-2 는 대기(todo).')
w('')
w('### SDTv3 ImageNet-1K — 공식 KD 체크포인트 (171M 1×8) 파인튜닝 ρ 보정 (서버 23, GPU 0–7)')
w('')
w('| 팔 | ρ | 단계 | 에폭 | train loss | train acc | val loss | val acc best | val acc 마지막 | spikes | spikes vs 대조 |')
w('|---|---|---|---|---|---|---|---|---|---|---|')
sd = d['sdtv3_imagenet_calib']
ctrl = [j for j in sd if j['arm'] == 'baseline'][0]
order = ['0 (보정)', '3e-3 (보정)', '1e-2 (보정)', '3e-2 (보정)', '0.1 (보정)', '0.3 (보정)', '1.0 (보정)']
for k in order:
    for j in sd:
        if j['knob'] == k:
            stage = '대조' if j['arm'] == 'baseline' else ('1단계' if k in order[1:4] else '2단계')
            spv = '' if j['spikes'] is None or j is ctrl else f"{100*(j['spikes']/ctrl['spikes']-1):+.2f}%"
            w(f"| {'대조 (규제 없음)' if j is ctrl else 'ours_inter_ch'} | {k.split(' ')[0]} | {stage} | {j['epochs']}/{j['total']} ({j['status']}) | "
              f"{f(j['train_loss'],4)} | {f(j['train_acc'])} | {f(j['val_loss'],4)} | {f(j['val_acc'])} | {f(j['val_last'])} | {sp(j['spikes'])} | {spv} |")
w('')
w(f"- 전부 n=1. **비교는 같은 열끼리 해야 한다:** 대조 best 85.91 / 마지막 85.61. ρ 3e-3·1e-2·3e-2 의 best 는 85.95/85.88/85.88, 마지막은 85.64/85.60/85.55. "
  "(\"대조 85.61 vs ρ 85.95\" 처럼 마지막과 best 를 섞으면 +0.3 차이가 생겨 보인다.)")
w('- 1단계 스파이크 변화는 −0.04% ~ −0.38% — 규제가 사실상 작동하지 않았다. 2단계 (ρ 0.1/0.3/1.0) 는 진행 중이고 ρ 1.0 의 2에폭 값이 −1.0% (대조 3에폭 값 대비라 에폭이 다르다).')
w('- 1단계 세 런의 에폭이 4/3 으로 찍힌다 — 원장 집계 쪽 표기 문제로 보이나 확인하지 않았다.')
w('- 기준점: 같은 체크포인트를 우리 평가 코드로 잰 값 85.21% / 2,045,688 spikes (10-08, n=1). 논문 Table 1 은 85.1%.')
w('')

# ================================================================ 7. excluded
w('## 7. 배제·미완주 (평균에서 뺌)')
w('')
w('| 데이터 | 팔 | ρ | seed | 서버 | 에폭 | val acc | spikes | 사유 |')
w('|---|---|---|---|---|---|---|---|---|')
for r in d['excluded']:
    why = r.get('exclude_reason') or ('2에폭 연기시험' if r['seed'] >= 900 else r['status'])
    w(f"| {r['ds']} | {r['arm']} | {r['knob']} | s{r['seed']} | {r['server']} | {r['epochs']} | {f(r['val_acc'])} | {sp(r['spikes'])} | {why} |")
w('')

# ================================================================ limitation
w('## [LIMITATION]')
w('')
w('- 어블레이션·파생 팔은 거의 전부 n=1 이다. 개별 Δ 의 SE 가 ±0.10%p 라 0.2%p 보다 작은 효과는 이 표로 말할 수 없다. "요소가 필요 없다" 는 결론이 아니다.')
w('- Δacc (같은 스파이크) 는 inter 곡선을 ln(spikes) 직선으로 근사한다. 10-05 분석에서 inter 곡선은 "평평 → 300K 아래 급락" 모양이었으므로 '
  '양 끝 (≈420K, ≈220K) 런의 Δ 는 직선 오차를 포함한다. 280~420K 재적합으로 바뀌는 건 ±0.01 이하였다.')
w('- BPSR 비교의 "같은 정확도 스파이크 비" 는 BPSR 곡선이 300~420K 에서 평평해 구간이 매우 넓다. 논문에 숫자로 쓰려면 같은 스파이크 Δacc 를 주 지표로 써야 한다.')
w('- C100 대표점 (2e-3) 은 n=1, QKFormer inter 는 각 n=1, VGGSNN 은 중간 에폭, SDTv3 는 3에폭 보정이다. 이 넷에 대해서는 어떤 결론도 내지 않았다.')
w('- QKFormer baseline 스파이크는 n=2 (나머지 baseline 은 스파이크 측정을 안 함). 정확도 기준 (best / 마지막 / 재측정) 에 따라 부호가 바뀐다.')
w('- 원장 값 (10-09 12:56) 을 그대로 썼다. 원장 이후 끝난 런 (QK 1.2e-2 제외) 은 반영 안 됨.')
open(P.with_suffix('.md'), 'w').write('\n'.join(L) + '\n')
print('wrote', P.with_suffix('.md'))
