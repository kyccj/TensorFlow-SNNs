"""inter_deep_261005.json -> inter_deep_261005.md (한국어 보고). 수치는 JSON 에서만 읽는다.

결론 절(0장)의 문장은 JSON 값으로 채운다. 해석 문장은 여기서 쓴다.
쓰는 법: /home/kyccj/anaconda3/envs/venv_1/bin/python analysis/inter_deep_261005_md.py
"""
import json, os, math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = json.load(open(os.path.join(ROOT, 'analysis', 'inter_deep_261005.json')))
INV = {r['tag']: r for r in json.load(open(
    '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad/ledger/inventory.json'))['runs']}
INTER = 'ours_inter_ch'
L = []
w = L.append


def f(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '–'
    return (f'{x:+.{d}f}' if sign else f'{x:.{d}f}')


def K(x):
    return '–' if x is None else f'{x / 1e3:,.0f}K'


def P(p):
    if p is None:
        return '–'
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def ci(c, d=3):
    return f"{f(c['est'], d, True)} [{f(c['lo'], d, True)}, {f(c['hi'], d, True)}]"


SD = O['seed_sd']
A1 = O['A']['A1']
C10 = O['A']['curves_C10']
C100 = O['A']['curves_C100']
G = O['G']; H = O['H']; B = O['B']; C = O['C']; D = O['D']; E = O['E']; F = O['F']

# ------------------------------------------------------------------------------------------
w('# ours_inter_ch 채택 근거와 \'안 된 시도\' 재분석 (R19, T=4, 26-10-05)\n')
w(f"원장 {O['meta']['ledger']} 집계, 310 에폭 완주·배제 아님·seed<900 만. 정확도 = val acc (베스트 에폭; CIFAR 에서 val = test 1만 장), "
  f"스파이크 = 그 에폭의 test 샘플당 스파이크 수 (원장). 생성: `analysis/inter_deep_261005.py` → `.json`, 이 문서는 `analysis/inter_deep_261005_md.py`.\n")
w(f"시드 SD (같은 팔·같은 ρ 묶음 안 편차를 합동): C10 **{SD['C10_pooled']:.3f}%p** (자유도 {SD['C10_df']}), "
  f"C100 **{SD['C100_pooled']:.3f}%p** (자유도 {SD['C100_df']}). baseline 만: C10 {SD['C10_base']:.3f} (n={SD['C10_base_n']}), "
  f"C100 {SD['C100_base']:.3f} (n={SD['C100_base_n']}).\n")
w("이름: ours_inter_ch = 런 키 `ours_ch`, ours_intra_ch = `prop` (원래 제안법, w²), ours_intra_ch (w¹) = `ours_w1`, "
  "ours_layer (w¹) = `ours_layer_w1`, ours_layer (w²) = `ours_layer` (민감도 분석에만). 오프셋 부호: **'다른 팔 − inter'** 이 음수면 inter 가 높다.\n")

# ------------------------------------------------------------------------------------------
w('## 0. 결론 (먼저)\n')
a = A1['C10_320-420K']; a2 = A1['C10_250-420K']; a3 = A1['C10_180-420K']
po = a['inter_vs_pooled_others']; po2 = a2['inter_vs_pooled_others']; po3 = a3['inter_vs_pooled_others']
hk = H['checkpoint']
l320 = hk['last|320-420K']['inter_vs_pooled_others']; l10_320 = hk['last10|320-420K']['inter_vs_pooled_others']
w(f"1. **C10 높은 구간(320~420K)의 inter 우위는 '있다' 쪽이지만 약하다.** 같은 스파이크로 보정하면 다른 네 팔을 합친 것보다 "
  f"{f(-po['est'], 3)}%p 높다 (95% CI [{f(-po['hi'], 3)}, {f(-po['lo'], 3)}], p={P(po['p'])}, n={a['common']['n']}: inter {a['common']['n_arm'][INTER]}). "
  f"팔 하나씩 보면 어느 것도 Holm 보정 후 유의하지 않다 (최소 p_holm={P(min(a['common']['coef'][x]['p_holm'] for x in a['common']['others']))}). "
  f"대역을 250~420K 로 넓히면 {f(-po2['est'], 3)}%p (p={P(po2['p'])}), 180~420K 이면 {f(-po3['est'], 3)}%p (p={P(po3['p'])}) 로 사라진다. "
  f"**주간 정리의 '0.08~0.13 위, 우연으로 보기 어렵다'는 과장이다** — 스파이크 보정 후 0.05~0.09, 단일 대비로 p≈0.03, 대역은 사후 선택.")
be = H['best_epoch']
w(f"2. **체크포인트 선택에 민감하다.** 마지막 에폭 정확도로 바꾸면 같은 320~420K 대비가 {f(-l320['est'], 3)}%p (p={P(l320['p'])}), "
  f"마지막 10 에폭 평균이면 {f(-l10_320['est'], 3)}%p (p={P(l10_320['p'])}) 로 약 1/3 줄고 유의하지 않다. "
  f"베스트 에폭 위치 자체는 팔 간 같다 (inter 중앙값 {be['by_arm'][INTER]['median']:.0f}, n={be['by_arm'][INTER]['n']}; 다른 팔 306~309; Mann-Whitney p={P(be['inter_vs_others_mw_p'])}). "
  "즉 inter 가 특별히 운 좋은 에폭을 고른 것은 아니지만, 0.05~0.08%p 수준의 차이는 체크포인트 정의 하나로 유의성이 바뀌는 크기다.")
cr = C10['crossing']
w(f"3. **교차점은 약 300K.** inter − 다른 팔 곡선이 + 에서 − 로 바뀌는 스파이크 수: "
  + ', '.join(f"{x} {K(v['median'])} [{K(v['lo'])}, {K(v['hi'])}]" for x, v in cr.items() if v['median'] and v['frac_cross'] >= 0.5)
  + ". 300K 아래에서는 inter 가 가장 낮고 (240K: −0.16~−0.24%p, 부트스트랩 CI 대부분 0 미포함, 단 inter 는 n=1 점들), "
  f"200~450K 전체 하이퍼볼륨은 팔 간 차이가 없다 (inter − 다른 팔 모두 CI 가 0 포함). **'곡선 전체가 높다'가 아니라 '높은 구간에서만 약간 높고 낮은 구간에서 확실히 낮다'.**")
cg = C['250-420K']['metrics']
w(f"4. **'채널을 더 죽여서 정확도가 높다'는 지지되지 않는다 — 방향이 반대다.** 같은 팔·같은 스파이크에서 죽은 채널이 많은 런일수록 정확도가 낮다 "
  f"(부분상관 r={f(cg['dead_ch']['partial_r_within_arm'])}, p={P(cg['dead_ch']['p_within_arm'])}, n={cg['dead_ch']['n']}). "
  f"inter 효과에 죽은 채널을 넣으면 효과가 줄지 않고 커진다 ({f(cg['dead_ch']['inter_effect_total'], 3, True)} → {f(cg['dead_ch']['inter_effect_direct'], 3, True)}%p). "
  f"정확도 잔차와 가장 강하게 묶인 지표는 '살아남은 채널 안 지니'(낮을수록 정확도 높음, 팔 안 r={f(cg['alive_ch_gini']['partial_r_within_arm'])}, p={P(cg['alive_ch_gini']['p_within_arm'])}).")
w(f"5. **층 배분·SynOps 는 팔 간 거의 같다.** ~340K 에서 감소분의 conv2 비중 68~71%, SynOps/스파이크 차이 1.3% 이내. inter 가 비싼 층을 더 깎는 증거는 없다.")
w("6. **안 된 시도는 통계로도 '효과 없음'.** st30 (n=3 대 n=3, Welch p=" + P(G['st30_summary']['welch_inter3e-3']['p'])
  + f"), Taylor (4점 역분산 평균 잔차 {f(G['pooled']['Taylor']['mean_resid_ivw'], 3, True)}%p, CI [{f(G['pooled']['Taylor']['ci'][0], 2, True)}, {f(G['pooled']['Taylor']['ci'][1], 2, True)}]), "
  f"가중 모양 변형 15점 (평균 잔차 {f(G['pooled']['가중 모양 변형 (sm, abl_*, β, l2_lr)']['mean_resid_ivw'], 3, True)}%p, p={P(G['pooled']['가중 모양 변형 (sm, abl_*, β, l2_lr)']['p'])}). "
  "곡선을 0.1%p 이상 움직인 처치는 없다.")
pc = O['A']['power_C100']
w(f"7. **C100 은 판단 불가.** inter 는 전부 n=1, 같은 스파이크 오프셋의 CI 폭이 ±0.2~0.4%p. 0.15%p 차이를 80% 검정력으로 보려면 팔당 약 {pc['n_per_arm_two_sample']:.0f}런 (시드 SD {pc['sd']:.2f} 기준; baseline SD {pc['sd_base']:.2f} 이면 {pc['n_per_arm_two_sample_base_sd']:.0f}런).")
w('')

# ------------------------------------------------------------------------------------------
w('## 이번 분석으로 바뀐 해석\n')
w('| 주간 정리 (10-05) | 재분석 | 판정 |')
w('|---|---|---|')
w(f"| 405K·386K·342K 세 점 모두 다른 팔보다 0.08~0.13 위, 우연 아님 | 스파이크 보정 후 320~420K 단일 대비 {f(-po['est'], 3)}%p, p={P(po['p'])}; 팔별 Holm 후 유의 없음; 마지막 에폭 기준이면 {f(-l320['est'], 3)}%p | **약화** — '높은 구간에서 약간 높을 수 있다' |")
w(f"| 스파이크 −17~−30% 에서 baseline 과 같은 정확도 | 맞다 (inter 3점 평균이 baseline {SD['C10_base_mean']:.2f}±{SD['C10_base']:.2f} 안). 단 다른 팔도 그 구간에서 baseline −0.1 안 | 유지 (inter 고유 성질 아님) |")
w("| inter 곡선 표 (8점) | 원장에 4.5e-3 (294K, 96.53), 7e-3 (221K, 96.42), 1.2e-2 (147K, 96.30) 세 점이 더 있다 (전부 n=1, 유효). 4.5e-3 은 주간 표의 4e-3·5e-3 보다 낮다 | **표 누락 정정** |")
w(f"| 낮은 스파이크에서 가장 빨리 떨어진다 | 240K 에서 inter − 다른 팔 −0.16~−0.24 (pooled 부트스트랩), 교차점 ~300K | 유지·정량화 |")
w(f"| 채널을 가장 많이 죽이고 그것이 이점의 기전 (암시) | 팔 안에서는 죽은 채널↑ → 정확도↓ (r={f(cg['dead_ch']['partial_r_within_arm'])}); 매개 비율 음수 | **기각** (구조적 희소성 이득은 별개로 유효) |")
dcI = D['decomp']['by_arm'][INTER]
w(f"| inter 는 채널을 통째로 끄는 것 말고는 줄일 방법이 없어 저스파이크에서 꺾인다 | inter 의 스파이크 감소 중 채널 사망 몫은 {100 * dcI['share_mean']:.0f}% ±{100 * dcI['share_sd']:.0f} (n={dcI['n']}); 나머지는 산 채널 안 발화 감소. 300K 아래 사망 가속은 ours_layer (w¹)·BPSR 에도 있다 | **기각** (꺾임의 기전은 미확인) |")
bo = F['offline_npz']['baseline']
w(f"| 죽은 채널 제거로 MACs 33~40% 감소 | 그중 baseline 자체의 죽은 채널이 이미 {bo['macs_pct_b'][0]:.1f}% (n={bo['n']}, 테스트 판정). 규제가 더한 몫은 그 차이. 같은 스파이크에서 inter 가 다른 팔보다 MACs 를 더 줄이는 것은 유의 (F 절) | 유지·맥락 추가 |")
w(f"| st30 효과 없음 | Welch p={P(G['st30_summary']['welch_inter3e-3']['p'])}, 첫 시드 세 팔 평균 +{G['st30_summary']['first_seed_signal']['mean']:.3f} 은 null 에서 {G['st30_summary']['p_mean3_ge_obs_under_null']:.2f} 확률 | 유지 |")
w(f"| Taylor 효과 없음 | 역분산 평균 {f(G['pooled']['Taylor']['mean_resid_ivw'], 3, True)}, CI ±{1.96 * G['pooled']['Taylor']['se']:.2f} | 유지 (CI 가 ±0.13 이라 0.1 효과는 배제 못함) |")
w(f"| 곡선을 움직인 것은 경쟁 범위 하나 | 경쟁 범위도 곡선 '전체'를 올리지는 못했다 (A1 팔 효과 F p={P(a2['p_arm'])} @250~420K, 하이퍼볼륨 차 0) — 곡선 '모양'(높은 구간↑·낮은 구간↓)을 바꿨다 | **수정** |")
w('')

# ------------------------------------------------------------------------------------------
w('## A. 곡선 수준 비교\n')
w('### A1. ANCOVA: acc = a_팔 + b·ln(spikes)\n')
w('기준 팔 = ours_inter_ch. 계수 = 같은 스파이크에서 (다른 팔 − inter), 95% CI, p (t), Holm 보정 p (팔 수만큼). '
  '`inter 대 나머지` = 다른 팔을 하나로 묶은 단일 대비. 상호작용 F = 팔마다 기울기가 다른가. 팔 F = 모든 팔이 한 곡선인가. '
  '잔차 SD 를 같은 (팔, ρ) 안 순수오차 SD 와 비교하고 적합결여(LOF) F 를 낸다.\n')
w('| 데이터·대역 | n (inter) | b (ln 스파이크) | ' + ' | '.join(['ours_intra_ch', 'ours_intra_ch (w¹)', 'ours_layer (w¹)', 'plain L2', 'BPSR']) +
  ' | inter 대 나머지 | 상호작용 F (p) | 팔 F (p) | 잔차 SD / 순수오차 SD | LOF p |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for key in ['C10_320-420K', 'C10_250-420K', 'C10_180-420K', 'C100_250-420K', 'C100_180-420K']:
    v = A1[key]; c = v['common']; co = c['coef']
    cells = []
    for arm in ['ours_intra_ch', 'ours_intra_ch (w¹)', 'ours_layer (w¹)', 'plain L2', 'BPSR']:
        if arm in co:
            cells.append(f"{ci(co[arm])} p={P(co[arm]['p'])} (holm {P(co[arm]['p_holm'])}, n={c['n_arm'][arm]})")
        else:
            cells.append('–')
    io = v['inter_vs_pooled_others']
    w(f"| {key.replace('_', ' ')} | {c['n']} ({c['n_arm'][INTER]}) | {f(co['b']['est'], 3)} ±{f(co['b']['se'], 3)} | " + ' | '.join(cells) +
      f" | {ci(io)} p={P(io['p'])} | {f(v['F_interaction'])} ({P(v['p_interaction'])}) | {f(v['F_arm'])} ({P(v['p_arm'])}) | "
      f"{f(c['resid_sd'], 3)} / {f(v['pure_error_sd'], 3)} | {P(v['p_lof'])} |")
w('')
w(f"- ours_layer (w²) 를 넣은 민감도: 320~420K 단일 대비 {ci(A1['C10_320-420K+layer_w2']['inter_vs_pooled_others'])} p={P(A1['C10_320-420K+layer_w2']['inter_vs_pooled_others']['p'])}, "
  f"250~420K {ci(A1['C10_250-420K+layer_w2']['inter_vs_pooled_others'])} p={P(A1['C10_250-420K+layer_w2']['inter_vs_pooled_others']['p'])} — 결론 같음.")
w(f"- 잔차 SD 가 순수오차(시드) SD 와 같고 LOF 가 유의하지 않다 → 선형-로그 모형으로 설명 못 하는 계통 차가 없다. "
  f"그 안에서 팔 차이는 시드 산포({SD['C10_pooled']:.2f}) 의 절반 이하.")
sl = A1['C10_180-420K']['slopes']
w('- 팔별 기울기 (180~420K, %p / ln 스파이크): ' + ', '.join(f"{k} {f(v['b'], 2)}±{f(v['se'], 2)} (n={v['n']})" for k, v in sl.items())
  + f". inter 가 가장 가파르다 (공통 기울기 F p={P(A1['C10_180-420K']['p_interaction'])} 로 기각은 못 함). "
  "inter 곡선은 직선이 아니라 '평평 → 300K 아래 급락' 모양이라 넓은 대역의 선형 오프셋은 inter 의 높은 구간 이점과 낮은 구간 손실을 상쇄해 0 근처가 된다.")
pw = O['A']['power_C10']
w(f"- 검정력: 250~420K 에서 inter 대 각 팔 오프셋 SE 0.036~0.072 → 진짜 차이가 0.10%p 일 때 검정력 "
  + ', '.join(f"{k.split('|')[1]} {v['power_0.10']:.2f}" for k, v in pw.items() if k.startswith('C10_250')) + '.\n')

w('### A2. 같은 스파이크에서의 정확도 (단조 곡선 보간, 부트스트랩 2000회)\n')
w('곡선 = 팔마다 ρ 점 평균을 ln(스파이크) 축에서 단조(스파이크↑ → 정확도 비감소) 로 맞춘 뒤 선형 보간. 범위 밖은 비움. '
  'pooled = 각 런을 ρ 점 평균 + 합동 시드 잔차로 재생성 (n=1 점에도 시드 산포 부여, **주 결과**); seed = ρ 점 안 시드 복원추출 (n=1 점은 고정되어 CI 가 과소).\n')
w('| 스파이크 | ' + ' | '.join(C10['curves'].keys()) + ' |')
w('|---|' + '---|' * len(C10['curves']))
for t, v in C10['at_spikes'].items():
    w(f"| {K(float(t))} | " + ' | '.join(f"{f(v['level'][a]['est'])}" + (f" ±{f(v['level'][a]['pooled_sd'], 2)}" if v['level'][a]['pooled_sd'] else '') for a in C10['curves']) + ' |')
w('')
w('inter − 다른 팔 (%p), pooled [95% CI] (P>0) / seed [95% CI]:\n')
w('| 스파이크 | ' + ' | '.join(a for a in C10['curves'] if a != INTER) + ' |')
w('|---|' + '---|' * (len(C10['curves']) - 1))
for t, v in C10['at_spikes'].items():
    cells = []
    for a in C10['curves']:
        if a == INTER:
            continue
        d = v['diff_inter_minus'][a]
        if d['pooled'] is None:
            cells.append('범위 밖'); continue
        p_, s_ = d['pooled'], d['seed']
        cells.append(f"{f(p_['est'], 2, True)} [{f(p_['lo'], 2, True)}, {f(p_['hi'], 2, True)}] ({p_['p_gt0']:.2f}) / [{f(s_['lo'], 2, True)}, {f(s_['hi'], 2, True)}]")
    w(f"| {K(float(t))} | " + ' | '.join(cells) + ' |')
w('')
w('- inter 의 240K·200K 값은 n=1 점(6e-3, 7e-3, 8e-3)의 보간이다. pooled CI 는 이 점들에 시드 SD 를 준 것이라 "n=1 이지만 시드 하나 차이로는 설명 안 되는 하락" 이라는 뜻이다. 그래도 n=1 점 셋이 모두 낮게 나온 것이므로 재현(시드 2·3)이 있어야 확정이다.')
w('- inter 곡선 점들: ' + ', '.join(f"{p['knob']} {K(p['spikes'])}/{p['acc']:.2f} (n={p['n']})" for p in reversed(C10['curves'][INTER]['points'])) + '.\n')

w('### A3. 파레토 전선과 하이퍼볼륨\n')
w('ρ 점 평균 기준 전선 (스파이크 오름차순, baseline 포함): ' + ' → '.join(f"{p['arm']} {p['knob']} ({K(p['spikes'])}, {p['acc']:.2f}, n={p['n']})" for p in C10['pareto_means']) + '\n')
w('개별 런 기준 전선: ' + ' → '.join(f"{p['arm']} {p['knob']} s{p.get('seed')} ({K(p['spikes'])}, {p['acc']:.2f})" for p in C10['pareto_runs']) + '\n')
w('대역별 전선 점유 확률 (pooled 부트스트랩, 그 대역에 전선 점이 있는 비율):\n')
w('| 대역 | ' + ' | '.join(C10['curves'].keys()) + ' |')
w('|---|' + '---|' * len(C10['curves']))
for b, d in C10['pareto_occupancy_pooled'].items():
    w(f"| {b} | " + ' | '.join(f"{d.get(a, 0):.2f}" for a in C10['curves']) + ' |')
w('')
w(f"하이퍼볼륨 = 200~450K 에서 '예산 x 로 얻는 최고 정확도 − 기준선({C10['hv_ref_acc']:.2f} = baseline − 0.3)' 의 평균 (%p). "
  "계단(실제 점) 과 단조곡선 두 방식. 실제 점 계단은 ρ 점이 많은 팔이 유리하다 (최대값의 잡음).\n")
w('| 팔 | 계단 HV [95% CI] | 단조 HV [95% CI] | inter − 이 팔 (계단) | inter − 이 팔 (단조) |')
w('|---|---|---|---|---|')
for a in C10['curves']:
    hs = C10['hv']['stair'][a]; hi_ = C10['hv']['iso'][a]
    ds_ = C10['hv_diff_inter_minus']['stair'].get(a); di = C10['hv_diff_inter_minus']['iso'].get(a)
    w(f"| {a} | {f(hs['est'], 3)} [{f(hs['pooled']['lo'], 3)}, {f(hs['pooled']['hi'], 3)}] | {f(hi_['est'], 3)} [{f(hi_['pooled']['lo'], 3)}, {f(hi_['pooled']['hi'], 3)}] | "
      + (f"{f(ds_['est'], 3, True)} [{f(ds_['lo'], 3, True)}, {f(ds_['hi'], 3, True)}]" if ds_ else '–') + ' | '
      + (f"{f(di['est'], 3, True)} [{f(di['lo'], 3, True)}, {f(di['hi'], 3, True)}]" if di else '–') + ' |')
w('')

w('### A4. 교차점 (inter 가 위에서 아래로 내려가는 스파이크 수, pooled 부트스트랩)\n')
w('| 상대 팔 | 점추정 | 부트스트랩 중앙값 [95% CI] | 교차가 생긴 비율 |')
w('|---|---|---|---|')
for x, v in cr.items():
    w(f"| {x} | {K(v['est']) if v['est'] else v['status']} | {K(v['median'])} [{K(v['lo'])}, {K(v['hi'])}] | {v['frac_cross']:.2f} |")
w('- ours_intra_ch (w²) 는 곡선 범위가 162~289K 라 inter 의 높은 구간과 겹치지 않는다. 겹치는 234~289K 에서 inter 는 늘 아래 (점추정).\n')

w('### A5. C100\n')
w('| 스파이크 | ' + ' | '.join(C100['curves'].keys()) + ' |')
w('|---|' + '---|' * len(C100['curves']))
for t, v in C100['at_spikes'].items():
    w(f"| {K(float(t))} | " + ' | '.join(f(v['level'][a]['est']) for a in C100['curves']) + ' |')
w('')
w('- inter 점: ' + ', '.join(f"{p['knob']} {K(p['spikes'])}/{p['acc']:.2f} (n={p['n']})" for p in reversed(C100['curves'][INTER]['points'])) + '. '
  f"inter 곡선 범위가 313~367K 뿐이라 300K 이하는 비교 불가. 340K 에서 inter − ours_layer (w¹) "
  f"{f(C100['at_spikes']['340000']['diff_inter_minus']['ours_layer (w¹)']['pooled']['est'], 2, True)} "
  f"[{f(C100['at_spikes']['340000']['diff_inter_minus']['ours_layer (w¹)']['pooled']['lo'], 2, True)}, {f(C100['at_spikes']['340000']['diff_inter_minus']['ours_layer (w¹)']['pooled']['hi'], 2, True)}].")
w(f"- 검정력 (두 표본, α=0.05, 80%): δ=0.15%p 를 보려면 팔당 {pc['n_per_arm_two_sample']:.0f}런 (σ={pc['sd']:.3f}, 합동) / {pc['n_per_arm_two_sample_base_sd']:.0f}런 (σ={pc['sd_base']:.3f}, baseline). "
  f"δ=0.22 이면 {pc['n_per_arm_delta_0.22']:.0f}런. 지금 inter n=4 (각 ρ n=1). **아직 판단 불가.**\n")

# ------------------------------------------------------------------------------------------
w('## B. 스파이크를 어디서 줄였나 (단계별 배분, SynOps)\n')
w('프로토콜: `csv` = reg_detail.csv 의 베스트 에폭 층별 spike_count (학습 모드·증강 포함, 샘플당), `npz` = 테스트 1만 장 추론 (뉴런별 합). '
  'baseline 은 같은 프로토콜의 baseline 평균 (csv n=' + str(B['baseline']['csv']['n']) + ', npz n=' + str(B['baseline']['npz']['n']) + '). '
  'SynOps = Σ 스파이크 × fan-out (경계 패딩 반영; 32×32 층 1,104~1,232 / 16×16 2,116 / 8×8 3,872; 단계 출구 층은 stride-2 로 616·1,186). '
  'npz 는 위치별 정확한 fan-out, csv 는 층 평균 fan-out.\n')
for band in ('~340K', '~240K', '~200K'):
    w(f'**{band}**\n')
    w('| 방법 | 프로토콜 | n | 서버 | val acc | spikes | 스파이크 감소 % | SynOps 감소 % | SynOps/스파이크 | 감소분 비중 conv1 / conv2 / conv3 / conv4 % | 단계별 감소율 conv1 / conv2 / conv3 % |')
    w('|---|---|---|---|---|---|---|---|---|---|---|')
    for arm, d in B[band].items():
        for proto in ('csv', 'npz'):
            if proto not in d:
                continue
            x = d[proto]; sh = x['stage_red_share_pct']; rb = x['stage_red_pct_of_base']
            w(f"| {arm} | {proto} | {x['n']} | {'/'.join(x['servers'])} | {x['val_acc']:.2f} | {x['spikes_ledger']:,.0f} | "
              f"{x['spk_red_pct'][0]:.1f}" + (f" ±{x['spk_red_pct'][1]:.1f}" if x['spk_red_pct'][1] else '') + f" | {x['syn_red_pct'][0]:.1f}" +
              (f" ±{x['syn_red_pct'][1]:.1f}" if x['syn_red_pct'][1] else '') + f" | {x['syn_per_spk']:,.0f} | "
              f"{sh['conv1']:.1f} / {sh['conv2']:.1f} / {sh['conv3']:.1f} / {sh['conv4']:.1f} | {rb['conv1']:.1f} / {rb['conv2']:.1f} / {rb['conv3']:.1f} |")
    w('')
w('SynOps/스파이크 (csv) inter 대 다른 팔, Welch (~340K 만 양쪽 n=3):\n')
w('| 대역 | 상대 | inter | 상대 | 차 % | p | n |')
w('|---|---|---|---|---|---|---|')
for k, v in B['synps_tests'].items():
    b_, a_ = k.split('|')
    w(f"| {b_} | {a_} | {v['inter']:,.0f} | {v['other']:,.0f} | {v['diff_pct']:+.2f} | {P(v['p'])} | {v['n'][0]} vs {v['n'][1]} |")
w('')
sy = B['ancova_synops_250-420K']
w(f"- x 축을 SynOps 로 바꾼 ANCOVA (250~420K, n={sy['n']}): " + ', '.join(f"{a} {f(sy['coef'][a]['est'], 3, True)} (p={P(sy['coef'][a]['p'])})" for a in sy['others'])
  + ' — 스파이크 축과 사실상 같다.')
w('- 해석: 모든 팔이 감소분의 약 2/3 를 conv2(32×32, 싼 층)에서, 1/5 을 conv1 에서 얻는다. inter 는 BPSR·plain L2 보다 conv1 을 덜, conv2 를 약간 더 깎는다. '
  'SynOps/스파이크가 inter 에서 0.4~1.3% 낮아 "남은 스파이크가 조금 더 싼 층에 있다" 이지만 등가 정확도로 0.01%p 미만이다. **층 배분은 inter 를 다른 팔과 구분하는 축이 아니다.**\n')

# ------------------------------------------------------------------------------------------
w('## C. 기전: 분포 지표가 정확도 잔차를 설명하는가\n')
for key, c in C.items():
    w(f"**대역 {key}** — npz 가 있는 런 n={c['n']} (" + ', '.join(f"{a} {n}" for a, n in sorted(c['n_arm'].items())) + '). '
      f"잔차 = val acc − (공통 곡선 {c['curve_no_arm']['a']:.2f} + {c['curve_no_arm']['b']:.3f}·ln 스파이크; 원장 전체로 적합, 팔 항 없음).\n")
    w('| 지표 | 부분상관 r (ln 스파이크 통제) | p | 팔 안 부분상관 r (ln 스파이크 + 팔 통제) | p (df) | 팔 안 기울기 (%p/단위) | inter 효과: 지표 없이 → 넣고 | 매개 비율 [95% CI] |')
    w('|---|---|---|---|---|---|---|---|')
    for mk, v in c['metrics'].items():
        w(f"| {mk} | {f(v['partial_r_spk'])} | {P(v['p_spk'])} | {f(v['partial_r_within_arm'])} | {P(v['p_within_arm'])} ({v['df_within']}) | "
          f"{v['slope_within_arm']:+.4f} ±{v['slope_se']:.4f} | {f(v['inter_effect_total'], 3, True)} → {f(v['inter_effect_direct'], 3, True)} | "
          f"{f(v['prop_mediated'])} [{f(v['prop_mediated_ci'][0])}, {f(v['prop_mediated_ci'][1])}] |")
    w('')
w('- "inter 효과" = (inter − 다른 팔 전체) 의 같은 스파이크 차이 (이 npz 부분집합에서). 매개 비율이 음수 = 지표를 넣으면 inter 효과가 **커진다** (억제 변수).')
w('- 죽은 채널: 팔 안에서 음의 상관. "채널을 더 죽인 런이 더 정확하다"가 아니라 반대. inter 가 채널을 더 죽이는 것은 정확도 이점의 원인이 아니라 그것과 **별개로** 나타나는 성질이다 (구조적 희소성 이득은 F 절에서 따로 본다).')
w(f'- 살아남은 채널 안 지니: 유일하게 팔 안·전체 모두 강한 음의 상관 (산 채널 안에서 발화가 고를수록 정확도↑). inter 효과의 일부(250~420K 점추정 {100 * C["250-420K"]["metrics"]["alive_ch_gini"]["prop_mediated"]:.0f}%, CI 0 포함)를 설명. '
  '단 지니는 스파이크 수와 정확도 둘 다에 걸린 지표라 인과 방향은 모른다.')
w(f"- n 이 {C['250-420K']['n']}~{C['180-420K']['n']} 이고 팔당 {min(min(c['n_arm'].values()) for c in C.values())}~{max(max(c['n_arm'].values()) for c in C.values())} 라 매개 분석은 탐색 수준이다. 180~420K 에서는 inter 효과 자체가 {C['180-420K']['metrics']['dead_ch']['inter_effect_total']:+.3f}%p 로 0 이라 매개 비율은 의미가 없다.\n")

# ------------------------------------------------------------------------------------------
w('## D. 저스파이크 붕괴: 죽은 채널 곡선\n')
w('죽은 채널 = 테스트 1만 장에서 한 번도 발화하지 않은 채널 (conv 4,480 + fc1 256 = 4,736). npz 프로토콜.\n')
w('분절 회귀 dead = a + b_hi·(ln S − ln 300K)⁺ + b_lo·(ln S − ln 300K)⁻ (팔별, 런 단위). b 는 "스파이크가 e 배 줄 때 죽은 채널 변화" 의 부호 반대 (음수 = 스파이크가 줄면 더 죽음).\n')
w('| 팔 | n | 스파이크 범위 | b (300K 위) | b (300K 아래) | 아래 − 위 [p] | 선형 b |')
w('|---|---|---|---|---|---|---|')
for a, v in D['fits'].items():
    w(f"| {a} | {v['n']} | {K(v['spk_range'][0])}~{K(v['spk_range'][1])} | "
      + (f"{v['b_above300']:+.0f} ±{v['se_above']:.0f} | {v['b_below300']:+.0f} ±{v['se_below']:.0f} | {v['diff']:+.0f} ±{v['se_diff']:.0f} [{P(v['p_diff'])}]" if 'b_above300' in v else '– | – | –')
      + f" | {v['b_linear']:+.0f} ±{v['se_linear']:.0f} |")
w('')
w('같은 스파이크에서의 죽은 채널 (ρ 점 평균의 로그 보간; 범위 밖은 빈칸):\n')
arms = sorted({a for t in D['at'].values() for a in t})
w('| 스파이크 | ' + ' | '.join(arms) + ' |')
w('|---|' + '---|' * len(arms))
for t, v in D['at'].items():
    w(f"| {K(float(t))} | " + ' | '.join((f"{v[a]['dead_ch']:.0f}" if a in v else '') for a in arms) + ' |')
w('')
w('단계별 죽은 채널 @240K / @200K (같은 보간):\n')
w('| 스파이크 | 팔 | conv1 /128 | conv2 /768 | conv3 /1536 | conv4 /2048 | fc /256 | val acc (보간) |')
w('|---|---|---|---|---|---|---|---|')
for t in ('340000', '240000', '200000'):
    for a, v in D['at'].get(t, {}).items():
        sd_ = v['stage_dead']
        w(f"| {K(float(t))} | {a} | {sd_['conv1']:.0f} | {sd_['conv2']:.0f} | {sd_['conv3']:.0f} | {sd_['conv4']:.0f} | {sd_['fc']:.0f} | {v['val_acc']:.2f} |")
w('')
w('inter ρ 순서 (스파이크 내림차순; 런 단위 개별값은 부록 D):\n')
w('| ρ | n | val acc | spikes | 죽은 채널 | conv1 / conv2 / conv3 / conv4 / fc | 50% 집중도 % | 산 채널 안 침묵 % | MACs 제거 % (b, 테스트 판정) |')
w('|---|---|---|---|---|---|---|---|---|')
for s_ in D['inter_sequence']:
    sd_ = s_['stage_dead']
    w(f"| {s_['knob']} | {s_['n']} | {s_['val_acc']:.2f} | {s_['spikes']:,.0f} | {s_['dead_ch']:.0f} | {sd_['conv1']:.0f} / {sd_['conv2']:.0f} / {sd_['conv3']:.0f} / {sd_['conv4']:.0f} / {sd_['fc']:.0f} | "
      f"{s_['half_share_pct']:.1f} | {s_['alive_ch_silent_pct']:.2f} | {s_['macs_pct_b']:.1f} |")
w('')
w('연속 ρ 점 사이 변화 (inter): ' + '; '.join(f"{m['from']}→{m['to']}: Δspk {m['d_spikes'] / 1e3:+.0f}K, Δdead {m['d_dead']:+.0f}, Δacc {m['d_acc']:+.2f}" for m in D['inter_marginal']) + '\n')
dc = D['decomp']
w(f"**분해: conv 스파이크 = 산 conv 채널 수 × 산 채널당 스파이크.** baseline (n={dc['n_base']}): 산 conv 채널 {dc['baseline_conv_alive_ch']:,.0f} / 4,480, 산 채널당 {dc['baseline_spk_per_alive_ch']:.1f} 스파이크/샘플. "
  "런마다 baseline 대비 ln 감소를 두 항으로 나눠 '채널 사망이 감소의 몇 %인가' 와, ln(산 채널 수) 를 ln(conv 스파이크) 에 회귀한 탄성(1 이면 전부 채널 사망, 0 이면 전부 채널 안 감소)을 낸다.\n")
w('| 팔 | n | 채널 사망 몫 (평균 ± SD) | 탄성 ± SE |')
w('|---|---|---|---|')
for a_, v in dc['by_arm'].items():
    w(f"| {a_} | {v['n']} | {100 * v['share_mean']:.1f}% ±{100 * (v['share_sd'] or 0):.1f} | {v['elasticity_alive']:.3f} ±{v['se']:.3f} |")
w('')
fi = D['fits']; at = D['at']
def _rng(t, arms_):
    v = [at[t][x]['dead_ch'] for x in arms_ if x in at.get(t, {})]
    return (min(v), max(v)) if v else (None, None)
oth = [x for x in at.get('240000', {}) if x != INTER]
lo240, hi240 = _rng('240000', oth); lo340, hi340 = _rng('340000', [x for x in at.get('340000', {}) if x != INTER])
w(f"- **inter 는 같은 스파이크에서 죽은 채널이 늘 가장 많다** (340K: {at['340000'][INTER]['dead_ch']:.0f} vs 다른 팔 {lo340:.0f}~{hi340:.0f}; "
  f"240K: {at['240000'][INTER]['dead_ch']:.0f} vs {lo240:.0f}~{hi240:.0f}). 추가 사망은 conv2·conv3 에 몰리고 conv4·fc 는 팔 간 같다.")
w(f"- **그러나 inter 도 스파이크 감소의 약 {100 * dc['by_arm'][INTER]['share_mean']:.0f}% 만 채널 사망으로 얻는다** (나머지 ~{100 - 100 * dc['by_arm'][INTER]['share_mean']:.0f}% 는 살아남은 채널의 발화 감소). "
  f"다른 팔 {min(100 * v['share_mean'] for k_, v in dc['by_arm'].items() if k_ != INTER):.0f}~{max(100 * v['share_mean'] for k_, v in dc['by_arm'].items() if k_ != INTER):.0f}% 보다 크지만 주 경로가 아니다. "
  "**주간 정리의 '채널을 통째로 끄는 것 말고는 줄일 방법이 없다' 는 틀렸다.** 채널 안 가중이 균일하면 채널 전체의 발화가 같이 줄 뿐, 꺼지는 것은 일부다.")
if INTER in fi and 'b_above300' in fi[INTER]:
    acc_ = [f"{k_} {v['diff']:+.0f} (p={P(v['p_diff'])})" for k_, v in fi.items() if 'diff' in v]
    w(f"- 300K 아래 사망 가속 (아래 − 위 기울기): " + ', '.join(acc_) + ". **가속은 inter 만의 현상이 아니다** — ours_layer (w¹)·BPSR 도 가속한다. "
      "따라서 '없어도 되는 채널이 소진되면 필요한 채널이 죽기 시작한다' 는 그림은 모든 팔에 적용되고, inter 를 구분하는 것은 가속 여부가 아니라 **같은 스파이크에서 이미 더 많이 죽어 있다**는 것(곡선이 왼쪽 위로 평행 이동)이다.")
w("- 정확도 하락과의 동시성: inter 의 연속 ρ 점 표에서 정확도 하락(−0.1~−0.25)은 스파이크가 300K 아래로 내려가는 구간에서 시작하고 같은 구간에서 죽은 채널이 700 을 넘는다. "
  "그러나 ours_layer (w¹) 는 200K 에서 879 개가 죽어도 96.48 이고 inter 는 비슷한 수(856, 7e-3)에서 96.42 다 → **죽은 채널 '수' 만으로는 하락을 설명할 수 없다.** C 절의 팔 안 음의 상관과 합치면, 죽은 채널은 저스파이크 하락의 표지일 수는 있어도 충분한 설명은 아니다. inter 저스파이크 점이 n=1 이라 이 이상은 판단 불가.\n")


# ------------------------------------------------------------------------------------------
w('## E. 학습 동역학\n')
w('에폭별 값: val acc·s_count·λ·R·loss_ratio 는 train.log (eval 프로토콜), 침묵 비율은 reg_neuron_detail.csv 의 dead_neuron_ratio '
  '(**학습 배치 기준 표본 단위 침묵 비율** — npz 의 "테스트 전체에서 한 번도 안 쏜" 정의와 다르다; 단계 안 층 평균).\n')
CH = [1, 10, 30, 60, 100, 200, 280, 310]
w('| 팔 ρ | n | 서버 | 지표 | ' + ' | '.join(f'ep{e}' for e in CH) + ' |')
w('|---|---|---|---|' + '---|' * len(CH))
for nm, rec in E['arms'].items():
    s = rec['series']
    for fld, fmt in (('val_acc', '{:.2f}'), ('s_count', '{:,.0f}'), ('lam', '{:.2e}'), ('loss_ratio', '{:.4f}'), ('dead_conv1', '{:.3f}'), ('dead_conv2', '{:.3f}'), ('dead_conv3', '{:.3f}')):
        if fld not in s:
            continue
        vals = s[fld]['mean']
        if all(v is None for v in vals):
            continue
        w(f"| {nm} | {rec['n']} | {'/'.join(rec['servers'])} | {fld} | " + ' | '.join(('–' if vals[e - 1] is None else fmt.format(vals[e - 1])) for e in CH) + ' |')
w('')
w('침묵 비율이 (초기→마지막 10 에폭 평균) 변화의 50%·90% 에 처음 닿는 에폭:\n')
w('| 팔 ρ | n | conv1 t50 / t90 | conv2 t50 / t90 | conv3 t50 / t90 | conv4 t50 / t90 | 베스트 에폭 |')
w('|---|---|---|---|---|---|---|')
for nm, rec in E['arms'].items():
    t = rec['t_dead']
    w(f"| {nm} | {rec['n']} | " + ' | '.join(f"{t.get(s + '_t50', '–')} / {t.get(s, '–')}" for s in ('conv1', 'conv2', 'conv3', 'conv4')) + f" | {E['best_epoch'].get(nm)} |")
w('')
w('inter 3e-3 (n=3) 대비 정확도 격차, 에폭 구간별 (시드마다 구간 평균 → Welch). 양수 = inter 가 높다. 괄호 = 같은 구간 스파이크 비 (inter/상대 − 1).\n')
wins = [k for k in next(iter(E['gap_windows'].values())).keys() if k != 'n']
w('| 상대 (n) | ' + ' | '.join(f'ep {k}' for k in wins) + ' |')
w('|---|' + '---|' * len(wins))
for nm, g in E['gap_windows'].items():
    w(f"| {nm} ({g['n'][1]}) | " + ' | '.join(f"{g[k]['acc_gap']:+.2f} ±{f(g[k]['se'], 2)} p={P(g[k]['p'])} ({g[k]['spk_ratio_minus1_pct']:+.1f}%)" for k in wins) + ' |')
w('')
w('체크포인트 규칙대로 "에폭 e 까지의 최고 val acc" 의 inter − 상대: ' + '; '.join(
    f"{nm}: " + ', '.join(f"ep{e} {v:+.2f}" for e, v in g.items()) for nm, g in E['bestsofar_gap'].items()) + '\n')
w('- inter 3e-3 는 101~200 에폭에 다른 팔보다 **낮고** (ours_layer (w¹) 대비 p=0.03), 이점은 201 에폭 이후(학습률이 거의 0 으로 갈 때)에만 나타난다. '
  '초반 30 에폭 차이는 SE 가 ±0.8~1.1 로 판단 불가.')
w('- λ 는 loss-ratio 로 ρ 에 맞춰 자동 조절되어 loss_ratio 가 0.003 (inter 3e-3) / 0.001 (w¹ 1e-3) 로 일정. inter 의 R 이 같은 단계 다른 팔의 약 1.3~1.8 배 (채널 안 가중이 균일해 0 이 아닌 가중이 많음).')
w('- 학습 배치 기준 침묵 비율의 시간 경로는 팔 간 거의 같다 (conv2·conv3 의 t50 은 모두 13~17 에폭, conv4 는 10). 이 정의로는 "inter 가 채널을 일찍 죽인다" 를 볼 수 없다 — '
  '에폭별 채널 단위 침묵 기록이 없다 (한계).\n')

# ------------------------------------------------------------------------------------------
w('## F. 실제 제거 이득 (죽은 채널 물리 제거)\n')
w('`analysis/prune_measure.py`: 죽은 채널 판정 = **학습셋** 앞 1만 장, 정확도 = 테스트 1만 장 (순환 판정 차단). (a) residual 합류점 `*_out_n` 제외, (b) 모든 스파이킹 층. '
  'params/MACs 는 타임스텝 1회분 밀집 연산 기준.\n')
w('| 방법 | ρ | 시드 | 서버 | train loss | train acc | val loss | val acc | spikes | 죽은 ch (a) | Δacc (a) | params % (a) | MACs % (a) | 죽은 ch (b) | Δacc (b) | params % (b) | MACs % (b) | 테스트에서 쏜 제거 ch (b) |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
grp = {}
for r in sorted(F['runs'], key=lambda r: (-(r['spikes'] or 0), r['arm'] or '', r['seed'] or 0)):
    grp.setdefault((r['arm'], r['knob']), []).append(r)
order = sorted(grp, key=lambda k: -np.mean([r['spikes'] for r in grp[k]]))
for k in order:
    rs = sorted(grp[k], key=lambda r: r['seed'])
    for r in rs:
        w(f"| {r['arm']} | {r['knob']} | s{r['seed']} | {r['server']} | {f(r['train_loss'], 4)} | {f(r['train_acc'])} | {f(r['val_loss'], 4)} | {f(r['val_acc'])} | {r['spikes']:,.0f} | "
          f"{r['a']['n_dead']} | {r['a']['d_acc']:+.2f} | {r['a']['params_pct']:.1f} | {r['a']['macs_pct']:.1f} | {r['b']['n_dead']} | {r['b']['d_acc']:+.2f} | {r['b']['params_pct']:.1f} | {r['b']['macs_pct']:.1f} | {r['b']['n_dead_fired_test']} |")
    if len(rs) > 1:
        m = lambda g: np.mean([g(r) for r in rs]); s = lambda g: np.std([g(r) for r in rs], ddof=1)
        w(f"| **{k[0]} 평균** | {k[1]} | **n={len(rs)}** | | {m(lambda r: r['train_loss']):.4f} | {m(lambda r: r['train_acc']):.2f} | {m(lambda r: r['val_loss']):.4f} | **{m(lambda r: r['val_acc']):.2f}** | **{m(lambda r: r['spikes']):,.0f}** | "
          f"{m(lambda r: r['a']['n_dead']):.0f} ±{s(lambda r: r['a']['n_dead']):.0f} | {m(lambda r: r['a']['d_acc']):+.2f} | {m(lambda r: r['a']['params_pct']):.1f} | **{m(lambda r: r['a']['macs_pct']):.1f}** ±{s(lambda r: r['a']['macs_pct']):.1f} | "
          f"{m(lambda r: r['b']['n_dead']):.0f} ±{s(lambda r: r['b']['n_dead']):.0f} | {m(lambda r: r['b']['d_acc']):+.2f} | {m(lambda r: r['b']['params_pct']):.1f} | **{m(lambda r: r['b']['macs_pct']):.1f}** ±{s(lambda r: r['b']['macs_pct']):.1f} | {m(lambda r: r['b']['n_dead_fired_test']):.1f} |")
w('')
oc = F['offline_check']
if oc:
    err = max(max(abs(x['params_json'] - x['params_offline']), abs(x['macs_json'] - x['macs_offline'])) for x in oc)
    w(f"오프라인 계산 검증: 측정 JSON 의 층별 죽은 채널 수만으로 params·MACs % 를 다시 계산하면 최대 오차 {err:.4f}%p ({len(oc)}건) → 개수만으로 정해진다. "
      "그래서 npz(테스트 판정)로 모든 팔의 같은 대역 MACs 를 계산할 수 있다 (아래, 정확도 손실은 측정 안 함).\n")
w('| 대역 | 방법 ρ | n | val acc | spikes | 죽은 ch (테스트) | params % (b) | MACs % (b) | MACs % (a) |')
w('|---|---|---|---|---|---|---|---|---|')
for k, v in F['offline_npz'].items():
    if k == 'baseline':
        continue
    b_, a_ = k.split('|')
    w(f"| {b_} | {a_} | {v['n']} | {v['val_acc']:.2f} | {v['spikes']:,.0f} | {v['dead_ch'][0]:.0f}" + (f" ±{v['dead_ch'][1]:.0f}" if v['dead_ch'][1] else '') +
      f" | {v['params_pct_b'][0]:.1f} | {v['macs_pct_b'][0]:.1f}" + (f" ±{v['macs_pct_b'][1]:.1f}" if v['macs_pct_b'][1] else '') + f" | {v['macs_pct_a'][0]:.1f} |")
bb = F['offline_npz']['baseline']
w(f"| – | baseline | {bb['n']} | | | {bb['dead_ch'][0]:.0f} ±{bb['dead_ch'][1]:.0f} | | {bb['macs_pct_b'][0]:.1f} ±{bb['macs_pct_b'][1]:.1f} | |")
w('')
for key in ('ancova_macs_250-420K', 'ancova_macs_180-420K'):
    if key in F:
        c = F[key]
        w(f"- 같은 스파이크에서 MACs 제거 % (b) 의 팔 차이, ANCOVA {key.split('_')[-1]} (n={c['n']}, inter {c['n_arm'].get(INTER)}): "
          + ', '.join(f"{a} {ci(c['coef'][a], 1)} p={P(c['coef'][a]['p'])}" for a in c['others']) + ' (음수 = inter 가 더 많이 제거).')
w('')
# ---- F 마무리 (26-10-05): inter 실측 행 해석 + 같은 스파이크 MACs 통계 ----
ds_ = F['dacc_summary']
w('### F1. inter 실측과 Δacc\n')
w(f"- 실측 {ds_['n_runs']}런 (inter {ds_['n_inter']}: 2e-3·3e-3 각 s1–s3, 8e-3 s1; inter s1 은 23, 나머지 138 런) × 시나리오 2개, **Δacc 는 전부 0.00** (|Δacc| 최대 {ds_['max_abs_dacc']:.2f}%p, 0 아닌 것 {ds_['n_nonzero']}건). "
  f"학습셋 판정 집합 = 테스트 판정 집합 = npz 판정 집합이 {ds_['n_meas_eq_npz']}/{ds_['n_runs']} 런에서 개수까지 같고, 제거 채널이 테스트에서 쏜 횟수 합 {ds_['fired_test_total']}. "
  "즉 제거된 채널은 테스트에서 한 번도 안 쏘므로 Δacc = 0 은 구조상 정해진다 — **Δacc 로는 팔을 가를 수 없다.** 팔을 가르는 것은 같은 스파이크에서 몇 채널(→ MACs) 을 죽였는가뿐이다.")
w(f"- 측정 스크립트의 제거 전 정확도와 원장 val acc 의 차이 최대 {ds_['max_abs_accbase_minus_ledger']:.2f}%p (CPU 추론 vs 학습 중 평가; 시드 SD 0.10~0.13 보다 작다).\n")
w('### F2. ~340K 대역: inter 3e-3 (실측 n=3) − 같은 대역 다른 팔 (npz 테스트 판정)\n')
w('양수 = inter 가 더 많이 제거. Welch t, 스파이크 보정 없음 — inter 쪽이 스파이크가 더 많아(스파이크가 많을수록 죽은 채널이 적다) 이 비교는 inter 에 불리한 방향이다. '
  '다른 팔의 Δacc 는 측정하지 않았지만 F1 에 따라 테스트 판정 제거는 Δacc = 0 이다.\n')
w('| 비교 팔 | n (inter / 팔) | Δspikes | Δval acc %p | ΔMACs % (b) [95% CI] p | ΔMACs % (a) [95% CI] p | Δparams % (b) [95% CI] p |')
w('|---|---|---|---|---|---|---|')
for nm, v in F['band340'].items():
    def cc(x, d=1):
        return f"{x['diff']:+.{d}f} [{x['lo']:+.{d}f}, {x['hi']:+.{d}f}] p={P(x['p'])}" if 'p' in x else f"{x['diff']:+.{d}f} (n≤2, 판단 불가)"
    w(f"| {nm} | {v['n_inter']} / {v['n_other']} | {v['spikes']['diff'] / 1e3:+,.1f}K | {v['val_acc']['diff']:+.2f} | {cc(v['macs_pct_b'])} | {cc(v['macs_pct_a'])} | {cc(v['params_pct_b'])} |")
w('')
w('### F3. 같은 스파이크에서 inter 가 MACs 를 더 줄이는가 (스파이크 보정 회귀, npz 테스트 판정 전 런)\n')
w('모형 y = a + b·ln(spikes) + c·[다른 팔] (다른 팔 = 곡선 비교 다섯 팔 합동). 아래는 부호를 뒤집어 **inter − 다른 팔** (양수 = inter 가 더 많이 제거). '
  '기울기가 팔마다 다를 수 있어(아래 "기울기 차이 p") 기울기를 따로 둔 모형에서 고정 스파이크 점의 대비도 함께 적는다. 서버(23) 더미를 넣어도 대비는 바뀌지 않는다.\n')
w('| 구간 | n (inter) | 지표 | 공통 기울기 대비 [95% CI] p | 서버 보정 대비 p | 기울기 차이 p | 기울기 따로: 고정 스파이크 대비 [95% CI] p |')
w('|---|---|---|---|---|---|---|')
neg = lambda c: f"{-c['est']:+.1f} [{-c['hi']:+.1f}, {-c['lo']:+.1f}]"
for band, v in F['pooled_macs'].items():
    for yk, lab in (('macs_pct_b', 'MACs % (b)'), ('macs_pct_a', 'MACs % (a)'), ('params_pct_b', 'params % (b)'), ('dead_ch', '죽은 ch')):
        x = v[yk]; d = 0 if yk == 'dead_ch' else 1
        ng = lambda c: f"{-c['est']:+.{d}f} [{-c['hi']:+.{d}f}, {-c['lo']:+.{d}f}]"
        w(f"| {band} | {v['n']} ({v['n_inter']}) | {lab} | {ng(x['others_minus_inter'])} p={P(x['others_minus_inter']['p'])} | {-x['with_server']['est']:+.{d}f} p={P(x['with_server']['p'])} | "
          f"{P(x['slope_diff']['p'])} | " + '; '.join(f"@{k} {ng(c)} p={P(c['p'])}" for k, c in x['at'].items()) + ' |')
w('')
for band, v in F['pooled_macs'].items():
    w(f"- {band}: 팔별 n = " + ', '.join(f"{a} {n}" for a, n in v['n_arm_detail'].items() if n) +
      f"; 스파이크 범위 inter {v['spk_range_inter'][0] / 1e3:,.0f}–{v['spk_range_inter'][1] / 1e3:,.0f}K, 다른 팔 {v['spk_range_others'][0] / 1e3:,.0f}–{v['spk_range_others'][1] / 1e3:,.0f}K.")
ma = F['measured_ancova']
w(f"- 참고 — 실측 JSON 만 (Δacc 가 측정된 {ma['b']['n']}런, inter {ma['b']['n_arm'].get(INTER)}): MACs (b) inter − 다른 팔 {neg(ma['b']['others_minus_inter'])} p={P(ma['b']['others_minus_inter']['p'])}, "
  f"(a) {neg(ma['a']['others_minus_inter'])} p={P(ma['a']['others_minus_inter']['p'])}. 유의하지 않은 것은 스파이크 범위가 거의 안 겹치기 때문이다 "
  f"(inter {ma['spk_range_inter'][0] / 1e3:,.0f}–{ma['spk_range_inter'][1] / 1e3:,.0f}K, 다른 팔 {ma['spk_range_others'][0] / 1e3:,.0f}–{ma['spk_range_others'][1] / 1e3:,.0f}K; 겹치는 inter 는 8e-3 s1 한 런) — 이 회귀는 대비를 식별하지 못한다. 대비는 npz 전 런 회귀(위 표)가 맡는다.\n")
p250 = F['pooled_macs']['250-420K']['macs_pct_b']; p300 = F['pooled_macs']['300-420K']['macs_pct_b']; p180 = F['pooled_macs']['180-420K']['macs_pct_b']
b340 = F['band340']
w(f"**F 결론.** 죽은 채널 제거는 모든 팔에서 정확도를 바꾸지 않으므로(Δacc 0.00, 실측 {ds_['n_runs']}런) 이득은 MACs 만으로 비교된다. "
  f"두 팔 집합의 스파이크 범위가 겹치는 300~420K 에서 inter 는 다른 팔 합동보다 같은 스파이크에서 MACs (b) 를 {-p300['others_minus_inter']['est']:.1f}%p 더 제거한다 "
  f"(n={F['pooled_macs']['300-420K']['n']}, inter {F['pooled_macs']['300-420K']['n_inter']}, 95% CI [{-p300['others_minus_inter']['hi']:.1f}, {-p300['others_minus_inter']['lo']:.1f}], p={P(p300['others_minus_inter']['p'])}; "
  f"기울기를 따로 두면 340K 에서 {-p300['at']['340K']['est']:+.1f}%p, p={P(p300['at']['340K']['p'])}). "
  f"180~420K (n={F['pooled_macs']['180-420K']['n']}, inter {F['pooled_macs']['180-420K']['n_inter']}) 에서도 {-p180['others_minus_inter']['est']:+.1f}%p, p={P(p180['others_minus_inter']['p'])}. "
  f"inter 쪽 기울기가 더 가팔라(기울기 차이 p={P(p180['slope_diff']['p'])}, 180~420K) 차이는 스파이크가 낮을수록 커진다: 340K {-p180['at']['340K']['est']:+.1f} → 250K {-p180['at']['250K']['est']:+.1f} → 200K {-p180['at']['200K']['est']:+.1f}%p "
  f"(200K 의 inter 는 8e-3 s1 한 런뿐이라 그 끝은 불확실). 서버(23) 보정은 대비를 바꾸지 않는다. "
  f"~340K 에서 inter 3e-3 (n=3) 의 MACs (b) 는 개별 팔과 비교하면 BPSR 1e-10 보다만 유의하게 크고 "
  f"({b340['BPSR 1e-10']['macs_pct_b']['diff']:+.1f}%p, p={P(b340['BPSR 1e-10']['macs_pct_b']['p'])}), ours_layer (w¹)·ours_intra_ch (w¹) 와는 n=3 대 n=3 으로 구분되지 않으며 plain L2 1e-7 은 n=2 라 판단 불가. "
  f"절대 크기: baseline 자신의 죽은 채널이 이미 MACs {F['offline_npz']['baseline']['macs_pct_b'][0]:.1f}% (n={F['offline_npz']['baseline']['n']}) 이므로 규제가 더한 몫은 inter 3e-3 에서 "
  f"{F['offline_npz']['~340K|ours_inter_ch 3e-3']['macs_pct_b'][0] - F['offline_npz']['baseline']['macs_pct_b'][0]:.1f}%p 다.\n")

# ------------------------------------------------------------------------------------------
w('## G. 안 된 시도의 통계\n')
w(f"각 점을 해당 기준 팔의 단조 곡선(같은 스파이크) 과 비교. 잔차 = 점 − 기준 곡선, SE = √(σ²/n + 기준 곡선 부트스트랩 분산), σ = 시드 SD (C10 {SD['C10_pooled']:.3f}, C100 {SD['C100_pooled']:.3f}). "
  "기준 곡선 범위 밖이면 비교하지 않는다 (외삽 안 함). '같은 ρ' 열은 참고.\n")
w('| 데이터 | 시도 | 런 키 | ρ | 기준 팔 | n | 시드 (서버) | train loss | train acc | val loss | val acc | spikes | 기준 곡선 @같은 스파이크 | 잔차 %p | z | 같은 ρ 기준 (n) |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for x in G['rows']:
    sr = x['same_rho']
    w(f"| {x['ds']} | {x['variant']} | {x['key']} | {x['knob']} | {x['ref_arm']} | {x['n']} | " + ', '.join(f"s{s}({sv})" for s, sv, _, _ in x['indiv']) +
      f" | {x['train_loss']:.4f} | {x['train_acc']:.2f} | {x['val_loss']:.4f} | {x['val_acc']:.2f} | {x['spikes']:,.0f} | "
      f"{f(x['ref_pred'])}" + (f" ±{f(x['ref_pred_se'], 2)}" if x['ref_pred'] is not None else '') + f" | {f(x['resid'], 2, True)} | {f(x['z'], 2, True)} | "
      + (f"{sr['val_acc']:.2f} @{K(sr['spikes'])} ({sr['n']})" if sr else '–') + ' |')
w('')
w('개별 시드 (n≥2 묶음): ' + '; '.join(f"{x['key']} {x['knob']}: " + ', '.join(f"s{s}({sv}) {a:.2f}@{K(sp)}" for s, sv, a, sp in x['indiv'])
                                   for x in G['rows'] if x['n'] >= 2) + '\n')
w('묶음 요약 (z 가 있는 점만, 역분산 가중 평균 잔차; Stouffer z):\n')
w('| 묶음 | 점 수 | 런 수 | 평균 잔차 %p [95% CI] | Stouffer z | p |')
w('|---|---|---|---|---|---|')
for k, v in G['pooled'].items():
    w(f"| {k} | {v['k']} | {v['n_runs']} | {f(v['mean_resid_ivw'], 3, True)} [{f(v['ci'][0], 3, True)}, {f(v['ci'][1], 3, True)}] | {f(v['stouffer_z'], 2, True)} | {P(v['p'])} |")
w('')
st = G['st30_summary']; wt = st['welch_inter3e-3']
w(f"**st30 상세.** inter 3e-3: st30 {wt['st30_mean']:.2f} ±{wt['st30_sd']:.2f} (n={wt['n_st30']}) vs 기본 {wt['base_mean']:.2f} ±{wt['base_sd']:.2f} (n={wt['n_base']}), "
  f"차 {wt['diff']:+.3f}%p (95% CI [{wt['ci95'][0]:+.2f}, {wt['ci95'][1]:+.2f}], Welch p={P(wt['p'])}); 스파이크 {wt['spk_diff_pct']:+.1f}% (p={P(wt['p_spk'])}). "
  f"S30/S1 비 {st['s30_ratio']['st30']:.3f} vs {st['s30_ratio']['base']:.3f} (30 에폭까지 규제가 없어 초반 스파이크가 높다).")
fs = st['first_seed_signal']; rp = st['replication_seeds_inter3e-3']
w(f"- 첫 시드 신호: 세 팔 s1 의 '같은 ρ 기본 대비' 차 {', '.join(f'{d:+.2f}' for d in fs['diffs'])} (평균 {fs['mean']:+.3f}, 각 SE {fs['se_each']:.3f}, 합친 z={fs['z_pooled']:.2f}). "
  f"효과 0 에서 세 값의 평균이 이만큼 이상일 확률 {st['p_mean3_ge_obs_under_null']:.2f}, 셋 모두 +0.08 이상일 확률 {st['p_all3_ge_0.08_under_null']:.3f}.")
w(f"- 재현 시드 (inter 3e-3 s2, s3 — 첫 시드 결과를 보고 돌린 것이라 선택 편향이 없다): 같은 ρ 대비 {', '.join(f'{d:+.2f}' for d in rp['diffs'])} (평균 {rp['mean_diff']:+.3f}). "
  "첫 시드의 +0.11 은 '좋게 나온 시드를 보고 반복을 결정' 한 승자의 저주와 맞는다. 재현 시드만으로 추정한 효과는 0 근처다.")
w('- 셋 모두 +0.08 이상일 확률이 0.02 로 작지만, s1 세 런은 모두 서버 138·같은 시기라 독립이 아닐 수 있고, 이후 inter 재현에서 +0.11/−0.14 로 갈렸다. 합친 판정은 위 묶음 표 (st30 6점 평균 잔차, CI 0 포함).\n')

# ------------------------------------------------------------------------------------------
w('## H. 견고성\n')
w('### H1. 체크포인트 선택 (val = test 에서 최고 에폭을 고른 낙관 편향)\n')
w('| 정확도 정의 | 대역 | n (inter) | ' + ' | '.join(['ours_intra_ch', 'ours_intra_ch (w¹)', 'ours_layer (w¹)', 'plain L2', 'BPSR']) + ' | inter 대 나머지 | 잔차 SD |')
w('|---|---|---|---|---|---|---|---|---|---|')
for k, c in H['checkpoint'].items():
    mode, band = k.split('|')
    lab = {'best': '베스트 에폭 (원장)', 'last': '마지막 에폭', 'last10': '마지막 10 에폭 평균'}[mode]
    cells = [(f"{f(c['coef'][a]['est'], 3, True)} (p={P(c['coef'][a]['p'])})" if a in c['coef'] else '–') for a in ['ours_intra_ch', 'ours_intra_ch (w¹)', 'ours_layer (w¹)', 'plain L2', 'BPSR']]
    io = c['inter_vs_pooled_others']
    w(f"| {lab} | {band} | {c['n']} ({c['n_arm'].get(INTER)}) | " + ' | '.join(cells) + f" | {ci(io)} p={P(io['p'])} | {c['resid_sd']:.3f} |")
w('')
w('베스트 − 마지막 10 에폭 평균 (낙관 폭, %p): ' + ', '.join(f"{a} {v['mean']:.3f} ±{v['sd']:.3f} (n={v['n']})" for a, v in H['best_minus_last10'].items()) + '\n')
w('- 낙관 폭은 팔 간 같다 (0.15~0.18). 그런데 inter 의 높은 구간 이점은 마지막 에폭·마지막 10 에폭 평균에서 줄어든다 → 이점의 일부가 "최고점 에폭의 잡음" 에 실려 있다.\n')
w('### H2. 서버\n')
sv = H['server']
w(f"- baseline C10: 138 {sv['baseline_C10']['138'][0]:.2f} (n={sv['baseline_C10']['138'][1]}) vs 23 {sv['baseline_C10']['23'][0]:.2f} (n={sv['baseline_C10']['23'][1]}), 차 {sv['baseline_C10']['diff_23_minus_138']:+.3f} (p={P(sv['baseline_C10']['p'])}).")
fe = sv['within_group_fe']
w(f"- 두 서버가 섞인 (팔, ρ) 묶음 {fe['groups']}개, n={fe['n']} 의 고정효과: 23 − 138 = {fe['est_23_minus_138']:+.3f} ±{fe['se']:.3f}%p (p={P(fe['p'])}); 스파이크 {sv['within_group_fe_lnspk']['est_23_minus_138_pct']:+.2f}%.")
for k in ('ancova_with_server_250-420K', 'ancova_with_server_180-420K'):
    c = sv[k]
    w(f"- A1 에 서버 항 추가 ({k.split('_')[-1]}): 서버 {ci(c['coef']['server23'])} p={P(c['coef']['server23']['p'])}; 팔 오프셋 " + ', '.join(f"{a} {f(c['coef'][a]['est'], 3, True)}" for a in c['others']) + ' — 바뀌지 않음.')
w('')
w('### H3. 스파이크 프로토콜 (원장 val/test eval vs npz 테스트 추론)\n')
w('npz / 원장 스파이크 비: ' + ', '.join(f"{a} {v['mean']:.3f}" + (f" ±{v['sd']:.3f}" if v['sd'] else '') + f" (n={v['n']})" for a, v in H['protocol_ratio_npz_over_ledger'].items()) + '\n')
for k in [k for k in H if k.startswith('protocol_') and k != 'protocol_ratio_npz_over_ledger']:
    a_, b_ = H[k]['ledger_spk'], H[k]['npz_spk']
    w(f"- {k.split('_')[1]} (npz 있는 런 n={a_['n']}): 원장 스파이크 축 " + ', '.join(f"{x} {f(a_['coef'][x]['est'], 3, True)}" for x in a_['others'])
      + ' / npz 스파이크 축 ' + ', '.join(f"{x} {f(b_['coef'][x]['est'], 3, True)}" for x in b_['others']) + '.')
w('- 비는 팔마다 1.05~1.11 로 다르지만 (inter 가 가장 낮은 편), 같은 런 집합에서 축을 바꿔도 오프셋은 0.005%p 안에서 같다 → 결론에 영향 없음.\n')

# ------------------------------------------------------------------------------------------
w('## 그림으로 만들 것 (데이터 위치: `analysis/inter_deep_261005.json`)\n')
w('1. **정확도 × 스파이크 + 단조 곡선 + 교차 띠**: `A.curves_C10.curves.<팔>.points` (ρ 점 평균, n) 와 `.iso_spikes/.iso_acc` (단조 곡선), 교차점 `A.curves_C10.crossing` (중앙값·CI 를 세로 띠로). 개별 런은 `A.A1.C10_180-420K.points`.')
w('2. **같은 스파이크 차이 (forest plot)**: `A.curves_C10.at_spikes.<스파이크>.diff_inter_minus.<팔>.pooled` (est, lo, hi) — 340/300/240/200K 네 줄.')
w('3. **죽은 채널 × 스파이크 (팔별)**: `D.runs[]` (spikes, dead_ch, stage_dead, val_acc); 분절 기울기 `D.fits`. 오른쪽 축 또는 아래 패널에 val acc.')
w('4. **정확도 잔차 vs 죽은 채널 / 산 채널 안 지니 (팔 색)**: `C.250-420K.rows[]` (resid, dead_ch, alive_ch_gini).')
w('5. **학습 동역학**: `E.arms.<팔 ρ>.series.{val_acc,s_count,dead_conv2}.mean` (에폭 1~310), 격차 구간 `E.gap_windows`.')
w('6. **제거 이득**: `F.runs[]` (macs_pct, d_acc) 와 `F.offline_npz` — x=스파이크, y=MACs 제거 %.\n')

# ------------------------------------------------------------------------------------------
w('## [LIMITATION]\n')
w('- **inter 의 저스파이크 점(5e-3 이하)은 전부 n=1.** 240K·200K 의 하락 크기(−0.16~−0.24)와 교차점 위치는 이 점들에 기대고 있다. pooled 부트스트랩이 시드 산포를 넣었지만 재현 없이는 확정 아님.')
w('- **대역(320~420K)은 사후 선택이다.** 주간 정리가 inter 가 좋아 보이는 구간을 골랐고, 그 구간 단일 대비 p≈0.03 은 다중 비교·대역 선택을 반영하지 않는다. 미리 정한 대역(250~420K)에서는 p≈0.12.')
w('- 정확도는 val=test 1만 장에서 고른 베스트 에폭이다. 독립 검증셋이 없어 낙관 편향의 절대 크기(0.15~0.18%p)는 모든 팔에 있고, 팔 간 차이에는 H1 의 마지막 에폭 분석으로만 대응했다.')
w('- C 절 상관은 관찰이다. 팔 안 상관은 시드 간 차이에서 나오므로 "죽은 채널을 강제로 늘리면 정확도가 떨어진다" 는 인과로 읽을 수 없다.')
w('- E 절의 침묵 비율은 학습 배치·표본 단위 정의라 채널 사망 시점을 직접 보지 못한다. 에폭별 채널 단위 기록이 필요하다.')
w('- npz·제거 측정은 CPU 로 새로 뽑은 런을 포함한다 (아래 경로). GPU 와 수치 차이는 정확도 0.01%p 수준으로 예상되나 이번에 대조하지 않았다.')
w('- C100 은 inter n=1 × 4점이라 어떤 비교도 판단 불가. CIFAR-10·R19·T=4 밖으로 일반화하지 않는다.')
w('')
w('<!-- RUNS_APPENDIX -->')

# ------------------------------------------------------------------------------------------
# 부록: A 에 쓴 런 전부 (개별 + 평균 행)
w('\n## 부록 A. 곡선 비교에 쓴 런 전부 (C10·C100, 개별값 + 묶음 평균)\n')
w('| 데이터 | 방법 | ρ | 시드 | 서버 | train loss | train acc | val loss | val acc | spikes | S30/S1 |')
w('|---|---|---|---|---|---|---|---|---|---|---|')
NAME = O['meta']['name_map']
for ds in ('C10', 'C100'):
    for key in ('base', 'ours_ch', 'prop', 'ours_w1', 'ours_layer_w1', 'ours_layer', 'l2', 'bpsr'):
        grp = {}
        for r in INV.values():
            if r['ds'] == ds and r['key'] == key and r['seed'] < 900 and r.get('status') in ('done', 'excluded', 'stale'):
                grp.setdefault(r['knob'], []).append(r)
        for kn in sorted(grp, key=lambda k: -np.mean([r.get('spikes') or 0 for r in grp[k]])):
            rs = sorted(grp[kn], key=lambda r: r['seed'])
            ok = [r for r in rs if r.get('status') == 'done' and r.get('epochs') == 310 and not r.get('exclude')]
            for r in rs:
                note = '' if r in ok else f" ✗ {r.get('exclude_reason') or r.get('status')} ({r.get('epochs')}ep)"
                w(f"| {ds} | {NAME[key]} | {kn} | s{r['seed']}{note} | {r.get('server')} | {f(r.get('train_loss'), 4)} | {f(r.get('train_acc'))} | {f(r.get('val_loss'), 4)} | {f(r.get('val_acc'))} | "
                  f"{(r.get('spikes') or 0):,.0f} | {f(r.get('s30'), 3)} |")
            if len(ok) > 1:
                m = lambda k: np.mean([r[k] for r in ok]); s = lambda k: np.std([r[k] for r in ok], ddof=1)
                w(f"| **{ds}** | **{NAME[key]} 평균** | {kn} | **n={len(ok)}** | | {m('train_loss'):.4f} | {m('train_acc'):.2f} | {m('val_loss'):.4f} | **{m('val_acc'):.2f}** ±{s('val_acc'):.2f} | **{m('spikes'):,.0f}** | |")
w('\n✗ = 평균에서 뺌 (배제 규칙 S30/S1<0.19, 미완주 등). 행은 남긴다.\n')

w('## 부록 D. npz 런 단위 (죽은 채널 곡선의 점)\n')
w('| 방법 | ρ | 시드 | 서버 | train loss | train acc | val loss | val acc | spikes | test 스파이크 | test acc | 죽은 ch | conv1/2/3/4/fc 죽은 ch | 침묵 % | 50% 집중도 % | 산 채널 안 지니 | MACs % (b) | npz |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for r in sorted(D['runs'], key=lambda r: (r['arm'], -r['spikes'])):
    sd_ = r['stage_dead']
    w(f"| {r['arm']} | {r['knob']} | s{r['seed']} | {r['server']} | {f(r['train_loss'], 4)} | {f(r['train_acc'])} | {f(r['val_loss'], 4)} | {f(r['val_acc'])} | {r['spikes']:,.0f} | {r['test_spk']:,.0f} | {r['test_acc']:.2f} | "
      f"{r['dead_ch']} | {sd_['conv1']}/{sd_['conv2']}/{sd_['conv3']}/{sd_['conv4']}/{sd_['fc']} | {r['silent_pct']:.2f} | {r['half_share_pct']:.2f} | {r['alive_ch_gini']:.3f} | {r['macs_pct_b']:.1f} | "
      + ('새로' if '/deep1005/' in (r.get('npz') or '') else '') + ' |')

w('\n## 배제·미완주 (C10·C100, 곡선 팔)\n')
w('| tag | 서버 | 상태 | 에폭 | val acc | spikes | S30/S1 | 사유 |')
w('|---|---|---|---|---|---|---|---|')
for x in O['excluded']:
    w(f"| {x['tag']} | {x['server']} | {x['status']} | {x['epochs']} | {f(x['val_acc'])} | {(x['spikes'] or 0):,.0f} | {f(x['s30'], 3)} | {x['reason']} |")

open(os.path.join(ROOT, 'analysis', 'inter_deep_261005.md'), 'w').write(('\n'.join(L) + '\n').replace('=<0.001', '<0.001'))
print('md lines', len(L))
