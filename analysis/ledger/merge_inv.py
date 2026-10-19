"""두 서버 수집 CSV + 큐 상태 → 실험 추적판용 JSON."""
import csv, json, os, re, sys, time
D = sys.argv[1]
KEY2ARM = {'base':'baseline','prop':'ours_intra_ch','ours_w1':'ours_w1','ours_layer':'ours_layer',
  'ours_layer_w1':'ours_layer_w1','ours_ch':'ours_inter_ch','ours_ch_b7':'ours_ch_b7',
  'ours_ch_b4':'ours_ch_b4','ours_ch_taylor':'ours_inter_ch (taylor)','l2':'plain L2','l2_lr':'plain L2 + rho','bpsr':'BPSR','sm':'1-softmax',
  'abl_novmem':'- vmem','abl_nofinal':'- final_step','abl_noinv':'- 1- inversion',
  'abl_noinv_wc':'- 1- inversion (wc)','abl_nolr':'- loss-ratio',
  'lw1_novmem':'lw1 - vmem','lw1_nofinal':'lw1 - final_step','lw1_nolr':'lw1 - loss-ratio','lw1_noinv':'lw1 - 1- inversion',
  # 26-10-04 Suetake 대조군, ours_inter_ch(run 키 ours_ch) 어블레이션
  'suetake':'Suetake','ch_novmem':'- vmem (inter)','ch_nofinal':'- final_step (inter)',
  'ch_noinv':'- 1- inversion (inter)','ch_nolr':'- loss-ratio (inter)',
  # 09-30 규제 시작 지연 (loss-ratio start_ep=30/60)
  'ours_w1_st30':'ours_w1 st30','ours_w1_st60':'ours_w1 st60',
  'ours_ch_st30':'ours_inter_ch st30','ours_ch_st60':'ours_inter_ch st60',
  'ours_layer_w1_st30':'ours_layer_w1 st30','ours_layer_w1_st60':'ours_layer_w1 st60',
  # 26-10-05 끝 구간 규제 해제 (loss-ratio end_ep=40/30)
  'ours_ch_end40':'ours_inter_ch end40','ours_ch_end30':'ours_inter_ch end30',
  # 26-10-05 inter x intra 곱 가중 (maxnorm_group='channel_x_within')
  'ours_chxwc':'ours_inter_ch × intra',
  # 26-10-10 ours_ch 에 서로게이트 지정 -- 서로게이트는 split_sg 가 꼬리표로 붙인다 (예 'ours_inter_ch (ahf w0.5 h0.75)')
  'ours_ch_sg':'ours_inter_ch'}
# vggdvscm (10-09) = 2025-11 증강 레시피 복원판 (클립 단위 NDA + CutMix). vggdvs (3월 레시피, 74%) 는 같은 DVS 탭에
# '(3월 레시피)' 꼬리표로 남긴다 -- 정규식에서 vggdvscm 이 vggdvs 보다 앞에 와야 한다.
# 26-10-15 run_paper 조합 vggdvsf0 (vggdvs config + 첫 프레임 반복 입력), vggdvscmtd (vggdvscm + 단계별 BN:
# BN 통계는 T*B 에서, 정규화는 스텝별. flags.py bn_per_step) 추가. 긴 이름을 짧은 이름보다 앞에 둔다
# (vggdvscmtd > vggdvscm > vggdvsf0 > vggdvs). 팔 이름 꼬리표는 COMBO_SUFFIX.
# 26-10-10 base_sg 의 서로게이트 손잡이 (ahf|asym)-w<폭>-h<높이> 도 받는다 ('.'->'p' 변환 후, 예 ahf-w1p0-h0p75).
# 26-10-10 ours_ch_sg 의 '<rho>@<서로게이트>' (예 3e-3@ahf-w0p5-h0p75) 도 받는다. 원장 knob 칸에는 rho 만 남기고
# 서로게이트는 팔 이름 꼬리표로 붙인다 (split_sg). '@' 없는 기존 태그의 파싱은 그대로다.
NAME = re.compile(r'^(r19c10|r19c100|vggdvscmtd|vggdvscm|vggdvsf0|vggdvs)-(.+?)(?:-(-|[0-9p]+e-?[0-9]+|(?:ahf|asym)-w[0-9p]+-h[0-9p]+|[0-9p]+e-?[0-9]+@(?:ahf|asym)-w[0-9p]+-h[0-9p]+))?(-f0)?-s(\d+)$')  # -f0: 프레임0 반복 입력 (10-12 수동 런)
def split_sg(knob):
    """'3e-3@ahf-w0.5-h0.75' -> ('3e-3', ' (ahf w0.5 h0.75)'). '@' 가 없으면 (knob, '')."""
    rho, sep, sg = knob.partition('@')
    return (rho, ' (' + sg.replace('-', ' ') + ')') if sep else (knob, '')
DS = {'r19c10': 'C10', 'r19c100': 'C100', 'vggdvs': 'DVS', 'vggdvscm': 'DVS',
      'vggdvsf0': 'DVS', 'vggdvscmtd': 'DVS'}   # vggdvs = VGGSNN · CIFAR10-DVS (09-29)
OLD_DVS_SUFFIX = ' (3월 레시피)'
# 조합 -> 팔 이름 꼬리표. vggdvsf0 는 수동 런 vggdvs-*-f0-* 와 같은 이름이 되도록 ' (frame0) (3월 레시피)'.
COMBO_SUFFIX = {'vggdvs': OLD_DVS_SUFFIX, 'vggdvsf0': ' (frame0)' + OLD_DVS_SUFFIX, 'vggdvscmtd': ' (단계별 BN)'}
def f(x):
    try: return float(x)
    except: return None
runs = {}
for srv, fn in (('138','inv138.csv'),('23','inv23.csv')):
    for r in csv.DictReader(open(os.path.join(D, fn))):
        tag = os.path.basename(r['run_dir'].rstrip('/'))
        m = NAME.match(tag)
        if not m: continue
        ds, key, knob, f0, seed = m.groups()
        knob = knob or '-'   # 'vggdvscm-base-f0-s1' 처럼 knob 자리가 비는 수동 런
        knob, sgt = split_sg(knob.replace('p','.'))
        if f0: sgt += ' (frame0)'
        ep = int(f(r['epochs']) or 0)
        rec = dict(tag=tag, ds=DS[ds], key=key, arm=KEY2ARM.get(key, key) + sgt + COMBO_SUFFIX.get(ds, ''),
                   knob=knob, seed=int(seed), server=srv, epochs=ep,
                   completed=r['completed'] in ('True','1','true'),
                   train_loss=f(r['train_loss']), train_acc=f(r['train_acc']),
                   val_loss=f(r['val_loss']), val_acc=f(r['val_acc']), spikes=f(r['spikes']),
                   s30=f(r['S30_over_S1']), exclude=r['exclude'] in ('True','1','true'),
                   exclude_reason=r['exclude_reason'], arm_conf=r['arm'], finished=r.get('finished', ''))
        old = runs.get(tag)
        if old is None or rec['epochs'] > old['epochs']:
            runs[tag] = rec
# canus 러너는 상태칸에 GPU 대신 러너 pid 를 적는다 -> 돌고 있는 run_paper 명령에서 GPU 를 읽는다
import subprocess
GPU138 = {}
ROOT = '/home/kyccj/PycharmProjects/TensorFlow-SNNs'


def live_runs(lines, root=None):
    """프로세스 목록(ps -eo args 줄들) -> {런 태그: GPU 또는 None}.
    26-10-14: 큐를 거치지 않고 직접 띄운 런(run_paper 직접 실행, _ours/<tag>/main_sweep.py 수동 런)도
    '진행 중'으로 잡으려고 만들었다. 이전에는 큐의 run: 행만 보고 판정해서 이런 런이 '멈춤'으로 보였다.
    - main_sweep.py 경로(_paper/<tag>/ 또는 _ours/<tag>/)가 보이면 그 런은 살아 있다.
    - GPU 는 run_paper 명령의 --gpus 로, 없으면 (canus 로컬일 때만) 그 런의 config_sweep.py 에서 읽는다."""
    live = {}
    for line in lines:
        m = re.search(r'/(_paper|_ours)/([^/\s]+)/main_sweep\.py', line)
        if m:
            d, tag = m.groups()
            g = None
            if root:
                try:
                    g = re.search(r'CUDA_VISIBLE_DEVICES"\]\s*=\s*"(\d+)"',
                                  open(os.path.join(root, d, tag, 'config_sweep.py')).read()).group(1)
                except Exception:
                    pass
            live.setdefault(tag, g)
        m = re.search(r'run_paper\.py (\S+) (\S+) (\S+) --seeds (\d+) --gpus (\d+)', line)
        if m:
            cb, mt, kb, sd, g = m.groups()
            live[f"{cb}-{mt}-{kb.replace('.','p')}-s{sd}"] = g
    return live


try:
    LIVE138 = live_runs(subprocess.check_output(['ps','-eo','args'], text=True).splitlines(), ROOT)
except Exception:
    LIVE138 = {}
GPU138 = {t: g for t, g in LIVE138.items() if g is not None}
# sejong 프로세스 목록은 refresh.sh 가 ps23.txt 로 받아 둔다 (없으면 sejong 직접 런은 판정 못 함 -> 예전처럼 '멈춤')
try:
    LIVE23 = live_runs(open(os.path.join(D, 'ps23.txt'), errors='ignore').read().splitlines())
except Exception:
    LIVE23 = {}
qline = 0
# 큐 상태
Q = '/home/kyccj/PycharmProjects/TensorFlow-SNNs/_queue/QUEUE.tsv'
for l in open(Q):
    if l.startswith('#') or not l.strip(): continue
    c = l.rstrip('\n').split('\t')
    if len(c) < 6: continue
    st, pri, combo, key, knob, seed = c[0], c[1], c[2], c[3], c[4], c[5]
    land = c[6] if len(c) > 6 else ''
    tag = f"{combo}-{key}-{knob.replace('.','p')}-s{seed}"
    s = st.split(':')[0]
    qline += 1
    if combo not in DS: continue
    if tag not in runs and s in ('todo','run','hold'):
        qknob, sgt = split_sg(knob)
        runs[tag] = dict(tag=tag, ds=DS[combo], key=key,
                         arm=KEY2ARM.get(key,key) + sgt + COMBO_SUFFIX.get(combo, ''), knob=qknob, seed=int(seed), server='-',
                         epochs=0, completed=False)
    if tag in runs:
        r = runs[tag]; r['queue'] = s; r['qline'] = qline; r['land'] = land
        r['pri'] = int(pri) if pri.lstrip('-').isdigit() else 9
        if s == 'run':
            host, who = (st.split(':') + ['', ''])[1:3]
            r['server'] = '138' if host == 'canus' else '23'
            r['gpu'] = who[3:] if who.startswith('gpu') else GPU138.get(tag, '?')
for r in runs.values():
    q = r.get('queue','')
    if r['completed']:
        r['status'] = 'excluded' if r.get('exclude') else 'done'
    elif q == 'run':   r['status'] = 'running'
    elif r['tag'] in LIVE138 or r['tag'] in LIVE23:
        # 큐 행 없이 직접 띄운 런 (26-10-14). 서버는 프로세스가 보인 쪽, GPU 는 명령/설정에서 읽은 값.
        srv, live = ('138', LIVE138) if r['tag'] in LIVE138 else ('23', LIVE23)
        r['status'] = 'running'; r['server'] = srv
        r['gpu'] = live[r['tag']] if live[r['tag']] is not None else '?'
    elif q == 'todo':  r['status'] = 'todo'
    elif q == 'hold':  r['status'] = 'hold'      # 큐에 있지만 실행기가 집지 않는 후순위 (사람이 todo 로 바꿈)
    else:              r['status'] = 'stale'
    # 26-10-10 서로게이트 꼬리표는 collect_paper_v2 의 arm(설정 판정)에 없으므로 ours_ch_sg 는 꼬리표 뺀 이름과 비교한다
    if r['arm_conf' if 'arm_conf' in r else 'arm'] != (KEY2ARM['ours_ch_sg'] if r.get('key') == 'ours_ch_sg' else r['arm']) and r.get('arm_conf'):
        r['mismatch'] = r['arm_conf']
out = dict(generated=time.strftime('%Y-%m-%d %H:%M'), runs=sorted(runs.values(),
           key=lambda r:(r['ds'], r['arm'], f(r['knob']) or 0, r['seed'])))
json.dump(out, open(os.path.join(D,'inventory.json'),'w'), ensure_ascii=False, indent=0)
from collections import Counter
print('총', len(runs), dict(Counter(r['status'] for r in runs.values())))
mm = [r for r in runs.values() if r.get('mismatch')]
print('이름-설정 불일치', len(mm), [(r['tag'], r['mismatch']) for r in mm][:8])
