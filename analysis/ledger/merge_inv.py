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
  'ours_chxwc':'ours_inter_ch × intra'}
NAME = re.compile(r'^(r19c10|r19c100|vggdvs)-(.+)-(-|[0-9p]+e-?[0-9]+)-s(\d+)$')
DS = {'r19c10': 'C10', 'r19c100': 'C100', 'vggdvs': 'DVS'}   # vggdvs = VGGSNN · CIFAR10-DVS (09-29)
def f(x):
    try: return float(x)
    except: return None
runs = {}
for srv, fn in (('138','inv138.csv'),('23','inv23.csv')):
    for r in csv.DictReader(open(os.path.join(D, fn))):
        tag = os.path.basename(r['run_dir'].rstrip('/'))
        m = NAME.match(tag)
        if not m: continue
        ds, key, knob, seed = m.groups()
        ep = int(f(r['epochs']) or 0)
        rec = dict(tag=tag, ds=DS[ds], key=key, arm=KEY2ARM.get(key, key),
                   knob=knob.replace('p','.'), seed=int(seed), server=srv, epochs=ep,
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
try:
    for line in subprocess.check_output(['ps','-eo','args'], text=True).splitlines():
        m = re.search(r'run_paper\.py (\S+) (\S+) (\S+) --seeds (\d+) --gpus (\d+)', line)
        if m:
            cb, mt, kb, sd, g = m.groups()
            GPU138[f"{cb}-{mt}-{kb.replace('.','p')}-s{sd}"] = g
except Exception:
    pass
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
        runs[tag] = dict(tag=tag, ds=DS[combo], key=key,
                         arm=KEY2ARM.get(key,key), knob=knob, seed=int(seed), server='-',
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
    elif q == 'todo':  r['status'] = 'todo'
    elif q == 'hold':  r['status'] = 'hold'      # 큐에 있지만 실행기가 집지 않는 후순위 (사람이 todo 로 바꿈)
    else:              r['status'] = 'stale'
    if r['arm_conf' if 'arm_conf' in r else 'arm'] != r['arm'] and r.get('arm_conf'):
        r['mismatch'] = r['arm_conf']
out = dict(generated=time.strftime('%Y-%m-%d %H:%M'), runs=sorted(runs.values(),
           key=lambda r:(r['ds'], r['arm'], f(r['knob']) or 0, r['seed'])))
json.dump(out, open(os.path.join(D,'inventory.json'),'w'), ensure_ascii=False, indent=0)
from collections import Counter
print('총', len(runs), dict(Counter(r['status'] for r in runs.values())))
mm = [r for r in runs.values() if r.get('mismatch')]
print('이름-설정 불일치', len(mm), [(r['tag'], r['mismatch']) for r in mm][:8])
