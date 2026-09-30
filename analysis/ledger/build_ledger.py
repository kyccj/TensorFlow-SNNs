"""inventory.json -> 실험 원장 HTML (Artifact 로 게시).

실험 목록은 페이지에 박아 넣는다. 서버를 다시 긁을 때마다 이 스크립트로 다시 만들고
같은 파일 경로로 재게시하면 같은 주소가 갱신된다.
'노트 반영' 체크만 artifact db 의 notes/<tag> 문서에 산다 (폰에서 눌러도 남고, Claude 가
read_db 로 읽어 '노트에 없는 것'을 뽑는다).

쓰는 법:  python analysis/ledger/build_ledger.py <inventory.json> <out.html>
"""
import json, sys

src, out = sys.argv[1], sys.argv[2]
inv = json.load(open(src))
import datetime as dt, heapq
# 예상 시각 — 도는 런은 남은 에폭 x 에폭당 시간, 대기 런은 큐 순서(우선, 줄번호)대로
# 가장 먼저 비는 GPU 에 넣는 모의 배정. GPU 14장 (138 의 0~5 + 23 의 0~7, 138 의 6·7 은 juyun).
EPOCH_S = 345          # R19 에폭당 초. 09-23 로그 중앙값(329~369s)
SLOTS = 14
now = dt.datetime.strptime(inv['generated'], '%Y-%m-%d %H:%M')
fmt = lambda t: t.strftime('%m-%d %H:%M')
running = [r for r in inv['runs'] if r['status'] == 'running']
todo = sorted([r for r in inv['runs'] if r['status'] == 'todo'],
              key=lambda r: (r.get('pri', 9), r.get('qline', 0)))
free = []
for r in running:
    fin = now + dt.timedelta(seconds=max(0, (200 if r['ds'] == 'DVS' else 310) - (r['epochs'] or 0)) * EPOCH_S)
    r['eta'] = fmt(fin); free.append(fin)
free += [now] * max(0, SLOTS - len(running))
heapq.heapify(free)
for i, r in enumerate(todo, 1):
    st = heapq.heappop(free); fin = st + dt.timedelta(seconds=(200 if r['ds'] == 'DVS' else 310) * EPOCH_S)
    r['order'], r['start'], r['eta'] = i, fmt(st), fmt(fin)
    heapq.heappush(free, fin)

KEEP = ('tag', 'ds', 'arm', 'knob', 'seed', 'server', 'epochs', 'status', 'train_loss',
        'train_acc', 'val_loss', 'val_acc', 'spikes', 's30', 'exclude_reason',
        'gpu', 'pri', 'land', 'order', 'start', 'eta', 'finished')
runs = [{k: r.get(k) for k in KEEP} for r in inv['runs']]
import os
_qk = os.path.join(os.path.dirname(os.path.abspath(src)), 'qk.json')
qk = json.load(open(_qk)) if os.path.exists(_qk) else None
# QKFormer 학습 런도 결과 표(QKFormer 탭)에 넣는다. 이름 예: 'ours_layer_w1 ρ 1e-2 s43', 'baseline', 'plain L2 λ 3e-6 s44'
import re
if qk:
    for j in qk['jobs']:
        if j.get('kind') != '학습':
            continue
        m = re.match(r'^(.*?)(?: [ρλ] (\S+))?(?: s(\d+))?$', j['name'])
        arm, knob, seed = m.group(1), m.group(2) or '-', int(m.group(3) or 42)
        qds = 'QK'
        if arm.startswith('c100_'):                # QKFormer CIFAR-100 (09-29 baseline 반복)
            qds, arm = 'QK100', arm[5:]
        runs.append({'tag': f'{qds.lower()}-{arm.replace(" ", "_")}-{knob}-s{seed}', 'ds': qds, 'arm': arm, 'knob': knob,
                     'seed': seed, 'server': j.get('server', '138'), 'epochs': j.get('epochs') or 0,
                     'status': j['status'], 'val_acc': j.get('acc'), 'spikes': j.get('spikes'),
                     'gpu': j.get('gpu'), 'eta': j.get('eta'), 'finished': j.get('finished')})
# 실행기 목록 밖에서 돌린 QKFormer plain L2 스윕(l2_<λ>, 시드 42)도 대조군으로 붙인다
from pathlib import Path
_QR = Path('/home/kyccj/runs/qkformer_wta_rev')
_have = {r['tag'] for r in runs}
for lg in sorted(_QR.glob('l2_*.log')):
    nm = lg.stem
    mm = re.match(r'^l2_([0-9.e-]+)(?:_s(\d+))?$', nm)
    if not mm:
        continue                                   # l2_1e-6_NaN버그 등 제외
    tag = f'qk-plain_L2-{mm.group(1)}-s{mm.group(2) or 42}'
    if tag in _have:
        continue
    best = re.findall(r'Best metric: ([\d.]+)', lg.read_text(errors='ignore'))
    sj = _QR / f'spikes_{nm}.json'
    summ = _QR / nm / 'summary.csv'
    ep = sum(1 for _ in summ.open()) - 1 if summ.exists() else 0
    runs.append({'tag': tag, 'ds': 'QK', 'arm': 'plain L2', 'knob': mm.group(1), 'seed': int(mm.group(2) or 42),
                 'server': '138', 'epochs': ep, 'status': 'done' if ep >= 400 else 'stale',
                 'finished': dt.datetime.fromtimestamp(summ.stat().st_mtime).strftime('%Y-%m-%d %H:%M') if (summ.exists() and ep >= 400) else None,
                 'val_acc': float(best[-1]) if best else None,
                 'spikes': json.load(open(sj))['totals']['spikes_per_sample'] if sj.exists() else None})
if qk:                                      # 09-29 사용자 지시: 스파이크 측정 작업은 원장에 보이지 않는다
    qk['jobs'] = [j for j in qk['jobs'] if j.get('kind') != '측정']
_c100 = Path('/home/kyccj/runs/qkformer_baseline/c100_baseline.log')
if _c100.exists() and not any(r['tag'] == 'qk100-baseline---s42' for r in runs):
    _b = re.findall(r'Best metric: ([\d.]+)', _c100.read_text(errors='ignore'))
    _sj = _QR / 'spikes_c100_base.json'
    runs.append({'tag': 'qk100-baseline---s42', 'ds': 'QK100', 'arm': 'baseline', 'knob': '-', 'seed': 42,
                 'server': '138', 'epochs': 410, 'status': 'done', 'val_acc': float(_b[-1]) if _b else None,
                 'finished': dt.datetime.fromtimestamp(_c100.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
                 'spikes': json.load(open(_sj))['totals']['spikes_per_sample'] if _sj.exists() else None})
data = json.dumps({'generated': inv['generated'], 'runs': runs, 'qk': qk}, ensure_ascii=False,
                  separators=(',', ':'))

HTML = r"""<title>EIP 실험 원장</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap">
<style>
:root{
  --bg:#F2F5F6; --surface:#FFFFFF; --sunk:#E9EEF0; --ink:#14202A; --muted:#5A6A76; --rule:#D5DEE3;
  --accent:#0A6F7A; --accent-ink:#FFFFFF;
  --done:#1D7F53; --done-bg:#E2F2E9; --excl:#9A5F0E; --excl-bg:#FAEDD6;
  --run:#2152C4; --run-bg:#E3EAFB; --todo:#6F7B86; --todo-bg:#EDF1F3;
  --stale:#AF2F27; --stale-bg:#FAE3E1; --miss:#C0450F;
  --sans:"IBM Plex Sans KR","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  color-scheme:dark;
  --bg:#0E1418; --surface:#151E24; --sunk:#1B262D; --ink:#E4ECF0; --muted:#8C9DA8; --rule:#26343C;
  --accent:#3DB3BF; --accent-ink:#06282C;
  --done:#5CCB93; --done-bg:#16342A; --excl:#E7B25C; --excl-bg:#3A2B12;
  --run:#86A8F5; --run-bg:#1C2A4A; --todo:#8C98A2; --todo-bg:#1F2A31;
  --stale:#F08A80; --stale-bg:#3C1D1B; --miss:#F2955E;
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --bg:#0E1418; --surface:#151E24; --sunk:#1B262D; --ink:#E4ECF0; --muted:#8C9DA8; --rule:#26343C;
  --accent:#3DB3BF; --accent-ink:#06282C;
  --done:#5CCB93; --done-bg:#16342A; --excl:#E7B25C; --excl-bg:#3A2B12;
  --run:#86A8F5; --run-bg:#1C2A4A; --todo:#8C98A2; --todo-bg:#1F2A31;
  --stale:#F08A80; --stale-bg:#3C1D1B; --miss:#F2955E;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding-inline:20px;padding-block:28px 64px}
header h1{font-size:26px;font-weight:700;letter-spacing:-.01em;margin:0 0 4px;text-wrap:balance}
header p{margin:0;color:var(--muted);font-size:13px}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}

.strip{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 14px}
.stat{display:flex;align-items:baseline;gap:6px;padding:6px 12px;border-radius:999px;background:var(--surface);border:1px solid var(--rule);font-size:13px}
.stat b{font-family:var(--mono);font-weight:500;font-size:15px}
.stat i{width:9px;height:9px;border-radius:50%;display:inline-block;align-self:center}
.stat.miss{border-color:var(--miss)} .stat.miss b{color:var(--miss)}

.controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin-bottom:22px;padding:10px 12px;background:var(--surface);border:1px solid var(--rule);border-radius:10px}
.seg{display:inline-flex;border:1px solid var(--rule);border-radius:8px;overflow:hidden}
.seg button{font:inherit;font-size:13px;padding:6px 14px;border:0;background:transparent;color:var(--ink);cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--accent);color:var(--accent-ink)}
.seg button+button{border-left:1px solid var(--rule)}
label.tg{display:inline-flex;align-items:center;gap:6px;font-size:13px;cursor:pointer}
.note-state{margin-left:auto;font-size:12px;color:var(--muted)}

.queue{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:14px;margin-bottom:6px}
.qcol{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:12px 14px}
.qcol h2{font-size:14px;font-weight:600;margin:0 0 8px;display:flex;align-items:baseline;gap:8px}
.qcol h2 b{font-family:var(--mono);font-weight:500;color:var(--muted)}
.qcol h2 small{margin-left:auto;font-weight:400;font-size:11.5px;color:var(--muted)}
.qcol h2 .seg{margin-left:auto}
.qcol h2 .seg button{font-size:11.5px;padding:3px 9px}
.qk{grid-column:1/-1}
.qk .st{font-size:11px;padding:1px 7px;border-radius:9px;white-space:nowrap}
.qk .st.done{background:var(--done-bg);color:var(--done)} .qk .st.running{background:var(--run-bg);color:var(--run)}
.qk .st.todo{background:var(--todo-bg);color:var(--todo)} .qk .st.stale{background:var(--stale-bg);color:var(--stale)}
.qk .last{font-family:var(--mono);font-size:11px;color:var(--muted);margin:8px 0 0;overflow-wrap:anywhere}
table.q{width:100%;border-collapse:collapse;font-size:12.5px}
table.q th{font-weight:500;font-size:11px;color:var(--muted);text-align:left;padding:4px 6px;border-bottom:1px solid var(--rule)}
table.q td{padding:5px 6px;border-bottom:1px solid var(--rule);vertical-align:middle}
table.q tr:last-child td{border-bottom:0}
table.q td.n{font-family:var(--mono);white-space:nowrap}
table.q .run{font-weight:500} table.q .run small{color:var(--muted);font-weight:400}
.bar{position:relative;height:6px;border-radius:3px;background:var(--sunk);min-width:70px}
.bar i{position:absolute;inset:0 auto 0 0;border-radius:3px;background:var(--run)}
.pr{font-family:var(--mono);font-size:11px;color:var(--muted)}
.empty-q{color:var(--muted);font-size:13px;padding:6px 0}
.group{margin-top:26px}
.group>h2{font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 10px;padding-bottom:6px;border-bottom:1px solid var(--rule)}
.arms{display:flex;flex-direction:column;gap:14px}
.arm{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:14px 16px}
.arm-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 12px;margin-bottom:10px}
.arm-head h3{margin:0;font-size:16px;font-weight:600}
.arm-head .desc{color:var(--muted);font-size:13px}
.arm-head .cnt{margin-left:auto;font-size:12px;color:var(--muted)}
.scroll{overflow-x:auto}
table.m{border-collapse:separate;border-spacing:4px;font-size:12.5px}
table.m th{font-weight:500;color:var(--muted);font-size:11.5px;text-align:center;padding:0 2px}
table.m th.k{text-align:right;padding-right:8px;white-space:nowrap}
table.m td.k{font-family:var(--mono);text-align:right;padding-right:8px;white-space:nowrap;color:var(--ink)}
table.m td.mean{font-family:var(--mono);font-size:12px;white-space:nowrap;padding-left:10px;color:var(--ink)}
table.m td.mean small{color:var(--muted)}
.c{position:relative;min-width:66px;height:52px;border-radius:7px;border:1px solid transparent;font:inherit;font-family:var(--mono);font-size:12px;
   display:flex;flex-direction:column;align-items:center;justify-content:center;line-height:1.15;cursor:pointer;padding:0 4px}
.c span{font-size:10px;opacity:.8}
.c .fin{font-size:9.5px;opacity:.65}
.mdbox{margin:14px 0;background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:12px 14px}
.mdbox[hidden]{display:none!important}
.mdbox textarea{width:100%;min-height:260px;font-family:var(--mono);font-size:12px;color:var(--ink);background:var(--bg);border:1px solid var(--rule);border-radius:8px;padding:8px;box-sizing:border-box}
.mdbox .row{display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap}
.mdbox .row small{color:var(--muted)}
.c.empty{background:transparent;border:1px dashed var(--rule);cursor:default}
.c.done{background:var(--done-bg);color:var(--done)}
.c.excluded{background:var(--excl-bg);color:var(--excl)}
.c.running{background:var(--run-bg);color:var(--run)}
.c.todo{background:var(--todo-bg);color:var(--todo)}
.c.hold{background:transparent;color:var(--todo);border:1px dashed var(--todo)}
.c.stale{background:var(--stale-bg);color:var(--stale)}
.c.sel{outline:2px solid var(--accent);outline-offset:1px}
.c:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.c .flag{position:absolute;top:3px;right:4px;font-size:10px;line-height:1}
.c .flag.ok{color:var(--done)}
.c .flag.no{width:7px;height:7px;border-radius:50%;background:var(--miss);top:4px;right:4px}
.dim{opacity:.28}

.detail{position:sticky;bottom:calc(env(safe-area-inset-bottom,0px) + 12px);margin-top:22px;background:var(--surface);border:1px solid var(--accent);
        border-radius:12px;padding:14px 16px;box-shadow:0 6px 24px rgba(0,0,0,.12)}
.detail[hidden]{display:none!important}
.dh{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;margin-bottom:10px}
.dh .tag{font-family:var(--mono);font-size:14px;font-weight:500}
.pill{font-size:11.5px;padding:2px 9px;border-radius:999px;font-weight:600}
.dgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px 16px}
.dgrid div{display:flex;flex-direction:column}
.dgrid dt{font-size:11px;color:var(--muted)}
.dgrid dd{margin:0;font-family:var(--mono);font-size:14px}
.btn{font:inherit;font-size:13px;padding:7px 14px;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:var(--accent-ink);cursor:pointer}
.btn.off{background:transparent;color:var(--accent)}
.btn[disabled]{opacity:.45;cursor:not-allowed}
.x{margin-left:auto;font:inherit;background:none;border:0;color:var(--muted);cursor:pointer;font-size:18px;line-height:1}
.foot{margin-top:28px;color:var(--muted);font-size:12px}
@media (max-width:520px){ header h1{font-size:22px} .note-state{margin-left:0;width:100%} }
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="wrap">
  <header>
    <h1>EIP 실험 원장</h1>
    <p id="sub"></p>
  </header>
  <div class="strip" id="strip"></div>
  <div class="controls">
    <div class="seg" role="group" aria-label="데이터셋">
      <button id="ds-C10" type="button" aria-pressed="true">CIFAR-10</button>
      <button id="ds-C100" type="button" aria-pressed="false">CIFAR-100</button>
      <button id="ds-QK" type="button" aria-pressed="false">QKFormer C10</button>
      <button id="ds-QK100" type="button" aria-pressed="false">QKFormer C100</button>
      <button id="ds-DVS" type="button" aria-pressed="false">VGGSNN DVS</button>
    </div>
    <label class="tg"><input type="checkbox" id="only-miss"> 노트에 없는 완주 런만 강조</label>
    <button type="button" class="btn off" id="md-btn">이 탭 결과 MD로 내보내기</button>
    <span class="note-state" id="note-state">노트 체크 불러오는 중…</span>
  </div>
  <section class="mdbox" id="mdbox" hidden>
    <div class="row"><b>Markdown</b><small id="md-note"></small>
      <button type="button" class="btn" id="md-copy">복사</button>
      <button type="button" class="btn off" id="md-dl">.md 파일로 저장</button>
      <button type="button" class="btn off" id="md-close">닫기</button></div>
    <textarea id="md-text" readonly></textarea>
  </section>
  <div class="queue" id="queue"></div>
  <div id="groups"></div>
  <section class="detail" id="detail" hidden aria-live="polite"></section>
  <p class="foot">칸 색 — <b style="color:var(--done)">완주</b> · <b style="color:var(--excl)">완주·배제</b>(310에폭 AND S30/S1 ≥ 0.19 위반) ·
  <b style="color:var(--run)">진행 중</b>(에폭/310) · <b style="color:var(--todo)">대기</b> · <b style="color:var(--stale)">중단</b>.
  칸 위 숫자는 best epoch 의 val acc, 아래는 그 에폭의 스파이크 수. 오른쪽 평균은 배제되지 않은 완주분만. 큐의 예상 시각은 에폭당 345초 · GPU 14장 가정의 모의 배정이라 실제와 몇 시간 어긋날 수 있다.
  <b style="color:var(--miss)">●</b> = 완주했는데 노트에 아직 안 옮김, <b style="color:var(--done)">✓</b> = 옮김.</p>
</div>

<script>
const DATA = __DATA__;
const GROUPS = [
  ['제안법 계열', ['ours_layer_w1','ours_layer','ours_intra_ch','ours_w1','ours_inter_ch','ours_ch_b7','ours_ch_b4',
                 'ours_layer_w1 st30','ours_layer_w1 st60','ours_w1 st30','ours_w1 st60','ours_inter_ch st30','ours_inter_ch st60']],
  ['대조군', ['baseline','plain L2','plain L2 + rho','BPSR','1-softmax']],
  ['제안법 어블레이션 (ours_layer_w1 에서 뺌)', ['lw1 - vmem','lw1 - final_step','lw1 - loss-ratio','lw1 - 1- inversion']],
  ['옛 어블레이션 (ours_intra_ch 에서 뺌)', ['- vmem','- final_step','- loss-ratio','- 1- inversion','- 1- inversion (wc)']],
];
const DESC = {
  'ours_layer_w1':'제안법 (09-23 확정) — 층 전체 경쟁 + w¹ 기울기',
  'ours_layer':'층 전체 경쟁, w²',
  'ours_intra_ch':'채널 안 경쟁, w² — 원래 방법 (런 키 prop)',
  'ours_w1':'채널 안 경쟁 + w¹',
  'ours_inter_ch':'채널 합끼리 경쟁 (런 키 ours_ch)',
  'ours_ch_b7':'ours_inter_ch + β=0.7','ours_ch_b4':'ours_inter_ch + β=0.4',
  'baseline':'규제 없음','plain L2':'‖spike‖₂, 매 스텝, 고정 λ','plain L2 + rho':'plain L2 + loss-ratio (런 키 l2_lr)',
  'BPSR':'Yan et al. 2022 식(4)','1-softmax':'1 − softmax 가중',
  '- vmem':'vmem 준비도 제거','- final_step':'매 스텝 규제','- loss-ratio':'고정 λ',
  '- 1- inversion':'반전 + 채널경쟁 + vmem 동시 제거 (maxnorm_plain 가지)',
  '- 1- inversion (wc)':'반전만 제거 (abl_noinv_wc)',
  'lw1 - vmem':'제안법에서 vmem 준비도 제거','lw1 - final_step':'제안법, 매 스텝 규제',
  'lw1 - loss-ratio':'제안법, 고정 λ','lw1 - 1- inversion':'제안법에서 1− 반전만 제거',
  'ours_layer_w1 st30':'ours_layer_w1, 규제 30ep 시작','ours_layer_w1 st60':'ours_layer_w1, 규제 60ep 시작',
  'ours_w1 st30':'ours_intra_ch w¹, 규제 30ep 시작','ours_w1 st60':'ours_intra_ch w¹, 규제 60ep 시작',
  'ours_inter_ch st30':'ours_inter_ch, 규제 30ep 시작','ours_inter_ch st60':'ours_inter_ch, 규제 60ep 시작',
};
const LAMBDA = new Set(['plain L2','BPSR','1-softmax','- loss-ratio','lw1 - loss-ratio']);
const STATUS_KO = {done:'완주', excluded:'완주·배제', running:'진행 중', todo:'대기', hold:'보류', stale:'중단'};

let ds = 'C10', onlyMiss = false, sel = null;
let notes = new Set(), dbRef = null, noteMode = 'loading';   // loading | live | off
try { const v = localStorage.getItem('ledger.ds'); if (['C10', 'C100', 'QK', 'QK100', 'DVS'].includes(v)) ds = v; } catch (e) {}
const TOT = r => r.ds.startsWith('QK') ? 410 : r.ds === 'DVS' ? 200 : 310;
let runSort = 'eta';
try { const v = localStorage.getItem('ledger.runSort'); if (['eta','server','gpu'].includes(v)) runSort = v; } catch (e) {}
function setRunSort(v){ runSort = v; try { localStorage.setItem('ledger.runSort', v); } catch (e) {} renderQueue(); }

const $ = id => document.getElementById(id);
const fmtK = k => k === '-' ? '—' : k;
const num = k => (k === '-' ? 0 : parseFloat(k));
const f2 = v => v == null ? '–' : v.toFixed(2);
const f4 = v => v == null ? '–' : v.toFixed(4);
const int = v => v == null ? '–' : Math.round(v).toLocaleString('en-US');
const isDoneish = r => r.status === 'done' || r.status === 'excluded';
const missing = r => isDoneish(r) && noteMode === 'live' && !notes.has(r.tag);

function render(){
  const rs = DATA.runs.filter(r => r.ds === ds);
  $('sub').textContent = ds.startsWith('QK')
    ? `QKFormer · CIFAR-${ds === 'QK' ? '10' : '100'} · 410에폭 · val acc 는 학습 로그 최고값(AMP), 스파이크는 테스트 전체 fp32 측정 · 목록 갱신 ${DATA.generated}`
    : ds === 'DVS' ? `VGGSNN · CIFAR10-DVS · 200에폭 · 검증 992장 (정확도 1σ ≈ ±1.4%p) · 목록 갱신 ${DATA.generated}`
    : `ResNet19 · ${ds === 'C10' ? 'CIFAR-10' : 'CIFAR-100'} · 310에폭 · 서버 138/23 · 목록 갱신 ${DATA.generated}`;
  const cnt = {done:0,excluded:0,running:0,todo:0,hold:0,stale:0};
  rs.forEach(r => cnt[r.status] = (cnt[r.status]||0) + 1);
  const miss = rs.filter(missing).length;
  const colors = {done:'var(--done)',excluded:'var(--excl)',running:'var(--run)',todo:'var(--todo)',hold:'var(--todo)',stale:'var(--stale)'};
  $('strip').innerHTML = Object.keys(cnt).map(s =>
      `<span class="stat"><i style="background:${colors[s]}"></i>${STATUS_KO[s]} <b>${cnt[s]}</b></span>`).join('')
    + (noteMode === 'live' ? `<span class="stat miss">노트 미기재 <b>${miss}</b></span>` : '');

  const byArm = {};
  rs.forEach(r => (byArm[r.arm] = byArm[r.arm] || []).push(r));
  let html = '';
  for (const [gname, arms] of GROUPS){
    const present = arms.filter(a => byArm[a]);
    if (!present.length) continue;
    html += `<section class="group"><h2>${gname}</h2><div class="arms">`;
    for (const a of present) html += armCard(a, byArm[a]);
    html += `</div></section>`;
  }
  const extra = Object.keys(byArm).filter(a => !GROUPS.some(g => g[1].includes(a)));
  if (extra.length){
    html += `<section class="group"><h2>기타</h2><div class="arms">` + extra.map(a => armCard(a, byArm[a])).join('') + `</div></section>`;
  }
  $('groups').innerHTML = html;
  renderQueue();
  document.querySelectorAll('.c[data-tag]').forEach(b => b.addEventListener('click', () => select(b.dataset.tag)));
  if (sel) renderDetail();
}

const knobLabel = r => r.arm === 'baseline' ? '' : `${LAMBDA.has(r.arm) ? 'λ' : 'ρ'} ${r.knob}`;
const runName = r => `<span class="run">${r.arm}</span> <small>${knobLabel(r)} · s${r.seed}${r.ds === 'C100' ? ' · C100' : r.ds === 'QK' ? ' · QKFormer' : r.ds === 'QK100' ? ' · QKFormer C100' : r.ds === 'DVS' ? ' · DVS' : ''}</small>`;
function renderQueue(){
  const g = r => (r.gpu == null || r.gpu === '' ? 99 : +r.gpu);
  const cmp = {
    eta: (a,b) => (a.eta||'').localeCompare(b.eta||''),
    server: (a,b) => a.server.localeCompare(b.server) || g(a) - g(b),
    gpu: (a,b) => g(a) - g(b) || a.server.localeCompare(b.server),
  }[runSort];
  const run = DATA.runs.filter(r => r.status === 'running').sort((a,b) => cmp(a,b) || (a.eta||'').localeCompare(b.eta||''));
  const sortBtn = (k, l) => `<button type="button" aria-pressed="${runSort === k}" onclick="setRunSort('${k}')">${l}</button>`;
  const todo = DATA.runs.filter(r => r.status === 'todo').sort((a,b) => (a.order||0) - (b.order||0));
  let h = `<div class="qcol"><h2>지금 도는 것 <b>${run.length}</b><span class="seg" role="group" aria-label="정렬">${sortBtn('eta','종료 시간')}${sortBtn('server','서버')}${sortBtn('gpu','GPU')}</span></h2>`;
  if (!run.length) h += `<p class="empty-q">돌고 있는 런이 없습니다.</p>`;
  else {
    h += `<div class="scroll"><table class="q"><thead><tr><th>서버·GPU</th><th>런</th><th>진행</th><th>예상 완료</th></tr></thead><tbody>`;
    for (const r of run){
      const pct = Math.min(100, Math.round((r.epochs || 0) / TOT(r) * 100));
      h += `<tr><td class="n">${r.server} · ${r.gpu ?? '?'}</td><td>${runName(r)}</td>
            <td><div class="bar"><i style="width:${pct}%"></i></div><span class="pr">${r.epochs}/${TOT(r)}</span></td>
            <td class="n">${r.eta || '–'}</td></tr>`;
    }
    h += `</tbody></table></div>`;
  }
  const hold = DATA.runs.filter(r => r.status === 'hold');
  h += `</div><div class="qcol"><h2>대기 <b>${todo.length}</b><small>들어갈 순서${hold.length ? ` · 보류 ${hold.length}` : ''}</small></h2>`;
  if (hold.length) h += `<p class="empty-q">보류 ${hold.length}개 — 큐에 있지만 실행기가 집지 않는 후순위 (${[...new Set(hold.map(r => r.arm + ' · ' + (r.ds === 'C100' ? 'C100' : r.ds)))].join(', ')}). 앞 순위가 GPU 를 받은 뒤 todo 로 바꾼다.</p>`;
  if (!todo.length) h += `<p class="empty-q">대기 중인 런이 없습니다. GPU 가 비면 놀게 됩니다.</p>`;
  else {
    h += `<div class="scroll"><table class="q"><thead><tr><th>#</th><th>우선</th><th>런</th><th>예상 착지</th><th>시작 → 완료</th></tr></thead><tbody>`;
    for (const r of todo){
      h += `<tr><td class="n">${r.order}</td><td class="n">${r.pri ?? ''}</td><td>${runName(r)}</td>
            <td class="n">${r.land || ''}</td><td class="n">${r.start || '–'} → ${r.eta || '–'}</td></tr>`;
    }
    h += `</tbody></table></div>`;
  }
  h += `</div>`;
  const qk = DATA.qk;
  if (qk){
    const ST = {done:'완료', running:'진행 중', todo:'대기', stale:'중단'};
    h += `<div class="qcol qk"><h2>QKFormer <b>${qk.jobs.length}</b><small>전용 실행기 ${qk.runner ? qk.runner + ' 동작 중' : '꺼짐'} · 138 GPU 0–5 / 23 GPU 0–7 · 한 장에 2개까지 · TF 대기 0개일 때만</small></h2>`;
    h += `<div class="scroll"><table class="q"><thead><tr><th>상태</th><th>작업</th><th>GPU</th><th>진행</th><th>val acc</th><th>스파이크/샘플</th><th>예상 완료</th></tr></thead><tbody>`;
    for (const j of qk.jobs){
      const prog = j.epochs == null ? '–' : `${j.epochs}/${qk.total}`;
      h += `<tr><td><span class="st ${j.status}">${ST[j.status] || j.status}</span></td><td>${j.name} <small style="color:var(--muted)">${j.kind}</small></td>
            <td class="n">${j.gpu == null ? (j.server || '–') : (j.server || '138') + ' · ' + j.gpu}</td><td class="n">${prog}</td>
            <td class="n">${j.acc == null ? '–' : j.acc.toFixed(2)}</td>
            <td class="n">${j.spikes == null ? '–' : Math.round(j.spikes).toLocaleString('en-US')}</td>
            <td class="n">${j.eta || '–'}</td></tr>`;
    }
    h += `</tbody></table></div>`;
    if (qk.last) h += `<p class="last">${qk.last.replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}</p>`;
    h += `</div>`;
  }
  $('queue').innerHTML = h;
}

function armCard(arm, list){
  const seeds = [...new Set(list.map(r => r.seed))].sort((a,b) => a-b);
  const knobs = [...new Set(list.map(r => r.knob))].sort((a,b) => num(a)-num(b));
  const at = {}; list.forEach(r => at[r.knob + '|' + r.seed] = r);
  const kname = arm === 'baseline' ? '' : (LAMBDA.has(arm) ? 'λ' : 'ρ');
  let t = `<div class="scroll"><table class="m"><thead><tr><th class="k">${kname}</th>`
        + seeds.map(s => `<th>s${s}</th>`).join('') + `<th style="text-align:left;padding-left:10px">평균 (유효 완주)</th></tr></thead><tbody>`;
  for (const k of knobs){
    t += `<tr><td class="k">${fmtK(k)}</td>`;
    const ok = [];
    for (const s of seeds){
      const r = at[k + '|' + s];
      if (!r){ t += `<td><div class="c empty"></div></td>`; continue; }
      if (r.status === 'done') ok.push(r);
      let main = '', subl = '';
      if (isDoneish(r)){ main = f2(r.val_acc); subl = (r.spikes ? int(r.spikes) : '') + (r.finished ? `</span><span class="fin">${r.finished.slice(5)}` : ''); }
      else if (r.status === 'running'){ main = `${r.epochs}/${TOT(r)}`; subl = r.server; }
      else if (r.status === 'todo'){ main = '대기'; }
      else if (r.status === 'hold'){ main = '보류'; subl = '후순위'; }
      else { main = `중단`; subl = `${r.epochs}ep`; }
      let flag = '';
      if (isDoneish(r) && noteMode === 'live') flag = notes.has(r.tag) ? `<b class="flag ok">✓</b>` : `<b class="flag no"></b>`;
      const dim = onlyMiss && !missing(r) ? ' dim' : '';
      t += `<td><button type="button" class="c ${r.status}${sel === r.tag ? ' sel' : ''}${dim}" data-tag="${r.tag}"
              aria-label="${r.tag} ${STATUS_KO[r.status]}">${flag}${main}${subl ? `<span>${subl}</span>` : ''}</button></td>`;
    }
    if (ok.length){
      const m = x => ok.reduce((s,r) => s + r[x], 0) / ok.length;
      let sd = '';
      if (ok.length > 1){ const mu = m('val_acc'); sd = ' ±' + Math.sqrt(ok.reduce((s,r) => s + (r.val_acc-mu)**2, 0) / (ok.length-1)).toFixed(2); }
      const sp = ok.filter(r => r.spikes != null);
      const spm = sp.length ? int(sp.reduce((s,r) => s + r.spikes, 0) / sp.length) + (sp.length < ok.length ? ` <small>(스파이크 n=${sp.length})</small>` : '') : '스파이크 미측정';
      t += `<td class="mean">${m('val_acc').toFixed(2)}<small>${sd}</small> · ${spm} <small>(n=${ok.length})</small></td>`;
    } else t += `<td class="mean"><small>—</small></td>`;
    t += `</tr>`;
  }
  t += `</tbody></table></div>`;
  const done = list.filter(isDoneish).length;
  return `<article class="arm"><div class="arm-head"><h3>${arm}</h3><span class="desc">${DESC[arm] || ''}</span>
          <span class="cnt">완주 ${done} / 전체 ${list.length}</span></div>${t}</article>`;
}

function select(tag){ sel = (sel === tag ? null : tag); render(); if (!sel) $('detail').hidden = true; }

function renderDetail(){
  const r = DATA.runs.find(x => x.tag === sel);
  if (!r){ $('detail').hidden = true; return; }
  const d = $('detail');
  const colorVar = {done:'done',excluded:'excl',running:'run',todo:'todo',hold:'todo',stale:'stale'}[r.status];
  const inNotes = notes.has(r.tag);
  let noteBtn = '';
  if (isDoneish(r)){
    if (noteMode === 'live') noteBtn = `<button type="button" class="btn${inNotes ? ' off' : ''}" id="note-btn">${inNotes ? '노트 반영 취소' : '노트에 옮김'}</button>`;
    else if (noteMode === 'off') noteBtn = `<button type="button" class="btn" disabled>노트 체크 저장 불가</button>`;
  }
  const s30 = r.s30 == null ? '–' : r.s30.toFixed(3) + (r.s30 < 0.19 ? ' (< 0.19)' : '');
  d.innerHTML = `<div class="dh"><span class="tag">${r.tag}</span>
      <span class="pill" style="background:var(--${colorVar}-bg);color:var(--${colorVar})">${STATUS_KO[r.status]}</span>
      <span class="mono" style="color:var(--muted);font-size:12.5px">서버 ${r.server} · ${r.epochs}/${TOT(r)} 에폭</span>
      ${noteBtn}<button type="button" class="x" id="close" aria-label="닫기">×</button></div>
    <dl class="dgrid">
      <div><dt>train loss</dt><dd>${f4(r.train_loss)}</dd></div>
      <div><dt>train acc</dt><dd>${f2(r.train_acc)}</dd></div>
      <div><dt>val loss</dt><dd>${f4(r.val_loss)}</dd></div>
      <div><dt>val acc</dt><dd>${f2(r.val_acc)}</dd></div>
      <div><dt>spikes</dt><dd>${int(r.spikes)}</dd></div>
      <div><dt>S30/S1</dt><dd>${s30}</dd></div>
      <div><dt>종료 시각</dt><dd>${r.finished || '–'}</dd></div>
    </dl>`;
  d.hidden = false;
  $('close').onclick = () => { sel = null; d.hidden = true; render(); };
  const b = $('note-btn');
  if (b) b.onclick = () => toggleNote(r.tag);
}

async function toggleNote(tag){
  if (!dbRef) return;
  const b = $('note-btn'); if (b) b.disabled = true;
  try {
    const doc = dbRef.doc('notes/' + tag);
    if (notes.has(tag)) await doc.delete();
    else await doc.set({on: true, at: new Date().toISOString()});
  } catch (e) {
    noteMode = 'off';
    $('note-state').textContent = '노트 체크를 저장하지 못했습니다 — 이 보기에서는 읽기만 됩니다';
    render();
  }
}

// ── MD 내보내기 (09-29): 지금 탭의 완주 결과를 방법·knob 별 표로. 열 순서는 보고 규약 그대로
const DSNAME = {C10:'R19-C10', C100:'R19-C100', QK:'QKFormer-C10', QK100:'QKFormer-C100', DVS:'VGGSNN-CIFAR10DVS'};
function buildMd(){
  const rs = DATA.runs.filter(r => r.ds === ds && isDoneish(r));
  const byArm = {}; rs.forEach(r => (byArm[r.arm] = byArm[r.arm] || []).push(r));
  const order = GROUPS.flatMap(g => g[1]).filter(a => byArm[a]).concat(Object.keys(byArm).filter(a => !GROUPS.some(g => g[1].includes(a))));
  const n = v => v == null ? '–' : v;
  let md = `# ${DSNAME[ds]} 실험 결과 (원장 ${DATA.generated})\n\n`;
  for (const a of order){
    const kn = a === 'baseline' ? '' : (LAMBDA.has(a) ? 'λ' : 'ρ');
    md += `### ${DSNAME[ds]} · ${a}\n\n| ${kn} | 시드 | 서버 | train loss | train acc | val loss | val acc | spikes | 종료 시각 | 비고 |\n|---|---|---|---|---|---|---|---|---|---|\n`;
    const knobs = [...new Set(byArm[a].map(r => r.knob))].sort((x,y) => num(x)-num(y));
    for (const k of knobs){
      const g = byArm[a].filter(r => r.knob === k).sort((x,y) => x.seed - y.seed);
      const kl = k === '-' ? '–' : k;
      for (const r of g) md += `| ${kl} | s${r.seed} | ${r.server} | ${f4(r.train_loss)} | ${f2(r.train_acc)} | ${f4(r.val_loss)} | ${f2(r.val_acc)} | ${r.spikes == null ? '–' : Math.round(r.spikes)} | ${n(r.finished)} | ${r.status === 'excluded' ? '제외 (S30/S1<0.19)' : ''} |\n`;
      const ok = g.filter(r => r.status === 'done');
      if (ok.length){
        const m = x => { const v = ok.filter(r => r[x] != null); return v.length ? v.reduce((s,r) => s + r[x], 0) / v.length : null; };
        const mu = m('val_acc'); const sd = ok.length > 1 ? ' ±' + Math.sqrt(ok.reduce((s,r) => s + (r.val_acc-mu)**2, 0) / (ok.length-1)).toFixed(2) : '';
        md += `| **${kl} 평균** | **n=${ok.length}** | | ${f4(m('train_loss'))} | ${f2(m('train_acc'))} | ${f4(m('val_loss'))} | **${f2(mu)}${sd}** | **${m('spikes') == null ? '–' : Math.round(m('spikes'))}** | | |\n`;
      }
    }
    md += `\n`;
  }
  return md;
}
$('md-btn').onclick = () => {
  const md = buildMd(); $('md-text').value = md; $('mdbox').hidden = false;
  $('md-note').textContent = `${DSNAME[ds]} · 완주 ${DATA.runs.filter(r => r.ds === ds && isDoneish(r)).length}런`;
};
$('md-close').onclick = () => { $('mdbox').hidden = true; };
$('md-copy').onclick = async () => {
  const t = $('md-text'); t.select();
  try { await navigator.clipboard.writeText(t.value); $('md-note').textContent = '복사했습니다'; }
  catch (e) { try { document.execCommand('copy'); $('md-note').textContent = '복사했습니다'; } catch (e2) { $('md-note').textContent = '자동 복사가 막혀 있습니다 — 상자를 선택해 직접 복사하세요'; } }
};
$('md-dl').onclick = async () => {
  const name = `${DSNAME[ds]}_results_${DATA.generated.slice(0,10)}.md`, text = $('md-text').value;
  let dl = null;
  try { dl = window.claude && window.claude.use ? await window.claude.use('downloads') : null; } catch (e) { dl = null; }
  if (dl){
    try { await dl.save({filename: name, data: text}); $('md-note').textContent = '저장했습니다'; }
    catch (e) { $('md-note').textContent = e && e.code === 'declined' ? '저장을 취소했습니다' : '이 보기에서는 파일 저장이 안 됩니다 — 복사를 쓰세요'; }
    return;
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], {type: 'text/markdown'})); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
};

function setDs(v){
  ds = v; sel = null; $('detail').hidden = true;
  $('ds-C10').setAttribute('aria-pressed', v === 'C10');
  $('ds-C100').setAttribute('aria-pressed', v === 'C100');
  $('ds-QK').setAttribute('aria-pressed', v === 'QK');
  $('ds-QK100').setAttribute('aria-pressed', v === 'QK100');
  $('ds-DVS').setAttribute('aria-pressed', v === 'DVS');
  try { localStorage.setItem('ledger.ds', v); } catch (e) {}
  render();
}
$('ds-C10').onclick = () => setDs('C10');
$('ds-C100').onclick = () => setDs('C100');
$('ds-QK').onclick = () => setDs('QK');
$('ds-QK100').onclick = () => setDs('QK100');
$('ds-DVS').onclick = () => setDs('DVS');
$('only-miss').onchange = e => { onlyMiss = e.target.checked; render(); };
setDs(ds);

(async () => {
  let db = null;
  try { db = window.claude && window.claude.use ? await window.claude.use('db') : null; } catch (e) { db = null; }
  if (!db){
    noteMode = 'off';
    $('note-state').textContent = '노트 체크는 claude.ai 에서 열 때만 보입니다';
    $('only-miss').disabled = true;
    render(); return;
  }
  dbRef = db;
  db.collection('notes').onSnapshot(snap => {
    notes = new Set(snap.docs.filter(d => d.exists && (d.data() || {}).on).map(d => d.id));
    noteMode = 'live';
    $('note-state').textContent = `노트에 옮긴 런 ${notes.size}개 · 칸을 눌러 체크`;
    render();
  }, () => {
    noteMode = 'off';
    $('note-state').textContent = '노트 체크를 불러오지 못했습니다';
    render();
  });
})();
</script>
"""
open(out, 'w').write(HTML.replace('__DATA__', data))
print('wrote', out, len(runs), 'runs')
