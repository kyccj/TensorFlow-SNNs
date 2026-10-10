"""베스트 체크포인트에서 뉴런별 발화 횟수를 뽑는다 (26-09-02).

CSV(`reg_neuron_detail.csv`) 에는 집계값(침묵률·지니·상위10%)만 있고 뉴런별 분포가 없다.
여기서는 체크포인트를 올려 테스트셋을 한 번 돌면서 각 뉴런의 spike_count 를 그대로 받아
`.npz` 로 떨군다. 그 뒤 분포(히스토그램·정렬곡선)를 그릴 수 있다.

쓰는 법:
    python extract_neuron_spikes.py <run_dir> <weights.h5> <out.npz> [gpu] [batches]
    예) python extract_neuron_spikes.py v3-g100_3e-7 v3-g100_3e-7/VGG16.../ep-0305.weights.h5 \
            _npz/vmem.npz 4 100

run_dir 은 그 실험의 `config_sweep.py` 가 있는 디렉토리다 (`_vmsil3/g100_3e-7` 같은).
설정을 그대로 써야 뉴런 배치·T·전처리가 학습 때와 같아진다.

저장 내용 — 층마다 두 배열:
    '<layer>'        각 뉴런이 테스트셋 전체에서 발화한 총 횟수  (모양 = 층의 뉴런 모양)
    '<layer>__fired' 각 뉴런이 '한 번이라도 발화한' 표본 수
그리고 'n_samples'(표본 수), 'T'(시간 단계).
발화율 = 총 발화 / (n_samples * T).

26-10-10 추가 키 (기존 키는 그대로, 추가만):
    '<layer>__t'       스텝별 발화 횟수, 테스트셋 전체 합  (모양 = [T, *뉴런 모양]).
                       '<layer>__t'.sum(0) == '<layer>'
    '<layer>__first_t' (표본, 뉴런) 쌍을 처음 발화한 스텝으로 센 히스토그램 [T+1].
                       칸 k = 스텝 k+1 에 처음 발화, 마지막 칸 = T 스텝 동안 한 번도 안 쏨.
                       합 = n_samples * 뉴런 수
    'logits'           출력층(loc=='OUT') 의 softmax 전 출력 [n_samples, classes].
                       모델의 snn_output_processing (last_ts / avg) 을 그대로 따른다
    'logits_t'         같은 값의 스텝별 원본 [n_samples, T, classes]
    'labels', 'pred'   정답·예측 클래스 [n_samples]  (pred 는 모델 최종 출력의 argmax)
predictions_n(출력층) 은 스파이크가 아니라 막전위를 내므로 그 층의 __t·__first_t 는
발화로 읽으면 안 된다 (기존 '<layer>' 도 마찬가지).
스텝별 값은 추가 forward 없이 얻는다: 뉴런은 매 스텝 spike 를 self.out(TensorArray,
크기 T) 에 쓰고 배치가 끝나도 그대로 들고 있다. forward 직후 그걸 stack 해서 읽는다.
"""
import sys, os

run_dir = sys.argv[1]
weights = sys.argv[2]
out_npz = sys.argv[3]
gpu = sys.argv[4] if len(sys.argv) > 4 else '4'
max_batches = int(sys.argv[5]) if len(sys.argv) > 5 else 0    # 0 = 전부

os.environ['CUDA_VISIBLE_DEVICES'] = gpu
sys.path.insert(0, os.path.abspath(run_dir))                  # 그 실험의 config_sweep 을 먼저 잡는다

import numpy as np
import tensorflow as tf

# TF 는 기본적으로 GPU 메모리를 거의 다 선점한다. 같은 GPU 에 다른 추출기가 겹치면
# 뒤에 뜬 쪽이 1GB 만 보고 죽는다 (실제로 그렇게 3개가 조용히 실패했다).
for _g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass

from config_sweep import config                               # 실험 설정 그대로
conf = config.flags
# config.train / batch_size / eager 는 전부 conf.mode 에서 파생된다 (config.py:122).
# config_sweep 끝에서 이미 config.set() 이 돌았으므로, mode 만 바꿔 다시 부른다.
conf.mode = 'inference'
# inference 모드는 filepath_load 를 conf.root_model_load(기본 ./models_ckpt)에서 찾는데,
# 실험은 conf.root_model_save(=exp_set_name)에 저장했다. 같은 곳을 보게 맞춘다.
# 26-09-14: 산출물을 /media/hdd1/kyccj/EIP 로 옮겼다. 옮기기 전에 만들어진 config 는
# root_model_save 가 저장소 루트 기준 상대경로(=exp_set_name)라 더 이상 존재하지 않는다.
# 그 경우 HDD 쪽을 본다. weights 는 인자로 직접 받으므로 이건 config.set() 용이다.
_rms = conf.root_model_save
if not os.path.isdir(_rms):
    for _root in ('/media/hdd1/kyccj/EIP/paper', '/media/hdd1/kyccj/EIP/archive'):
        _c = os.path.join(_root, os.path.basename(_rms))
        if os.path.isdir(_c):
            conf.root_model_save = _c
            break
conf.root_model_load = conf.root_model_save
config.set()
assert not config.train, 'train 모드로 남아 있음'

import lib_snn, datasets, callbacks                           # noqa: E402

lib_snn.utils.set_gpu()

train_ds, valid_ds, test_ds, train_n, valid_n, test_n, num_class, steps = datasets.datasets.load()
model = lib_snn.model_builder.model_builder(num_class, steps, valid_ds)
model.load_weights(weights)
print(f'[로드] {weights}', flush=True)

# 스파이크를 세는 뉴런 층만 고른다
acts = []
for l in model.layers:
    a = getattr(l, 'act', None)
    if isinstance(a, lib_snn.neurons.Neuron) and hasattr(a, 'spike_count'):
        acts.append((l.name, a))
print(f'[층] {len(acts)}개: ' + ', '.join(n for n, _ in acts), flush=True)

T = conf.time_step
tot = {}          # 뉴런별 총 발화 횟수
fired = {}        # 뉴런별 '한 번이라도 발화한 표본' 수
tot_t = {}        # 스텝별 발화 횟수 [T, ...]
first_t = {}      # 최초 발화 스텝 히스토그램 [T+1]
n_seen = 0
n_correct = 0

# 출력층 = loc=='OUT' 인 뉴런 (ResNet/VGG 의 predictions_n). 없으면 마지막 뉴런 층.
out_acts = [(n, a) for n, a in acts if getattr(a, 'loc', None) == 'OUT']
out_name, out_act = out_acts[-1] if out_acts else acts[-1]
out_proc = getattr(model, 'snn_output_processing', 'last_ts')
print(f'[출력층] {out_name}  (snn_output_processing={out_proc})', flush=True)
logits_l, logits_t_l, labels_l, pred_l = [], [], [], []


def _steps(a):
    # forward 직후 뉴런의 스텝별 출력 [T, batch, ...]. self.out 은 eager TensorArray 다.
    o = a.out
    if not hasattr(o, 'stack'):
        raise RuntimeError(f'{a.name}.out 이 TensorArray 가 아님 ({type(o)}) -- 스텝별 값을 못 읽는다')
    return o.stack().numpy().astype(np.float32)

for bi, (x, y) in enumerate(test_ds):
    if max_batches and bi >= max_batches:
        break
    # spike_count 는 tf.Variable 이고 배치마다 자동으로 안 지워진다. 학습/평가 경로에서는
    # 콜백이 지워 주지만 여기서는 직접 부른다. 안 하면 배치마다 누적되어 발화율이 1 을 넘는다.
    for _, _a in acts:
        _a.reset_spike_count()
    out = model(x, training=False)
    p = tf.argmax(out, axis=-1).numpy()
    t = tf.argmax(y, axis=-1).numpy() if len(y.shape) > 1 else y.numpy().reshape(-1)
    n_correct += int((p == t).sum())
    n_seen += int(x.shape[0])
    for name, a in acts:
        sc = a.spike_count.numpy()                            # [batch, ...] T 동안의 누적
        s = sc.sum(axis=0)
        f = (sc > 0).sum(axis=0)
        st = _steps(a)                                        # [T, batch, ...]
        s_t = st.sum(axis=1)
        nz = st > 0
        ft = np.where(nz.any(axis=0), nz.argmax(axis=0), T)   # 표본·뉴런별 최초 발화 스텝 (0-기준, T=안 쏨)
        h = np.bincount(ft.ravel(), minlength=T + 1)
        del st, nz, ft
        if name in tot:
            tot[name] += s; fired[name] += f
            tot_t[name] += s_t; first_t[name] += h
        else:
            tot[name] = s.astype(np.float64); fired[name] = f.astype(np.int64)
            tot_t[name] = s_t.astype(np.float64); first_t[name] = h.astype(np.int64)
    lt = np.moveaxis(_steps(out_act), 0, 1)                   # [batch, T, classes]
    logits_t_l.append(lt)
    logits_l.append(lt[:, T - 1] if out_proc == 'last_ts' else lt.mean(axis=1))
    labels_l.append(t.astype(np.int64)); pred_l.append(p.astype(np.int64))
    if bi % 20 == 0:
        print(f'  batch {bi}  표본 {n_seen}  정확도 {100*n_correct/max(1,n_seen):.2f}%', flush=True)

print(f'[완료] 표본 {n_seen}, 정확도 {100*n_correct/max(1,n_seen):.2f}%', flush=True)

os.makedirs(os.path.dirname(out_npz) or '.', exist_ok=True)
save = {k: v for k, v in tot.items()}
save.update({k + '__fired': v for k, v in fired.items()})
save['n_samples'] = np.array(n_seen)
save['T'] = np.array(T)
save['acc'] = np.array(100 * n_correct / max(1, n_seen))
# 26-10-10 추가 키. 카운트는 1만 이하 정수라 float32 로 정확하다 (용량 절반).
save.update({k + '__t': v.astype(np.float32) for k, v in tot_t.items()})
save.update({k + '__first_t': v for k, v in first_t.items()})
save['logits'] = np.concatenate(logits_l).astype(np.float32)
save['logits_t'] = np.concatenate(logits_t_l).astype(np.float32)
save['labels'] = np.concatenate(labels_l)
save['pred'] = np.concatenate(pred_l)
# 자체 점검: 스텝별 합 == 총합, 히스토그램 합 == 표본×뉴런
for k in tot:
    d = np.abs(tot_t[k].sum(axis=0) - tot[k]).max()
    hs = int(first_t[k].sum())
    if d > 1e-3 * max(1.0, np.abs(tot[k]).max()) or hs != n_seen * tot[k].size:
        print(f'  [경고] {k}: |__t.sum(0)-총합| 최대 {d:.3g}, __first_t 합 {hs} != {n_seen * tot[k].size}', flush=True)
np.savez_compressed(out_npz, **save)
print(f'[저장] {out_npz}', flush=True)
for name, _ in acts:
    v = tot[name]
    rate = v / (n_seen * T)
    print(f'  {name:16s} 모양 {str(v.shape):18s} 뉴런 {v.size:7d}  '
          f'평균 발화율 {rate.mean():.4f}  침묵 {100*(v==0).mean():5.1f}%')
