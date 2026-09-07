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
n_seen = 0
n_correct = 0

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
        if name in tot:
            tot[name] += s; fired[name] += f
        else:
            tot[name] = s.astype(np.float64); fired[name] = f.astype(np.int64)
    if bi % 20 == 0:
        print(f'  batch {bi}  표본 {n_seen}  정확도 {100*n_correct/max(1,n_seen):.2f}%', flush=True)

print(f'[완료] 표본 {n_seen}, 정확도 {100*n_correct/max(1,n_seen):.2f}%', flush=True)

os.makedirs(os.path.dirname(out_npz) or '.', exist_ok=True)
save = {k: v for k, v in tot.items()}
save.update({k + '__fired': v for k, v in fired.items()})
save['n_samples'] = np.array(n_seen)
save['T'] = np.array(T)
save['acc'] = np.array(100 * n_correct / max(1, n_seen))
np.savez_compressed(out_npz, **save)
print(f'[저장] {out_npz}', flush=True)
for name, _ in acts:
    v = tot[name]
    rate = v / (n_seen * T)
    print(f'  {name:16s} 모양 {str(v.shape):18s} 뉴런 {v.size:7d}  '
          f'평균 발화율 {rate.mean():.4f}  침묵 {100*(v==0).mean():5.1f}%')
