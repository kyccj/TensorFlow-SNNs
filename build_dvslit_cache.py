"""vggdvslit 조합의 CIFAR10-DVS 캐시 빌더 (26-10-26). CPU 전용.

tfds cifar10_dvs (0.0.3) 를 순서대로 읽어 문헌 프로토콜의 프레임을 만든다:
이벤트 수 T 등분 -> 128->48 bicubic+antialias -> float32 개수 (눈금 1.0, 로더가 0.1 을 곱한다).
프로토콜·좌표 규칙은 datasets/cifar10_dvs_lit.py 머리 주석에 있다 (이 스크립트는 그 함수를 쓴다).

산출 (<root>/<key>/, key = T10_s48_bicubic_aa):
    train_x.npy [9000,T,48,48,2] f32   train_y.npy  train_id.npy  train_nev.npy (표본별 이벤트 수)
    test_x.npy  [1000,T,48,48,2] f32   test_y.npy   test_id.npy   test_nev.npy
    manifest.json (프로토콜 항목, 표본 수, 이벤트 수 통계, 파일 sha256, 코드 sha256)
놓는 자리: 학습 표본은 class*900 + 사전순 순위, 시험은 class*100 + (순위-900). tfds 읽는 순서와 무관하다.
임시 디렉토리에 다 쓴 뒤 rename 으로 한 번에 나타난다. 이미 있으면 멈춘다 (지우지 않는다).

    python build_dvslit_cache.py                       # 전체 (canus 기본 /media/hdd1/kyccj/EIP/dvslit_cache)
    python build_dvslit_cache.py --out-root /x/y       # 다른 루트 (로더는 env EIP_DVSLIT_CACHE 로 찾는다)
    python build_dvslit_cache.py --out-root /tmp/z --limit-train 2 --limit-test 1   # 검증용 부분 캐시

부분 캐시는 클래스마다 처음 만난 학습 K장·시험 M장만 담는다 (놓는 자리는 만난 순서). manifest 에
limit 이 남고, 로더는 env EIP_DVSLIT_ALLOW_PARTIAL=1 일 때만 받는다. 기본 루트에는 못 쓴다.

원격 서버(sejong·55)에는 이 스크립트로 다시 빌드하지 않는다 (26-10-26 리뷰). TF 버전이 다르면 bicubic 결과가
비트 단위로 달라질 수 있어 서버 효과와 방법 효과가 섞인다. canus 캐시를 복사하고 (원격 작업은 3등급),
로더가 파일 sha256 을 datasets/cifar10_dvs_lit.py 의 REF_SHA256 (canus 기준) 과 대조한다. 여기서 만든 전체 캐시가
기준과 다르면 빌드 끝에 경고를 찍는다 (로더가 거부한다. 지우지는 않는다).
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''         # GPU 는 학습 런이 쓴다. 이 스크립트는 CPU 만.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import argparse
import hashlib
import importlib.util
import json
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIT_PATH = os.path.join(HERE, 'datasets', 'cifar10_dvs_lit.py')


def load_lit():
    # datasets/__init__ 은 config·lib_snn 을 끌어오므로 파일 경로로 이 모듈만 불러온다
    spec = importlib.util.spec_from_file_location('cifar10_dvs_lit', LIT_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    lit = load_lit()
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=lit.T_DEFAULT)
    ap.add_argument('--size', type=int, default=lit.SIZE_DEFAULT)
    ap.add_argument('--out-root', default=None, help='캐시 루트 (기본: cifar10_dvs_lit.cache_root())')
    ap.add_argument('--limit-train', type=int, default=0, help='검증용: 클래스마다 학습 K장만')
    ap.add_argument('--limit-test', type=int, default=0, help='검증용: 클래스마다 시험 M장만')
    ap.add_argument('--threads', type=int, default=8, help='TF CPU 스레드 (다른 런의 입력 파이프라인 몫을 남긴다)')
    a = ap.parse_args()

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(a.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    import tensorflow_datasets as tfds
    import events_tfds.events.cifar10_dvs  # noqa: F401  (빌더 등록)

    T, S = a.T, a.size
    partial = bool(a.limit_train or a.limit_test)
    root = a.out_root or lit.cache_root()
    if partial and not (a.limit_train > 0 and a.limit_test > 0):
        raise SystemExit('부분 캐시는 --limit-train 과 --limit-test 를 둘 다 1 이상으로 준다')
    if partial:
        # 실제 런이 읽는 기본 루트(dirname(EIP_STORE)/dvslit_cache)에는 부분 캐시를 두지 않는다
        store = os.environ.get('EIP_STORE', '/media/hdd1/kyccj/EIP/paper')
        prod = os.path.abspath(os.path.join(os.path.dirname(os.path.normpath(store)), 'dvslit_cache'))
        if not a.out_root or os.path.abspath(root) == prod:
            raise SystemExit('부분 캐시는 --out-root 로 기본 루트가 아닌 곳에 만든다')
    key = lit.cache_key(T, S)
    final = os.path.join(root, key)
    if os.path.exists(final):
        raise SystemExit(f'이미 있음: {final} (지우지 않는다. 다른 --out-root 를 쓰거나 사용자 확인 후 정리)')
    os.makedirs(root, exist_ok=True)
    tmp = os.path.join(root, f'.tmp_{key}_{os.getpid()}')
    os.makedirs(tmp)

    K = a.limit_train if partial else lit.N_TRAIN_PER_CLASS
    M = a.limit_test if partial else lit.PER_CLASS - lit.N_TRAIN_PER_CLASS
    n = {'train': lit.N_CLASS * K, 'test': lit.N_CLASS * M}
    X = {s: np.lib.format.open_memmap(os.path.join(tmp, f'{s}_x.npy'), mode='w+', dtype=np.float32,
                                      shape=(n[s], T, S, S, 2)) for s in n}
    Y = {s: np.full(n[s], -1, np.int64) for s in n}
    ID = {s: np.full(n[s], -1, np.int64) for s in n}
    NEV = {s: np.zeros(n[s], np.int64) for s in n}
    filled = {s: np.zeros(n[s], bool) for s in n}
    seen = {s: np.zeros(lit.N_CLASS, np.int64) for s in n}     # 부분 캐시: 클래스별 만난 수
    rank = lit.lex_rank()
    pairs = set()

    builder = tfds.builder('cifar10_dvs')
    tfds_version = str(builder.info.version)
    ds = tfds.load('cifar10_dvs', split='train', shuffle_files=False)
    t0 = time.time()
    n_read = 0
    for ex in ds:
        lab = int(ex['label'])
        eid = int(ex['example_id'])
        if not (0 <= eid < lit.PER_CLASS):
            raise SystemExit(f'example_id 범위 밖: label {lab} id {eid}')
        if (lab, eid) in pairs:
            raise SystemExit(f'(label,id) 중복: {(lab, eid)}')
        pairs.add((lab, eid))
        r = int(rank[eid])
        split = 'train' if r < lit.N_TRAIN_PER_CLASS else 'test'
        if partial:
            cap = K if split == 'train' else M
            if seen[split][lab] >= cap:
                if all((seen[s] >= (K if s == 'train' else M)).all() for s in n):
                    break
                continue
            pos = lab * cap + int(seen[split][lab])
            seen[split][lab] += 1
        else:
            pos = lab * K + r if split == 'train' else lab * M + (r - lit.N_TRAIN_PER_CLASS)
        if filled[split][pos]:
            raise SystemExit(f'자리 중복: {split} {pos}')
        ev = ex['events']
        coords = ev['coords'].numpy()
        pol = ev['polarity'].numpy()
        fr = lit.events_to_frames(coords, pol, T)
        X[split][pos] = lit.resize_frames(fr, S)
        Y[split][pos], ID[split][pos], NEV[split][pos] = lab, eid, coords.shape[0]
        filled[split][pos] = True
        n_read += 1
        if n_read % 500 == 0:
            el = time.time() - t0
            print(f'  {n_read} 표본  {el:.0f}s  ({el / n_read:.3f} s/표본)', flush=True)
        if partial and all((seen[s] >= (K if s == 'train' else M)).all() for s in n):
            break

    for s in n:
        if not filled[s].all():
            raise SystemExit(f'{s}: 빈 자리 {int((~filled[s]).sum())}개 -- 캐시를 만들지 않는다 (임시 디렉토리 {tmp} 남김)')
        X[s].flush()
        del X[s]
        np.save(os.path.join(tmp, f'{s}_y.npy'), Y[s])
        np.save(os.path.join(tmp, f'{s}_id.npy'), ID[s])
        np.save(os.path.join(tmp, f'{s}_nev.npy'), NEV[s])
    if not partial:
        # 분할 규칙 재확인: 클래스마다 학습 900 / 시험 100, 시험 id 집합이 사전순 뒤 100
        tid = set(lit.test_ids())
        for c in range(lit.N_CLASS):
            te = set(ID['test'][Y['test'] == c].tolist())
            tr = set(ID['train'][Y['train'] == c].tolist())
            if not (len(tr) == 900 and len(te) == 100 and te == tid and not (tr & te)):
                raise SystemExit(f'분할 규칙 불일치: class {c} train {len(tr)} test {len(te)} (임시 디렉토리 {tmp} 남김)')

    nev_all = np.concatenate([NEV['train'], NEV['test']])
    man = dict(lit.expected_fields(T, S))
    man.update({
        'n_train': int(n['train']), 'n_test': int(n['test']),
        'limit': ({'train_per_class': K, 'test_per_class': M, 'positions': 'encounter order'} if partial else None),
        'positions': 'train: class*900 + lex rank; test: class*100 + (lex rank - 900)' if not partial else 'encounter order',
        'label_order': 'tfds cifar10_dvs CLASSES (airplane, automobile, bird, cat, deer, dog, ship, truck, frog, horse)',
        'tfds': {'name': 'cifar10_dvs', 'version': tfds_version, 'split': 'train', 'n_read': n_read},
        'events_per_sample': {'min': int(nev_all.min()), 'median': float(np.median(nev_all)),
                              'mean': float(nev_all.mean()), 'max': int(nev_all.max())},
        'test_ids_per_class': lit.test_ids() if not partial else None,
        'files': {f: sha256(os.path.join(tmp, f)) for f in sorted(os.listdir(tmp)) if f.endswith('.npy')},
        'code_sha256': {'datasets/cifar10_dvs_lit.py': sha256(LIT_PATH),
                        'build_dvslit_cache.py': sha256(os.path.abspath(__file__))},
        'tf_version': tf.__version__, 'numpy_version': np.__version__,
        'host': socket.gethostname(), 'built': time.strftime('%Y-%m-%d %H:%M:%S'),
        'build_sec': round(time.time() - t0, 1),
    })
    with open(os.path.join(tmp, 'manifest.json'), 'w') as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    os.rename(tmp, final)
    lit.check_cache(final, T, S, allow_partial=partial)
    print(f'완료: {final}  train {n["train"]} / test {n["test"]}  ({man["build_sec"]}s)', flush=True)
    ref = lit.REF_SHA256.get(key)
    if not partial:
        if ref is None:
            print(f'** 경고: 키 {key} 의 기준 sha256 이 없다 -- 로더가 거부한다 (REF_SHA256)', flush=True)
        elif man['files'] != ref:
            diff = sorted(f for f in set(ref) | set(man['files']) if ref.get(f) != man['files'].get(f))
            print(f'** 경고: canus 기준 캐시와 비트가 다른 파일 {diff} -- 로더가 거부한다. canus 캐시를 복사해 쓴다.',
                  flush=True)
        else:
            print('canus 기준 캐시와 sha256 동일', flush=True)


if __name__ == '__main__':
    sys.exit(main())
