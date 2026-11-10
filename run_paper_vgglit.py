"""VGGSNN x CIFAR10-DVS 문헌 데이터 프로토콜 (vggdvslit) 의 학습 레시피 탐색 진입점 (26-10-30, 사용자 승인 10-30).

run_paper.py 를 고치지 않는다. 새 조합을 run_paper 의 조합 표에 런타임에 등록한 뒤 run_paper.main() 을 그대로 부른다.
방법 (METHODS)·덮어쓰기 블록·verify·preflight·run_one·실패 시 종료코드는 run_paper 것 그대로다.
  왜 새 파일인가: 학습 경로 4파일 (lib_snn/neurons.py, flags.py, run_paper.py, config_snn_training.py) 의 sha256 을
  R19 C100 대기 스크립트가 감시하고 있고 (바뀌면 그 런들을 띄우지 않는다), run_paper.py 변경은 두 서버 배포다.
  이 스크립트·새 config·새 모듈은 기존 조합이 읽지 않는다. 기존 조합을 이 스크립트로 띄워도 생성되는 config_sweep.py·
  main_sweep.py 는 run_paper.py 로 띄운 것과 바이트 단위로 같다 (make_main 은 PAD 조합에서만 글자를 바꾼다).
  run_paper.py 의 vggdvslit 장치 (SOURCES·COMBO_FLAGS·LIT, 26-10-26 canus 작업 트리) 가 있어야 한다 -- 없으면 멈춘다.

조합 (원본 config 는 configs_paper/ 아래, 각 파일 머리 주석에 vggdvslit 와의 차이)
    vggdvslit300  r1: vggdvslit 와 같고 train_epoch 200 -> 300.
    vggdvslitsgd  r2: 문헌식 최적화기 -- SGD momentum 0.9 (nesterov 아님), lr 0.025, cos 워밍업 0·하한 0, 감쇠 0,
                  300에폭, 배치 64. 1000 % 64 != 0 이라 평가셋을 채운다 (아래 PAD).
    vggdvslitlr1e3·vggdvslitlr3e3·vggdvslittet  2단계 (26-10-31): lr 1e-3 / 3e-3, TET 식 Adam (NEW 의 줄 주석).
    vggdvslitlk5·vggdvslitlk5tet  (26-11-03): 각각 vggdvslit·vggdvslittet 에서 누설만 0.9 -> 0.5 (reset_by_sub 그대로).
  모두 vggdvslit 장치를 그대로 받는다: 단계별 BN (COMBO_FLAGS bn_per_step), LIT_PATCHES (문헌 로더·TET out=step·
  [MEM]·[PFLOOR]·[LITSPK]), preflight 의 lit 파일·캐시 sha256 대조.
  방법은 run_paper 의 모든 METHODS 를 쓸 수 있다 (base, ours_ch <rho>, ...). 다음 단계의 ours_inter_ch 는
    python run_paper_vgglit.py vggdvslitsgd ours_ch 5e-3 --seeds 1 --gpus 1
  다만 rho 눈금은 옵티마이저·에폭이 바뀌면 다시 찾아야 할 수 있다 (lambda = rho*L_task/R 은 같지만, SGD 는 기울기 크기를
  그대로 쓰고 AdamW 는 정규화한다).
  원장: 26-11-03 부터 NEW 의 7조합을 analysis/ledger/merge_inv.py (NAME·LIT_VARIANT_SUFFIX) 와
  analysis/collect_paper_v2.py (NAME_DVS) 가 안다. vggdvslit 와 같은 DVS 탭에서 레시피마다 다른 꼬리표
  ' (문헌 데이터 프로토콜 T10, <차이>)' 로 갈리고 vggdvslit 행·표는 그대로다. NEW 에 조합을 더하면 그 두 곳에도 더한다
  (빠뜨리면 그 런은 원장·대시보드에 조용히 안 나온다).

PAD (vggdvslitsgd): 모델 배치가 고정이라 평가 배치를 따로 둘 수 없다. main_sweep.py 에서 LIT 패치의 세 곳을 바꾼다.
    로더   datasets.cifar10_dvs_lit -> datasets.cifar10_dvs_litpad (시험 1000장 + 채움 24행 가중 0 = 16 x 64)
    compile  tet_loss.apply 뒤에 lib_snn/lit_spike_metrics_pad.use_weighted_metrics -- 정확도 메트릭을 weighted_metrics 로
           (Keras 는 metrics= 에 가중을 안 쓴다. 안 바꾸면 val_acc 가 채움 행까지 1024 행 평균이 된다)
    콜백   LitSpikeLogger -> lib_snn/lit_spike_metrics_pad.LitSpikeLoggerPad (채움 행 빼고 세기 + [EVALN] 줄)
  근거·정확성 (val_acc·val_loss 가 1000장 평균과 같은 이유) 은 datasets/cifar10_dvs_litpad.py 머리 주석,
  CPU 검증은 실제 TETLoss·make_metrics 로 했다 (직접 1000장 평균과 loss 차 1e-6, acc 같음).

이 스크립트 전용 옵션 (나머지 인자는 run_paper.py 와 같다)
    --min-free-mib N   다른 작업과 GPU 를 나눠 쓸 때. run_paper 는 '사용 200MiB 이하' GPU 만 빈 것으로 본다.
                       이 옵션을 주면 '남은 메모리 >= N MiB' 인 GPU 를 쓰고, TF_FORCE_GPU_ALLOW_GROWTH=true 를 자식에게
                       넘긴다 (안 넘기면 TF 가 남은 메모리를 통째로 잡아 같이 쓰는 작업이 죽는다).
                       N 의 기준 (26-10-30 GPU 1 공유 연기시험): 배치 40 은 [MEM] peak 7.7 GiB, nvidia-smi 예약
                       10.9 GB 였고 2에폭째 예약을 늘리다 OOM 났다 (남은 메모리 약 2 GB). -> 배치 40 은 14000 이상,
                       배치 64 (활성값 1.6배) 는 20000 이상. 공유 GPU 는 같이 도는 작업이 바뀔 때 남은 메모리가 출렁이고,
                       100% 사용 중이면 스텝이 약 5배 느리다 (에폭 1: 734 s vs 빈 GPU 175 s). 본 런은 빈 GPU 를 권한다.
예)
    python run_paper_vgglit.py --list
    python run_paper_vgglit.py vggdvslitsgd base - --seeds 901 --gpus 1 --dir _smoke --epochs 2 --min-free-mib 20000
    python run_paper_vgglit.py vggdvslit300 base - --seeds 1 --gpus 3 --dir _paper          (빈 GPU)
"""
import argparse, os, subprocess, sys

import run_paper as rp

NEW = {
    # 조합: (원본 config, 모델, 데이터셋)
    'vggdvslit300': ('configs_paper/vggdvs_lit300', 'VGGSNN', 'CIFAR10_DVS'),
    'vggdvslitsgd': ('configs_paper/vggdvs_litsgd', 'VGGSNN', 'CIFAR10_DVS'),
    # 26-10-31 레시피 탐색 2단계 (문헌 lr 1e-3 쪽, 배치 40 이라 PAD 불필요)
    'vggdvslitlr1e3': ('configs_paper/vggdvs_lr1e3', 'VGGSNN', 'CIFAR10_DVS'),   # lr 6e-3 -> 1e-3 만
    'vggdvslitlr3e3': ('configs_paper/vggdvs_lr3e3', 'VGGSNN', 'CIFAR10_DVS'),   # lr 6e-3 -> 3e-3 만
    'vggdvslittet':   ('configs_paper/vggdvs_tet', 'VGGSNN', 'CIFAR10_DVS'),     # TET 식 Adam 1e-3·감쇠 0·워밍업 0·cos 하한 0·300에폭
    # 26-11-03 baseline 레시피 선별 (사용자 결정 11-03 '누설 0.5 만'). 누설 0.9 -> 0.5 만, reset_by_sub 그대로 (각 원본 config 머리 주석)
    'vggdvslitlk5':    ('configs_paper/vggdvs_litlk5', 'VGGSNN', 'CIFAR10_DVS'),      # vggdvslit + 누설 0.5 (200에폭)
    'vggdvslitlk5tet': ('configs_paper/vggdvs_litlk5tet', 'VGGSNN', 'CIFAR10_DVS'),   # vggdvslittet + 누설 0.5 (300에폭)
}
PAD = {'vggdvslitsgd'}
PAD_FILES = ('datasets/cifar10_dvs_litpad.py', 'lib_snn/lit_spike_metrics_pad.py')
# (LIT 패치가 main_sweep.py 에 넣은 글자, 바꿀 글자). 각각 정확히 1번 나와야 하고 아니면 투입 전에 멈춘다.
PAD_SWAPS = (
    ("import datasets.cifar10_dvs_lit as _dvslit\n"
     "datasets.cifar10_dvs = _dvslit\n"
     "print('[DVSLIT] loader = datasets.cifar10_dvs_lit', flush=True)\n",
     "import datasets.cifar10_dvs_litpad as _dvslit    # dvslitpad (26-10-30): 같은 프로토콜, 평가셋을 배치 배수로 채움\n"
     "datasets.cifar10_dvs = _dvslit\n"
     "print('[DVSLIT] loader = datasets.cifar10_dvs_litpad (평가셋 채움 행 가중 0)', flush=True)\n"),
    ("    _tet.apply(model, config, out='step')\n",
     "    _tet.apply(model, config, out='step')\n"
     "    import lib_snn.lit_spike_metrics_pad as _litpad    # dvslitpad (26-10-30): 정확도도 표본 가중을 쓰게 다시 compile\n"
     "    _litpad.use_weighted_metrics(model, config, out='step')\n"),
    # 뒤에 붙은 원래 주석 ('# dvslit: 문헌 정의 스파이크 (n_in 제외)') 은 그대로 남는다
    ("    callbacks_train.append(_litspk.LitSpikeLogger(model, config))",
     "    callbacks_train.append(_litpad.LitSpikeLoggerPad(model, config, _dvslit.LAYOUT))"),
)


_registered = False


def register():
    """새 조합을 run_paper 의 표에 넣는다. 기존 항목은 건드리지 않는다. 두 번 불려도 한 번만 한다."""
    global _registered
    if _registered:
        return
    _registered = True
    need =('SOURCES', 'COMBO_FLAGS', 'LIT', 'LIT_PATCHES', 'make_main', 'preflight', 'free_gpus', 'main')
    miss = [n for n in need if not hasattr(rp, n)]
    if miss or 'vggdvslit' not in rp.SOURCES or 'vggdvslit' not in rp.LIT:
        raise SystemExit(f'중단: 이 run_paper.py 에 vggdvslit 장치가 없다 ({miss or "vggdvslit 조합 없음"}). '
                         f'canus 작업 트리 (26-10-26 설치본) 에서만 쓴다.')
    for combo, src in NEW.items():
        if combo in rp.SOURCES and rp.SOURCES[combo] != src:
            raise SystemExit(f'중단: run_paper.py 에 이미 {combo} 가 다른 정의로 있다: {rp.SOURCES[combo]}')
        rp.SOURCES[combo] = src
        # vggdvslit 와 같은 조합 플래그 (단계별 BN). dict 를 복사해 vggdvslit 항목과 객체를 나누지 않는다.
        rp.COMBO_FLAGS[combo] = dict(rp.COMBO_FLAGS['vggdvslit'])
        rp.LIT.add(combo)

    orig_make_main, orig_preflight = rp.make_main, rp.preflight

    def make_main(combo):
        m = orig_make_main(combo)
        if combo not in PAD:
            return m                       # 기존 조합·vggdvslit300 은 run_paper 결과 그대로
        for old, new in PAD_SWAPS:
            n = m.count(old)
            if n != 1:
                raise RuntimeError(f'{combo}: main_sweep.py 의 LIT 패치 글자 {old.strip()[:60]!r} 이 {n}개 (1개여야 함)')
            m = m.replace(old, new)
        if '_litspk.LitSpikeLogger(' in m or 'cifar10_dvs_litpad' not in m:
            raise RuntimeError(f'{combo}: 채움 평가 패치가 반영되지 않았다')
        return m

    def preflight(jobs, sweep_dir, epochs=0, fix_seed=False):
        bad = []
        if any(j[0] in PAD for j in jobs):
            for rel in PAD_FILES:
                p = os.path.join(rp.PROJECT_ROOT, rel)
                if not os.path.exists(p):
                    bad.append(f'{rel} 없음')
                    continue
                try:
                    with open(p, encoding='utf-8') as f:
                        compile(f.read(), p, 'exec')     # 문법만 본다 (파일을 쓰지 않는다)
                except SyntaxError as e:
                    bad.append(f'{rel} 문법 오류: {e}')
        if bad:
            raise SystemExit('중단:\n  ' + '\n  '.join(bad))
        return orig_preflight(jobs, sweep_dir, epochs, fix_seed)

    rp.make_main = make_main
    rp.preflight = preflight


def shared_free_gpus(min_free_mib):
    """'남은 메모리 >= min_free_mib' 인 GPU. run_paper.free_gpus 와 같은 인자를 받는다 (max_mib 는 안 쓴다)."""
    def free_gpus(allowed, exclude, max_mib=200):
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                timeout=60).decode()
        except Exception:
            return []
        res = []
        for line in out.strip().split('\n'):
            try:
                idx, used, total = [int(x.strip()) for x in line.split(',')]
            except ValueError:          # '[N/A]' 등 못 읽는 줄은 빈 GPU 로 치지 않는다
                continue
            if idx in allowed and idx not in exclude and total - used >= min_free_mib:
                res.append(idx)
        return res
    return free_gpus


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--min-free-mib', type=int, default=0)
    mine, rest = ap.parse_known_args(sys.argv[1:])
    register()
    if mine.min_free_mib > 0:
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'    # run_one 이 os.environ 을 복사해 자식에게 넘긴다
        rp.free_gpus = shared_free_gpus(mine.min_free_mib)
        print(f'[vgglit] GPU 공유 모드: 남은 메모리 >= {mine.min_free_mib} MiB 인 GPU 사용, '
              f'TF_FORCE_GPU_ALLOW_GROWTH=true', flush=True)
    sys.argv = [sys.argv[0]] + rest
    rp.main()


if __name__ == '__main__':
    main()
