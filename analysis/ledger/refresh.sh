#!/bin/bash
# EIP 실험 원장 갱신: 두 서버 수집 -> 큐와 병합 -> HTML 생성.
# 생성된 HTML 을 Artifact 로 재게시하면 같은 주소가 갱신된다:
#   https://claude.ai/artifact/WCndwAPhuymJggUDdyD6nh   ('노트 반영' 체크는 그 artifact 의 db notes/<tag>)
# 쓰는 법:  bash analysis/ledger/refresh.sh <작업폴더>
set -e
R=/home/kyccj/PycharmProjects/TensorFlow-SNNs; W=${1:-/tmp/eip_ledger}; mkdir -p $W; cd $R
P=~/anaconda3/envs/venv_1/bin/python
$P analysis/collect_paper_v2.py '_paper/r19c*' '_paper/vggdvs*' '_ours/vggdvs*' --out $W/inv138.csv   # _ours: 수동 런 (10-12 DVS 프레임0 4런)
scp -q -P 23456 analysis/collect_paper_v2.py kyccj@161.122.23.23:/tmp/collect_paper_v2.py
ssh -n -p 23456 -o BatchMode=yes kyccj@161.122.23.23 'cd ~/PycharmProjects/TensorFlow-SNNs && ~/anaconda3/envs/venv_eip/bin/python /tmp/collect_paper_v2.py "_paper/r19c*" "_paper/vggdvs*" --store /srv2/kyccj/EIP/paper --out /tmp/inv23.csv' 2>/dev/null
scp -q -P 23456 kyccj@161.122.23.23:/tmp/inv23.csv $W/inv23.csv
# 26-10-14: sejong 프로세스 목록 — 큐 없이 직접 띄운 sejong 런을 '진행 중'으로 잡는다 (merge_inv.py live_runs). 실패해도 원장은 계속
ssh -n -p 23456 -o BatchMode=yes kyccj@161.122.23.23 'ps -eo args' > $W/ps23.txt 2>/dev/null || : > $W/ps23.txt
$P analysis/ledger/merge_inv.py $W
$P analysis/ledger/qk_status.py $W
$P analysis/ledger/sdt_status.py $W || echo 'sdt_status 실패 — SDT 칸은 비어 있음'  # 원격 읽기 실패가 원장 전체를 죽이지 않게
$P analysis/ledger/build_ledger.py $W/inventory.json $W/eip_ledger.html
