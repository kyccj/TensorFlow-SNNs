#!/bin/bash
# EIP 실험 원장 갱신: 세 서버(138 canus, 23 sejong, 55 neuro) 수집 -> 큐와 병합 -> HTML 생성.
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
# 26-10-19 서버 55 (neuro, 161.122.23.55, ssh 23456) — sejong 과 같은 방식으로 수집한다 (inv55.csv, ps55.txt).
# 새 서버라 55 가 닿지 않아도 원장은 만든다:
#  - 원격 호출을 전부 if 안에 둬서 set -e 가 스크립트를 끝내지 않게 하고, timeout 으로 멈춘 ssh 가
#    serve_live.py (refresh.sh 를 600초 제한으로 부른다) 를 막지 않게 한다.
#  - 지난 실행의 inv55.csv·ps55.txt 가 낡은 값으로 섞이지 않게 먼저 지운다 (merge_inv.py 는 inv55.csv 가 없으면 55 를 뺀다).
#  - collect_paper_v2 는 55 의 repo 안 _paper/<tag> 만 읽는다 (train.log 는 EIP_STORE=/srv2/kyccj/EIP/paper 로 가는 링크).
#    EIP_STORE 옆의 smoke 같은 시험 디렉토리는 읽지 않는다.
O55='-o BatchMode=yes -o ConnectTimeout=15'; H55=kyccj@161.122.23.55
rm -f $W/inv55.csv $W/ps55.txt $W/err55.txt
if timeout 120 scp -P 23456 $O55 analysis/collect_paper_v2.py $H55:/tmp/collect_paper_v2.py 2>>$W/err55.txt \
   && timeout 300 ssh -n -p 23456 $O55 $H55 'cd ~/PycharmProjects/TensorFlow-SNNs && ~/anaconda3/envs/venv_1/bin/python /tmp/collect_paper_v2.py "_paper/r19c*" "_paper/vggdvs*" --store /srv2/kyccj/EIP/paper --out /tmp/inv55.csv' 2>>$W/err55.txt \
   && timeout 120 scp -P 23456 $O55 $H55:/tmp/inv55.csv $W/inv55.csv 2>>$W/err55.txt; then
  timeout 60 ssh -n -p 23456 $O55 $H55 'ps -eo args' > $W/ps55.txt 2>>$W/err55.txt \
    || { : > $W/ps55.txt; echo '[경고] 서버 55 프로세스 목록 실패 — 55 에서 직접 띄운 런은 진행 중으로 안 잡힘'; }
else
  rm -f $W/inv55.csv
  echo '[경고] 서버 55 수집 실패 — 이번 원장에는 55 가 없음. 마지막 오류:'; tail -n 2 $W/err55.txt 2>/dev/null | sed 's/^/    /'
fi
$P analysis/ledger/merge_inv.py $W
$P analysis/ledger/qk_status.py $W
$P analysis/ledger/sdt_status.py $W || echo 'sdt_status 실패 — SDT 칸은 비어 있음'  # 원격 읽기 실패가 원장 전체를 죽이지 않게
$P analysis/ledger/build_ledger.py $W/inventory.json $W/eip_ledger.html
