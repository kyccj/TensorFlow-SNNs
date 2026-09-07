#!/bin/bash
# prop 5런이 끝나면 sm 비교군을 넣는다 (26-09-04, 무시드판).
#
# **시드는 쓰지 않는다.** run_paper.py 기본이 conf.run_seed=-1 이라 RNG 를 안 건드린다.
# 전역 시드를 박았더니 tf.data 증강이 에폭마다 같은 시퀀스를 재생해 R19-C10 baseline 이
# 96.647% -> 95.720% (sd 0.050 -> 0.429) 로 떨어졌다. train-val 격차 부호가 -1.12 에서
# +0.73 으로 뒤집힌 것이 증거다. 시드 숫자는 태그 구분용으로만 남는다.
#
# lambda=1e-7 근거 — R19-C10 1-softmax 기존 스윕:
#     5e-08 -> 96.71% @ 274,826  무손실
#     1e-07 -> 96.59% @ 202,255  무손실, 58.7% 감소  <- 무손실 경계의 가장 깊은 유효점
#     5e-07 -> 95.74% @  76,248  S30/S1=0.112 -> 사전 규칙으로 배제
# 이 점이 제안법 rho=4e-3 의 예상 착지(208,902)와 3% 안에서 만난다 — 짝비교의 핵심.
cd /home/kyccj/PycharmProjects/TensorFlow-SNNs

echo "[$(date +%m-%d\ %H:%M)] prop 5런 종료 대기..."
while : ; do
    n=0
    for s in 1 2 3 4 5; do
        # grep -c 는 0건일 때 종료코드 1 이라 `|| echo 0` 을 쓰면 "0\n0" 이 되어
        # [ 가 integer expression error 를 낸다. 파이프로 받아 첫 줄만 쓴다.
        c=$( { grep -c val_loss "_paper/r19c10-prop-4e-3-s$s/train.log" 2>/dev/null; } | head -1 )
        [ "${c:-0}" -ge 310 ] && n=$((n+1))
    done
    [ "$n" -ge 5 ] && break
    sleep 300
done
echo "[$(date +%m-%d\ %H:%M)] prop 완료. sm lambda=1e-7 5런 투입."

exec /home/kyccj/anaconda3/envs/venv_1/bin/python -u run_paper.py \
    r19c10 sm 1e-7 --seeds 1,2,3,4,5 --gpus 0,2,3,4 --dir _paper
