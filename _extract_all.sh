#!/bin/bash
# 베스트 체크포인트에서 뉴런별 발화 횟수를 뽑는다 (26-09-02). GPU 4 에서 순차 실행.
# vm 은 이미 끝나서 뺐다. 로그는 런마다 따로 남긴다 -- 필터로 파이프하면 오류가 같이 지워진다
# (실제로 그렇게 OOM 세 건을 놓쳤다).
cd /home/kyccj/PycharmProjects/TensorFlow-SNNs
export LD_LIBRARY_PATH=/home/kyccj/anaconda3/envs/venv_1/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1
PY=/home/kyccj/anaconda3/envs/venv_1/bin/python
OUT=_npz
mkdir -p $OUT

run () {   # $1 이름  $2 run_dir  $3 weights
  echo "===== $1  ($(date +%H:%M))"
  $PY -u extract_neuron_spikes.py "$2" "$3" "$OUT/$1.npz" 4 0 > "$OUT/$1.log" 2>&1
  rc=$?
  if [ -f "$OUT/$1.npz" ]; then
    grep -E "^\[완료\]" "$OUT/$1.log"
  else
    echo "  실패 rc=$rc — $OUT/$1.log 확인"
    grep -E "Error|Traceback|ResourceExhausted" "$OUT/$1.log" | head -3
  fi
}

run base "_grow/baseline_run1" \
  "grow-sweep-baseline_run1/VGG16_AP_CIFAR10/ep-310_bat-100_opt-ADAMW_lr-COS-6E-03_wd-2E-02_sc_ra_cm_re_ts-4_nc-R-R_nr-s/ep-0308.weights.h5"

run sm "_shoulder/sm15-r1" \
  "sh-sm15-r1/VGG16_AP_CIFAR10/ep-310_bat-100_opt-ADAMW_lr-COS-6E-03_wd-2E-02_sc_ra_cm_re_ts-4_nc-R-R_nr-s_r-sc-n-sm-1.5e-07_7/ep-0303.weights.h5"

run mx "_mxg/mx_wc_1e-7" \
  "mxg-mx_wc_1e-7/VGG16_AP_CIFAR10/ep-310_bat-100_opt-ADAMW_lr-COS-6E-03_wd-2E-02_sc_ra_cm_re_ts-4_nc-R-R_nr-s_r-sc-n-sm-1e-07_0.5/ep-0303.weights.h5"

echo "===== 전부 완료  ($(date +%H:%M))"
