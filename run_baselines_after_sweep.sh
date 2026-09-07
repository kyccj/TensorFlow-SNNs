#!/bin/bash
# Wait for R19 C100 sweep to finish, then launch all baselines

echo "Waiting for R19 C100 sweep to finish..."
while ps aux | grep 'r19_c100.*main_sweep' | grep -v grep > /dev/null 2>&1; do
    sleep 60
done
echo "R19 C100 sweep done! Launching baselines..."

export LD_LIBRARY_PATH=/home/kyccj/anaconda3/envs/venv_1/lib:$LD_LIBRARY_PATH
export XLA_FLAGS='--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'

cd /home/kyccj/PycharmProjects/TensorFlow-SNNs
/home/kyccj/anaconda3/envs/venv_1/bin/python run_baselines_all.py
