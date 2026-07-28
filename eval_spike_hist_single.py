"""
Run single experiment: collect per-neuron spike counts and save to npz.
Usage: python eval_spike_hist_single.py <exp_name> <ckpt_path> <channel_wise:0|1>
"""

import os
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np

BASE = '/home/kyccj/PycharmProjects/TensorFlow-SNNs'

exp_name = sys.argv[1]
ckpt_path = sys.argv[2]
channel_wise = (sys.argv[3] == '1')

import tensorflow as tf
from config import config
conf = config.flags

conf.model = 'VGG16'
conf.dataset = 'CIFAR10'
conf.pooling_vgg = 'avg'
conf.nn_mode = 'SNN'
conf.n_reset_type = 'reset_by_sub'
conf.leak_const_init = 0.9
conf.optimizer = 'ADAMW'
conf.lr_schedule = 'COS'
conf.n_init_vth = 1.0
conf.train_epoch = 310
conf.learning_rate_init = 1E-5
conf.learning_rate = 6E-3
conf.weight_decay_AdamW = 2E-2
conf.batch_size = 100
conf.label_smoothing = 0.1
conf.lmb = 1E-3
conf.regularizer = None
conf.randaug_en = True
conf.randaug_mag = 0.9
conf.randaug_mag_std = 0.4
conf.randaug_n = 1
conf.randaug_rate = 0.5
conf.rand_erase_en = True
conf.fire_surro_grad_func = 'asym'
conf.surrogate_bias = 0.6
conf.mix_off_iter = 500*200
conf.mix_alpha = 0.5

conf.reg_spike_out = True
conf.reg_spike_out_alpha = 4
conf.reg_spike_out_sc = True
conf.reg_spike_out_sc_sm = True
conf.reg_spike_out_norm = True
conf.reg_spike_out_wta_rev = True
conf.reg_spike_log_detail = True
conf.sc_loss_scd = False

if channel_wise:
    conf.reg_spike_out_const = 1e-07
    conf.reg_spike_channel_wise = True
else:
    conf.reg_spike_out_const = 1e-08
    conf.reg_spike_adaptive = True
    conf.reg_spike_adaptive_start_ep = 50
    conf.reg_spike_adaptive_lambda_init = 1e-8
    conf.reg_spike_adaptive_lambda_max = 1e-6
    conf.reg_spike_adaptive_step_up = 0.5
    conf.reg_spike_adaptive_step_down = 0.3
    conf.reg_spike_adaptive_margin = 0.005

conf.exp_set_name = 'eval-spike-hist'
conf.root_model_save = '/tmp/eval_spike_hist_tmp'
os.makedirs('/tmp/eval_spike_hist_tmp', exist_ok=True)

config.set()

import datasets
import lib_snn

_, _, test_ds, _, _, _, num_class, train_steps_per_epoch = datasets.datasets.load()

model = lib_snn.model_builder.model_builder(num_class, train_steps_per_epoch, test_ds)
model.load_weights(ckpt_path)
print(f'Loaded: {ckpt_path}')

# Find neuron layers
neuron_layers = []
for layer in model.layers:
    if hasattr(layer, 'spike_count'):
        neuron_layers.append(layer)
    elif hasattr(layer, 'act') and hasattr(layer.act, 'spike_count'):
        neuron_layers.append(layer)

if not neuron_layers:
    # Try model.layers_w_neuron
    if hasattr(model, 'layers_w_neuron') and model.layers_w_neuron:
        neuron_layers = model.layers_w_neuron
    else:
        # Deeper search
        for layer in model.layers:
            for sub in getattr(layer, '_layers', []):
                if hasattr(sub, 'spike_count'):
                    neuron_layers.append(sub)

print(f'  Found {len(neuron_layers)} neuron layers')
for nl in neuron_layers:
    print(f'    {nl.name} (type={type(nl).__name__})')

all_spike_counts = {}
num_batches = 10

for batch_idx, data_batch in enumerate(test_ds):
    if batch_idx >= num_batches:
        break
    images = data_batch[0]

    for layer in neuron_layers:
        neuron = layer.act if hasattr(layer, 'act') and hasattr(layer.act, 'spike_count') else layer
        neuron.reset_spike_count()

    _ = model(images, training=False)

    for layer in neuron_layers:
        neuron = layer.act if hasattr(layer, 'act') and hasattr(layer.act, 'spike_count') else layer
        sc = neuron.spike_count.numpy()
        name = layer.name
        if name not in all_spike_counts:
            all_spike_counts[name] = []
        all_spike_counts[name].append(sc.flatten())

    if batch_idx % 5 == 0:
        print(f'  batch {batch_idx}/{num_batches}')

result = {name: np.concatenate(arrays) for name, arrays in all_spike_counts.items()}

# Print summary
for layer_name in sorted(result.keys()):
    sc = result[layer_name]
    counts = np.bincount(sc.astype(int).clip(0, 4), minlength=5)
    total = len(sc)
    pcts = counts / total * 100
    print(f'  {layer_name}: ' + ' '.join(f'{i}:{pcts[i]:.1f}%' for i in range(5)))

# Save
out = f'{BASE}/EIP_figure/_spike_hist_{exp_name}.npz'
np.savez(out, **result)
print(f'Saved: {out}')
