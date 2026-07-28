'''
    Analyze per-neuron spike count histogram.
    Usage: python analyze_spike_histogram.py <config_dir> <ckpt_path>
'''
import os
import sys
import re
import types

config_dir = sys.argv[1]
ckpt_path = sys.argv[2]

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, config_dir))
sys.path.insert(0, PROJECT_ROOT)

# Force CPU BEFORE anything imports TF
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load config_sweep.py with CUDA_VISIBLE_DEVICES line disabled
# (config_sweep sets GPU which would override our CPU setting before TF init)
config_file = os.path.join(PROJECT_ROOT, config_dir, 'config_sweep.py')
with open(config_file) as f:
    src = f.read()
src = re.sub(r'os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=\s*"[^"]*"',
             '# CUDA_VISIBLE_DEVICES overridden for CPU analysis', src)

config_mod = types.ModuleType('config_sweep')
config_mod.__file__ = config_file
sys.modules['config_sweep'] = config_mod
exec(compile(src, config_file, 'exec'), config_mod.__dict__)

config = config_mod.config
conf = config.flags

import numpy as np
import h5py

# Detect checkpoint variable count for first neuron layer
ckpt_var_count = 0
with h5py.File(ckpt_path, 'r') as f:
    act_path = '_layer_checkpoint_dependencies/activation/act/vars'
    if act_path in f:
        ckpt_var_count = len(f[act_path])
print(f'Checkpoint first neuron var count: {ckpt_var_count}')

import tensorflow as tf

# Must import lib_snn BEFORE models (same order as main_snn_training.py)
import lib_snn
from models.models import model_sel_sc

# Load CIFAR10 test data
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test[:200].astype(np.float32) / 255.0

# Auto-detect model from config
model_name = conf.model
dataset_name = conf.dataset
num_classes = 10 if 'CIFAR10' in dataset_name and 'CIFAR100' not in dataset_name else 100
model_cls = model_sel_sc[model_name]

# Build model
model = model_cls(
    batch_size=100,
    input_shape=(32, 32, 3),
    conf=conf,
    model_name=model_name,
    dataset_name=dataset_name,
    classes=num_classes,
    weights=None,
    include_top=True,
)

# Populate layers_w_neuron (normally done by init_snn during training setup)
model.set_layers_w_neuron()

# Build by forward pass (must match batch_size), then load weights
model(x_test[:100], training=False)

# Monkey-patch _load_own_variables to handle variable count mismatch
# (checkpoints saved before/after neuron_metrics code addition)
_orig_load = tf.keras.layers.Layer._load_own_variables
def _lenient_load(self, store):
    try:
        _orig_load(self, store)
    except ValueError:
        # Load as many matching variables as possible
        self._update_trackables()
        all_vars = self._trainable_weights + self._non_trainable_weights
        keys = list(store.keys())
        n_load = min(len(all_vars), len(keys))
        for i in range(n_load):
            all_vars[i].assign(np.array(store[keys[i]]))
tf.keras.layers.Layer._load_own_variables = _lenient_load
try:
    model.load_weights(ckpt_path)
finally:
    tf.keras.layers.Layer._load_own_variables = _orig_load

# Debug: check model structure
print(f'layers_w_neuron count: {len(model.layers_w_neuron)}')
for i, layer in enumerate(model.layers_w_neuron[:3]):
    print(f'  [{i}] {layer.name}, has act: {hasattr(layer, "act")}', end='')
    if hasattr(layer, 'act'):
        print(f', act type: {type(layer.act).__name__}', end='')
        print(f', has sci: {hasattr(layer.act, "spike_count_int")}', end='')
    print()

# Forward pass in batches and collect spike_count_int
layer_data = {}
for batch_start in range(0, 200, 100):
    batch = x_test[batch_start:batch_start+100]
    _ = model(batch, training=False)

    for layer in model.layers_w_neuron:
        if hasattr(layer, 'act') and hasattr(layer.act, 'spike_count_int'):
            sci = layer.act.spike_count_int.numpy()
            name = layer.name
            if name not in layer_data:
                layer_data[name] = []
            layer_data[name].append(sci.copy())

# Print histogram
ts = conf.time_step
header = f'{"Layer":<20} {"Neurons":>8}'
for v in range(ts + 1):
    header += f'  {"sc="+str(v):>7}'
header += f'  {"mean":>6}'
print(header)
print('-' * len(header))

for name, batches in layer_data.items():
    sci_all = np.concatenate(batches, axis=0)
    flat = sci_all.reshape(sci_all.shape[0], -1)
    n_neurons = flat.shape[1]

    parts = f'{name:<20} {n_neurons:>8}'
    for v in range(ts + 1):
        frac = np.mean(np.sum(flat == v, axis=1) / n_neurons)
        parts += f'  {frac:>6.1%}'
    mean_sc = np.mean(flat)
    parts += f'  {mean_sc:>6.3f}'
    print(parts)
