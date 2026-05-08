
def _try_import(import_fn):
    try:
        return import_fn()
    except Exception:
        return None

# Core models - lazy imports to avoid keras.engine dependency in old files
VGG16_TR = VGG11 = VGG16 = VGG_SPECK = None
ResNet18 = ResNet19 = ResNet34 = ResNet50 = ResNet101 = ResNet152 = None
ResNet20 = ResNet32 = ResNet44 = ResNet56 = None
ResNet18V2 = ResNet34V2 = ResNet50V2 = None
ResNet20V2 = ResNet32V2 = ResNet44V2 = ResNet56V2 = None
ResNet20_SEW = ResNet34_MS = ResNet18_MS = ResNet32_MS = None
MobileNetV2 = EfficientNetV2S = EfficientNetV2M = EfficientNetV2L = None
spikformer = spikformer_tb = None

try:
    from models.vgg16_tr import VGG16_TR
except Exception: pass
try:
    from models.vgg11_func import VGG11
except Exception: pass
try:
    from models.vgg16_func import VGG16
except Exception: pass
try:
    from models.vgg_speck import VGG_SPECK
except Exception: pass
try:
    from models.resnet import (ResNet18, ResNet19, ResNet34, ResNet50,
                                ResNet101, ResNet152, ResNet20, ResNet32,
                                ResNet44, ResNet56, ResNet18V2, ResNet34V2,
                                ResNet50V2, ResNet20V2, ResNet32V2, ResNet44V2,
                                ResNet56V2, ResNet20_SEW, ResNet34_MS,
                                ResNet18_MS, ResNet32_MS)
except Exception: pass
try:
    from models.spikformer import spikformer
except Exception: pass
try:
    from models.spikformer_tb import spikformer_tb
except Exception: pass
try:
    from models.mobilenet_v2 import MobileNetV2
except Exception: pass
try:
    from models.efficientnet_v2 import EfficientNetV2S, EfficientNetV2M, EfficientNetV2L
except Exception: pass

from models.sdtv3 import sdtv3
from models.sdtv3_mae import sdtv3_large_cls

# model selector
model_sel_tr = {
    'VGG16': VGG16_TR,
}

model_sel_sc = {
    'VGG11': VGG11,
    'VGG16': VGG16,
    'VGG_SPECK': VGG_SPECK,
    'ResNet18': ResNet18,
    'ResNet19': ResNet19,
    'ResNet20': ResNet20,
    'ResNet32': ResNet32,
    'ResNet34': ResNet34,
    'ResNet44': ResNet44,
    'ResNet50': ResNet50,
    'ResNet56': ResNet56,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
    'ResNet18V2': ResNet18V2,
    'ResNet20V2': ResNet20V2,
    'ResNet32V2': ResNet32V2,
    'ResNet34V2': ResNet34V2,
    'ResNet44V2': ResNet44V2,
    'ResNet50V2': ResNet50V2,
    'ResNet56V2': ResNet56V2,
    'ResNet20_SEW': ResNet20_SEW,
    'ResNet34_MS': ResNet34_MS,
    'ResNet32_MS': ResNet32_MS,
    'ResNet18_MS': ResNet18_MS,
    'MobileNetV2': MobileNetV2,
    'EfficientNetV2S': EfficientNetV2S,
    'EfficientNetV2M': EfficientNetV2M,
    'EfficientNetV2L': EfficientNetV2L,
    'Spikformer': spikformer,
    'Spikformer_tb': spikformer_tb,
    'SDT_V3_tiny': sdtv3,
    'SDT_V3_small': sdtv3,
    'SDT_V3_medium': sdtv3,
    'SDT_V3_large': sdtv3,
    'SDT_V3_MAE_base': sdtv3_large_cls,
    'SDT_V3_MAE_large': sdtv3_large_cls,
}


def model_sel(model_name, train_type):

    if train_type == 'transfer':
        model_top = model_sel_tr[model_name]
    elif train_type == 'scratch':
        model_top = model_sel_sc[model_name]
    else:
        assert False

    return model_top
