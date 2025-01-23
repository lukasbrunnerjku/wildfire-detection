import torch.nn as nn
import math


WEIGHT_DECAY_MODULES = (
    nn.Linear,
    nn.LazyLinear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.LazyConv1d,
    nn.LazyConv2d,
    nn.LazyConv3d,
    nn.LazyConvTranspose1d,
    nn.LazyConvTranspose2d,
    nn.LazyConvTranspose3d,
)

ALL_NORMS = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LazyBatchNorm1d,
    nn.LazyBatchNorm2d,
    nn.LazyBatchNorm3d,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.GroupNorm,
    nn.LocalResponseNorm,
)

NO_WEIGHT_DECAY_MODULES = (
    *ALL_NORMS,
    nn.Embedding,
)


def adjusted_weight_decay(
    weight_decay: float,
    batch_size: int,
    total_steps: int,
) -> float:
    return weight_decay * math.sqrt(batch_size / total_steps)


def weight_decay_parameter_split(model: nn.Module) -> tuple[set, set]:
    decay = set()
    no_decay = set()

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, WEIGHT_DECAY_MODULES):
                # weights of modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, NO_WEIGHT_DECAY_MODULES):
                # weights of modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # parameter lists for decay/no_decay group in optimizer
    decay = [param_dict[pn] for pn in sorted(list(decay))]
    no_decay = [param_dict[pn] for pn in sorted(list(no_decay))]

    return decay, no_decay
