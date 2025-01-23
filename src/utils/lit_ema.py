from copy import deepcopy
from ema_pytorch import EMA
import torch.nn as nn
import logging
from lightning.pytorch.callbacks import Callback
from typing import Dict, Any

from .lit_logging import setup_logger, log_main_process


class EMACallback(Callback):
    """
    Model Exponential Moving Average. Empirically it has been found that using the moving average
    of the trained parameters of a deep network is better than using its trained parameters directly.

    The ema parameters of the network is set after training end.

    Adapted from lightning callback
    https://github.com/benihime91/gale/blob/master/gale/collections/callbacks/ema.py
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_every: int = 10,
        update_after_step: int = 100,
    ):
        self.ema = EMA(model, beta=decay, update_every=update_every, update_after_step=update_after_step)
        self.temp_model = None
        self.logger = None

    def on_fit_start(self, trainer, pl_module):
        self.ema.to(device=pl_module.device)
        self.logger = setup_logger(trainer.global_rank)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx,
    ):
        """
        Update the stored parameters using a moving average.

        The ema update should be calculated after the optimizer step.
        https://stackoverflow.com/questions/73985576/pytorchlightning-model-calls-order
        """
        self.ema.update()

    def store(self, model: nn.Module):
        self.temp_model = deepcopy(model)

    def copy_params_from_temp_to_model(self):
        # see EMA.copy_params_from_ema_to_model()
        copy = self.ema.inplace_copy

        for (_, ma_params), (_, current_params) in zip(self.ema.get_params_iter(self.temp_model), self.ema.get_params_iter(self.ema.model)):
            copy(current_params.data, ma_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(self.ema.get_buffers_iter(self.temp_model), self.ema.get_buffers_iter(self.ema.model)):
            copy(current_buffers.data, ma_buffers.data)

    def restore(self):
        self.copy_params_from_temp_to_model()
        self.temp_model = None  # free memory

    def on_validation_epoch_start(self, trainer, pl_module):
        # save original parameters before replacing with EMA version
        self.store(self.ema.model)

        # copy EMA parameters to model
        self.ema.copy_params_from_ema_to_model()

        log_main_process(
            self.logger,
            logging.INFO,
            "Using EMA weights in validation loop.",
        )
        
    def on_validation_end(self, trainer, pl_module):
        "Restore original parameters to resume training later"
        self.restore()
            
    def on_save_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> None:
        # checkpoint: the checkpoint dictionary that will be saved.
        checkpoint["state_dict_ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> None:
        # https://github.com/zyinghua/uncond-image-generation-ldm/blob/main/train.py#L339
        self.ema.load_state_dict(checkpoint["state_dict_ema"])
