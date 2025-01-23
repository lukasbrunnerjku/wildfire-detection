import sys
import functools
import logging
from lightning.pytorch.utilities import rank_zero_only


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(distributed_rank=0, *, level=logging.DEBUG):
    """
    Initialize the logger and set its verbosity level to `level`.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


@rank_zero_only
def log_main_process(logger, lvl, msg):
    """
    Logs `msg` using `logger` only on the main process
    """
    logger.log(lvl, msg)
