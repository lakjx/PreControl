import json
import logging
import argparse
from typing import List, Tuple, Union, Dict, Any

from stefutil.prettier import style as s, get_logger, get_logging_handler



_logger = get_logger(__name__)


def unwrap_model(model: 'torch.nn.Module') -> 'torch.nn.Module':
    # taken from https://github.com/huggingface/transformers/issues/6821#issuecomment-1223452983
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, 'module'):
        return unwrap_model(model.module)
    else:
        return model


def set_seed(seed: int = 42):
    import random
    import numpy as np
    import torch

    _logger.info(f'Setting seed to {s.i(seed)}... ')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def override_std_handler(logger: logging.Logger, handler: logging.Handler):
    std_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(std_handlers) == 1  # sanity check
    logger.removeHandler(std_handlers[0])
    logger.addHandler(handler)


# taken from HF
log_str2log_level = {
    "detail": logging.DEBUG,  # will also print filename and line number
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def style_transformers_logging(log_level: str = 'info'):
    import transformers
    log_level_ = log_str2log_level[log_level]
    transformers.logging.set_verbosity(log_level_)
    logger_ts = transformers.logging._get_library_root_logger()
    override_std_handler(logger_ts, get_logging_handler(kind='stdout'))


def model_generation_config2dict(conf) -> Dict[str, Any]:
    return json.loads(conf.to_json_string())


def argparse_str2bool(value: Union[str, bool]) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")

def argparse_str2int_list(value: str) -> List[int]:
    ret = json.loads(value)
    assert isinstance(ret, list) and all(isinstance(i, int) for i in ret)
    return ret
    # return [float(i) for i in ret]

def argparse_str2float_list(value: str) -> List[float]:
    ret = json.loads(value)
    assert isinstance(ret, list) and all(isinstance(i, float) for i in ret)
    return [float(i) for i in ret]

# def process_hidden_states(outputs, input_ids_repeated):
def get_last_layer_output_token_hidden_states(hidden_states: Tuple[Tuple['torch.Tensor']]) -> List['torch.Tensor']:
    last_hidden_states = []
    # for idx, hidden_state in enumerate(outputs.hidden_states):
    for idx, hidden_state in enumerate(hidden_states):
        # ============================ Begin of Added ============================
        # for each generated token, get the last layer hidden state
        #   per huggingface generate API, each tensor is of shape (batch_size, seq_len, hidden_dim), where
        #       in 1st iter, 2nd dim is length of prompt + 1
        #       in 2nd and further, 2nd dim is always 1
        # ============================ End of Added ============================
        last_hidden_states.append(hidden_state[-1][:, -1, :])
    return last_hidden_states

