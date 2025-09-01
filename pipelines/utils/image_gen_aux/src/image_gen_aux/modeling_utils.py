import itertools
from typing import List, Tuple

import torch
from torch import Tensor


def get_parameter_device(parameter: torch.nn.Module) -> torch.device:
    """
    Gets the device of a PyTorch module's parameters or buffers.

    Args:
        parameter (`torch.nn.Module`): The PyTorch module from which to get the device.

    Returns:
        `torch.device`: The device of the module's parameters or buffers.
    """
    try:
        parameters_and_buffers = itertools.chain(parameter.parameters(), parameter.buffers())
        return next(parameters_and_buffers).device
    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5
        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def get_parameter_dtype(parameter: torch.nn.Module) -> torch.dtype:
    """
    Gets the data type of a PyTorch module's parameters or buffers.

    Args:
        parameter (`torch.nn.Module`): The PyTorch module from which to get the data type.

    Returns:
        `torch.dtype`: The data type of the module's parameters or buffers.
    """
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5
        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


class ModelMixin(torch.nn.Module):
    """
    Provides convenient properties to access the device and data type
    of a PyTorch module.

    By inheriting from this class, your custom PyTorch modules can access these properties
    without manual retrieval of device and data type information.

    These properties assume that all module parameters and buffers reside
    on the same device and have the same data type, respectively.
    """

    def __init__(self):
        super().__init__()

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)
