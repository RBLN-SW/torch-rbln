"""RNG state of the RBLN default generators, in the shape of ``torch.cuda.random``.

torch reaches these through the device module: ``torch.manual_seed`` calls ``_is_in_bad_fork``
and ``manual_seed_all``, and ``torch.random.fork_rng``, ``torch.utils.checkpoint`` and
``torch.testing._utils.freeze_rng_state`` call ``get_rng_state`` / ``set_rng_state``. None of
them claims a device, so seeding before a launcher assigns ``RBLN_DEVICES`` is safe.
"""

from typing import Iterable, List, Union  # noqa: UP035

import torch

import torch_rbln._C
from torch_rbln.memory import _normalize_device


__all__ = [
    "_is_in_bad_fork",
    "manual_seed",
    "manual_seed_all",
    "get_rng_state",
    "get_rng_state_all",
    "set_rng_state",
    "set_rng_state_all",
]

_UINT64_MASK = (1 << 64) - 1


def _is_in_bad_fork() -> bool:
    """Always ``False``: the generators live on the host, so seeding is safe in any fork."""
    return False


def _default_generator(device: Union[int, str, torch.device, None]) -> torch.Generator:
    return torch_rbln._C.get_default_generator(_normalize_device(device).index)


def manual_seed(seed: int) -> None:
    """Seed the default generator of the current RBLN device."""
    _default_generator(None).manual_seed(int(seed))


def manual_seed_all(seed: int) -> None:
    """Seed the default generator of every RBLN device, including devices not yet used.

    Never raises for a seed ``torch.manual_seed`` accepted: that calls this on every process
    that imports torch_rbln, NPU or not.
    """
    # Wrap a negative seed the way torch.Generator.manual_seed does.
    torch_rbln._C.manual_seed_all(int(seed) & _UINT64_MASK)


def get_rng_state(device: Union[int, str, torch.device] = "rbln") -> torch.Tensor:
    """Return the RNG state of ``device`` as a ByteTensor; ``"rbln"`` means the current device."""
    return _default_generator(device).get_state()


def get_rng_state_all() -> List[torch.Tensor]:
    """Return the RNG state of every device, in device-index order."""
    return [get_rng_state(index) for index in range(torch_rbln._C.device_count())]


def set_rng_state(new_state: torch.Tensor, device: Union[int, str, torch.device] = "rbln") -> None:
    """Set the RNG state of ``device`` from a ByteTensor produced by :func:`get_rng_state`."""
    _default_generator(device).set_state(new_state.clone(memory_format=torch.contiguous_format))


def set_rng_state_all(new_states: Iterable[torch.Tensor]) -> None:
    """Set the RNG state of every device from states ordered like :func:`get_rng_state_all`."""
    for index, state in enumerate(new_states):
        set_rng_state(state, index)
