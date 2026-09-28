from torch_rbln import programs as _programs
from torch_rbln.device.device import *  # noqa: F403
from torch_rbln.device.device_tensor_utils import *  # noqa: F403
from torch_rbln.device.streams import *  # noqa: F403
from torch_rbln.memory import *  # noqa: F403
from torch_rbln.profiler import *  # noqa: F403


def __getattr__(name: str):
    # torch.rbln.capture_programs / CompiledProgram: resolved lazily, see torch_rbln/programs.py.
    if name in _programs.__all__:
        return getattr(_programs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
