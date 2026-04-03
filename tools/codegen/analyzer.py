"""
Function schema analyzer.

Extracts relevant information from function schemas for code generation.

Responsibilities:
- Extract information needed for code generation from function schemas
- Find self argument position
- Check for specific kwarg existence

Usage examples:
- CodeGenerator: Analyze function and select appropriate code generation logic
- OpCodeGenerator: Determine input extraction approach when generating Out functions
"""

from typing import Optional

from torchgen.model import Arguments  # type: ignore[import-untyped]


class FunctionAnalyzer:
    """Analyzes function schemas to extract relevant information."""

    @staticmethod
    def get_self_arg_position_index(arguments: Arguments) -> Optional[int]:  # type: ignore[no-any-unimported]
        """
        Returns the index of `self` argument within the full *positional* argument list.
        That is: pre_self_positional + [self] + post_self_positional
        If `self` does not exist, returns None.
        """
        if arguments.self_arg is not None:
            for idx, arg in enumerate(arguments.flat_positional):
                if arg.name == "self":
                    return idx
        return None

    @staticmethod
    def check_has_kwarg(arguments: Arguments, name: str) -> bool:  # type: ignore[no-any-unimported]
        """Check if function has a specific keyword argument."""
        for arg in arguments.pre_tensor_options_kwarg_only:
            if arg.name == name:
                return True
        return False
