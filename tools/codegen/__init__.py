"""
Code generator for RBLN device operations.

This package generates register_ops.py from native_functions.yaml, enabling
RBLN device acceleration for ATen operations defined in the YAML file.

The generated code is used at runtime to register and execute operations
on the RBLN device backend.
"""

from .main import main


__all__ = ["main"]
