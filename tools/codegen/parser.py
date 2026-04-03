"""
YAML parser for native functions.

Handles YAML parsing and preprocessing for torchgen compatibility.

Responsibilities:
- Parse native_functions.yaml and tags.yaml
- Preprocess for torchgen compatibility (fix out ops grouping issues)

Processing steps:
1. Load YAML files
2. Find out ops (those with grouping issues)
3. Add to torchgen's OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY
4. Call parse_native_yaml

Usage examples:
- main.py: Parse YAML and pass to CodeGenerator
"""

import re
from typing import Any, List, Optional  # noqa: UP035

import yaml
from torchgen.gen import parse_native_yaml  # type: ignore[import-untyped]
from torchgen.model import dispatch_keys, DispatchKey  # type: ignore[import-untyped]
from torchgen.native_function_generation import OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY  # type: ignore[import-untyped]


class YamlParser:
    """Handles YAML parsing and preprocessing for torchgen compatibility."""

    @staticmethod
    def find_out_ops_that_dont_group(file_path: str) -> Optional[List[str]]:
        """
        Find operations with 'out' parameter that don't get grouped properly by torchgen.
        This is a workaround for torchgen compatibility.
        """
        with open(file_path, "r", encoding="utf-8") as f:  # noqa: UP015
            data = yaml.safe_load(f)

        out_ops = []
        # catches all patterns where 'out' appears after the first '.'
        custom = ["max.dim_max", "min.dim_min"]  # TODO: don't have out signature and need to treat it specially
        custom_pattern = "|".join(map(re.escape, custom))
        pattern = re.compile(rf"^((?:{custom_pattern}|[^.]+\..*?out.*?))\(")
        if data is None:
            return None
        for item in data:
            if "func" in item:
                match = pattern.match(item["func"])
                if match:
                    out_ops.append(match.group(1))

        for op in out_ops:
            OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY.append(op)  # WA for torchgen works
        return out_ops

    @staticmethod
    def parse_native_functions(yaml_file: str, tags_file: str) -> Any:
        """Parse native functions YAML file with torchgen."""
        if YamlParser.find_out_ops_that_dont_group(yaml_file) is not None:
            dispatch_keys.append(DispatchKey.PrivateUse1)
            return parse_native_yaml(yaml_file, tags_file)
        else:
            return None
