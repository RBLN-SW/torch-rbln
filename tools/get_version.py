"""Compute the project version using setuptools-scm with custom schemes.

Called by hatchling's 'code' version source plugin during builds.
Passes custom version_scheme and local_scheme callables directly to
setuptools-scm, eliminating the need for a separate entry-point package.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from setuptools_scm import get_version as _scm_get_version


if TYPE_CHECKING:
    from setuptools_scm.version import ScmVersion

TAG_REGEX = r"^v(?P<version>\d+\.\d+\.\d+(?:(?:a|b|rc)\d*)?)"


def _version_scheme(version: ScmVersion) -> str:
    """Like guess-next-dev, but strip pre-release suffixes before bumping.

    - distance == 0 (on tag): return tag version as-is, preserving pre-release.
      Use distance, not exact, so dirty working tree still gets tag version.
      e.g. v0.10.0a0  → 0.10.0a0
    - distance > 0 (off tag): strip pre-release, bump patch, add .devN
      e.g. v0.10.0a0, distance=5 → 0.10.1.dev5
           v0.10.0,   distance=5 → 0.10.1.dev5
    """
    if version.distance == 0:
        return version.format_with("{tag}")

    tag_str = str(version.tag)
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", tag_str)
    if not match:
        raise ValueError(f"Cannot parse base version from tag: {tag_str}")

    major, minor, patch = match.group(1), match.group(2), match.group(3)
    return f"{major}.{minor}.{int(patch) + 1}.dev{version.distance}"


def _local_scheme(version: ScmVersion) -> str:
    """Return +{node} only (no date component). Omit when on tag (distance=0)."""
    if version.distance == 0 or version.node is None:
        return ""
    return version.format_choice("+{node}", "+{node}")


def compute_version() -> str:
    """Compute version from git tags using setuptools-scm."""
    return _scm_get_version(
        version_scheme=_version_scheme,
        local_scheme=_local_scheme,
        tag_regex=TAG_REGEX,
    )
