"""Compute the project version from git tags using setuptools-scm.

Invoked by hatchling's ``code`` version source at build time. setuptools-scm's
defaults apply except for a custom local scheme (``_local_scheme``).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from setuptools_scm import get_version


if TYPE_CHECKING:
    from setuptools_scm.version import ScmVersion


def _local_scheme(version: ScmVersion) -> str:
    """PEP 440 local segment: ``+g<sha>`` off a tag, empty on a tag, with a
    ``.debug`` / ``+debug`` marker for ``TORCH_RBLN_BUILD_TYPE=Debug`` builds.

    Omits the date that setuptools-scm's default ``node-and-date`` would add, so
    a dirty tree yields the same segment as a clean one.
    """
    if version.distance == 0 or version.node is None:
        local = ""
    else:
        local = version.format_with("+{node}")
    if os.environ.get("TORCH_RBLN_BUILD_TYPE") == "Debug":
        local = f"{local}.debug" if local else "+debug"
    return local


def compute_version() -> str:
    """Return the project version."""
    return get_version(local_scheme=_local_scheme)
