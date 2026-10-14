# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = [
#   "packaging==26.3",
# ]
# ///
"""Print the kind of a release tag.

The kind is final, post, pre-release, development, or local. A tag is v
followed by a PEP 440 version in normalized form, the only form from
which setuptools-scm derives the same version; any other tag exits 2.
"""

from __future__ import annotations

import argparse

from packaging.version import InvalidVersion, Version


def kind(version: Version) -> str:
    if version.local is not None:
        return "local"
    if version.is_devrelease:
        return "development"
    if version.is_prerelease:
        return "pre-release"
    return "post" if version.is_postrelease else "final"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tag")
    tag = parser.parse_args().tag
    try:
        version: Version | None = Version(tag.removeprefix("v"))
    except InvalidVersion:
        version = None
    if not tag.startswith("v") or version is None or str(version) != tag[1:]:
        parser.error(
            f"'{tag}' is not an accepted tag: expected v and a PEP 440 version in its normalized form"
            " (for example, v0.12.0, v0.12.0rc1, v0.12.0.post1, v0.13.0.dev0)"
        )
    print(kind(version))


if __name__ == "__main__":
    main()
