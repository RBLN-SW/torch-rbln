#!/usr/bin/env -S uv run --isolated --only-group ci --locked python
"""Print a release of a package on pypi.org.

With --version, print that version if it is a release, and fail otherwise.
Without it, print the latest release.

A release has at least one file that is not yanked. Pre-releases count, since
the pins this version feeds are often one; development releases and versions
that are not in normalized public form do not. PyPI's own `info.version` leaves
out pre-releases, so this reads every release instead.
"""

from __future__ import annotations

import argparse

import requests
from packaging.version import InvalidVersion, Version


PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"


def releases(package: str) -> dict[Version, str]:
    try:
        response = requests.get(PYPI_JSON_URL.format(package=package), timeout=30)
        response.raise_for_status()
        listed = response.json()["releases"]
    except (requests.RequestException, ValueError, KeyError) as error:
        msg = f"Could not read the releases of {package} from pypi.org: {error}"
        raise SystemExit(msg) from None
    found = {}
    for text, files in listed.items():
        try:
            version = Version(text)
        except InvalidVersion:
            continue
        if version.is_devrelease or version.local is not None or str(version) != text:
            continue
        if any(not file.get("yanked", False) for file in files):
            found[version] = text
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", required=True)
    parser.add_argument("--version", help="the release to check for, in normalized form")
    args = parser.parse_args()
    found = releases(args.package)
    if args.version is not None:
        if args.version not in found.values():
            msg = f"pypi.org has no release {args.version} of {args.package}"
            raise SystemExit(msg)
        print(args.version)
        return
    if not found:
        msg = f"pypi.org has no release of {args.package}"
        raise SystemExit(msg)
    print(found[max(found)])


if __name__ == "__main__":
    main()
