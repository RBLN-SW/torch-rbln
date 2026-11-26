#!/usr/bin/env -S uv run --isolated --only-group ci --locked python
"""Print the latest version of a package on pypi.org.

Pre-releases count, since the pins this version feeds are often one. Development
releases and releases whose files are all yanked do not. PyPI's own `info.version`
leaves out pre-releases, so this reads every release instead.
"""

from __future__ import annotations

import argparse

import requests
from packaging.version import InvalidVersion, Version


PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", required=True)
    args = parser.parse_args()
    try:
        response = requests.get(PYPI_JSON_URL.format(package=args.package), timeout=30)
        response.raise_for_status()
        releases = response.json()["releases"]
    except (requests.RequestException, ValueError, KeyError) as error:
        msg = f"Could not read the releases of {args.package} from pypi.org: {error}"
        raise SystemExit(msg) from None
    candidates = []
    for text, files in releases.items():
        try:
            version = Version(text)
        except InvalidVersion:
            continue
        if version.is_devrelease or not any(not file.get("yanked", False) for file in files):
            continue
        candidates.append((version, text))
    if not candidates:
        msg = f"pypi.org has no installable release of {args.package}"
        raise SystemExit(msg)
    print(max(candidates)[1])


if __name__ == "__main__":
    main()
