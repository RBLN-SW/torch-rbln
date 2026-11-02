#!/usr/bin/env -S uv run --isolated --only-group ci --locked python
"""Print true when an index already lists a wheel, else false."""

from __future__ import annotations

import argparse
import http.client
import os
import posixpath
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from packaging.utils import InvalidWheelFilename, parse_wheel_filename


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


TIMEOUT_SECONDS = 60


class ProjectPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.file_names: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        href = dict(attrs).get("href")
        if tag == "a" and href:
            path = urllib.parse.urlsplit(href).path
            self.file_names.append(urllib.parse.unquote(posixpath.basename(path)))


def authenticated_opener(url: str, username: str, password: str) -> urllib.request.OpenerDirector:
    # HTTPPasswordMgr ignores the scheme, so the explicit port keeps the
    # credentials off plain HTTP.
    parts = urllib.parse.urlsplit(url)
    port = parts.port or {"http": 80, "https": 443}[parts.scheme]
    origin = f"{parts.scheme}://{parts.hostname}:{port}/"
    passwords = urllib.request.HTTPPasswordMgrWithPriorAuth()
    passwords.add_password(None, origin, username, password, is_authenticated=True)
    return urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(passwords))


def fetch_page(url: str, username: str, password: str) -> str | None:
    request = urllib.request.Request(url, headers={"Accept": "text/html"})  # noqa: S310
    try:
        with authenticated_opener(url, username, password).open(request, timeout=TIMEOUT_SECONDS) as response:
            body: bytes = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        msg = f"Could not fetch {url}: HTTP {error.code}"
        raise SystemExit(msg) from None
    except (OSError, http.client.HTTPException) as error:
        msg = f"Could not fetch {url}: {error}"
        raise SystemExit(msg) from None
    try:
        return body.decode()
    except UnicodeDecodeError as error:
        msg = f"Could not decode the page at {url}: {error}"
        raise SystemExit(msg) from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, metavar="NAME")
    parser.add_argument("--wheel-filename", required=True)
    args = parser.parse_args()
    indexes = (
        tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8")).get("tool", {}).get("uv", {}).get("index", [])
    )
    index_entry = next((index for index in indexes if index.get("name") == args.index), None)
    if index_entry is None:
        parser.error(f"no [[tool.uv.index]] named {args.index}")
    index_url = str(index_entry["url"])
    try:
        target_wheel = parse_wheel_filename(args.wheel_filename)
    except InvalidWheelFilename as error:
        msg = f"Could not parse the wheel filename: {error}"
        raise SystemExit(msg) from None
    project_name = target_wheel[0]
    prefix = "UV_INDEX_" + re.sub(r"[^A-Z0-9]", "_", args.index.upper())
    username = os.environ.get(f"{prefix}_USERNAME")
    password = os.environ.get(f"{prefix}_PASSWORD")
    if not (username and password):
        msg = f"{prefix}_USERNAME and {prefix}_PASSWORD must both be set"
        raise SystemExit(msg)
    url = f"{index_url.rstrip('/')}/{project_name}/"

    page = fetch_page(url, username, password)
    if page is None:
        print(f"{url} returned HTTP 404, so the index lists no {project_name} files", file=sys.stderr)
        print("false")
        return
    listing = ProjectPageParser()
    listing.feed(page)
    if not any(name.endswith((".whl", ".tar.gz", ".zip")) for name in listing.file_names):
        msg = f"{url} lists no distribution files, so it may not be a project page"
        raise SystemExit(msg)
    unparsed = []
    for name in listing.file_names:
        if not name.endswith(".whl"):
            continue
        try:
            if parse_wheel_filename(name) == target_wheel:
                print("true")
                return
        except InvalidWheelFilename:
            unparsed.append(name)
    if unparsed:
        msg = f"Could not parse {', '.join(unparsed)} on {url}, so {args.wheel_filename} may be listed"
        raise SystemExit(msg)
    print("false")


if __name__ == "__main__":
    main()
