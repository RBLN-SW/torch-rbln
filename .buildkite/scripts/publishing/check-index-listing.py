# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = [
#   "packaging==26.3",
# ]
# ///
"""Print true when a PEP 503 project page lists a wheel, else false.

A missing page (HTTP 404) prints false; any other failure exits 1.
INDEX_USERNAME and INDEX_PASSWORD, set together, supply HTTP Basic
credentials. Wheels compare by their parsed name, version, build tag,
and tags, not by spelling.
"""

from __future__ import annotations

import argparse
import http.client
import os
import posixpath
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from packaging.utils import InvalidWheelFilename, parse_wheel_filename


TIMEOUT_SECONDS = 60
RETRY_DELAY_SECONDS = 10
TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}


class LinkedFiles(HTMLParser):
    """Collect the file name each anchor's href points at."""

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        href = dict(attrs).get("href")
        if tag == "a" and href:
            path = urllib.parse.urlsplit(href).path
            self.names.append(urllib.parse.unquote(posixpath.basename(path)))


def opener(url: str) -> urllib.request.OpenerDirector:
    username = os.environ.get("INDEX_USERNAME")
    password = os.environ.get("INDEX_PASSWORD")
    if bool(username) != bool(password):
        given = "INDEX_USERNAME" if username else "INDEX_PASSWORD"
        raise SystemExit(f"Expected INDEX_USERNAME and INDEX_PASSWORD together, got only {given}")
    if not (username and password):
        return urllib.request.build_opener()
    # HTTPPasswordMgr ignores the scheme, so an explicit port keeps the
    # credentials off plain HTTP.
    parts = urllib.parse.urlsplit(url)
    port = parts.port or {"http": 80, "https": 443}[parts.scheme]
    origin = f"{parts.scheme}://{parts.hostname}:{port}/"
    passwords = urllib.request.HTTPPasswordMgrWithPriorAuth()
    passwords.add_password(None, origin, username, password, is_authenticated=True)
    return urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(passwords))


def fetch(url: str) -> str | None:
    """Return the page, or None for HTTP 404.

    A transient failure is retried once.
    """
    request = urllib.request.Request(url, headers={"Accept": "text/html"})
    open_url = opener(url).open
    for attempt in (1, 2):
        try:
            with open_url(request, timeout=TIMEOUT_SECONDS) as response:
                body: bytes = response.read()
            break
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if error.code not in TRANSIENT_STATUSES or attempt == 2:
                raise SystemExit(f"Could not fetch {url}: HTTP {error.code}") from None
            reason = f"HTTP {error.code}"
        except (OSError, http.client.HTTPException) as error:
            if attempt == 2:
                raise SystemExit(f"Could not fetch {url}: {error}") from None
            reason = str(error)
        print(f"Could not fetch {url}: {reason}; retrying in {RETRY_DELAY_SECONDS} s", file=sys.stderr)
        time.sleep(RETRY_DELAY_SECONDS)
    try:
        return body.decode()
    except UnicodeDecodeError as error:
        raise SystemExit(f"Cannot decode the page at {url}: {error}") from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="the project page, such as https://pypi.org/simple/<project>/")
    parser.add_argument("wheel", help="the wheel's file name")
    args = parser.parse_args()
    try:
        wanted = parse_wheel_filename(args.wheel)
    except InvalidWheelFilename as error:
        raise SystemExit(f"Cannot parse the wheel name: {error}") from None

    page = fetch(args.url)
    if page is None:
        print(f"{args.url} returned HTTP 404", file=sys.stderr)
        print("false")
        return
    links = LinkedFiles()
    links.feed(page)
    if not any(name.endswith((".whl", ".tar.gz", ".zip")) for name in links.names):
        raise SystemExit(f"{args.url} lists no distribution files; it is not a project page")
    unparsed = []
    for name in links.names:
        if not name.endswith(".whl"):
            continue
        try:
            if parse_wheel_filename(name) == wanted:
                print("true")
                return
        except InvalidWheelFilename:
            unparsed.append(name)
    if unparsed:
        raise SystemExit(f"Cannot parse {', '.join(unparsed)}; the wheel may be one of them")
    print("false")


if __name__ == "__main__":
    main()
