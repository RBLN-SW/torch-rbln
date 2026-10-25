"""Report the results of a command-line linter or formatter to lintrunner."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path


def report(
    code: str,
    path: str | None,
    name: str,
    description: str | None,
    *,
    line: int | None = None,
    char: int | None = None,
    original: str | None = None,
    replacement: str | None = None,
) -> None:
    message = {
        "path": path,
        "line": line,
        "char": char,
        "code": code,
        "severity": "error",
        "name": name,
        "original": original,
        "replacement": replacement,
        "description": description,
    }
    print(json.dumps(message), flush=True)  # noqa: T201


def run(code: str, path: str | None, argv: list[str], stdin: bytes = b"") -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(argv, check=False, input=stdin, capture_output=True, env={**os.environ, "NO_COLOR": "1"})  # noqa: S603
    except OSError as err:
        report(code, path, "command-failed", f"{shlex.join(argv)}: {err}")
        return None


def report_failure(code: str, path: str | None, proc: subprocess.CompletedProcess[bytes]) -> None:
    output = (proc.stdout + proc.stderr).decode(errors="replace").strip()
    report(code, path, "command-failed", f"{shlex.join(proc.args)} exited {proc.returncode}:\n{output}")


def lint(code: str, command: str, pattern: re.Pattern[str], paths: list[str]) -> None:
    argv: list[str] = []
    for arg in shlex.split(command):
        argv.extend(paths if arg == "{paths}" else [arg])
    proc = run(code, None, argv)
    if proc is None:
        return
    output = (proc.stdout + proc.stderr).decode(errors="replace")
    matches = list(pattern.finditer(output))
    for match in matches:
        report(
            code,
            match["path"],
            match["name"],
            match["description"],
            line=int(match["line"]),
            char=int(match["char"]),
        )
    unmatched = pattern.sub("", output).strip()
    if proc.returncode != 0 and (not matches or unmatched):
        report_failure(code, None, proc)


def format_file(code: str, command: str, path: str) -> None:
    original = Path(path).read_bytes()
    proc = run(code, path, [arg.replace("{path}", path) for arg in shlex.split(command)], stdin=original)
    if proc is None:
        return
    if proc.returncode != 0:
        report_failure(code, path, proc)
    elif proc.stdout != original:
        report(
            code,
            path,
            "format",
            None,
            original=original.decode(),
            replacement=proc.stdout.decode(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, fromfile_prefix_chars="@"
    )
    parser.add_argument("--code", required=True)
    parser.add_argument("--lint-command", help="command to run once; {paths} expands to the files")
    parser.add_argument(
        "--lint-pattern", help="regex for one finding, with groups path, line, char, name, and description"
    )
    parser.add_argument(
        "--format-command",
        help="command that reads one file on stdin and writes it formatted to stdout; {path} expands to its path",
    )
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    if bool(args.lint_command) != bool(args.lint_pattern):
        parser.error("--lint-command and --lint-pattern must be used together")
    if args.lint_command:
        pattern = re.compile(args.lint_pattern, re.MULTILINE)
        if missing := {"path", "line", "char", "name", "description"}.difference(pattern.groupindex):
            parser.error(f"--lint-pattern is missing groups: {', '.join(sorted(missing))}")
        lint(args.code, args.lint_command, pattern, args.paths)
    if args.format_command:
        for path in args.paths:
            format_file(args.code, args.format_command, path)


if __name__ == "__main__":
    main()
