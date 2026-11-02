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


def lint(
    code: str, command: str, pattern: re.Pattern[str], ignore_pattern: re.Pattern[str] | None, paths: list[str]
) -> None:
    argv: list[str] = []
    for arg in shlex.split(command):
        argv.extend(paths if arg == "{paths}" else [arg])
    proc = run(code, None, argv)
    if proc is None:
        return
    output = (proc.stdout + proc.stderr).decode(errors="replace")
    matches = list(pattern.finditer(output))
    for match in matches:
        finding = match.groupdict()
        report(
            code,
            finding.get("path"),
            finding["name"],
            finding["description"],
            line=int(finding["line"]) if finding.get("line") else None,
            char=int(finding["char"]) if finding.get("char") else None,
        )
    unmatched = pattern.sub("", output)
    if ignore_pattern is not None:
        unmatched = ignore_pattern.sub("", unmatched)
    if proc.returncode != 0 and (not matches or unmatched.strip()):
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
        "--lint-pattern",
        help="regex for a finding, with groups name and description, plus optional path, line, and char",
    )
    parser.add_argument("--lint-ignore-pattern", help="regex for output to ignore")
    parser.add_argument("--format-command", help="command that formats stdin to stdout; {path} expands to the file")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    if bool(args.lint_command) != bool(args.lint_pattern):
        parser.error("--lint-command and --lint-pattern must be used together")
    if args.lint_ignore_pattern and not args.lint_command:
        parser.error("--lint-ignore-pattern requires --lint-command")
    if args.lint_command:
        pattern = re.compile(args.lint_pattern, re.MULTILINE)
        if missing := {"name", "description"}.difference(pattern.groupindex):
            parser.error(f"--lint-pattern is missing groups: {', '.join(sorted(missing))}")
        ignore_pattern = re.compile(args.lint_ignore_pattern, re.MULTILINE) if args.lint_ignore_pattern else None
        lint(args.code, args.lint_command, pattern, ignore_pattern, args.paths)
    if args.format_command:
        for path in args.paths:
            format_file(args.code, args.format_command, path)


if __name__ == "__main__":
    main()
