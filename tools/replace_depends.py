import argparse
import re


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace a dependency spec in pyproject.toml and optionally set [tool.uv.sources] index"
    )
    parser.add_argument("--source", required=True, help="Package name of old dependency")
    parser.add_argument("--target", required=True, help="New dependency specification")
    parser.add_argument("--pyproject-path", required=True, help="Path to pyproject.toml")
    parser.add_argument(
        "--set-index",
        metavar="INDEX",
        help="Set [tool.uv.sources].<source> to { index = INDEX } (e.g. rbln)",
    )
    options = parser.parse_args()

    with open(options.pyproject_path, encoding="utf-8") as f:
        content = f.read()

    # Replace any double-quoted dependency string that starts with the source
    # package name followed by a version specifier or end-of-string.
    # The negative lookahead prevents matching longer names that share the
    # same prefix (e.g. --source "torch" must NOT match "torch-rbln").
    pattern = re.compile(r'"' + re.escape(options.source) + r'(?![a-zA-Z0-9._-])[^"]*"')
    new_content = pattern.sub(f'"{options.target}"', content)

    set_index = options.set_index

    if set_index is not None:
        # Update [tool.uv.sources] for this package: name = { index = "INDEX" }
        # Match existing line like:  rebel-compiler = { index = "rbln" }
        # or:  torch = { index = "pytorch-cpu" }
        source_escaped = re.escape(options.source)
        sources_pattern = re.compile(
            r"^(\s*" + source_escaped + r'\s*=\s*\{\s*index\s*=\s*)"[^"]*"(\s*\})',
            re.MULTILINE,
        )
        replacement = rf'\g<1>"{set_index}"\g<2>'
        if sources_pattern.search(new_content):
            new_content = sources_pattern.sub(replacement, new_content)
        else:
            # Append or ensure the line exists (section must exist)
            # Insert after "rebel-compiler = { index = "rbln" }" or similar
            line_re = re.compile(
                r"^(\s*)" + source_escaped + r"\s*=\s*\{[^}]+\}",
                re.MULTILINE,
            )
            if line_re.search(new_content):
                new_content = line_re.sub(
                    f'\\1{options.source} = {{ index = "{set_index}" }}',
                    new_content,
                )
            else:
                # Add under [tool.uv.sources]
                marker = "[tool.uv.sources]"
                idx = new_content.find(marker)
                if idx != -1:
                    insert = idx + len(marker)
                    end_of_line = new_content.find("\n", insert)
                    if end_of_line != -1:
                        new_content = (
                            new_content[: end_of_line + 1]
                            + f'{options.source} = {{ index = "{set_index}" }}\n'
                            + new_content[end_of_line + 1 :]
                        )

    if new_content == content and options.set_index is None:
        print(f"Warning: no dependency matching '{options.source}' found in {options.pyproject_path}")
    with open(options.pyproject_path, "w", encoding="utf-8") as f:
        f.write(new_content)


if __name__ == "__main__":
    main()
