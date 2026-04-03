"""
Main entry point for code generation.

CLI interface and main function for generating register_ops.py.

Execution flow:
1. Parse YAML files (YamlParser.parse_native_functions)
2. Generate code (CodeGenerator.generate_python_registration)
3. Write to file

For detailed flow, refer to CodeGenerator class in codegen.py.
"""

import argparse
from sys import exit

from .codegen import CodeGenerator
from .parser import YamlParser


def main(yaml_file: str, tags_file: str, generated_file: str) -> int:
    """
    Main entry point for code generation.

    Overall flow:
    1. Parse YAML: Parse native_functions.yaml and tags.yaml
    2. Generate code: Generate register_ops.py code from parsed data
    3. Write file: Save generated code to file

    Args:
        yaml_file: Path to native_functions.yaml
        tags_file: Path to tags.yaml
        generated_file: Path to output register_ops.py

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Step 1: Parse YAML files
    # Detailed implementation: See YamlParser in parser.py
    parsed_yaml = YamlParser.parse_native_functions(yaml_file, tags_file)

    if parsed_yaml is not None:
        # Step 2: Generate Python code
        # Detailed implementation: See CodeGenerator in codegen.py
        generator = CodeGenerator()
        python_code = generator.generate_python_registration(parsed_yaml)

        # Step 3: Write to file (POSIX text files end with newline)
        with open(generated_file, "w") as file:
            file.write(python_code)
            if python_code and not python_code.endswith("\n"):
                file.write("\n")
        return 0
    else:
        # fill file to empty
        with open(generated_file, "w") as file:
            file.write("")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process YAML files and generate Python code.")
    parser.add_argument("yaml_file", type=str, help="Path to the native functions YAML file")
    parser.add_argument("tags_file", type=str, help="Path to the tags YAML file")
    parser.add_argument("generated_file", type=str, help="Path to the generated Python file")

    args = parser.parse_args()
    exit(main(args.yaml_file, args.tags_file, args.generated_file))
