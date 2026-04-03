"""
Legacy entry point for code generation.

This file redirects to the new modular structure in the codegen package.
"""

import sys
from pathlib import Path


# Add tools directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    # Import here to avoid issues when this file is imported as a module
    from codegen.main import main  # type: ignore[import-not-found]

    if len(sys.argv) < 4:
        print("Usage: run_codegen.py <yaml_file> <tags_file> <generated_file>")
        sys.exit(1)

    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
