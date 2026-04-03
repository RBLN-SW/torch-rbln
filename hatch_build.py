"""
Custom build hook for hatchling that builds the C++ extension using CMake.

This replaces Poetry's custom build script (tools/build-libtorch-rbln.py).
The hook is invoked during wheel building to compile the native extension.

Usage:
    This file should be placed in the project root alongside pyproject.toml.
    It is referenced in [tool.hatch.build.hooks.custom] in pyproject.toml.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CMakeBuildHook(BuildHookInterface):
    """Build hook that compiles the C++ extension using CMake."""

    PLUGIN_NAME = "cmake-build"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """
        Called before the build starts.

        Args:
            version: The version being built
            build_data: Dictionary containing build configuration
        """
        if self.target_name != "wheel":
            # Only build the extension for wheel, not sdist
            return

        self._build_cmake_extension(version, build_data)

    def _build_cmake_extension(self, version: str, build_data: dict[str, Any]) -> None:
        """
        Build the CMake-based C++ extension.

        This is equivalent to running tools/build-libtorch-rbln.py which:
        1. Runs codegen to generate register_ops.py
        2. Configures CMake with appropriate settings
        3. Builds the native extension (_C.so)
        4. Copies built artifacts to torch_rbln/
        """
        project_root = Path(self.root)
        build_dir = project_root / "build"

        # Ensure build directory exists
        build_dir.mkdir(parents=True, exist_ok=True)

        # Run code generation for register_ops.py
        self._run_codegen(project_root)

        # Get build configuration from environment
        build_type = os.environ.get("TORCH_RBLN_BUILD_TYPE", "Release")
        install_dir = project_root / "torch_rbln"
        python_executable = sys.executable

        # Find ninja executable
        ninja_path = self._find_ninja()

        # Only remove CMakeCache.txt if ninja path has changed (stale cache)
        # This preserves incremental builds while handling uv's isolated environments
        cmake_cache = build_dir / "CMakeCache.txt"
        if cmake_cache.exists() and ninja_path:
            try:
                cache_content = cmake_cache.read_text()
                if "CMAKE_MAKE_PROGRAM" in cache_content and ninja_path not in cache_content:
                    print("Ninja path changed, removing stale CMake cache", file=sys.stderr)
                    cmake_cache.unlink()
            except OSError:
                pass  # Ignore read errors, CMake will handle it
        if not ninja_path:
            raise RuntimeError("Ninja build tool not found. Install it via: pip install ninja")

        # CMake configuration — mirrors tools/build-libtorch-rbln.sh
        cmake_args = [
            "cmake",
            "-S",
            str(project_root),
            "-B",
            str(build_dir),
            "-G",
            "Ninja",
            "-DBUILD_SHARED_LIBS=ON",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DPython_EXECUTABLE={python_executable}",
            f"-DCMAKE_MAKE_PROGRAM={ninja_path}",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ]

        # Enable ccache for faster rebuilds if available
        ccache_path = shutil.which("ccache")
        if ccache_path:
            cmake_args.extend(
                [
                    f"-DCMAKE_C_COMPILER_LAUNCHER={ccache_path}",
                    f"-DCMAKE_CXX_COMPILER_LAUNCHER={ccache_path}",
                ]
            )

        # Add custom CMake options from environment
        if os.environ.get("TORCH_RBLN_DEPLOY") == "ON":
            cmake_args.append("-DTORCH_RBLN_DEPLOY=ON")

        # Configure
        self._run_command(cmake_args, cwd=project_root)

        # Build
        build_args = [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            build_type,
            "-j",
            str(os.cpu_count() or 4),
        ]
        self._run_command(build_args, cwd=project_root)

        # Install artifacts into torch_rbln/ via cmake --install.
        # This copies _C.so, lib/*.so, _C.pyi stubs, and (optionally) test
        # binaries — matching the layout produced by tools/build-libtorch-rbln.sh.
        install_args = [
            "cmake",
            "--install",
            str(build_dir),
            "--config",
            build_type,
        ]
        self._run_command(install_args, cwd=project_root)

        # Add built/installed artifacts to wheel (they are gitignored)
        self._update_build_data(build_data, project_root)

    def _find_ninja(self) -> str | None:
        """
        Find ninja executable.

        Tries to locate ninja in the following order:
        1. In PATH (via shutil.which)
        2. From ninja Python package (if installed)

        Returns:
            Path to ninja executable, or None if not found
        """
        # Try to find ninja in PATH
        ninja_path = shutil.which("ninja")
        if ninja_path:
            return ninja_path

        # Try to import ninja package and use its bundled executable
        try:
            import ninja

            ninja_executable = Path(ninja.BIN_DIR) / "ninja"
            if ninja_executable.exists():
                return str(ninja_executable)
        except (ImportError, AttributeError):
            pass

        return None

    def _run_command(self, cmd: list[str], cwd: Path) -> None:
        """Run a command and check for errors."""
        print(f"Running: {' '.join(cmd)}", file=sys.stderr)
        subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=False,
            check=True,
        )

    def _run_codegen(self, project_root: Path) -> None:
        """
        Run code generation to create register_ops.py.

        This generates the operator registration module from YAML definitions.
        """
        yaml_file = project_root / "aten" / "src" / "ATen" / "native" / "native_functions.yaml"
        tags_file = project_root / "aten" / "src" / "ATen" / "native" / "tags.yaml"
        generated_file = project_root / "torch_rbln" / "_internal" / "register_ops.py"

        # Ensure the _internal directory exists
        generated_file.parent.mkdir(parents=True, exist_ok=True)

        codegen_script = project_root / "tools" / "run_codegen.py"

        cmd = [
            sys.executable,
            str(codegen_script),
            str(yaml_file),
            str(tags_file),
            str(generated_file),
        ]

        print("Generating register_ops.py...", file=sys.stderr)
        self._run_command(cmd, cwd=project_root)

    @staticmethod
    def _get_manylinux_platform_tag() -> str | None:
        """Detect the manylinux platform tag from the system's glibc version.

        Returns a tag like 'manylinux_2_39_x86_64', or None on non-glibc systems.
        """
        if sys.platform != "linux":
            return None

        try:
            libc_info = os.confstr("CS_GNU_LIBC_VERSION")
            _, version_str = libc_info.split()
        except (ValueError, OSError, AttributeError):
            _, version_str = platform.libc_ver()
            if not version_str:
                return None

        major, minor = version_str.split(".")[:2]
        machine = platform.machine()
        return f"manylinux_{major}_{minor}_{machine}"

    def _update_build_data(self, build_data: dict[str, Any], project_root: Path) -> None:
        """Update build_data to include cmake-installed artifacts.

        All build artifacts are gitignored (*.so*, torch_rbln/lib/,
        torch_rbln/test/, torch_rbln/_C/__init__.pyi) so they must be
        added via force_include to appear in the wheel.
        """
        torch_rbln_dir = project_root / "torch_rbln"

        # Force include gitignored build artifacts in the wheel
        force_include = build_data.setdefault("force_include", {})

        def _include(path: Path) -> None:
            rel = path.relative_to(project_root)
            force_include[str(path)] = str(rel)

        # _C extension module (installed to torch_rbln/ root)
        for so_file in torch_rbln_dir.glob("_C*.so*"):
            _include(so_file)

        # Shared libraries (installed to torch_rbln/lib/)
        lib_dir = torch_rbln_dir / "lib"
        if lib_dir.exists():
            for so_file in lib_dir.glob("*.so*"):
                _include(so_file)

        # _C.pyi type stub (installed to torch_rbln/_C/__init__.pyi)
        pyi_stub = torch_rbln_dir / "_C" / "__init__.pyi"
        if pyi_stub.exists():
            _include(pyi_stub)

        # C++ test binaries (installed to torch_rbln/test/ when
        # RBLN_INSTALL_TESTING=ON)
        test_dir = torch_rbln_dir / "test"
        if test_dir.exists():
            for test_bin in test_dir.iterdir():
                if test_bin.is_file():
                    _include(test_bin)

        # Generated register_ops.py (gitignored but required)
        register_ops = torch_rbln_dir / "_internal" / "register_ops.py"
        if register_ops.exists():
            _include(register_ops)

        build_data["pure_python"] = False
        manylinux_platform = self._get_manylinux_platform_tag()
        if manylinux_platform:
            python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
            build_data["tag"] = f"{python_tag}-{python_tag}-{manylinux_platform}"
        else:
            build_data["infer_tag"] = True


# Export the hook class for hatchling discovery
def get_build_hook() -> type[BuildHookInterface]:
    """Entry point for hatchling to discover the build hook."""
    return CMakeBuildHook
