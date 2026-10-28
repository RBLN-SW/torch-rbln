import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging.version import Version


class CMakeBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:  # noqa: ARG002
        install_manifest = self._build_cmake_extension()
        build_data["pure_python"] = False
        self._include_installed_files(build_data, install_manifest)
        self._set_platform_tag(build_data)

    def _build_cmake_extension(self) -> Path:
        project_root = Path(self.root)
        build_dir = project_root / "build"

        build_type = os.environ.get("TORCH_RBLN_BUILD_TYPE", "Release")
        build_testing = os.environ.get("TORCH_RBLN_BUILD_TESTING", "ON")
        install_prefix = project_root / "torch_rbln"
        python_executable = sys.executable
        ninja_path = self._find_ninja()

        configure_args = [
            "cmake",
            "-S",
            str(project_root),
            "-B",
            str(build_dir),
            "-G",
            "Ninja",
            f"-DBUILD_TESTING={build_testing}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
            # Without this, CMake can build the extension for the first Python on PATH
            # instead of the interpreter running this hook.
            f"-DPython3_EXECUTABLE={python_executable}",
            f"-DCMAKE_MAKE_PROGRAM={ninja_path}",
            f"-DTORCH_RBLN_VERSION={Version(self.metadata.version).base_version}",
        ]
        # A cache whose CMAKE_MAKE_PROGRAM names a different Ninja executable comes from
        # another build environment, such as an earlier isolated build.
        cmake_cache = build_dir / "CMakeCache.txt"
        if cmake_cache.exists():
            cache_content = cmake_cache.read_text()
            if "CMAKE_MAKE_PROGRAM" in cache_content and ninja_path not in cache_content:
                self.app.display_info("Ninja path changed, removing stale CMake cache")
                cmake_cache.unlink()
        self._run_command(configure_args)

        build_args = [
            "cmake",
            "--build",
            str(build_dir),
        ]
        self._run_command(build_args)

        install_args = [
            "cmake",
            "--install",
            str(build_dir),
        ]
        self._run_command(install_args)

        return build_dir / "install_manifest.txt"

    def _find_ninja(self) -> str:
        ninja_path = shutil.which("ninja")
        if ninja_path:
            return ninja_path

        try:
            import ninja  # noqa: PLC0415

            ninja_executable = Path(ninja.BIN_DIR) / "ninja"
            if ninja_executable.exists():
                return str(ninja_executable)
        except (ImportError, AttributeError):
            pass

        msg = "Ninja build tool not found. Install the ninja package."
        raise RuntimeError(msg)

    def _run_command(self, cmd: list[str]) -> None:
        self.app.display_waiting(f"Running: {' '.join(cmd)}")
        subprocess.run(  # noqa: S603
            cmd,
            cwd=self.root,
            check=True,
        )

    @staticmethod
    def _get_manylinux_platform_tag() -> str | None:
        if sys.platform != "linux":
            return None

        try:
            libc_info = os.confstr("CS_GNU_LIBC_VERSION")
            _, version_str = (libc_info or "").split()
        except (ValueError, OSError, AttributeError):
            _, version_str = platform.libc_ver()
            if not version_str:
                return None

        major, minor = version_str.split(".")[:2]
        machine = platform.machine()
        return f"manylinux_{major}_{minor}_{machine}"

    def _include_installed_files(self, build_data: dict[str, Any], install_manifest: Path) -> None:
        project_root = Path(self.root)
        # The manifest keeps gitignored build leftovers out of the wheel.
        installed_files = install_manifest.read_text().splitlines()
        if not installed_files:
            msg = f"cmake --install recorded no files in {install_manifest}"
            raise RuntimeError(msg)

        build_data["artifacts"].extend(
            f"/{Path(installed_file).relative_to(project_root).as_posix()}" for installed_file in installed_files
        )

    def _set_platform_tag(self, build_data: dict[str, Any]) -> None:
        manylinux_platform = self._get_manylinux_platform_tag()
        if manylinux_platform:
            python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
            build_data["tag"] = f"{python_tag}-{python_tag}-{manylinux_platform}"
        else:
            build_data["infer_tag"] = True
