"""Default RBLN network-related environment before native runtime loads.

Control-plane addresses:

- If ``RBLN_LOCAL_IP`` / ``RBLN_ROOT_IP`` are **unset or empty**, they default to
  ``127.0.0.1``. Values set in the environment **before** ``torch_rbln`` is
  imported (e.g. multi-node jobs) are kept.
- Set ``RBLN_FORCE_LOOPBACK_CONTROL_PLANE=1`` to force both to ``127.0.0.1``
  even when they were already set (single-node / debugging).

``RBLN_RDMA_IP`` is filled by probing RoCEv2-capable RDMA netdevs (sysfs mapping
+ optional ``rdma link``) when unset, matching the manual steps used on
Ubuntu 22.04+.

This module is not run at ``torch_rbln`` import time. Autoport distributed tests
call it via ``test.utils.configure_rbln_network_for_autoport_tests`` before
spawning subprocesses so each child inherits the variables when ``librbln``
loads.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

_SYSFS_INFINIBAND = Path("/sys/class/infiniband")
_RDMA_LINK_TIMEOUT_SEC = 5.0
_LOOPBACK = "127.0.0.1"


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _apply_control_plane_ips() -> None:
    """Default LOCAL/ROOT to loopback; respect pre-set env; optional force flag."""
    if _env_truthy("RBLN_FORCE_LOOPBACK_CONTROL_PLANE"):
        os.environ["RBLN_LOCAL_IP"] = _LOOPBACK
        os.environ["RBLN_ROOT_IP"] = _LOOPBACK
        return
    if not os.environ.get("RBLN_LOCAL_IP", "").strip():
        os.environ["RBLN_LOCAL_IP"] = _LOOPBACK
    if not os.environ.get("RBLN_ROOT_IP", "").strip():
        os.environ["RBLN_ROOT_IP"] = _LOOPBACK


def _read_operstate(netdev_dir: Path) -> str | None:
    try:
        return (netdev_dir / "operstate").read_text().strip()
    except OSError:
        return None


def _ipv4_for_iface(iface: str) -> str | None:
    ip_bin = shutil.which("ip")
    if not ip_bin:
        return None
    try:
        proc = subprocess.run(
            [ip_bin, "-4", "addr", "show", iface],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)/", proc.stdout)
    return m.group(1) if m else None


def _rdma_port_active(dev_name: str) -> bool | None:
    """Return True/False from ``rdma link``, or None if ``rdma`` is missing or unparsable."""
    rdma_bin = shutil.which("rdma")
    if not rdma_bin:
        return None
    try:
        proc = subprocess.run(
            [rdma_bin, "link", "show", f"{dev_name}/1"],
            capture_output=True,
            text=True,
            timeout=_RDMA_LINK_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = proc.stdout + proc.stderr
    if "state ACTIVE" in out:
        return True
    if "state DOWN" in out or "state DISABLED" in out:
        return False
    return None


def probe_roce_rdma_ipv4() -> str | None:
    """Pick an IPv4 on an up netdev under ``/sys/class/infiniband/*/device/net``.

    Prefers interfaces whose RDMA link is ACTIVE when ``rdma`` from rdma-core
    is available; otherwise returns the first usable IPv4.
    """
    if not _SYSFS_INFINIBAND.is_dir():
        return None

    active_first: list[tuple[str, str, str]] = []
    fallback: list[tuple[str, str, str]] = []

    for rdma_dev in sorted(_SYSFS_INFINIBAND.iterdir()):
        if not rdma_dev.is_dir():
            continue
        dev_name = rdma_dev.name
        net_root = rdma_dev / "device" / "net"
        if not net_root.is_dir():
            continue
        for netdev in sorted(net_root.iterdir()):
            if not netdev.is_dir():
                continue
            if _read_operstate(netdev) != "up":
                continue
            iface = netdev.name
            ipv4 = _ipv4_for_iface(iface)
            if not ipv4:
                continue
            link_ok = _rdma_port_active(dev_name)
            if link_ok is False:
                continue
            tup = (dev_name, iface, ipv4)
            if link_ok is True:
                active_first.append(tup)
            else:
                fallback.append(tup)

    if active_first:
        return active_first[0][2]
    if fallback:
        return fallback[0][2]
    return None


def apply_default_rbln_network_environment() -> None:
    """Apply control-plane defaults and auto-fill ``RBLN_RDMA_IP`` when unset.

    See module docstring for ``RBLN_LOCAL_IP`` / ``RBLN_ROOT_IP`` and
    ``RBLN_FORCE_LOOPBACK_CONTROL_PLANE``. Safe to call once before loading
    ``librbln``. Non-empty ``RBLN_RDMA_IP`` is left unchanged.
    """
    _apply_control_plane_ips()

    existing = os.environ.get("RBLN_RDMA_IP", "").strip()
    if existing:
        return

    discovered = probe_roce_rdma_ipv4()
    if discovered:
        os.environ["RBLN_RDMA_IP"] = discovered
