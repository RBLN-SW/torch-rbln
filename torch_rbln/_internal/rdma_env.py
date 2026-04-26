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
import shutil
import socket
import struct
import subprocess
import sys
from pathlib import Path


try:  # POSIX-only; Linux is the only RBLN runtime target.
    import fcntl as _fcntl
except ImportError:
    _fcntl = None  # type: ignore[assignment]


_SYSFS_INFINIBAND = Path("/sys/class/infiniband")
_RDMA_LINK_TIMEOUT_SEC = 5.0
_LOOPBACK = "127.0.0.1"
_DIAG_PREFIX = "[rbln_rdma_probe]"
# Linux SIOCGIFADDR ioctl number — get the IPv4 address bound to an iface.
# Lets us read the IPv4 without depending on the `ip` (iproute2) binary,
# which may be missing from minimal CI containers.
_SIOCGIFADDR = 0x8915
_IFNAMSIZ = 16  # IFNAMSIZ from <linux/if.h> (15-byte name + NUL)


def _diag(msg: str) -> None:
    """Print a diagnostic line to stderr with a greppable prefix.

    Unconditional so CI logs capture every probe decision (no env toggle).
    The emitted lines let an operator identify which filter stage (sysfs
    missing, operstate down, no IPv4, rdma link DOWN/DISABLED, missing
    `ip` or `rdma` binary) caused an empty RBLN_RDMA_IP result without
    shell access to the CI host.
    """
    print(f"{_DIAG_PREFIX} {msg}", file=sys.stderr, flush=True)


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
    """Read the IPv4 bound to ``iface`` via ``SIOCGIFADDR`` ioctl.

    Avoids the iproute2 ``ip`` binary so the probe still works in minimal
    CI containers. ``OSError`` here typically means the kernel responded
    with ``EADDRNOTAVAIL`` (interface has no IPv4) or ``ENODEV``
    (interface vanished) — both are treated as "no IPv4 available".
    """
    if _fcntl is None:
        _diag(f"iface={iface} fcntl module unavailable (non-POSIX runtime)")
        return None
    iface_bytes = iface.encode("utf-8")
    if len(iface_bytes) >= _IFNAMSIZ:
        _diag(f"iface={iface} name too long for ifreq (max {_IFNAMSIZ - 1} bytes)")
        return None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except OSError as exc:
        _diag(f"iface={iface} AF_INET socket() failed: {exc!r}")
        return None
    try:
        # ifreq layout: char ifr_name[IFNAMSIZ]; union { struct sockaddr ifr_addr; ... }
        # We send a 256-byte buffer with the iface name; kernel writes sockaddr_in into bytes 16:.
        ifreq = struct.pack("256s", iface_bytes)
        try:
            packed = _fcntl.ioctl(sock.fileno(), _SIOCGIFADDR, ifreq)
        except OSError as exc:
            _diag(f"iface={iface} SIOCGIFADDR ioctl failed: {exc!r}")
            return None
    finally:
        sock.close()
    # sockaddr_in: sa_family(2) + sin_port(2) + sin_addr(4) at offset 20 within ifreq.
    return socket.inet_ntoa(packed[20:24])


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
    except (OSError, subprocess.TimeoutExpired) as exc:
        _diag(f"dev={dev_name} `rdma link show` failed: {exc!r}")
        return None
    out = proc.stdout + proc.stderr
    if "state ACTIVE" in out:
        return True
    if "state DOWN" in out or "state DISABLED" in out:
        return False
    return None


def _diag_environment_preamble() -> None:
    """One-time snapshot of the bits that drive the RoCE probe."""
    _diag(f"sysfs={_SYSFS_INFINIBAND} exists={_SYSFS_INFINIBAND.is_dir()}")
    _diag(f"fcntl_module={'present' if _fcntl is not None else 'missing'}  rdma_bin={shutil.which('rdma')}")
    if _SYSFS_INFINIBAND.is_dir():
        try:
            entries = sorted(e.name for e in _SYSFS_INFINIBAND.iterdir())
        except OSError as exc:
            _diag(f"sysfs iterdir failed: {exc!r}")
            return
        _diag(f"sysfs entries (n={len(entries)}): {entries}")


def probe_roce_rdma_ipv4() -> str | None:
    """Pick an IPv4 on an up netdev under ``/sys/class/infiniband/*/device/net``.

    Prefers interfaces whose RDMA link is ACTIVE when ``rdma`` from rdma-core
    is available; otherwise returns the first usable IPv4. Emits one
    ``[rbln_rdma_probe]`` line per filter decision on stderr so CI logs
    surface the reason an empty result was produced without needing shell
    access to the runner.
    """
    _diag_environment_preamble()

    if not _SYSFS_INFINIBAND.is_dir():
        _diag(f"result=None reason=sysfs-missing path={_SYSFS_INFINIBAND}")
        return None

    active_first: list[tuple[str, str, str]] = []
    fallback: list[tuple[str, str, str]] = []

    for rdma_dev in sorted(_SYSFS_INFINIBAND.iterdir()):
        if not rdma_dev.is_dir():
            _diag(f"skip dev={rdma_dev.name} reason=not-a-directory")
            continue
        dev_name = rdma_dev.name
        net_root = rdma_dev / "device" / "net"
        if not net_root.is_dir():
            _diag(f"skip dev={dev_name} reason=no-device/net (path={net_root})")
            continue
        netdevs = sorted(net_root.iterdir())
        _diag(f"dev={dev_name} netdevs={[n.name for n in netdevs]}")
        for netdev in netdevs:
            if not netdev.is_dir():
                _diag(f"skip dev={dev_name} netdev={netdev.name} reason=not-a-directory")
                continue
            iface = netdev.name
            operstate = _read_operstate(netdev)
            if operstate != "up":
                _diag(f"skip dev={dev_name} iface={iface} reason=operstate={operstate!r} (need 'up')")
                continue
            ipv4 = _ipv4_for_iface(iface)
            if not ipv4:
                _diag(f"skip dev={dev_name} iface={iface} reason=no-ipv4-assigned")
                continue
            link_ok = _rdma_port_active(dev_name)
            if link_ok is False:
                _diag(f"skip dev={dev_name} iface={iface} ipv4={ipv4} reason=rdma-link-DOWN/DISABLED")
                continue
            tup = (dev_name, iface, ipv4)
            if link_ok is True:
                _diag(f"candidate dev={dev_name} iface={iface} ipv4={ipv4} bucket=active")
                active_first.append(tup)
            else:
                _diag(f"candidate dev={dev_name} iface={iface} ipv4={ipv4} bucket=fallback (rdma link unknown)")
                fallback.append(tup)

    if active_first:
        chosen = active_first[0]
        _diag(f"result={chosen[2]} via dev={chosen[0]} iface={chosen[1]} bucket=active")
        return chosen[2]
    if fallback:
        chosen = fallback[0]
        _diag(f"result={chosen[2]} via dev={chosen[0]} iface={chosen[1]} bucket=fallback")
        return chosen[2]
    _diag("result=None reason=no-candidate-passed-filters")
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
        print(f"RBLN_RDMA_IP={existing}")
        return

    discovered = probe_roce_rdma_ipv4()
    if discovered:
        os.environ["RBLN_RDMA_IP"] = discovered
    print(f"RBLN_RDMA_IP={os.environ.get('RBLN_RDMA_IP', '')}")
