"""torch.rbln.ipc: empty_exportable / export_tensor / open_tensor across processes."""

import json
import multiprocessing
import os
from multiprocessing import reduction

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401


def _accel_node_accessible() -> bool:
    return os.access("/dev/accel/accel0", os.R_OK | os.W_OK)


def _opener_main(conn):
    """Child: receive the handle and its fds, open the tensor, report its values."""
    import torch  # noqa: PLC0415
    import torch_rbln  # noqa: F401, PLC0415

    handle = conn.recv()
    fds = [reduction.recv_handle(conn) for _ in handle.fds]
    try:
        t = torch.rbln.ipc.open_tensor(handle, fds, device=handle.device_index)
        conn.send(("ok", t.cpu().tolist()))
        del t
    except Exception as e:  # noqa: BLE001
        conn.send(("err", repr(e)))
    finally:
        for fd in fds:
            os.close(fd)
        conn.close()
    os._exit(0)


def _open_in_child(handle) -> list:
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    p = ctx.Process(target=_opener_main, args=(child,))
    p.start()
    try:
        parent.send(handle._replace(fds=[-1] * len(handle.fds)))
        for fd in handle.fds:
            reduction.send_handle(parent, fd, p.pid)
        status, payload = parent.recv()
    finally:
        for fd in handle.fds:
            os.close(fd)
        p.join(timeout=120)
    assert status == "ok", payload
    return payload


def _two_chiplet_layout(n: int) -> str:
    half = n // 2
    return json.dumps(
        {
            "on_device": {
                "allocs": [
                    {"node_id": 0, "chiplet_id": 0, "size": half * 4},
                    {"node_id": 0, "chiplet_id": 1, "size": half * 4},
                ],
                "tensor_transform": {
                    "user_tensor_meta": {"dtype": 7, "shape": [n]},
                    "physical_tensor_meta": {"dtype": 7, "shape": [n]},
                    "op_replay": [
                        {
                            "src_elem_count": n,
                            "dst_elem_counts": [half, half],
                            "relocations": [
                                {
                                    "original_elems_start_index": 0,
                                    "elems_count": half,
                                    "device_alloc_idx": 0,
                                    "device_elems_start_index": 0,
                                },
                                {
                                    "original_elems_start_index": half,
                                    "elems_count": half,
                                    "device_alloc_idx": 1,
                                    "device_elems_start_index": 0,
                                },
                            ],
                            "empty_elements_value": [],
                        }
                    ],
                },
            }
        }
    )


def _flat_layout(n: int) -> str:
    return json.dumps({"on_device": {"allocs": [{"node_id": 0, "chiplet_id": 0, "size": n * 4}]}})


def _exportable_arange(n: int, layout: str = None) -> torch.Tensor:
    x = torch.rbln.ipc.empty_exportable((n,), dtype=torch.float32, device="rbln:0")
    if layout is not None:
        from rebel import _C  # noqa: PLC0415

        _C.vmem.debug.set_device_alloc_by_json(x.data_ptr(), layout)
    x.copy_(torch.arange(n, dtype=torch.float32))
    torch.rbln.synchronize()
    return x


@pytest.mark.test_set_ci
@pytest.mark.skipif(not _accel_node_accessible(), reason="/dev/accel/accel0 not accessible (export/import need the DRM accel node)")
class TestIpc(TestCase):
    def test_empty_exportable_is_an_rbln_tensor(self):
        x = torch.rbln.ipc.empty_exportable((4, 8), dtype=torch.bfloat16, device="rbln:0")
        self.assertEqual(x.shape, torch.Size([4, 8]))
        self.assertEqual(x.dtype, torch.bfloat16)
        self.assertEqual(x.device, torch.device("rbln:0"))

    def test_export_describes_the_logical_tensor(self):
        x = _exportable_arange(1024)
        h = torch.rbln.ipc.export_tensor(x)
        try:
            self.assertEqual(h.nbytes, 1024 * 4)
            self.assertEqual(h.shape, x.shape)
            self.assertEqual(h.dtype, torch.float32)
            self.assertEqual(h.device_index, 0)
            self.assertIsInstance(h.payload, bytes)
            self.assertEqual(len(h.fds), 1)
        finally:
            for fd in h.fds:
                os.close(fd)

    def test_export_rejects_non_exportable_tensor(self):
        x = torch.arange(1024, dtype=torch.float32, device="rbln:0")
        with self.assertRaisesRegex(RuntimeError, "empty_exportable"):
            torch.rbln.ipc.export_tensor(x)

    def test_export_rejects_interior_view(self):
        x = _exportable_arange(1024)
        with self.assertRaisesRegex(RuntimeError, "whole storage"):
            torch.rbln.ipc.export_tensor(x[16:])

    def test_export_rejects_cpu_tensor(self):
        with self.assertRaisesRegex(RuntimeError, "expected an rbln tensor"):
            torch.rbln.ipc.export_tensor(torch.ones(4))

    def test_open_rejects_wrong_fd_count(self):
        x = _exportable_arange(1024)
        h = torch.rbln.ipc.export_tensor(x)
        try:
            with self.assertRaisesRegex(RuntimeError, "fd"):
                torch.rbln.ipc.open_tensor(h, fds=[])
        finally:
            for fd in h.fds:
                os.close(fd)

    def test_cross_process_open_reads_exporter_data(self):
        x = _exportable_arange(1024)
        got = _open_in_child(torch.rbln.ipc.export_tensor(x))
        self.assertEqual(got, list(range(1024)))

    def test_cross_process_open_of_chiplet_sharded_tensor(self):
        x = _exportable_arange(1024, _two_chiplet_layout(1024))
        h = torch.rbln.ipc.export_tensor(x)
        self.assertEqual(len(h.fds), 2)
        got = _open_in_child(h)
        self.assertEqual(got, list(range(1024)))

    def test_layout_is_fixed_after_export(self):
        from rebel import _C  # noqa: PLC0415

        x = _exportable_arange(1024, _two_chiplet_layout(1024))
        h = torch.rbln.ipc.export_tensor(x)
        for fd in h.fds:
            os.close(fd)
        with self.assertRaisesRegex(RuntimeError, "dma-buf"):
            _C.vmem.debug.set_device_alloc_by_json(x.data_ptr(), _flat_layout(1024))
        self.assertEqual(x.cpu().tolist(), list(range(1024)))


if __name__ == "__main__":
    run_tests()
