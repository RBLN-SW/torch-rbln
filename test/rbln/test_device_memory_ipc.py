"""Cross-process device memory handles: torch.rbln.empty_exportable / export_device_memory / import_device_memory."""

import multiprocessing
import os
from multiprocessing import reduction

import pytest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_rbln  # noqa: F401


def _accel_node_accessible() -> bool:
    return os.access("/dev/accel/accel0", os.R_OK | os.W_OK)


def _importer_main(conn, size, offset, shape, dtype, device_index, expected_sum):
    """Child: receive fd, import, read back, report."""
    import torch  # noqa: PLC0415
    import torch_rbln  # noqa: F401, PLC0415

    fd = reduction.recv_handle(conn)
    try:
        t = torch.rbln.import_device_memory(fd, size, offset, shape, dtype, device=device_index)
        got = t.cpu()
        conn.send(("ok", float(got.sum()), got[:4].tolist()))
        del t
    except Exception as e:  # noqa: BLE001
        conn.send(("err", repr(e), None))
    finally:
        os.close(fd)
        conn.close()
    os._exit(0)


def _exportable_arange(n: int) -> torch.Tensor:
    x = torch.rbln.empty_exportable((n,), dtype=torch.float32, device="rbln:0")
    x.copy_(torch.arange(n, dtype=torch.float32))
    torch.rbln.synchronize()
    return x


@pytest.mark.test_set_ci
@pytest.mark.skipif(not _accel_node_accessible(), reason="/dev/accel/accel0 not accessible (export/import need the DRM accel node)")
class TestDeviceMemoryIPC(TestCase):
    def test_empty_exportable_allocates_on_device(self):
        x = torch.rbln.empty_exportable((4, 8), dtype=torch.bfloat16, device="rbln:0")
        self.assertEqual(x.shape, torch.Size([4, 8]))
        self.assertEqual(x.dtype, torch.bfloat16)
        self.assertEqual(x.device, torch.device("rbln:0"))

    def test_export_returns_handle_covering_allocation(self):
        x = _exportable_arange(1024)
        h = torch.rbln.export_device_memory(x)
        try:
            self.assertGreater(h.fd, 0)
            self.assertGreaterEqual(h.size, x.numel() * x.element_size())
            self.assertEqual(h.offset, 0)
            self.assertEqual(h.shape, x.shape)
            self.assertEqual(h.dtype, torch.float32)
            self.assertEqual(h.device_index, 0)
        finally:
            os.close(h.fd)

    def test_export_rejects_non_exportable_tensor(self):
        x = torch.arange(1024, dtype=torch.float32, device="rbln:0")
        with self.assertRaisesRegex(RuntimeError, "empty_exportable"):
            torch.rbln.export_device_memory(x)

    def test_export_rejects_interior_view(self):
        x = _exportable_arange(1024)
        with self.assertRaisesRegex(RuntimeError, "whole storage"):
            torch.rbln.export_device_memory(x[16:])

    def test_export_rejects_cpu_tensor(self):
        with self.assertRaisesRegex(RuntimeError, "expected an rbln tensor"):
            torch.rbln.export_device_memory(torch.ones(4))

    def test_cross_process_import_reads_exporter_data(self):
        x = _exportable_arange(1024)
        h = torch.rbln.export_device_memory(x)

        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe()
        p = ctx.Process(
            target=_importer_main,
            args=(child, h.size, h.offset, tuple(h.shape), h.dtype, h.device_index, float(x.cpu().sum())),
        )
        p.start()
        try:
            reduction.send_handle(parent, h.fd, p.pid)
            status, payload, head = parent.recv()
        finally:
            os.close(h.fd)
            p.join(timeout=120)
        self.assertEqual(status, "ok", payload)
        self.assertEqual(payload, float(sum(range(1024))))
        self.assertEqual(head, [0.0, 1.0, 2.0, 3.0])
        del x


if __name__ == "__main__":
    run_tests()
