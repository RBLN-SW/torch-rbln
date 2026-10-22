"""torch.rbln.ipc end to end: a second process opens exported tensors and works on them.

One opener process serves the whole class over a pipe, standing in for an LMCache cache
server next to a vLLM worker. The exporter (this process) plays the worker: it owns the
tensors, the opener reads them (store), writes them (retrieve) and runs device ops on them.
"""

import json
import multiprocessing
import os
import pickle
from multiprocessing import reduction

import pytest
import torch
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase, instantiate_parametrized_tests

import torch_rbln  # noqa: F401


DEVICE = "rbln:0"


def _accel_node_accessible() -> bool:
    return os.access("/dev/accel/accel0", os.R_OK | os.W_OK)


def _opener_main(conn):
    """Child: open handles it is sent and run commands on the opened tensors."""
    import torch  # noqa: PLC0415
    import torch_rbln  # noqa: F401, PLC0415

    tensors = {}
    while True:
        cmd, *args = conn.recv()
        if cmd == "exit":
            break
        try:
            if cmd == "open":
                key, handle = args
                fds = [reduction.recv_handle(conn) for _ in handle.fds]
                try:
                    tensors[key] = torch.rbln.ipc.open_tensor(handle, fds, device=handle.device_index)
                finally:
                    for fd in fds:
                        os.close(fd)
                result = None
            elif cmd == "read":
                result = tensors[args[0]].cpu()
            elif cmd == "write":
                key, src = args
                tensors[key].copy_(src)
                result = None
            elif cmd == "add_":
                key, value = args
                tensors[key].add_(value)
                result = None
            elif cmd == "gather":
                key, dim, index = args
                result = tensors[key].index_select(dim, index.to(tensors[key].device)).cpu()
            elif cmd == "scatter":
                key, dim, index, src = args
                t = tensors[key]
                t.index_copy_(dim, index.to(t.device), src.to(t.device))
                result = None
            elif cmd == "copy_to_local":
                local = torch.empty_like(tensors[args[0]])
                local.copy_(tensors[args[0]])
                result = local.cpu()
            elif cmd == "compiled_double":
                f = torch.compile(lambda t: t * 2, backend="rbln", dynamic=False)
                result = f(tensors[args[0]]).cpu()
            elif cmd == "export":
                torch.rbln.ipc.export_tensor(tensors[args[0]])
                result = None
            elif cmd == "drop":
                del tensors[args[0]]
                result = None
            else:
                raise ValueError(cmd)
            torch.rbln.synchronize()
            conn.send(("ok", result))
        except Exception as e:  # noqa: BLE001
            conn.send(("err", repr(e)))
    conn.close()
    os._exit(0)


def _sharded_layout(n: int, chiplets: int, dtype_code: int = 7, elem_size: int = 4) -> str:
    per = n // chiplets
    return json.dumps(
        {
            "on_device": {
                "allocs": [{"node_id": 0, "chiplet_id": c, "size": per * elem_size} for c in range(chiplets)],
                "tensor_transform": {
                    "user_tensor_meta": {"dtype": dtype_code, "shape": [n]},
                    "physical_tensor_meta": {"dtype": dtype_code, "shape": [n]},
                    "op_replay": [
                        {
                            "src_elem_count": n,
                            "dst_elem_counts": [per] * chiplets,
                            "relocations": [
                                {
                                    "original_elems_start_index": c * per,
                                    "elems_count": per,
                                    "device_alloc_idx": c,
                                    "device_elems_start_index": 0,
                                }
                                for c in range(chiplets)
                            ],
                            "empty_elements_value": [],
                        }
                    ],
                },
            }
        }
    )


def _exportable(src: torch.Tensor, layout: str = None) -> torch.Tensor:
    x = torch.rbln.ipc.empty_exportable(src.shape, dtype=src.dtype, device=DEVICE)
    if layout is not None:
        from rebel import _C  # noqa: PLC0415

        _C.vmem.debug.set_device_alloc_by_json(x.data_ptr(), layout)
    x.copy_(src)
    torch.rbln.synchronize()
    return x


def _values(shape, dtype, seed=0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if dtype.is_floating_point:
        return torch.randn(shape, generator=g).to(dtype)
    return torch.randint(-100, 100, shape, generator=g).to(dtype)


@pytest.mark.test_set_ci
@pytest.mark.skipif(not _accel_node_accessible(), reason="/dev/accel/accel0 not accessible (export/import need the DRM accel node)")
class TestIpcScenarios(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        ctx = multiprocessing.get_context("spawn")
        cls._conn, child = ctx.Pipe()
        cls._proc = ctx.Process(target=_opener_main, args=(child,))
        cls._proc.start()

    @classmethod
    def tearDownClass(cls):
        try:
            cls._conn.send(("exit",))
        finally:
            cls._proc.join(timeout=120)
        super().tearDownClass()

    def _call(self, *msg):
        self._conn.send(msg)
        status, result = self._conn.recv()
        self.assertEqual(status, "ok", result)
        return result

    def _call_expecting_error(self, *msg) -> str:
        self._conn.send(msg)
        status, result = self._conn.recv()
        self.assertEqual(status, "err", result)
        return result

    def _share(self, key: str, x: torch.Tensor):
        handle = torch.rbln.ipc.export_tensor(x)
        try:
            self._conn.send(("open", key, handle._replace(fds=[-1] * len(handle.fds))))
            for fd in handle.fds:
                reduction.send_handle(self._conn, fd, self._proc.pid)
            status, result = self._conn.recv()
            self.assertEqual(status, "ok", result)
        finally:
            for fd in handle.fds:
                os.close(fd)
        return handle

    # --- store direction: the opener reads what the exporter wrote ---------------------

    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int32, torch.int8])
    def test_opener_reads_every_dtype(self, dtype):
        src = _values((4, 256), dtype)
        x = _exportable(src)
        self._share(f"dtype-{dtype}", x)
        self.assertEqual(self._call("read", f"dtype-{dtype}"), src)
        self._call("drop", f"dtype-{dtype}")

    def test_opener_gathers_kv_blocks(self):
        # blocks-first KV cache of one layer: [num_blocks, kv, heads, 1, block_size, head_dim]
        src = _values((8, 2, 2, 1, 16, 64), torch.bfloat16)
        kv = _exportable(src)
        self._share("kv-gather", kv)
        blocks = torch.tensor([1, 3, 6])
        self.assertEqual(self._call("gather", "kv-gather", 0, blocks), src.index_select(0, blocks))
        self._call("drop", "kv-gather")

    def test_opener_sees_exporter_writes_after_open(self):
        x = _exportable(torch.zeros(1024))
        self._share("late-write", x)
        x.copy_(torch.arange(1024, dtype=torch.float32))
        torch.rbln.synchronize()
        self.assertEqual(self._call("read", "late-write"), torch.arange(1024, dtype=torch.float32))
        self._call("drop", "late-write")

    def test_opener_copies_into_its_own_memory(self):
        src = _values((64, 128), torch.bfloat16)
        x = _exportable(src)
        self._share("d2d", x)
        self.assertEqual(self._call("copy_to_local", "d2d"), src)
        self._call("drop", "d2d")

    def test_many_layers_stay_distinct(self):
        layers = [_exportable(torch.full((2, 512), float(i))) for i in range(8)]
        for i, layer in enumerate(layers):
            self._share(f"layer-{i}", layer)
        for i in range(8):
            self.assertEqual(self._call("read", f"layer-{i}"), torch.full((2, 512), float(i)))
            self._call("drop", f"layer-{i}")

    # --- retrieve direction: the exporter sees what the opener wrote ---------------------

    def test_exporter_sees_opener_scatter_into_kv_blocks(self):
        kv = _exportable(torch.zeros(8, 2, 2, 1, 16, 64, dtype=torch.bfloat16))
        self._share("kv-scatter", kv)
        blocks = torch.tensor([0, 5])
        payload = _values((2, 2, 2, 1, 16, 64), torch.bfloat16, seed=1)
        self._call("scatter", "kv-scatter", 0, blocks, payload)
        expected = torch.zeros(8, 2, 2, 1, 16, 64, dtype=torch.bfloat16)
        expected[blocks] = payload
        self.assertEqual(kv.cpu(), expected)
        self._call("drop", "kv-scatter")

    def test_exporter_sees_opener_host_write(self):
        x = _exportable(torch.zeros(1024))
        self._share("host-write", x)
        self._call("write", "host-write", torch.arange(1024, dtype=torch.float32))
        self.assertEqual(x.cpu(), torch.arange(1024, dtype=torch.float32))
        self._call("drop", "host-write")

    def test_exporter_sees_opener_device_op(self):
        x = _exportable(torch.ones(4096))
        self._share("device-op", x)
        self._call("add_", "device-op", 2.0)
        self.assertEqual(x.cpu(), torch.full((4096,), 3.0))
        self._call("drop", "device-op")

    def test_round_trip_both_ways(self):
        x = _exportable(torch.zeros(256))
        self._share("ping-pong", x)
        for step in range(3):
            x.add_(1.0)
            torch.rbln.synchronize()
            self.assertEqual(self._call("read", "ping-pong"), torch.full((256,), 2.0 * step + 1))
            self._call("add_", "ping-pong", 1.0)
            self.assertEqual(x.cpu(), torch.full((256,), 2.0 * step + 2))
        self._call("drop", "ping-pong")

    # --- chiplet-sharded layouts ----------------------------------------------------------

    @parametrize("chiplets", [2, 4])
    def test_sharded_tensor_both_ways(self, chiplets):
        n = 4096
        x = _exportable(torch.arange(n, dtype=torch.float32), _sharded_layout(n, chiplets))
        handle = self._share(f"sharded-{chiplets}", x)
        self.assertEqual(len(handle.fds), chiplets)
        self.assertEqual(self._call("read", f"sharded-{chiplets}"), torch.arange(n, dtype=torch.float32))
        self._call("add_", f"sharded-{chiplets}", 1.0)
        self.assertEqual(x.cpu(), torch.arange(n, dtype=torch.float32) + 1)
        index = torch.tensor([0, n // chiplets, n - 1])
        self.assertEqual(self._call("gather", f"sharded-{chiplets}", 0, index), index.to(torch.float32) + 1)
        self._call("drop", f"sharded-{chiplets}")

    # --- graph mode -----------------------------------------------------------------------

    def test_compiled_graph_on_exported_and_opened_tensor(self):
        f = torch.compile(lambda t: t + 1, backend="rbln", dynamic=False)
        x = _exportable(torch.arange(1024, dtype=torch.float32))
        self.assertEqual(f(x).cpu(), torch.arange(1024, dtype=torch.float32) + 1)
        self._share("graph", x)
        self.assertEqual(self._call("compiled_double", "graph"), torch.arange(1024, dtype=torch.float32) * 2)
        self.assertEqual(f(x).cpu(), torch.arange(1024, dtype=torch.float32) + 1)
        self._call("drop", "graph")

    # --- handles and lifetime -------------------------------------------------------------

    def test_two_exports_of_one_tensor_alias(self):
        x = _exportable(torch.zeros(512))
        self._share("alias-a", x)
        self._share("alias-b", x)
        self._call("write", "alias-a", torch.full((512,), 7.0))
        self.assertEqual(self._call("read", "alias-b"), torch.full((512,), 7.0))
        self._call("drop", "alias-a")
        self._call("drop", "alias-b")

    def test_opened_tensor_cannot_be_reexported(self):
        x = _exportable(torch.zeros(512))
        self._share("reexport", x)
        self.assertIn("empty_exportable", self._call_expecting_error("export", "reexport"))
        self._call("drop", "reexport")

    def test_exporter_keeps_working_after_opener_drops(self):
        x = _exportable(torch.zeros(512))
        self._share("dropped", x)
        self._call("drop", "dropped")
        x.add_(5.0)
        self.assertEqual(x.cpu(), torch.full((512,), 5.0))

    def test_handle_pickles_without_fds(self):
        x = _exportable(torch.zeros(3, 5, dtype=torch.bfloat16))
        h = torch.rbln.ipc.export_tensor(x)
        try:
            back = pickle.loads(pickle.dumps(h._replace(fds=[])))
            self.assertEqual(back.payload, h.payload)
            self.assertEqual(back.shape, h.shape)
            self.assertEqual(back.dtype, h.dtype)
            self.assertEqual(back.nbytes, 3 * 5 * 2)
        finally:
            for fd in h.fds:
                os.close(fd)

    def test_exporting_process_cannot_open_its_own_handle(self):
        x = _exportable(torch.zeros(256))
        h = torch.rbln.ipc.export_tensor(x)
        try:
            with self.assertRaisesRegex(RuntimeError, "rbln_import_device_memory failed"):
                torch.rbln.ipc.open_tensor(h)
        finally:
            for fd in h.fds:
                os.close(fd)
        x.fill_(4.0)
        self.assertEqual(x.cpu(), torch.full((256,), 4.0))

    def test_opener_releases_shared_import_with_last_reference(self):
        x = _exportable(torch.zeros(512))
        self._share("refs-a", x)
        self._share("refs-b", x)
        self._call("drop", "refs-a")
        self._call("write", "refs-b", torch.full((512,), 3.0))
        self.assertEqual(x.cpu(), torch.full((512,), 3.0))
        self._call("drop", "refs-b")

    def test_exported_tensor_keeps_normal_ops(self):
        x = _exportable(torch.arange(1024, dtype=torch.float32))
        h = torch.rbln.ipc.export_tensor(x)
        for fd in h.fds:
            os.close(fd)
        x.mul_(2.0)
        x[:10].fill_(-1.0)
        expected = torch.arange(1024, dtype=torch.float32) * 2
        expected[:10] = -1.0
        self.assertEqual(x.cpu(), expected)
        self.assertEqual((x + x).cpu(), expected * 2)


instantiate_parametrized_tests(TestIpcScenarios)


if __name__ == "__main__":
    run_tests()
