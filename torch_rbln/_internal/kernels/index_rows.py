"""Row kernel for ``x[(rows, cols)]`` on a 2-D tensor.

Serves ``x[(r, c)]``, ``x[(r, c)] = v`` (one value per point) and
``x.index_put_((r, c), v, accumulate=True)`` for the form ``try_index_rows`` in
RBLNTensorAdvancedIndexing.cpp lets through.

``x`` [B, V] is viewed as [B, V/64, 64]. A host plan, built when a pair of index
tensors is first seen and reused while neither changes, maps the K points onto
per-batch-row slots:

  idx    [B, T]          the rows each batch row touches, each named once: points
                         that share a row share a slot, and unused slots name
                         distinct untouched rows.
  lane   [J, B, T, 64]   layer j marks the lane of the j-th point, in index order,
                         that falls in a slot. J is the most points one slot holds;
                         a later layer overrides an earlier one, so the last write
                         to a point wins, as on the CPU.
  perm   [J, B*T]        the value each layer of a slot takes.
  eslot, elane [Kp, ...] each point's slot and lane, for the gather.

Values stay on the device: the program places them by gathering rows of the
values expanded to 64 lanes. One program per (kind, B, V/64, T, J, Kp) shape.

The program computes on the device, which rounds float16 to a shorter mantissa
as any compiled op does: a float16 value the device did not produce can change
in its last bit, anywhere in ``x`` for a put. Device-produced values (a model's
logits) and bfloat16 are unchanged.
"""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from typing import Any

import torch

import torch_rbln._C
from torch_rbln._internal.compile_cache import compile_rbln_cached_entry
from torch_rbln._internal.env_utils import use_device_group_num_devices
from torch_rbln._internal.log_utils import rbln_log_debug, rbln_log_warn_once
from torch_rbln._internal.ops_utils import _graph_input_order


_LANES = 64
# IndexRowsKind in RBLNTensorAdvancedIndexing.h
_GATHER, _PUT, _PUT_ACCUMULATE = 0, 1, 2
# Plans kept for distinct index pairs; vLLM holds one pair per logits processor.
_MAX_PLANS = 16


_row_ops_ok: bool | None = None


def _row_ops_available() -> bool:
    """Whether rebel has the ``rbln::batched_*_rows`` ops.

    Checked on first use so that importing torch_rbln does not import the compiler.
    """
    global _row_ops_ok
    if _row_ops_ok is None:
        import rebel.ops  # noqa: F401, PLC0415

        _row_ops_ok = hasattr(torch.ops.rbln, "batched_scatter_add_rows")
        if not _row_ops_ok:
            rbln_log_warn_once("index row kernel off: rebel has no rbln::batched_*_rows ops")
    return _row_ops_ok


# --- Programs -------------------------------------------------------------------


def _place(values, perm_j, shape):
    """Rows ``shape`` [B, T, L] holding ``values[perm_j[s]]`` in every lane."""
    n = values.shape[0]
    rows = torch.ops.rbln.batched_gather_rows(values.view(1, n, 1).expand(1, n, _LANES), perm_j.view(1, -1))
    return rows.view(shape)


def _gather_program(x3, idx, eslot, elane):
    b, t = idx.shape
    rows = torch.ops.rbln.batched_gather_rows(x3, idx).view(1, b * t, _LANES)
    picked = torch.ops.rbln.batched_gather_rows(rows, eslot.view(1, -1)).view(-1, _LANES)
    zero = torch.zeros((), dtype=x3.dtype, device=x3.device)
    # keepdim keeps the sum on the device.
    return torch.where(elane, picked, zero).sum(-1, keepdim=True).view(-1)


def _put_program(x3, idx, lane, perm, values):
    rows = torch.ops.rbln.batched_gather_rows(x3, idx)
    for j in range(lane.shape[0]):
        rows = torch.where(lane[j], _place(values, perm[j], rows.shape), rows)
    return torch.ops.rbln.batched_scatter_rows(x3, idx, rows)


def _put_accumulate_program(x3, idx, lane, perm, values):
    zero = torch.zeros((), dtype=values.dtype, device=values.device)
    update = None
    for j in range(lane.shape[0]):
        layer = torch.where(lane[j], _place(values, perm[j], lane[j].shape), zero)
        update = layer if update is None else update + layer
    return torch.ops.rbln.batched_scatter_add_rows(x3, idx, update)


_PROGRAMS = {_GATHER: _gather_program, _PUT: _put_program, _PUT_ACCUMULATE: _put_accumulate_program}


# Graph input order of each compiled program, by its cache key.
_input_orders: dict[tuple, tuple[int, ...]] = {}


def _run(kind, args, out):
    """Run the program for ``kind`` into ``out`` (no copy after the first call)."""
    options: dict[str, Any] = {"disable_logger": True, "_runtime_holder": []}
    if not use_device_group_num_devices():
        options["num_devices"] = 1
    key = (kind, out.device.index, *((tuple(a.shape), a.dtype) for a in args))
    entry = compile_rbln_cached_entry(_PROGRAMS[kind], dynamic=False, options=options, device_cache_key=key)
    runtime = entry.runtime()
    order = _input_orders.get(key)
    if runtime is not None and order is not None:
        runtime.run(*(args[i] for i in order), out=[out])
        return
    result = entry.compiled(*args)
    runtime = entry.runtime()
    if runtime is not None:
        order = _graph_input_order(runtime, list(args))
        if order is not None:
            _input_orders[key] = order
    out.copy_(result)


# --- Host plan ------------------------------------------------------------------


def _next_pow2(n: int, floor: int) -> int:
    return max(floor, 1 << (n - 1).bit_length())


def _host_index(index: torch.Tensor, size: int, dim: int) -> torch.Tensor:
    if index.numel() and (index.min() < -size or index.max() >= size):
        bad = index[(index < -size) | (index >= size)][0].item()
        msg = f"index {bad} is out of bounds for dimension {dim} with size {size}"
        raise IndexError(msg)
    return torch.where(index < 0, index + size, index)


class _Plan:
    __slots__ = ("device", "elane", "eslot", "idx", "k", "kp", "lane", "perm", "perm_zero")

    def __init__(self, device, k, kp, idx, lane, perm, eslot, elane):
        self.device = device
        self.k, self.kp = k, kp
        self.idx, self.lane, self.perm, self.eslot, self.elane = idx, lane, perm, eslot, elane
        self.perm_zero: torch.Tensor | None = None

    def perm_for(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == self.k:
            return self.perm
        if self.perm_zero is None:
            self.perm_zero = torch.zeros(self.perm.shape, dtype=torch.int32).to(self.device)
        return self.perm_zero


def _build_plan(rows: torch.Tensor, cols: torch.Tensor, b: int, v: int, device: torch.device) -> _Plan | None:
    """The plan for host int64 indices.

    None when a batch row touches more rows than ``x`` has.
    """
    r = _host_index(rows, b, 0)
    c = _host_index(cols, v, 1)
    k = r.numel()
    nrows = v // _LANES
    key = r * nrows + c // _LANES
    slot_key, slot_of = torch.unique(key, return_inverse=True)
    slot_b, slot_row = slot_key // nrows, slot_key % nrows
    count = torch.bincount(slot_b, minlength=b)
    # The row ops take at least 64 rows per batch row.
    t = _next_pow2(int(count.max()), 64)
    if t > nrows:
        return None
    slot_t = torch.arange(slot_key.numel()) - (count.cumsum(0) - count)[slot_b]
    idx = torch.empty(b, t, dtype=torch.int32)
    idx[slot_b, slot_t] = slot_row.to(torch.int32)
    # Fill each batch row's unused slots with its first untouched rows.
    free = torch.ones(b, nrows, dtype=torch.bool)
    free[slot_b, slot_row] = False
    rank = free.cumsum(1)
    pad_b, pad_row = (free & (rank <= (t - count).unsqueeze(1))).nonzero(as_tuple=True)
    idx[pad_b, count[pad_b] + rank[pad_b, pad_row] - 1] = pad_row.to(torch.int32)

    eslot = (slot_b * t + slot_t)[slot_of]
    sorted_slot, order = torch.sort(eslot, stable=True)
    layer = torch.empty(k, dtype=torch.long)
    layer[order] = torch.arange(k) - torch.searchsorted(sorted_slot, sorted_slot)
    j = int(layer.max()) + 1
    lane_of = c % _LANES
    lane = torch.zeros(j, b * t, _LANES, dtype=torch.bool)
    lane[layer, eslot, lane_of] = True
    perm = torch.zeros(j, b * t, dtype=torch.int32)
    perm[layer, eslot] = torch.arange(k, dtype=torch.int32)

    kp = _next_pow2(k, 16)
    eslot_p = torch.zeros(kp, dtype=torch.int32)
    eslot_p[:k] = eslot.to(torch.int32)
    elane = torch.zeros(kp, _LANES, dtype=torch.bool)
    elane[torch.arange(k), lane_of] = True

    rbln_log_debug(f"index_rows plan: shape=({b}, {v}) points={k} T={t} J={j} Kp={kp}")
    return _Plan(
        device,
        k,
        kp,
        idx.to(device),
        lane.view(j, b, t, _LANES).to(device),
        perm.to(device),
        eslot_p.to(device),
        elane.to(device),
    )


class _CachedPlan:
    __slots__ = ("cols", "cols_ref", "plan", "rows", "rows_ref", "versions")

    def __init__(self, rows, cols, host_rows, host_cols, versions, plan):
        self.rows_ref, self.cols_ref = weakref.ref(rows), weakref.ref(cols)
        self.rows, self.cols = host_rows, host_cols
        self.versions = versions
        self.plan = plan


def _versions(rows: torch.Tensor, cols: torch.Tensor) -> tuple[int, int] | None:
    """The index tensors' version counters, or None when one has none (an
    inference tensor, as vLLM's are), so a write to it cannot be seen.
    """
    if rows.is_inference() or cols.is_inference():
        return None
    return rows._version, cols._version


_plans_lock = threading.Lock()
# (id(rows), id(cols), shape, device) -> _CachedPlan
_plans: OrderedDict[tuple, _CachedPlan] = OrderedDict()


def _plan_for(self: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> _Plan | None:
    """The plan for these indices: reused while both index tensors are unchanged.

    The version counters tell; without them the indices are read back and
    compared, which is what the point-by-point path does on every call anyway.
    """
    key = (id(rows), id(cols), tuple(self.shape), self.device)
    versions = _versions(rows, cols)
    with _plans_lock:
        hit = _plans.get(key)
    if hit is not None and (hit.rows_ref() is not rows or hit.cols_ref() is not cols):
        hit = None
    if hit is not None and versions is not None and hit.versions == versions:
        return hit.plan
    host_rows, host_cols = rows.to("cpu", torch.long), cols.to("cpu", torch.long)
    if hit is not None and versions is None and torch.equal(hit.rows, host_rows) and torch.equal(hit.cols, host_cols):
        return hit.plan
    plan = _build_plan(host_rows, host_cols, self.shape[0], self.shape[1], self.device)
    with _plans_lock:
        _plans[key] = _CachedPlan(rows, cols, host_rows, host_cols, versions, plan)
        _plans.move_to_end(key)
        while len(_plans) > _MAX_PLANS:
            _plans.popitem(last=False)
    return plan


# Device buffers the values are padded into, by (device, dtype, length).
_value_buffers: dict[tuple, torch.Tensor] = {}


def _padded_values(values: torch.Tensor, plan: _Plan) -> torch.Tensor:
    if values.numel() != plan.k:
        return values.reshape(1)
    if plan.k == plan.kp:
        return values
    key = (values.device, values.dtype, plan.kp)
    buf = _value_buffers.get(key)
    if buf is None:
        buf = torch.zeros(plan.kp, dtype=values.dtype).to(values.device)
        _value_buffers[key] = buf
    buf[: plan.k].copy_(values)
    return buf


# --- Entry point (installed into the C++ kernels) ----------------------------------


def _index_rows(kind: int, self: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, values_or_out) -> bool:
    if not _row_ops_available():
        return False
    plan = _plan_for(self, rows, cols)
    if plan is None:
        return False
    x3 = self.view(self.shape[0], -1, _LANES)
    if kind == _GATHER:
        out = values_or_out
        if plan.k == plan.kp:
            _run(kind, (x3, plan.idx, plan.eslot, plan.elane), out)
        else:
            full = torch.empty(plan.kp, dtype=out.dtype, device=out.device)
            _run(kind, (x3, plan.idx, plan.eslot, plan.elane), full)
            out.copy_(full[: plan.k])
        return True
    values = values_or_out
    _run(kind, (x3, plan.idx, plan.lane, plan.perm_for(values), _padded_values(values, plan)), x3)
    return True


torch_rbln._C._set_index_rows_kernel(_index_rows)
