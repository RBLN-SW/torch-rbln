// =============================================================================
// TEMPORARY: auto-discover RBLN_RDMA_IP from /sys/class/infiniband RoCE v2 GIDs.
//
// This is a workaround for the period before librbln-ccl performs the
// discovery internally. Once the native runtime can resolve the local RoCE
// IPv4 itself (mirroring NCCL's ncclIbGetGidIndex), this file pair must be
// removed.
//
// Removal procedure (all steps in a single commit):
//   1. Delete this file and RdmaIpAutoDiscovery.cpp.
//   2. Delete the call to MaybeAutoDiscoverRbnRdmaIp() and the include of
//      this header in ProcessGroupRBLN.cpp.
//   3. Delete the RdmaIpAutoDiscovery.cpp entry in the sibling CMakeLists.txt.
//
// `grep -rn "MaybeAutoDiscoverRbnRdmaIp\|RdmaIpAutoDiscovery\|RBLN_DISABLE_AUTO_RDMA_IP"`
// must return zero matches once removal is complete.
//
// The rest of the codebase communicates with this code only via the
// RBLN_RDMA_IP environment variable, so removal does not require touching
// any other file beyond the three above.
// =============================================================================
#pragma once

// NOTE: must NOT be nested under namespace c10d -- a nested `rbln` namespace
// there would shadow the top-level `::rbln` (rebel-compiler runtime) and break
// every `rbln::MemoryInfo` / `rbln::DataType` reference inside c10d code.
namespace torch_rbln::detail {

// Resolve RBLN_RDMA_IP for the running process. Resolution priority:
//
//   1. RBLN_RDMA_IP already set -- no-op (caller-provided value wins).
//   2. RBLN_RDMA_HCA set -- HCA device name as listed under
//      /sys/class/infiniband (e.g. "rocep99s0", "mlx5_0"); an
//      optional ":<port>" / "/<port>" suffix is parsed and ignored.
//      The HCA's first bound netdev is resolved to an IPv4 and
//      written to RBLN_RDMA_IP. Failure is logged but non-fatal.
//      Mirrors ssw-common-umd PR #1930 in librbln-ccl.
//   3. RBLN_RDMA_HCA unset -- do nothing; RBLN_RDMA_IP stays unset.
//
// No auto-discovery: picking an HCA across mixed-vendor hosts is the
// caller's responsibility (e.g. RBLN_RDMA_HCA=rocep99s0 to pin a
// Broadcom NIC over a co-located Intel iRDMA one).
//
// Idempotent (c10::once_flag inside). Set RBLN_DISABLE_AUTO_RDMA_IP=1
// to skip every step above and leave the environment alone.
//
// Never throws: when no IP can be resolved, logs a warning (if
// RCCL_PORT_GEN is set) or an info diagnostic (otherwise) and lets
// librbln-ccl decide whether to fail at RCCL init -- recent
// librbln-ccl no longer requires RBLN_RDMA_IP on single-node runs.
void MaybeAutoDiscoverRbnRdmaIp();

} // namespace torch_rbln::detail
