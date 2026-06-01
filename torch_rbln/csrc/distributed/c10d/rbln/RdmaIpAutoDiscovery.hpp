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
//   1. RBLN_RDMA_HCA -- mirrors the explicit-HCA mechanism planned in
//      librbln-ccl (ssw-common-umd PR #1930). Value is the HCA device
//      name as listed under /sys/class/infiniband (e.g. "rocep99s0",
//      "mlx5_0"); an optional ":<port>" / "/<port>" suffix is parsed
//      and ignored. The HCA's first bound netdev is resolved to an
//      IPv4 and overrides any existing RBLN_RDMA_IP.
//   2. Existing RBLN_RDMA_IP -- left as-is.
//   3. Auto-discovery -- probes /sys/class/infiniband for a RoCE v2
//      capable device whose port is ACTIVE and whose netdev has an
//      IPv4 matching the GID table. Devices are ranked by vendor
//      priority (Broadcom bnxt_re first, Intel irdma last) to avoid
//      picking an iRDMA NIC on mixed-vendor hosts where it is known
//      to mis-bind to RBLN traffic.
//
// Idempotent (c10::once_flag inside). Set RBLN_DISABLE_AUTO_RDMA_IP=1
// to skip every step above and leave the environment alone.
//
// Never throws: when no IP can be found, logs a warning (if
// RCCL_PORT_GEN is set) or an info diagnostic (otherwise) and lets
// librbln-ccl decide whether to fail at RCCL init -- recent
// librbln-ccl no longer requires RBLN_RDMA_IP on single-node runs.
void MaybeAutoDiscoverRbnRdmaIp();

} // namespace torch_rbln::detail
