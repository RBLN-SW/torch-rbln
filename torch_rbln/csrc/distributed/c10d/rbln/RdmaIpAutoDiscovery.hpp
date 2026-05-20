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
namespace torch_rbln {
namespace detail {

// Auto-fill RBLN_RDMA_IP by probing /sys/class/infiniband for a RoCE v2
// capable device whose port is ACTIVE and whose netdev has an IPv4 matching
// the GID table. Idempotent (std::once_flag inside). Respects pre-set
// RBLN_RDMA_IP (does not overwrite). Set RBLN_DISABLE_AUTO_RDMA_IP=1 to skip.
// Throws (via RBLN_CHECK) when RCCL_PORT_GEN is set but no IP can be found.
void MaybeAutoDiscoverRbnRdmaIp();

} // namespace detail
} // namespace torch_rbln
