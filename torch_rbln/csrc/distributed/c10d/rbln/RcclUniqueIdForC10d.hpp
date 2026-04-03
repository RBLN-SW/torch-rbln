/**
 * @file RcclUniqueIdForC10d.hpp
 * @brief Minimal definitions for rccl_unique_id and rcclGetUniqueId for c10d broadcast.
 *
 * This header avoids including rebel/common/dynamic_library.h (which pulls in
 * TVM and other rebel deps). Layout must match uapi/rccl.h used by rebel.
 */
#pragma once

#ifndef RCCL_IP_STR_LEN
#define RCCL_IP_STR_LEN 16
#endif

struct rccl_unique_id {
  char root_ip[RCCL_IP_STR_LEN];
  char self_ip[RCCL_IP_STR_LEN];
  char self_rdma_ip[RCCL_IP_STR_LEN];
  int root_port;
  int rdma_base_port;
};

namespace rbln {

/** Provided by rebel runtime (dynamic_library.cc). Called by rank 0 to generate ID. */
extern int (*rcclGetUniqueId)(struct rccl_unique_id*);

} // namespace rbln
