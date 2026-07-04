#ifndef REBEL_RUNTIME_API_RBLN_RETCODE_H
#define REBEL_RUNTIME_API_RBLN_RETCODE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  RBLNRetCode_SUCCESS = 0,
  RBLNRetCode_FAILURE,
  RBLNRetCode_INVALID,
} RBLNRetCode;

#ifdef __cplusplus
}
#endif

#endif  // REBEL_RUNTIME_API_RBLN_RETCODE_H
