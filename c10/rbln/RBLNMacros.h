#pragma once

// Macros for controlling symbol visibility in shared libraries.
// We need one set of macros for every separate library we build.

#ifdef _WIN32
#if defined(C10_RBLN_BUILD_SHARED_LIBS)
#define C10_RBLN_EXPORT __declspec(dllexport)
#define C10_RBLN_IMPORT __declspec(dllimport)
#else
#define C10_RBLN_EXPORT
#define C10_RBLN_IMPORT
#endif
#else // _WIN32
#if defined(__GNUC__)
#define C10_RBLN_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define C10_RBLN_EXPORT
#endif // defined(__GNUC__)
#define C10_RBLN_IMPORT C10_RBLN_EXPORT
#endif // _WIN32

// This one is being used by libc10_rbln.so
#ifdef C10_RBLN_BUILD_MAIN_LIB
#define C10_RBLN_API C10_RBLN_EXPORT
#else
#define C10_RBLN_API C10_RBLN_IMPORT
#endif
