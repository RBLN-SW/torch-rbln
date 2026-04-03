#ifndef REBEL_RUNTIME_MEMORY_STATS_H
#define REBEL_RUNTIME_MEMORY_STATS_H

#include <map>
#include <stdexcept>
#include <string>

namespace rbln {

/**
 * @brief Memory statistics tracking class for RBLN caching allocator.
 *
 * This class provides comprehensive memory statistics tracking including:
 * - Allocated Memory: memory currently in use by tensors (allocated_current_)
 * - Reserved Memory: memory allocated from device (reserved_current_)
 * - Active Block Memory: memory blocks currently in use (active_current_)
 * - Cached Block Memory: reusable cached blocks from fragmentation (cached_current_)
 * - Allocation operation counters
 */
class MemoryStats {
 public:
  MemoryStats() = default;

  // Allocated memory statistics - memory currently in use by tensors
  uint64_t GetAllocatedCurrent() const;
  uint64_t GetAllocatedPeak() const;
  uint64_t GetAllocatedTotalAllocated() const;
  uint64_t GetAllocatedTotalFreed() const;

  // Reserved memory statistics - memory managed by the caching allocator
  uint64_t GetReservedCurrent() const;
  uint64_t GetReservedPeak() const;
  uint64_t GetReservedTotalAllocated() const;
  uint64_t GetReservedTotalFreed() const;

  // Active memory statistics - memory blocks currently in use
  uint64_t GetActiveCurrent() const;
  uint64_t GetActivePeak() const;

  // Cached block memory statistics - cached blocks from fragmentation that can be reused
  uint64_t GetCachedCurrent() const;
  uint64_t GetCachedPeak() const;

  // Allocation operation counters
  uint64_t GetNumAllocRetries() const;
  uint64_t GetNumOoms() const;
  uint64_t GetNumDeviceAlloc() const;
  uint64_t GetNumDeviceFree() const;

  // Setters for updating statistics
  void SetAllocatedCurrent(uint64_t value);
  void SetAllocatedPeak(uint64_t value);
  void SetAllocatedTotalAllocated(uint64_t value);
  void SetAllocatedTotalFreed(uint64_t value);

  void SetReservedCurrent(uint64_t value);
  void SetReservedPeak(uint64_t value);
  void SetReservedTotalAllocated(uint64_t value);
  void SetReservedTotalFreed(uint64_t value);

  void SetActiveCurrent(uint64_t value);
  void SetActivePeak(uint64_t value);

  void SetCachedCurrent(uint64_t value);
  void SetCachedPeak(uint64_t value);

  void SetNumAllocRetries(uint64_t value);
  void SetNumOoms(uint64_t value);
  void SetNumDeviceAlloc(uint64_t value);
  void SetNumDeviceFree(uint64_t value);

  // Helper methods for updating statistics
  void AddAllocated(uint64_t bytes);
  void SubtractAllocated(uint64_t bytes);
  void AddReserved(uint64_t bytes);
  void SubtractReserved(uint64_t bytes);
  void AddActive(uint64_t bytes);
  void SubtractActive(uint64_t bytes);
  void AddCached(uint64_t bytes);
  void SubtractCached(uint64_t bytes);
  void IncrementAllocRetries();
  void IncrementOoms();
  void IncrementDeviceAlloc();
  void IncrementDeviceFree();

  // Reset methods
  void ResetAccumulatedStats();
  void ResetPeakStats();

  // Print memory statistics
  void PrintStats() const;

  // Convert to map format for external API
  std::map<std::string, uint64_t> GetMemoryStats() const;

 private:
  // Allocated memory statistics - memory currently in use by tensors
  uint64_t allocated_current_ = 0;  // Current amount of memory used by tensors (bytes)
  uint64_t allocated_peak_ = 0;     // Peak amount of memory used by tensors (bytes)
  uint64_t allocated_total_allocated_ =
      0;  // Total amount of memory allocated to tensors over time (bytes)
  uint64_t allocated_total_freed_ =
      0;  // Total amount of memory freed from tensors over time (bytes)

  // Reserved memory statistics - total memory allocated from device
  uint64_t reserved_current_ = 0;  // Current amount of memory allocated from device (bytes)
  uint64_t reserved_peak_ = 0;     // Peak amount of memory allocated from device (bytes)
  uint64_t reserved_total_allocated_ =
      0;  // Total amount of memory allocated from device over time (bytes)
  uint64_t reserved_total_freed_ =
      0;  // Total amount of memory released to device over time (bytes)

  // Active memory statistics - memory blocks currently in use
  uint64_t active_current_ = 0;  // Current amount of active memory blocks (bytes)
  uint64_t active_peak_ = 0;     // Peak amount of active memory blocks (bytes)

  // Cached block memory statistics - cached blocks from fragmentation that can be reused
  uint64_t cached_current_ = 0;  // Current amount of cached blocks from fragmentation (bytes)
  uint64_t cached_peak_ = 0;     // Peak amount of cached blocks from fragmentation (bytes)

  // Allocation operation counters
  uint64_t num_alloc_retries_ = 0;  // Number of allocation retries due to cache flush
  uint64_t num_ooms_ = 0;           // Number of out-of-memory errors encountered
  uint64_t num_device_alloc_ = 0;   // Total number of device memory allocations
  uint64_t num_device_free_ = 0;    // Total number of device memory frees
};

}  // namespace rbln

#endif  // REBEL_RUNTIME_BASE_MEMORY_STATS_H
