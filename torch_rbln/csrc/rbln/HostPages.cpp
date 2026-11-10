#include <torch_rbln/csrc/rbln/HostPages.h>

#include <c10/util/Exception.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <linux/io_uring.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

namespace {

constexpr uint64_t kPage = 4096;
constexpr uint64_t kHuge = 2ULL * 1024 * 1024;
constexpr uint64_t kWindow = 1024ULL * 1024 * 1024;
constexpr uint64_t kPresent = 1 << 3;
constexpr uint64_t kSwapped = 1 << 4;
constexpr uint64_t kZero = 1 << 5;
constexpr uint64_t kHugePage = 1 << 6;

// Linux PAGEMAP_SCAN UAPI, also when the build host's headers predate it.
struct PageRegion {
  uint64_t start;
  uint64_t end;
  uint64_t categories;
};

struct PageScan {
  uint64_t size;
  uint64_t flags;
  uint64_t start;
  uint64_t end;
  uint64_t walk_end;
  uint64_t vec;
  uint64_t vec_len;
  uint64_t max_pages;
  uint64_t category_inverted;
  uint64_t category_mask;
  uint64_t category_anyof_mask;
  uint64_t return_mask;
};
static_assert(sizeof(PageScan) == 96);
constexpr auto kPageMapScan = _IOWR('f', 16, PageScan);

class FileDescriptor {
 public:
  FileDescriptor() = default;
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  FileDescriptor(FileDescriptor&&) = delete;
  FileDescriptor& operator=(FileDescriptor&&) = delete;
  ~FileDescriptor() {
    reset();
  }
  int get() const {
    return fd_;
  }
  void reset(int fd = -1) {
    if (fd_ >= 0) {
      ::close(fd_);
    }
    fd_ = fd;
  }

 private:
  int fd_ = -1;
};

class HostPageSnapshot {
 public:
  HostPageSnapshot(uint64_t ptr, uint64_t size, bool promote) {
    TORCH_CHECK_VALUE(
        ptr > 0 && size > 0 && size <= kWindow &&
            ptr <= std::numeric_limits<uint64_t>::max() - kHuge &&
            size <= std::numeric_limits<uint64_t>::max() - kHuge - ptr,
        "Invalid host page snapshot range");
    if (sysconf(_SC_PAGESIZE) != static_cast<long>(kPage)) {
      error_ = ENOTSUP;
      return;
    }
    const auto end = ptr + size;
    if (promote) {
      const auto first = (ptr + kPage - 1) & ~(kPage - 1);
      const auto last = end & ~(kPage - 1);
      if (last > first) {
        // Pool-local, best effort. No host-wide THP or mount settings change.
        // The address crosses the Python integer API boundary here.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        auto* address = reinterpret_cast<void*>(first);
        madvise(address, last - first, MADV_HUGEPAGE);
        constexpr int kMadvCollapse = 25;
        madvise(address, last - first, kMadvCollapse);
      }
    }

    io_uring_params params{};
    ring_.reset(static_cast<int>(syscall(__NR_io_uring_setup, 1U, &params)));
    if (ring_.get() < 0) {
      error_ = errno;
      return;
    }
    // One iovec per PMD lets Linux coalesce a huge folio's pin metadata.
    // One giant iovec instead retains a bvec for every base page on Linux 6.8.
    std::vector<iovec> buffers;
    for (auto cursor = ptr; cursor < end;) {
      const auto stop = std::min(end, (cursor / kHuge + 1) * kHuge);
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      buffers.push_back({reinterpret_cast<void*>(cursor), stop - cursor});
      cursor = stop;
    }
    if (syscall(
            __NR_io_uring_register,
            ring_.get(),
            IORING_REGISTER_BUFFERS,
            buffers.data(),
            static_cast<unsigned>(buffers.size())) < 0) {
      error_ = errno;
      close();
      return;
    }
    pinned_ = true;

    FileDescriptor pagemap;
    pagemap.reset(open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC));
    if (pagemap.get() < 0) {
      error_ = errno;
      close();
      return;
    }
    auto cursor = ptr & ~(kPage - 1);
    const auto scan_end = (end + kPage - 1) & ~(kPage - 1);
    while (cursor < scan_end) {
      std::array<PageRegion, 256> regions{};
      PageScan arg{};
      arg.size = sizeof(arg);
      arg.start = cursor;
      arg.end = scan_end;
      arg.vec = reinterpret_cast<uint64_t>(regions.data());
      arg.vec_len = regions.size();
      arg.return_mask = kPresent | kSwapped | kZero | kHugePage;
      const auto count = ioctl(pagemap.get(), kPageMapScan, &arg);
      if (count < 0 || static_cast<size_t>(count) > regions.size() ||
          arg.walk_end <= cursor || arg.walk_end > scan_end) {
        error_ = count < 0 ? errno : EIO;
        close();
        return;
      }
      for (int i = 0; i < count; ++i) {
        const auto& region = regions[i];
        if (region.start < cursor || region.end > arg.walk_end ||
            region.end <= region.start) {
          error_ = EIO;
          close();
          return;
        }
        if ((region.categories & (kPresent | kSwapped | kZero | kHugePage)) !=
            (kPresent | kHugePage)) {
          continue;
        }
        const auto first =
            (std::max(ptr, region.start) + kHuge - 1) & ~(kHuge - 1);
        const auto last = std::min(end, region.end) & ~(kHuge - 1);
        if (last > first) {
          if (!huge_.empty() &&
              huge_.back().first + huge_.back().second == first) {
            huge_.back().second += last - first;
          } else {
            huge_.emplace_back(first, last - first);
          }
        }
      }
      cursor = arg.walk_end;
    }
  }

  ~HostPageSnapshot() {
    close();
  }
  HostPageSnapshot(const HostPageSnapshot&) = delete;
  HostPageSnapshot& operator=(const HostPageSnapshot&) = delete;
  HostPageSnapshot(HostPageSnapshot&&) = delete;
  HostPageSnapshot& operator=(HostPageSnapshot&&) = delete;

  void close() {
    huge_.clear();
    if (pinned_) {
      // No I/O was submitted. Unregister is synchronous; closing the fd also
      // releases the pins if unregister is interrupted by a signal.
      syscall(
          __NR_io_uring_register,
          ring_.get(),
          IORING_UNREGISTER_BUFFERS,
          nullptr,
          0U);
      pinned_ = false;
    }
    ring_.reset();
  }
  const std::vector<std::pair<uint64_t, uint64_t>>& huge_ranges() const {
    return huge_;
  }
  int error() const {
    return error_;
  }

 private:
  FileDescriptor ring_;
  bool pinned_ = false;
  int error_ = 0;
  std::vector<std::pair<uint64_t, uint64_t>> huge_;
};

} // namespace

void register_host_page_api(pybind11::module_& module) {
  namespace py = pybind11;
  py::class_<HostPageSnapshot>(module, "_HostPageSnapshot")
      .def(
          py::init<uint64_t, uint64_t, bool>(),
          py::arg("ptr"),
          py::arg("size"),
          py::arg("promote"),
          py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("huge_ranges", &HostPageSnapshot::huge_ranges)
      .def_property_readonly("error", &HostPageSnapshot::error)
      .def(
          "close",
          &HostPageSnapshot::close,
          py::call_guard<py::gil_scoped_release>());
}
