// =============================================================================
// TEMPORARY: resolve RBLN_RDMA_IP from RBLN_RDMA_HCA when set; otherwise no-op.
// See RdmaIpAutoDiscovery.hpp for the removal procedure.
// =============================================================================
#include <torch_rbln/csrc/distributed/c10d/rbln/RdmaIpAutoDiscovery.hpp>

#include <arpa/inet.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <c10/rbln/RBLNLogging.h>
#include <c10/util/CallOnce.h>

namespace torch_rbln::detail {

namespace {

namespace fs = std::filesystem;

constexpr const char* kSysfsInfiniband = "/sys/class/infiniband";
constexpr const char* kDiagPrefix = "[rbln_rdma_probe]";
constexpr const char* kEnvRdmaIp = "RBLN_RDMA_IP";
constexpr const char* kEnvRdmaHca = "RBLN_RDMA_HCA";
constexpr const char* kEnvDisable = "RBLN_DISABLE_AUTO_RDMA_IP";

// Mirrors the Python helper's stderr-tagged probe lines so existing CI grep
// patterns ([rbln_rdma_probe] ...) keep working when the source of the
// diagnostics moves from Python to C++.
#define RDMA_DIAG(fmt_str, ...) RBLN_LOG_INFO("{} " fmt_str, kDiagPrefix, ##__VA_ARGS__)

bool EnvTruthy(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr) {
    return false;
  }
  std::string s(v);
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  // Trim surrounding whitespace.
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s == "1" || s == "true" || s == "yes" || s == "on";
}

// Read the IPv4 bound to ``iface`` via SIOCGIFADDR. Avoids the iproute2
// ``ip`` binary so the probe still works in minimal CI containers.
// EADDRNOTAVAIL / ENODEV is treated as "no IPv4 available".
std::optional<std::string> Ipv4ForIface(const std::string& iface) {
  if (iface.size() >= IFNAMSIZ) {
    RDMA_DIAG("iface={} name too long for ifreq (max {} bytes)", iface, IFNAMSIZ - 1);
    return std::nullopt;
  }
  int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) {
    RDMA_DIAG("iface={} AF_INET socket() failed errno={}", iface, errno);
    return std::nullopt;
  }
  struct ifreq ifr{};
  std::memcpy(ifr.ifr_name, iface.c_str(), iface.size());
  ifr.ifr_name[iface.size()] = '\0';
  int rc = ::ioctl(fd, SIOCGIFADDR, &ifr);
  int saved_errno = errno;
  ::close(fd);
  if (rc != 0) {
    RDMA_DIAG("iface={} SIOCGIFADDR ioctl failed errno={}", iface, saved_errno);
    return std::nullopt;
  }
  auto* sin = reinterpret_cast<struct sockaddr_in*>(&ifr.ifr_addr);
  std::array<char, INET_ADDRSTRLEN> buf{};
  if (::inet_ntop(AF_INET, &sin->sin_addr, buf.data(), buf.size()) == nullptr) {
    return std::nullopt;
  }
  return std::string(buf.data());
}

// Resolve an HCA device name (e.g. "rocep99s0", "mlx5_0") to its first
// bound netdev under /sys/class/infiniband/<hca>/device/net/. Strips an
// optional ":<port>" or "/<port>" suffix (NCCL-compatible syntax); the
// port number is logged and ignored. Warns when multiple netdevs are
// bound -- set RBLN_RDMA_IP explicitly to pick a specific one.
std::optional<std::string> HcaToNetdev(std::string_view hca) {
  size_t sep = hca.find_first_of(":/");
  std::string dev_name(hca.substr(0, sep));
  if (dev_name.empty()) {
    RDMA_DIAG("{}={} parse failed: empty device name", kEnvRdmaHca, hca);
    return std::nullopt;
  }
  if (sep != std::string_view::npos) {
    RDMA_DIAG("{}={} ignoring port suffix '{}' for HCA '{}'", kEnvRdmaHca, hca, std::string(hca.substr(sep)), dev_name);
  }

  fs::path net_root = fs::path(kSysfsInfiniband) / dev_name / "device" / "net";
  std::error_code ec;
  if (!fs::is_directory(net_root, ec)) {
    RDMA_DIAG("{}={} resolve failed: {} does not exist", kEnvRdmaHca, dev_name, net_root.string());
    return std::nullopt;
  }

  std::vector<std::string> netdevs;
  for (const auto& entry : fs::directory_iterator(net_root, ec)) {
    std::string name = entry.path().filename().string();
    if (name.empty() || name[0] == '.') {
      continue;
    }
    netdevs.push_back(std::move(name));
  }
  if (netdevs.empty()) {
    RDMA_DIAG("{}={} resolve failed: no netdev under {}", kEnvRdmaHca, dev_name, net_root.string());
    return std::nullopt;
  }
  std::sort(netdevs.begin(), netdevs.end());
  std::string chosen = netdevs.front();
  if (netdevs.size() > 1) {
    std::string others;
    for (size_t i = 1; i < netdevs.size(); ++i) {
      if (!others.empty()) {
        others += ",";
      }
      others += netdevs[i];
    }
    RBLN_LOG_WARN(
        "{} {}={} has multiple netdevs; selected '{}', also bound: [{}]. "
        "To select a different one, set {} explicitly.",
        kDiagPrefix,
        kEnvRdmaHca,
        dev_name,
        chosen,
        others,
        kEnvRdmaIp);
  }
  return chosen;
}

// Full RBLN_RDMA_HCA -> IPv4 resolution. Returns nullopt and emits a
// diagnostic on any failure; the caller decides whether to warn or
// continue.
std::optional<std::string> ResolveRbnRdmaHca(std::string_view hca) {
  auto netdev = HcaToNetdev(hca);
  if (!netdev) {
    return std::nullopt;
  }
  auto ipv4 = Ipv4ForIface(*netdev);
  if (!ipv4) {
    RDMA_DIAG("{}={} netdev={} resolve failed: no IPv4", kEnvRdmaHca, hca, *netdev);
    return std::nullopt;
  }
  RDMA_DIAG("{}={} netdev={} ipv4={}", kEnvRdmaHca, hca, *netdev, *ipv4);
  return ipv4;
}

void DoOnce() {
  if (EnvTruthy(kEnvDisable)) {
    RDMA_DIAG("{} resolution disabled by {}=1", kEnvRdmaIp, kEnvDisable);
  } else {
    // Resolution priority (see header for the full contract):
    //   1. RBLN_RDMA_IP already set -- no-op (caller-provided value wins).
    //   2. RBLN_RDMA_HCA set -- resolve to an IPv4 and setenv. Failure
    //      is logged but non-fatal.
    //   3. RBLN_RDMA_HCA unset -- do nothing.
    const char* existing = std::getenv(kEnvRdmaIp);
    if (existing != nullptr && existing[0] != '\0') {
      RDMA_DIAG("{} already set ({}), skipping resolution", kEnvRdmaIp, existing);
    } else {
      const char* hca = std::getenv(kEnvRdmaHca);
      if (hca != nullptr && hca[0] != '\0') {
        auto ipv4 = ResolveRbnRdmaHca(hca);
        if (ipv4.has_value()) {
          // overwrite=0 is sufficient: we just confirmed RBLN_RDMA_IP is empty.
          if (::setenv(kEnvRdmaIp, ipv4->c_str(), /*overwrite=*/0) != 0) {
            RDMA_DIAG("setenv({}, {}) failed errno={}", kEnvRdmaIp, *ipv4, errno);
          } else {
            RDMA_DIAG("{}={} (via {}={})", kEnvRdmaIp, *ipv4, kEnvRdmaHca, hca);
          }
        } else {
          RDMA_DIAG("{}={} resolution failed, leaving {} unset (see diagnostics above)", kEnvRdmaHca, hca, kEnvRdmaIp);
        }
      } else {
        RDMA_DIAG(
            "{} unset; not setting {} (set {} or {} explicitly to pin a NIC)",
            kEnvRdmaHca,
            kEnvRdmaIp,
            kEnvRdmaHca,
            kEnvRdmaIp);
      }
    }
  }

  // No IP found. Missing-IP is no longer fatal: we can't tell single- vs
  // multi-node here, so warn (when RCCL_PORT_GEN is set) and defer the
  // decision to the runtime. See the header for the full contract.
  const char* port_gen_env = std::getenv("RCCL_PORT_GEN");
  const bool use_autoport = (port_gen_env != nullptr && port_gen_env[0] != '\0');
  const char* final_ip = std::getenv(kEnvRdmaIp);
  const bool have_ip = (final_ip != nullptr && final_ip[0] != '\0');
  if (have_ip) {
    return;
  }
  if (use_autoport) {
    RBLN_LOG_WARN(
        "{} RCCL_PORT_GEN set but {} unresolved. Single-node runs continue; "
        "multi-node will fail at RCCL init. To pin it set {} (HCA device name "
        "as listed under /sys/class/infiniband) or {} (IPv4 string) explicitly.",
        kDiagPrefix,
        kEnvRdmaIp,
        kEnvRdmaHca,
        kEnvRdmaIp);
    return;
  }
  RDMA_DIAG("{} unresolved, but RCCL_PORT_GEN is not set -- RDMA IP not required, continuing", kEnvRdmaIp);
}

} // namespace

void MaybeAutoDiscoverRbnRdmaIp() {
  static c10::once_flag once;
  c10::call_once(once, &DoOnce);
}

} // namespace torch_rbln::detail
