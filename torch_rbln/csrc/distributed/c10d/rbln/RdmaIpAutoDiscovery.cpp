// =============================================================================
// TEMPORARY: auto-discover RBLN_RDMA_IP from /sys/class/infiniband RoCE v2 GIDs.
// See RdmaIpAutoDiscovery.hpp for the removal procedure.
// =============================================================================
#include <torch_rbln/csrc/distributed/c10d/rbln/RdmaIpAutoDiscovery.hpp>

#include <arpa/inet.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <c10/rbln/RBLNLogging.h>

namespace c10d {
namespace rbln {
namespace detail {

namespace {

namespace fs = std::filesystem;

constexpr const char* kSysfsInfiniband = "/sys/class/infiniband";
constexpr const char* kDiagPrefix = "[rbln_rdma_probe]";
constexpr const char* kEnvRdmaIp = "RBLN_RDMA_IP";
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

std::optional<std::string> ReadFileTrimmed(const fs::path& p) {
  std::ifstream f(p);
  if (!f.is_open()) {
    return std::nullopt;
  }
  std::stringstream ss;
  ss << f.rdbuf();
  std::string s = ss.str();
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
    s.pop_back();
  }
  return s;
}

// Parse an IPv4-mapped-IPv6 GID written by sysfs as eight 16-bit groups,
// e.g. "0000:0000:0000:0000:0000:ffff:0a0a:1235" -> "10.10.18.53".
// Returns nullopt for the all-zero entry or any non-IPv4-mapped layout.
std::optional<std::string> ParseIpv4MappedGid(std::string_view gid) {
  std::array<std::string, 8> parts;
  size_t idx = 0;
  size_t start = 0;
  for (size_t i = 0; i <= gid.size(); ++i) {
    if (i == gid.size() || gid[i] == ':') {
      if (idx >= 8) {
        return std::nullopt;
      }
      parts[idx++] = std::string(gid.substr(start, i - start));
      start = i + 1;
    }
  }
  if (idx != 8) {
    return std::nullopt;
  }
  for (size_t i = 0; i < 5; ++i) {
    if (parts[i] != "0000") {
      return std::nullopt;
    }
  }
  std::string p5 = parts[5];
  std::transform(p5.begin(), p5.end(), p5.begin(), [](unsigned char c) { return std::tolower(c); });
  if (p5 != "ffff") {
    return std::nullopt;
  }
  if (parts[6].size() != 4 || parts[7].size() != 4) {
    return std::nullopt;
  }
  auto hex_byte = [](const std::string& s, size_t off) -> std::optional<int> {
    try {
      return std::stoi(s.substr(off, 2), nullptr, 16);
    } catch (...) {
      return std::nullopt;
    }
  };
  auto b1 = hex_byte(parts[6], 0);
  auto b2 = hex_byte(parts[6], 2);
  auto b3 = hex_byte(parts[7], 0);
  auto b4 = hex_byte(parts[7], 2);
  if (!b1 || !b2 || !b3 || !b4) {
    return std::nullopt;
  }
  if ((*b1 | *b2 | *b3 | *b4) == 0) {
    return std::nullopt;
  }
  return std::to_string(*b1) + "." + std::to_string(*b2) + "." + std::to_string(*b3) + "." + std::to_string(*b4);
}

// Read IPv4 addresses registered as RoCE v2 GIDs on an RDMA device. If
// RBLN_RDMA_IP ends up not being in this set, rcclGetUniqueId will fail to
// resolve it later. Returns sorted, deduplicated IPv4 strings.
std::vector<std::string> ReadRoceV2GidIpv4s(const fs::path& rdma_dev) {
  std::set<std::string> out;
  fs::path ports_root = rdma_dev / "ports";
  std::error_code ec;
  if (!fs::is_directory(ports_root, ec)) {
    return {};
  }
  std::vector<fs::directory_entry> port_dirs;
  for (auto& entry : fs::directory_iterator(ports_root, ec)) {
    port_dirs.push_back(entry);
  }
  std::sort(port_dirs.begin(), port_dirs.end(), [](const auto& a, const auto& b) { return a.path() < b.path(); });
  for (const auto& port_entry : port_dirs) {
    if (!port_entry.is_directory(ec)) {
      continue;
    }
    fs::path gids_dir = port_entry.path() / "gids";
    fs::path types_dir = port_entry.path() / "gid_attrs" / "types";
    if (!fs::is_directory(gids_dir, ec) || !fs::is_directory(types_dir, ec)) {
      continue;
    }
    std::vector<fs::directory_entry> gid_files;
    for (auto& gid_entry : fs::directory_iterator(gids_dir, ec)) {
      gid_files.push_back(gid_entry);
    }
    std::sort(gid_files.begin(), gid_files.end(), [](const auto& a, const auto& b) { return a.path() < b.path(); });
    for (const auto& gid_entry : gid_files) {
      auto gid_type = ReadFileTrimmed(types_dir / gid_entry.path().filename());
      if (!gid_type) {
        continue;
      }
      if (gid_type->find("RoCE v2") == std::string::npos) {
        continue;
      }
      auto gid = ReadFileTrimmed(gid_entry.path());
      if (!gid) {
        continue;
      }
      if (auto ipv4 = ParseIpv4MappedGid(*gid); ipv4) {
        out.insert(*ipv4);
      }
    }
  }
  return std::vector<std::string>(out.begin(), out.end());
}

// Reads ``state`` and ``phys_state`` straight from sysfs so the probe works
// in containers that don't ship the `rdma` (rdma-core) binary. Each file is
// a short string like "4: ACTIVE" or "1: DOWN" written by ib_core.
std::pair<std::optional<std::string>, std::optional<std::string>> ReadRdmaPortState(
    const fs::path& rdma_dev,
    const std::string& port = "1") {
  fs::path port_dir = rdma_dev / "ports" / port;
  return {ReadFileTrimmed(port_dir / "state"), ReadFileTrimmed(port_dir / "phys_state")};
}

// Classify sysfs port state. Returns true for ACTIVE, false for DOWN/DISABLED,
// nullopt for unknown (INIT/ARMED/missing) so the caller can fall through to
// the secondary bucket.
std::optional<bool> ClassifyRdmaState(const std::optional<std::string>& state) {
  if (!state || state->empty()) {
    return std::nullopt;
  }
  std::string upper = *state;
  std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) { return std::toupper(c); });
  if (upper.find("ACTIVE") != std::string::npos) {
    return true;
  }
  if (upper.find("DOWN") != std::string::npos || upper.find("DISABLED") != std::string::npos) {
    return false;
  }
  return std::nullopt;
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
  char buf[INET_ADDRSTRLEN] = {};
  if (::inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf)) == nullptr) {
    return std::nullopt;
  }
  return std::string(buf);
}

std::optional<std::string> ReadOperstate(const fs::path& netdev_dir) {
  return ReadFileTrimmed(netdev_dir / "operstate");
}

// Core loop. Walks /sys/class/infiniband/*, picks the first IPv4 on an
// 'up' netdev whose RDMA port is ACTIVE. Falls back to any netdev IPv4
// when no port reports ACTIVE.
std::optional<std::string> ProbeRoceRdmaIpv4() {
  std::error_code ec;
  fs::path sysfs_root(kSysfsInfiniband);
  if (!fs::is_directory(sysfs_root, ec)) {
    RDMA_DIAG("result=None reason=sysfs-missing path={}", kSysfsInfiniband);
    return std::nullopt;
  }

  std::vector<fs::directory_entry> rdma_devs;
  for (auto& entry : fs::directory_iterator(sysfs_root, ec)) {
    rdma_devs.push_back(entry);
  }
  std::sort(rdma_devs.begin(), rdma_devs.end(), [](const auto& a, const auto& b) { return a.path() < b.path(); });

  using Candidate = std::tuple<std::string, std::string, std::string>; // dev, iface, ipv4
  std::vector<Candidate> active_first;
  std::vector<Candidate> fallback;

  for (const auto& rdma_dev_entry : rdma_devs) {
    const fs::path& rdma_dev = rdma_dev_entry.path();
    std::string dev_name = rdma_dev.filename().string();
    if (!rdma_dev_entry.is_directory(ec)) {
      RDMA_DIAG("skip dev={} reason=not-a-directory", dev_name);
      continue;
    }
    fs::path net_root = rdma_dev / "device" / "net";
    if (!fs::is_directory(net_root, ec)) {
      RDMA_DIAG("skip dev={} reason=no-device/net (path={})", dev_name, net_root.string());
      continue;
    }
    std::vector<fs::directory_entry> netdevs;
    for (auto& entry : fs::directory_iterator(net_root, ec)) {
      netdevs.push_back(entry);
    }
    std::sort(netdevs.begin(), netdevs.end(), [](const auto& a, const auto& b) { return a.path() < b.path(); });

    std::string netdev_names;
    for (const auto& nd : netdevs) {
      if (!netdev_names.empty()) {
        netdev_names += ",";
      }
      netdev_names += nd.path().filename().string();
    }
    RDMA_DIAG("dev={} netdevs=[{}]", dev_name, netdev_names);

    auto gid_ipv4s = ReadRoceV2GidIpv4s(rdma_dev);
    std::string gid_list;
    for (const auto& s : gid_ipv4s) {
      if (!gid_list.empty()) {
        gid_list += ",";
      }
      gid_list += s;
    }
    RDMA_DIAG("dev={} roce_v2_gid_ipv4s=[{}]", dev_name, gid_list);

    auto [port_state, phys_state] = ReadRdmaPortState(rdma_dev);
    RDMA_DIAG(
        "dev={} port_state={} phys_state={}",
        dev_name,
        port_state ? *port_state : std::string("<unknown>"),
        phys_state ? *phys_state : std::string("<unknown>"));
    auto sysfs_link_ok = ClassifyRdmaState(port_state);

    for (const auto& netdev_entry : netdevs) {
      if (!netdev_entry.is_directory(ec)) {
        RDMA_DIAG("skip dev={} netdev={} reason=not-a-directory", dev_name, netdev_entry.path().filename().string());
        continue;
      }
      std::string iface = netdev_entry.path().filename().string();
      auto operstate = ReadOperstate(netdev_entry.path());
      if (!operstate || *operstate != "up") {
        RDMA_DIAG(
            "skip dev={} iface={} reason=operstate={} (need 'up')",
            dev_name,
            iface,
            operstate ? *operstate : std::string("<none>"));
        continue;
      }
      auto ipv4 = Ipv4ForIface(iface);
      if (!ipv4) {
        RDMA_DIAG("skip dev={} iface={} reason=no-ipv4-assigned", dev_name, iface);
        continue;
      }
      if (!gid_ipv4s.empty() && std::find(gid_ipv4s.begin(), gid_ipv4s.end(), *ipv4) == gid_ipv4s.end()) {
        RDMA_DIAG(
            "warn dev={} iface={} ipv4={} NOT in RoCEv2 GID table [{}] -- "
            "RDMA unique_id setup may reject this address",
            dev_name,
            iface,
            *ipv4,
            gid_list);
      }
      if (sysfs_link_ok.has_value() && *sysfs_link_ok == false) {
        RDMA_DIAG(
            "skip dev={} iface={} ipv4={} reason=rdma-port-DOWN (port_state={} phys_state={})",
            dev_name,
            iface,
            *ipv4,
            port_state ? *port_state : std::string("<unknown>"),
            phys_state ? *phys_state : std::string("<unknown>"));
        continue;
      }
      Candidate tup{dev_name, iface, *ipv4};
      if (sysfs_link_ok.has_value() && *sysfs_link_ok == true) {
        RDMA_DIAG("candidate dev={} iface={} ipv4={} bucket=active", dev_name, iface, *ipv4);
        active_first.push_back(std::move(tup));
      } else {
        RDMA_DIAG(
            "candidate dev={} iface={} ipv4={} bucket=fallback (rdma link unknown)", dev_name, iface, *ipv4);
        fallback.push_back(std::move(tup));
      }
    }
  }

  auto pick = [](const Candidate& c, const char* bucket) {
    RDMA_DIAG("result={} via dev={} iface={} bucket={}", std::get<2>(c), std::get<0>(c), std::get<1>(c), bucket);
    return std::get<2>(c);
  };
  if (!active_first.empty()) {
    return pick(active_first.front(), "active");
  }
  if (!fallback.empty()) {
    return pick(fallback.front(), "fallback");
  }
  RDMA_DIAG("result=None reason=no-candidate-passed-filters");
  return std::nullopt;
}

void DoOnce() {
  if (EnvTruthy(kEnvDisable)) {
    RDMA_DIAG("auto-discovery disabled by {}=1", kEnvDisable);
  } else {
    const char* existing = std::getenv(kEnvRdmaIp);
    if (existing != nullptr && existing[0] != '\0') {
      RDMA_DIAG("{} already set ({}), skipping auto-discovery", kEnvRdmaIp, existing);
    } else {
      auto ipv4 = ProbeRoceRdmaIpv4();
      if (ipv4.has_value()) {
        // overwrite=0: respect any value a parent set between EnvTruthy and now
        // (extremely unlikely, but cheap to guarantee).
        if (::setenv(kEnvRdmaIp, ipv4->c_str(), /*overwrite=*/0) != 0) {
          RDMA_DIAG("setenv({}, {}) failed errno={}", kEnvRdmaIp, *ipv4, errno);
        } else {
          RDMA_DIAG("{}={} (auto-discovered)", kEnvRdmaIp, *ipv4);
        }
      } else {
        RDMA_DIAG("{}=<none> (auto-discovery produced no candidate)", kEnvRdmaIp);
      }
    }
  }

  // Gate: RCCL_PORT_GEN-enabled runs require RBLN_RDMA_IP. Without it, the
  // RCCL autoport init path in ProcessGroupRBLN cannot bind a remote
  // endpoint and InitRBLNWork would fail deep inside the runtime with a
  // less-actionable message. Surface the misconfiguration here.
  const char* port_gen_env = std::getenv("RCCL_PORT_GEN");
  const bool use_autoport = (port_gen_env != nullptr && port_gen_env[0] != '\0');
  const char* final_ip = std::getenv(kEnvRdmaIp);
  const bool have_ip = (final_ip != nullptr && final_ip[0] != '\0');
  if (have_ip) {
    return;
  }
  RBLN_CHECK(
      !use_autoport,
      "RCCL_PORT_GEN is set but RBLN_RDMA_IP could not be determined. "
      "Set RBLN_RDMA_IP explicitly, or ensure /sys/class/infiniband exposes a "
      "RoCE v2 capable device with an ACTIVE port and an IPv4-assigned netdev. "
      "See [rbln_rdma_probe] diagnostics above for the failure stage.");
  RDMA_DIAG("RBLN_RDMA_IP unresolved, but RCCL_PORT_GEN is not set -- RDMA IP not required, continuing");
}

} // namespace

void MaybeAutoDiscoverRbnRdmaIp() {
  static std::once_flag once;
  std::call_once(once, &DoOnce);
}

} // namespace detail
} // namespace rbln
} // namespace c10d
