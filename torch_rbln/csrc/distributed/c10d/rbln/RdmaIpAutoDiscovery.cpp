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
#include <optional>
#include <set>
#include <sstream>
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

// Auto-discovery vendor priority. Lower wins. Broadcom (bnxt_re) is the
// validated default on RBLN hosts; Intel iRDMA (irdma) on E810 is known to
// mis-bind to RBLN traffic on mixed-vendor hosts so it is pushed below
// every other driver. Unknown drivers sit in the middle.
constexpr int kVendorPriorityBroadcom = 0;
constexpr int kVendorPriorityUnknown = 1;
constexpr int kVendorPriorityIntelIrdma = 2;

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
  std::array<char, INET_ADDRSTRLEN> buf{};
  if (::inet_ntop(AF_INET, &sin->sin_addr, buf.data(), buf.size()) == nullptr) {
    return std::nullopt;
  }
  return std::string(buf.data());
}

std::optional<std::string> ReadOperstate(const fs::path& netdev_dir) {
  return ReadFileTrimmed(netdev_dir / "operstate");
}

// Read /sys/class/infiniband/<hca>/device/driver symlink basename. Returns
// the RDMA driver name ("bnxt_re", "irdma", "mlx5_core", ...) or nullopt
// when the link is missing/unreadable.
std::optional<std::string> ReadRdmaDriverName(const fs::path& rdma_dev) {
  fs::path driver_link = rdma_dev / "device" / "driver";
  std::error_code ec;
  fs::path target = fs::read_symlink(driver_link, ec);
  if (ec) {
    return std::nullopt;
  }
  std::string base = target.filename().string();
  if (base.empty()) {
    return std::nullopt;
  }
  return base;
}

int VendorPriority(std::string_view driver_name) {
  if (driver_name == "bnxt_re") {
    return kVendorPriorityBroadcom;
  }
  if (driver_name == "irdma") {
    return kVendorPriorityIntelIrdma;
  }
  return kVendorPriorityUnknown;
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

// Core loop. Walks /sys/class/infiniband/*, picks the first IPv4 on an
// 'up' netdev whose RDMA port is ACTIVE. Falls back to any netdev IPv4
// when no port reports ACTIVE. Within each bucket, candidates are ranked
// by vendor priority (Broadcom bnxt_re first, Intel irdma last) so that
// mixed-vendor hosts deterministically prefer the validated Broadcom NIC.
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

  struct Candidate {
    int priority;
    std::string dev;
    std::string iface;
    std::string ipv4;
  };
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

    auto driver = ReadRdmaDriverName(rdma_dev);
    int priority = driver.has_value() ? VendorPriority(*driver) : kVendorPriorityUnknown;
    RDMA_DIAG("dev={} driver={} vendor_priority={}", dev_name, driver.value_or(std::string("<unknown>")), priority);

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
      Candidate cand{priority, dev_name, iface, *ipv4};
      if (sysfs_link_ok.has_value() && *sysfs_link_ok == true) {
        RDMA_DIAG("candidate dev={} iface={} ipv4={} priority={} bucket=active", dev_name, iface, *ipv4, priority);
        active_first.push_back(std::move(cand));
      } else {
        RDMA_DIAG(
            "candidate dev={} iface={} ipv4={} priority={} bucket=fallback (rdma link unknown)",
            dev_name,
            iface,
            *ipv4,
            priority);
        fallback.push_back(std::move(cand));
      }
    }
  }

  auto by_priority = [](const Candidate& a, const Candidate& b) {
    if (a.priority != b.priority) {
      return a.priority < b.priority;
    }
    return a.dev < b.dev;
  };
  std::sort(active_first.begin(), active_first.end(), by_priority);
  std::sort(fallback.begin(), fallback.end(), by_priority);

  auto pick = [](const Candidate& c, const char* bucket) {
    RDMA_DIAG("result={} via dev={} iface={} priority={} bucket={}", c.ipv4, c.dev, c.iface, c.priority, bucket);
    return c.ipv4;
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
    // Resolution priority (see header for the full contract):
    //   1. RBLN_RDMA_HCA -- explicit override, wins over existing RBLN_RDMA_IP.
    //   2. Existing RBLN_RDMA_IP -- left as-is.
    //   3. Auto-discovery via /sys/class/infiniband (vendor-prioritized).
    const char* hca = std::getenv(kEnvRdmaHca);
    if (hca != nullptr && hca[0] != '\0') {
      auto ipv4 = ResolveRbnRdmaHca(hca);
      if (ipv4.has_value()) {
        const char* existing = std::getenv(kEnvRdmaIp);
        if (existing != nullptr && existing[0] != '\0' && std::string(existing) != *ipv4) {
          RBLN_LOG_WARN(
              "{} existing {}={} overridden by {}={} -> {}",
              kDiagPrefix,
              kEnvRdmaIp,
              existing,
              kEnvRdmaHca,
              hca,
              *ipv4);
        }
        // overwrite=1: RBLN_RDMA_HCA explicitly takes precedence over any
        // existing RBLN_RDMA_IP, matching ssw-common-umd PR #1930.
        if (::setenv(kEnvRdmaIp, ipv4->c_str(), /*overwrite=*/1) != 0) {
          RDMA_DIAG("setenv({}, {}) failed errno={}", kEnvRdmaIp, *ipv4, errno);
        } else {
          RDMA_DIAG("{}={} (via {}={})", kEnvRdmaIp, *ipv4, kEnvRdmaHca, hca);
        }
      } else {
        RDMA_DIAG(
            "{}={} resolution failed, leaving {} unchanged (see diagnostics above)", kEnvRdmaHca, hca, kEnvRdmaIp);
      }
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
        "multi-node will fail at RCCL init. To pin it set {}, or ensure "
        "/sys/class/infiniband has a RoCE v2 ACTIVE port with an IPv4 netdev. "
        "See [rbln_rdma_probe] above.",
        kDiagPrefix,
        kEnvRdmaIp,
        kEnvRdmaIp);
    return;
  }
  RDMA_DIAG("RBLN_RDMA_IP unresolved, but RCCL_PORT_GEN is not set -- RDMA IP not required, continuing");
}

} // namespace

void MaybeAutoDiscoverRbnRdmaIp() {
  static c10::once_flag once;
  c10::call_once(once, &DoOnce);
}

} // namespace torch_rbln::detail
