#!/usr/bin/env bash
# TEMPORARY, do not merge: records the network path a job's distributed tests use, so
# two jobs (a shard and the unsharded step) can be compared line by line.
#
# Usage:
#   network-path-probe.sh start <dir>   # before pytest: snapshot counters, start sampler
#   network-path-probe.sh stop <dir>    # after pytest: stop sampler, print the record
#
# The record lists the job's network settings, which address Gloo resolves the
# hostname to, every listening TCP socket and abstract Unix socket seen while the tests
# ran (ephemeral ports folded to "*"), the devices the test processes held open, and
# how many bytes each RDMA port and network interface moved during the run.
set -uo pipefail

action="$1"
dir="$2"

counters() {
  local port
  for port in /sys/class/infiniband/*/ports/*; do
    [[ -e "${port}" ]] || continue
    echo "rdma ${port#/sys/class/infiniband/} $(<"${port}/counters/port_xmit_data") $(<"${port}/counters/port_rcv_data")"
  done
  awk 'NR > 2 { gsub(/:/, " "); print "netdev", $1, $10, $2 }' /proc/net/dev
}

# One line per listening socket: "tcp <ip>:<port>", with the address in /proc/net/tcp*
# hex decoded and ports above the fixed ones folded.
listeners() {
  local file line local_address state hex_ip hex_port ip port
  for file in /proc/net/tcp /proc/net/tcp6; do
    while read -r _ local_address _ state _; do
      [[ "${state}" == "0A" ]] || continue
      hex_ip="${local_address%:*}"
      hex_port="${local_address#*:}"
      port=$((16#${hex_port}))
      case "${hex_ip}" in
        0100007F) ip="127.0.0.1" ;;
        00000000) ip="0.0.0.0" ;;
        00000000000000000000000000000000) ip="::" ;;
        0000000000000000FFFF00000100007F) ip="::ffff:127.0.0.1" ;;
        *)
          if [[ "${#hex_ip}" -eq 8 ]]; then
            ip="$((16#${hex_ip:6:2})).$((16#${hex_ip:4:2})).$((16#${hex_ip:2:2})).$((16#${hex_ip:0:2}))"
          else
            ip="ipv6:${hex_ip}"
          fi
          ;;
      esac
      if [[ "${port}" -ge 32768 ]]; then
        port="*"
      fi
      echo "tcp listen ${ip}:${port}"
    done < <(tail -n +2 "${file}")
  done
  # Abstract Unix sockets, with the trailing uid/pid folded.
  awk 'NR > 1 && $8 ~ /^@/ { sub(/-[0-9]+-[0-9]+$/, "-*", $8); print "unix", $8 }' /proc/net/unix
  # Devices held open by the test processes.
  local fd
  for fd in /proc/[0-9]*/fd/*; do
    line="$(readlink "${fd}" 2>/dev/null)" || continue
    case "${line}" in
      /dev/infiniband/* | /dev/rbln* | /dev/rsd* | /dev/accel/*) echo "device ${line}" ;;
      *) ;;
    esac
  done
}

case "${action}" in
  start)
    mkdir -p "${dir}"
    counters >"${dir}/counters.before"
    (
      while [[ ! -e "${dir}/stop" ]]; do
        listeners
        sleep 0.2
      done
    ) </dev/null 2>/dev/null | sort -u -o "${dir}/seen" >/dev/null 2>&1 &
    echo $! >"${dir}/sampler.pid"
    ;;
  stop)
    touch "${dir}/stop"
    wait "$(<"${dir}/sampler.pid")" 2>/dev/null
    # `wait` only works on our own children; poll for the sampler to exit otherwise.
    while kill -0 "$(<"${dir}/sampler.pid")" 2>/dev/null; do
      sleep 0.2
    done
    sleep 0.5
    counters >"${dir}/counters.after"
    echo "network path record"
    echo "  settings:"
    env | grep -E '^(MASTER_ADDR|MASTER_PORT|RBLN_|RCCL_|GLOO_|NCCL_|TP_SOCKET)' | sort | sed 's/^/    /'
    echo "  hostname $(</proc/sys/kernel/hostname) resolves to: $(getent ahostsv4 "$(</proc/sys/kernel/hostname)" | awk '{print $1}' | sort -u | tr '\n' ' ')"
    echo "  seen while the tests ran:"
    sort -u "${dir}/seen" | sed 's/^/    /'
    echo "  bytes moved during the tests (tx rx), nonzero only:"
    join <(awk '{print $1"_"$2, $3, $4}' "${dir}/counters.before" | sort) \
      <(awk '{print $1"_"$2, $3, $4}' "${dir}/counters.after" | sort) \
      | awk '{ tx = $4 - $2; rx = $5 - $3; if (tx || rx) printf "    %s %d %d\n", $1, tx, rx }'
    ;;
  *)
    echo "network-path-probe.sh: unknown action '${action}'" >&2
    exit 2
    ;;
esac
