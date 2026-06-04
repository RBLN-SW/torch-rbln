"""Compare engine, host, and gated builds. Compute miss rate against an
oracle that always picks the faster path."""

import json
import sys


def load(path):
    with open(path) as f:
        d = json.load(f)
    return {(r["outer"], r["inner_elems"], r["sf"]): r["us"] for r in d["rows"]}


def main():
    eng = load(sys.argv[1])
    host = load(sys.argv[2])
    gated = load(sys.argv[3])
    label = sys.argv[4] if len(sys.argv) > 4 else "gated"

    keys = sorted(eng.keys() & host.keys() & gated.keys())

    # Oracle picks min for each cell. Compute how much gated overpays vs oracle.
    total_oracle = sum(min(eng[k], host[k]) for k in keys)
    total_eng = sum(eng[k] for k in keys)
    total_host = sum(host[k] for k in keys)
    total_gated = sum(gated[k] for k in keys)

    print(f"Comparison across {len(keys)} cells, total μs:")
    print(f"  oracle (min eng/host) : {total_oracle:>12.0f}")
    print(f"  always-engine         : {total_eng:>12.0f}  ({total_eng/total_oracle:>5.2f}× oracle)")
    print(f"  always-host           : {total_host:>12.0f}  ({total_host/total_oracle:>5.2f}× oracle)")
    print(f"  {label:21s} : {total_gated:>12.0f}  ({total_gated/total_oracle:>5.2f}× oracle)")

    # Worst cells where gated overpays vs oracle.
    print(f"\nWorst gated overpayments (gated / oracle):")
    rows = []
    for k in keys:
        oracle = min(eng[k], host[k])
        g = gated[k]
        rows.append((k, g, oracle, eng[k], host[k]))
    rows.sort(key=lambda r: r[1] - r[2], reverse=True)
    print(f"{'outer':>6} {'inner_B':>8} {'sf':>4} {'oracle':>9} {'gated':>9} {'eng':>9} {'host':>9} {'Δμs':>8}")
    for k, g, o, e, h in rows[:15]:
        outer, inner_e, sf = k
        print(f"{outer:>6d} {inner_e*2:>8d} {sf:>4d} {o:>9.1f} {g:>9.1f} {e:>9.1f} {h:>9.1f} {g-o:>8.1f}")

    # Cells where gated wins by >100us vs always-host:
    print(f"\nGated wins >100μs vs always-host: ", end="")
    big_wins = [(k, gated[k], host[k]) for k in keys if host[k] - gated[k] > 100]
    print(f"{len(big_wins)} cells")

    # Cells where gated regresses >100us vs always-host:
    big_regress = [(k, gated[k], host[k]) for k in keys if gated[k] - host[k] > 100]
    print(f"Gated regresses >100μs vs always-host: {len(big_regress)} cells")
    for k, g, h in sorted(big_regress, key=lambda r: r[1] - r[2], reverse=True)[:10]:
        outer, inner_e, sf = k
        print(f"  outer={outer:>5d} inner={inner_e*2:>7d}B sf={sf:>3d}: gated={g:>9.1f} host={h:>9.1f} Δ={g-h:>9.1f}")


if __name__ == "__main__":
    main()
