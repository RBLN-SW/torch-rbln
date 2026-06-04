"""Compare engine vs host sweep JSON outputs and characterize the crossover."""

import json
import sys


def load(path):
    with open(path) as f:
        d = json.load(f)
    by_key = {}
    for r in d["rows"]:
        key = (r["outer"], r["inner_elems"], r["sf"])
        by_key[key] = r["us"]
    return by_key


def main():
    engine_path, host_path = sys.argv[1], sys.argv[2]
    eng = load(engine_path)
    host = load(host_path)

    keys = sorted(set(eng) & set(host))
    print(f"{'outer':>6} {'inner_e':>8} {'sf':>4} {'inner_B':>9} {'span_B':>11} "
          f"{'engine':>9} {'host':>9} {'win?':>5} {'ratio':>8}")
    for k in keys:
        outer, inner_e, sf = k
        inner_b = inner_e * 2
        span = ((outer - 1) * sf * inner_e + inner_e) * 2
        e = eng[k]
        h = host[k]
        win = "✓" if e < h else "✗"
        ratio = e / h if h > 0 else float("inf")
        print(f"{outer:>6d} {inner_e:>8d} {sf:>4d} {inner_b:>9d} {span:>11d} "
              f"{e:>9.1f} {h:>9.1f} {win:>5} {ratio:>8.2f}")

    # Summary: when does engine beat host?
    wins_only = [(k, eng[k], host[k]) for k in keys if eng[k] < host[k]]
    losses = [(k, eng[k], host[k]) for k in keys if eng[k] >= host[k]]
    print(f"\nWins: {len(wins_only)}/{len(keys)}, losses: {len(losses)}/{len(keys)}")

    # Losses where the engine is more than 2x slower than host bounce
    bad = [item for item in losses if item[1] / item[2] > 2.0]
    print(f"Bad losses (engine >2× host): {len(bad)}")
    if bad:
        print("Worst losses:")
        worst = sorted(bad, key=lambda x: x[1] / x[2], reverse=True)[:15]
        for k, e, h in worst:
            outer, inner_e, sf = k
            print(f"  outer={outer:>5d} inner={inner_e*2:>7d}B sf={sf:>3d}: engine={e:>9.1f} "
                  f"host={h:>9.1f} ratio={e/h:>7.2f}")

    # Wins by outer / inner bucket
    print("\nWins by outer bucket:")
    buckets = [(0, 64), (64, 256), (256, 1024), (1024, 4096), (4096, 10**9)]
    for lo, hi in buckets:
        sub_keys = [k for k in keys if lo < k[0] <= hi]
        sub_wins = [k for k in sub_keys if eng[k] < host[k]]
        sub_bad = [k for k in sub_keys if eng[k] > 2 * host[k]]
        print(f"  outer in ({lo},{hi}]: {len(sub_wins)}/{len(sub_keys)} wins, {len(sub_bad)} >2× losses")

    print("\nWins by inner_bytes bucket:")
    buckets_i = [(0, 256), (256, 1024), (1024, 4096), (4096, 16384), (16384, 65536), (65536, 10**9)]
    for lo, hi in buckets_i:
        sub_keys = [k for k in keys if lo < k[1] * 2 <= hi]
        sub_wins = [k for k in sub_keys if eng[k] < host[k]]
        sub_bad = [k for k in sub_keys if eng[k] > 2 * host[k]]
        print(f"  inner_B in ({lo},{hi}]: {len(sub_wins)}/{len(sub_keys)} wins, {len(sub_bad)} >2× losses")


if __name__ == "__main__":
    main()
