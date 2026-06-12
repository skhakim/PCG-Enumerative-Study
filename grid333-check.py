# pip install z3-solver networkx tqdm

from z3 import *
import networkx as nx
from itertools import combinations
from tqdm import tqdm
import time


def grid_3x3x3():
    G = nx.Graph()

    def vid(x, y, z):
        return x * 9 + y * 3 + z

    for x in range(3):
        for y in range(3):
            for z in range(3):
                G.add_node(vid(x, y, z))

    for x in range(3):
        for y in range(3):
            for z in range(3):
                u = vid(x, y, z)
                for dx, dy, dz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    x2, y2, z2 = x + dx, y + dy, z + dz
                    if x2 < 3 and y2 < 3 and z2 < 3:
                        G.add_edge(u, vid(x2, y2, z2))

    return G


def exact_pcg_4pc_solver(
    G,
    timeout_ms=24 * 60 * 60 * 1000,
    export_smt2="grid_3x3x3_exact_4pc.smt2",
):
    V = sorted(G.nodes())
    E = {tuple(sorted(e)) for e in G.edges()}

    s = SolverFor("QF_LRA")
    s.set("timeout", timeout_ms)

    D = {}
    for i, j in combinations(V, 2):
        D[(i, j)] = Real(f"d_{i}_{j}")

    def d(i, j):
        if i == j:
            return RealVal(0)
        if i > j:
            i, j = j, i
        return D[(i, j)]

    dmin = Real("dmin")
    dmax = Real("dmax")

    # Scale normalization: for nonempty PCG witnesses with dmin > 0,
    # distances can be rescaled so dmin = 1.
    s.add(dmin == 1)
    s.add(dmax >= dmin)

    pairs = list(combinations(V, 2))
    triples = list(combinations(V, 3))
    quads = list(combinations(V, 4))

    for i, j in tqdm(pairs, desc="Distance positivity"):
        s.add(d(i, j) > 0)

    for i, j, k in tqdm(triples, desc="Triangle inequalities"):
        s.add(d(i, j) <= d(i, k) + d(k, j))
        s.add(d(i, k) <= d(i, j) + d(j, k))
        s.add(d(j, k) <= d(j, i) + d(i, k))

    for i, j in tqdm(pairs, desc="PCG interval constraints"):
        dij = d(i, j)

        if tuple(sorted((i, j))) in E:
            s.add(dij >= dmin)
            s.add(dij <= dmax)
        else:
            s.add(Or(dij < dmin, dij > dmax))

    for a, b, c, e in tqdm(quads, desc="Four-point constraints"):
        X = d(a, b) + d(c, e)
        Y = d(a, c) + d(b, e)
        Z = d(a, e) + d(b, c)

        s.add(Or(
            And(X == Y, X >= Z),
            And(X == Z, X >= Y),
            And(Y == Z, Y >= X),
        ))

    if export_smt2:
        with open(export_smt2, "w") as f:
            f.write(s.to_smt2())
        print(f"\nSMT2 exported to {export_smt2}")

    print("\nSolving...")
    start = time.time()
    result = s.check()
    elapsed = time.time() - start

    print("Result:", result)
    print(f"Elapsed: {elapsed:.2f} seconds")

    print("\nZ3 statistics:")
    print(s.statistics())

    if result == unsat:
        print("\nUNSAT: graph is not a PCG under exact tree-metric 4PC formulation.")
        return {"result": "unsat"}

    if result == unknown:
        reason = s.reason_unknown()
        print("\nUNKNOWN: Z3 timed out or could not decide.")
        print("Reason:", reason)
        return {"result": "unknown", "reason": reason}

    m = s.model()

    witness = {
        "result": "sat",
        "dmin": m.evaluate(dmin, model_completion=True),
        "dmax": m.evaluate(dmax, model_completion=True),
        "D": {
            (i, j): m.evaluate(d(i, j), model_completion=True)
            for i, j in pairs
        },
    }

    print("\nSAT witness found.")
    print("dmin =", witness["dmin"])
    print("dmax =", witness["dmax"])

    return witness


if __name__ == "__main__":
    G = grid_3x3x3()

    print("|V| =", G.number_of_nodes())
    print("|E| =", G.number_of_edges())
    print("Pairs =", G.number_of_nodes() * (G.number_of_nodes() - 1) // 2)
    print("Triples =", len(list(combinations(G.nodes(), 3))))
    print("Quadruples =", len(list(combinations(G.nodes(), 4))))

    out = exact_pcg_4pc_solver(
        G,
        timeout_ms=24 * 60 * 60 * 1000,
        export_smt2="grid_3x3x3_exact_4pc.smt2",
    )

    if out["result"] == "unsat":
        print("\nCertified non-PCG.")
    elif out["result"] == "unknown":
        print("\nNo conclusion.")
    else:
        print("\nGraph is PCG.")