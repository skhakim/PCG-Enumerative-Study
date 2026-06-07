import multiprocessing as mp
from itertools import combinations, product, permutations
from concurrent.futures import ProcessPoolExecutor, as_completed

from z3 import (
    Context,
    SolverFor,
    Real,
    RealVal,
    Or,
    And,
    sat,
    unsat,
    unknown,
    set_param,
)


def build_q4():
    # Vertex i is the 4-bit tuple of i
    verts = [tuple((i >> k) & 1 for k in range(4)) for i in range(16)]
    vid = {v: i for i, v in enumerate(verts)}

    edges = []
    for a, b in combinations(verts, 2):
        if sum(x != y for x, y in zip(a, b)) == 1:
            edges.append(tuple(sorted((vid[a], vid[b]))))
    edges.sort()
    edge_set = set(edges)

    all_pairs = list(combinations(range(len(verts)), 2))
    nonedges = [p for p in all_pairs if p not in edge_set]
    return verts, vid, edges, edge_set, nonedges


def build_q4_automorphisms(verts, vid):
    autos = []
    seen = set()

    # Hypercube automorphisms:
    # permute coordinates, then flip any subset of coordinates
    for perm in permutations(range(4)):
        for flips in product([0, 1], repeat=4):
            mapping = []
            for v in verts:
                w = tuple(v[perm[i]] ^ flips[i] for i in range(4))
                mapping.append(vid[w])

            tup = tuple(mapping)
            if tup not in seen:
                seen.add(tup)
                autos.append(tup)

    return autos


def image_edge(edge, perm):
    a, b = edge
    return tuple(sorted((perm[a], perm[b])))


def edge_orbit_representatives(edges, autos):
    unseen = set(edges)
    reps = []

    while unseen:
        e = next(iter(unseen))
        orb = {image_edge(e, perm) for perm in autos}
        reps.append(min(orb))
        unseen -= orb

    reps.sort()
    return reps


def ordered_edge_pair_orbit_representatives(edges, autos):
    unseen = {(e1, e2) for e1 in edges for e2 in edges}
    reps = []

    while unseen:
        p = next(iter(unseen))
        e1, e2 = p
        orb = {(image_edge(e1, perm), image_edge(e2, perm)) for perm in autos}
        reps.append(min(orb))
        unseen -= orb

    reps.sort()
    return reps


def build_q4_cases():
    verts, vid, edges, edge_set, nonedges = build_q4()
    autos = build_q4_automorphisms(verts, vid)
    edge_reps = edge_orbit_representatives(edges, autos)
    pair_reps = ordered_edge_pair_orbit_representatives(edges, autos)

    cases = []

    # LPG branch
    for ub_edge in edge_reps:
        cases.append({
            "kind": "LPG",
            "lower_edge": None,
            "upper_edge": ub_edge,
        })

    # Genuine PCG branch
    for lb_edge, ub_edge in pair_reps:
        cases.append({
            "kind": "PCG",
            "lower_edge": lb_edge,
            "upper_edge": ub_edge,
        })

    return {
        "verts": verts,
        "edges": edges,
        "nonedges": nonedges,
        "autos": autos,
        "edge_reps": edge_reps,
        "pair_reps": pair_reps,
        "cases": cases,
    }


def stats_to_text(stats):
    try:
        return str(stats)
    except Exception:
        return "<unavailable>"


def make_base_solver_q4(ctx, n=16, timeout_ms=0):
    set_param("parallel.enable", True)

    s = SolverFor("QF_LRA", ctx=ctx)
    if timeout_ms > 0:
        s.set(timeout=timeout_ms)

    d = {}
    for i in range(n):
        for j in range(i + 1, n):
            v = Real(f"d_{i}_{j}", ctx=ctx)
            d[(i, j)] = v
            s.add(v > 0)

    def D(i, j):
        if i == j:
            return RealVal(0, ctx=ctx)
        return d[(i, j)] if i < j else d[(j, i)]

    # Triangle inequalities
    for i, j, k in combinations(range(n), 3):
        s.add(D(i, j) <= D(i, k) + D(k, j))
        s.add(D(i, k) <= D(i, j) + D(j, k))
        s.add(D(j, k) <= D(j, i) + D(i, k))

    # Four-point condition
    for a, b, c, e in combinations(range(n), 4):
        s1 = D(a, b) + D(c, e)
        s2 = D(a, c) + D(b, e)
        s3 = D(a, e) + D(b, c)
        s.add(Or(
            And(s1 == s2, s1 >= s3),
            And(s1 == s3, s1 >= s2),
            And(s2 == s3, s2 >= s1),
        ))

    return s, D


def worker_solve_case(case_id, total_cases, case, timeout_ms):
    # Each process creates its own Z3 context and solver
    ctx = Context()

    verts, vid, edges, edge_set, nonedges = build_q4()
    s, D = make_base_solver_q4(ctx=ctx, n=len(verts), timeout_ms=timeout_ms)

    kind = case["kind"]
    lb_edge = case["lower_edge"]
    ub_edge = case["upper_edge"]

    proc_name = mp.current_process().name
    print(
        f"[{proc_name}] [{case_id}/{total_cases}] START "
        f"{kind} lower={lb_edge} upper={ub_edge}",
        flush=True,
    )

    if kind == "LPG":
        # By scaling/tightening:
        # interval = [0,1], and some edge attains 1
        s.add(D(*ub_edge) == 1)

        for e in edges:
            s.add(D(*e) <= 1)
        for f in nonedges:
            s.add(D(*f) > 1)

    else:
        # By scaling/tightening:
        # interval = [1,U], some edge attains 1, some edge attains U
        U = D(*ub_edge)

        s.add(D(*lb_edge) == 1)
        s.add(U >= 1)

        for e in edges:
            s.add(D(*e) >= 1)
            s.add(D(*e) <= U)

        for f in nonedges:
            s.add(Or(D(*f) < 1, D(*f) > U))

    ans = s.check()
    ans_str = str(ans)
    stats_str = stats_to_text(s.statistics())

    print(
        f"[{proc_name}] [{case_id}/{total_cases}] DONE -> {ans_str}",
        flush=True,
    )
    print(
        f"[{proc_name}] [{case_id}/{total_cases}] STATS {stats_str}",
        flush=True,
    )

    if ans == sat:
        matrix = []
        for i in range(len(verts)):
            row = []
            for j in range(len(verts)):
                if i == j:
                    row.append("0")
                else:
                    row.append(str(s.model().evaluate(D(i, j))))
            matrix.append(row)

        result = {
            "case_id": case_id,
            "case": case,
            "status": "sat",
            "matrix": matrix,
            "upper_value": None if kind == "LPG" else str(s.model().evaluate(D(*ub_edge))),
        }
        return result

    if ans == unsat:
        return {
            "case_id": case_id,
            "case": case,
            "status": "unsat",
        }

    return {
        "case_id": case_id,
        "case": case,
        "status": "unknown",
    }


def solve_q4_pcg_multiprocessing(max_workers=4, timeout_ms=10000):
    data = build_q4_cases()
    cases = data["cases"]

    print(f"vertices      : {len(data['verts'])}", flush=True)
    print(f"edges         : {len(data['edges'])}", flush=True)
    print(f"nonedges      : {len(data['nonedges'])}", flush=True)
    print(f"automorphisms : {len(data['autos'])}", flush=True)
    print(f"edge orbits   : {len(data['edge_reps'])}", flush=True)
    print(f"pair orbits   : {len(data['pair_reps'])}", flush=True)
    print(f"total cases   : {len(cases)}", flush=True)
    print(f"max_workers   : {max_workers}", flush=True)
    print(f"timeout_ms    : {timeout_ms}", flush=True)

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for i, case in enumerate(cases, 1):
            fut = ex.submit(
                worker_solve_case,
                i,
                len(cases),
                case,
                timeout_ms,
            )
            futures.append(fut)

        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)

    sat_results = [r for r in results if r["status"] == "sat"]
    unknown_results = [r for r in results if r["status"] == "unknown"]

    if sat_results:
        sat_results.sort(key=lambda r: r["case_id"])
        return {
            "status": "sat",
            "winner": sat_results[0],
            "all_results": results,
        }

    if unknown_results:
        return {
            "status": "unknown",
            "all_results": results,
        }

    return {
        "status": "unsat",
        "all_results": results,
    }


def print_final_result(result):
    print("\noverall status:", result["status"], flush=True)

    if result["status"] != "sat":
        return

    w = result["winner"]
    case = w["case"]

    print("winning case id:", w["case_id"], flush=True)
    print("mode:", case["kind"], flush=True)
    print("lower boundary edge:", case["lower_edge"], flush=True)
    print("upper boundary edge:", case["upper_edge"], flush=True)

    if case["kind"] == "LPG":
        print("interval: [0, 1]", flush=True)
    else:
        print(f"interval: [1, {w['upper_value']}]", flush=True)

    print("distance matrix:", flush=True)
    for row in w["matrix"]:
        print(" ".join(row), flush=True)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    result = solve_q4_pcg_multiprocessing(
        max_workers=6,     # try 4, 6, or 8
        timeout_ms=10000000,  # 10 seconds per case
    )
    print_final_result(result)
