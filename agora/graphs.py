"""Exhaustive enumeration of undirected communication graphs (no external deps).

Graphs are non-isomorphic simple undirected graphs on n labelled-up-to-isomorphism
vertices. Counts are checked against OEIS A000088 (n=1..8: 1, 2, 4, 11, 34, 156,
1044, 12346). Canonical form = lexicographically smallest adjacency bitstring over
vertex permutations that respect a colour refinement (1-WL) partition, so it is
exact, not a heuristic.

A graph maps to an agora channel spec: each undirected edge {i, j} becomes a
pairwise channel readable and writable by exactly {i, j}. Agents with no edges
have no channel and are isolated (flagged in features).
"""

from __future__ import annotations

import itertools
from collections import deque
from typing import Dict, FrozenSet, Iterable, List, Sequence, Tuple

Edge = Tuple[int, int]


def _adj_from_edges(n: int, edges: Iterable[Edge]) -> List[int]:
    adj = [0] * n
    for i, j in edges:
        adj[i] |= 1 << j
        adj[j] |= 1 << i
    return adj


def _refine(n: int, adj: List[int]) -> List[int]:
    """1-WL colour refinement; returns a colour per vertex (canonically ordered)."""
    colour = [bin(adj[v]).count("1") for v in range(n)]
    while True:
        sig = [
            (colour[v], tuple(sorted(colour[u] for u in range(n) if adj[v] >> u & 1)))
            for v in range(n)
        ]
        ranks = {s: r for r, s in enumerate(sorted(set(sig)))}
        new = [ranks[s] for s in sig]
        if len(set(new)) == len(set(colour)):
            return new
        colour = new


def canonical(n: int, adj: List[int]) -> Tuple[int, ...]:
    """Exact canonical key: min adjacency code over colour-respecting permutations."""
    colour = _refine(n, adj)
    cells: Dict[int, List[int]] = {}
    for v, c in enumerate(colour):
        cells.setdefault(c, []).append(v)
    ordered_cells = [cells[c] for c in sorted(cells)]
    best = None
    for combo in itertools.product(*(itertools.permutations(cell) for cell in ordered_cells)):
        order = [v for cell in combo for v in cell]  # new label k -> old vertex order[k]
        pos = {old: new for new, old in enumerate(order)}
        code = []
        for a in range(n):
            for b in range(a + 1, n):
                code.append(1 if adj[order[a]] >> order[b] & 1 else 0)
        key = tuple(code)
        if best is None or key < best:
            best = key
    return (n,) + best


def _edges_from_key(key: Tuple[int, ...]) -> Tuple[int, List[Edge]]:
    n = key[0]
    bits = key[1:]
    edges, k = [], 0
    for a in range(n):
        for b in range(a + 1, n):
            if bits[k]:
                edges.append((a, b))
            k += 1
    return n, edges


def enumerate_graphs(max_n: int) -> Dict[int, List[Tuple[int, ...]]]:
    """All non-isomorphic graphs for n = 1..max_n, by vertex augmentation."""
    out: Dict[int, List[Tuple[int, ...]]] = {1: [(1,)]}
    for n in range(2, max_n + 1):
        seen = set()
        for key in out[n - 1]:
            m, edges = _edges_from_key(key)
            base = _adj_from_edges(n, edges)
            for mask in range(1 << (n - 1)):
                adj = base[:]
                for u in range(n - 1):
                    if mask >> u & 1:
                        adj[u] |= 1 << (n - 1)
                        adj[n - 1] |= 1 << u
                seen.add(canonical(n, adj))
        out[n] = sorted(seen)
    return out


def edges_of(key: Tuple[int, ...]) -> Tuple[int, List[Edge]]:
    return _edges_from_key(key)


def features(key: Tuple[int, ...]) -> Dict[str, object]:
    return features_from_edges(*_edges_from_key(key))


def features_from_edges(n: int, edges: Sequence[Edge]) -> Dict[str, object]:
    adj = _adj_from_edges(n, edges)
    nbrs = [[u for u in range(n) if adj[v] >> u & 1] for v in range(n)]
    deg = [len(x) for x in nbrs]
    m = len(edges)
    # diameter / connectivity via BFS per vertex
    diam, connected = 0, True
    for s in range(n):
        dist = {s: 0}
        q = deque([s])
        while q:
            v = q.popleft()
            for u in nbrs[v]:
                if u not in dist:
                    dist[u] = dist[v] + 1
                    q.append(u)
        if len(dist) < n:
            connected = False
        else:
            diam = max(diam, max(dist.values()))
    tri = sum(1 for a, b in edges for c in nbrs[a] if c in nbrs[b] and c > b)
    triples = sum(d * (d - 1) // 2 for d in deg)
    clustering = round(3 * tri / triples, 4) if triples else 0.0
    mean = sum(deg) / n
    var = sum((d - mean) ** 2 for d in deg) / n
    return {
        "n": n,
        "edges": m,
        "density": round(2 * m / (n * (n - 1)), 4) if n > 1 else 0.0,
        "connected": connected,
        "is_tree": connected and m == n - 1,
        "diameter": diam if connected else None,
        "max_degree": max(deg) if deg else 0,
        "degree_var": round(var, 4),
        "isolated": sum(1 for d in deg if d == 0),
        "triangles": tri,
        "clustering": clustering,
    }


def channel_spec(key: Tuple[int, ...], agents: Sequence[str]) -> Dict[str, Dict[str, List[str]]]:
    """Pairwise channels: one board per undirected edge, readable+writable by its ends."""
    n, edges = _edges_from_key(key)
    if n != len(agents):
        raise ValueError(f"graph has {n} vertices but {len(agents)} agents")
    spec = {}
    for a, b in edges:
        name = f"link_{agents[a]}_{agents[b]}"
        spec[name] = {
            "readers": [agents[a], agents[b]],
            "writers": [agents[a], agents[b]],
            "label": "PRIVATE LINK (visible only to the two connected agents)",
        }
    return spec


# ---- sampling for n > 8 ---------------------------------------------------
# Exact canonical forms are infeasible for symmetric graphs at n >= 9 (e.g. a
# 30-ring has 30! colour-respecting permutations), so sampled graphs are
# deduplicated by a 1-WL certificate. This can merge distinct regular graphs
# that WL cannot separate; sample counts are therefore upper bounds on distinct
# classes, and the sampling scheme (not the class count) is the reported object.

import random


def wl_certificate(n: int, edges: Sequence[Edge]) -> Tuple[object, ...]:
    adj = _adj_from_edges(n, edges)
    colour = _refine(n, adj)
    # iterate refinement to stable colouring, then emit sorted multiset
    return (n, len(edges), tuple(sorted(colour)))


def _tree_prufer(n: int, rng: random.Random) -> List[Edge]:
    if n == 2:
        return [(0, 1)]
    seq = [rng.randrange(n) for _ in range(n - 2)]
    deg = [1] * n
    for v in seq:
        deg[v] += 1
    edges = []
    for v in seq:
        leaf = min(i for i in range(n) if deg[i] == 1)
        edges.append((min(leaf, v), max(leaf, v)))
        deg[leaf] -= 1
        deg[v] -= 1
    u, w = [i for i in range(n) if deg[i] == 1]
    edges.append((u, w))
    return edges


def _gnp(n: int, p: float, rng: random.Random) -> List[Edge]:
    return [(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < p]


def _ba(n: int, m: int, rng: random.Random) -> List[Edge]:
    """Barabási–Albert preferential attachment (scale-free hubs)."""
    edges = set((a, b) for a in range(m + 1) for b in range(a + 1, m + 1))
    targets = [v for e in edges for v in e]
    for v in range(m + 1, n):
        chosen = set()
        while len(chosen) < m:
            chosen.add(rng.choice(targets))
        for u in chosen:
            edges.add((min(u, v), max(u, v)))
            targets += [u, v]
    return sorted(edges)


def _kary_tree(n: int, k: int) -> List[Edge]:
    """Balanced k-ary tree: a rigid manager -> report hierarchy."""
    return [((i - 1) // k, i) for i in range(1, n)]


def _chain_with_skips(n: int, rng: random.Random) -> List[Edge]:
    """Chain of command plus random skip-level links."""
    edges = {(i, i + 1) for i in range(n - 1)}
    for _ in range(max(1, n // 4)):
        a, b = sorted(rng.sample(range(n), 2))
        edges.add((a, b))
    return sorted(edges)


def _barbell(n: int) -> List[Edge]:
    """Two cliques joined by one bridge (two teams, one liaison)."""
    h = n // 2
    edges = [(a, b) for a in range(h) for b in range(a + 1, h)]
    edges += [(a, b) for a in range(h, n) for b in range(a + 1, n)]
    edges.append((h - 1, h))
    return edges


FAMILIES = {
    "tree": lambda n, r: _tree_prufer(n, r),
    "gnp_0.10": lambda n, r: _gnp(n, 0.10, r),
    "gnp_0.20": lambda n, r: _gnp(n, 0.20, r),
    "gnp_0.35": lambda n, r: _gnp(n, 0.35, r),
    "scale_free_m2": lambda n, r: _ba(n, 2, r),
    "kary_tree_k3": lambda n, r: _kary_tree(n, 3),
    "chain_skips": lambda n, r: _chain_with_skips(n, r),
    "barbell": lambda n, r: _barbell(n),
}


def sample_graphs(n: int, per_family: int, seed: int) -> List[Dict[str, object]]:
    """Deterministic sample: `per_family` draws from each family at size n, deduped by WL certificate.

    Deterministic families (kary_tree, barbell) contribute one graph each.
    """
    rng = random.Random(f"{seed}-{n}")
    seen, out = set(), []
    for fam, gen in FAMILIES.items():
        draws = 1 if fam in ("kary_tree_k3", "barbell") else per_family
        for i in range(draws):
            edges = sorted(set((min(a, b), max(a, b)) for a, b in gen(n, rng)))
            cert = wl_certificate(n, edges)
            if cert in seen:
                continue
            seen.add(cert)
            out.append({"family": fam, "n": n, "edges": edges, "cert": repr(cert)})
    return out


def key_from_edges(n: int, edges: Sequence[Edge]) -> Tuple[int, ...]:
    eset = {tuple(e) for e in edges}
    bits = []
    k = 0
    for a in range(n):
        for b in range(a + 1, n):
            bits.append(1 if (a, b) in eset or (b, a) in eset else 0)
            k += 1
    return (n,) + tuple(bits)


def channel_spec_edges(n: int, edges: Sequence[Edge], agents: Sequence[str]):
    return channel_spec(key_from_edges(n, edges), agents)
