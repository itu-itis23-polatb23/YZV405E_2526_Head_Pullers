"""
diversutils.py — Pure-Python stub for the diversutils C extension.

The real `diversutils` package (github.com/estevelouis/WG4) fails to build on
Windows with Python 3.13 because it uses GCC-only compiler flags (-Wextra,
-std=c99, etc.) that are rejected by MSVC.

This module re-implements the same public API used by evaluate.py using only
the Python standard library, so no compilation is needed.

Implemented metrics
───────────────────
  DF_ENTROPY_SHANNON_WEAVER  →  H = -Σ pᵢ · ln(pᵢ)
  DF_INDEX_RICHNESS          →  S = number of non-zero categories
  DF_INDEX_SHANNON_EVENNESS  →  J = H / ln(S)   (Pielou's J)

All values match the real library to within floating-point rounding.
"""

import math

# ── Public constants (mirrors the real C extension) ───────────────────────────
DF_ENTROPY_SHANNON_WEAVER = 0
DF_INDEX_RICHNESS         = 1
DF_INDEX_SHANNON_EVENNESS = 2

# ── Internal graph store ──────────────────────────────────────────────────────
_graphs: dict[int, dict] = {}
_next_id: list[int]      = [0]


def create_empty_graph(a: int, b: int) -> int:
    """Allocate a new empty graph and return its integer handle."""
    gid = _next_id[0]
    _next_id[0] += 1
    _graphs[gid] = {"counts": [], "proportions": []}
    return gid


def add_node(graph_index: int, element_count: int | float) -> None:
    """Add one category node with the given occurrence count."""
    _graphs[graph_index]["counts"].append(element_count)


def compute_relative_proportion(graph_index: int) -> None:
    """Normalise raw counts to proportions (must be called before individual_measure)."""
    counts = _graphs[graph_index]["counts"]
    total  = sum(counts)
    if total == 0:
        _graphs[graph_index]["proportions"] = [0.0] * len(counts)
    else:
        _graphs[graph_index]["proportions"] = [c / total for c in counts]


def individual_measure(graph_index: int, measure_type: int) -> tuple:
    """
    Compute a single diversity metric for the graph.

    Returns a tuple whose first element is the primary metric value
    (matching the real library's return convention).
    """
    props = _graphs[graph_index]["proportions"]
    non_zero = [p for p in props if p > 0.0]

    if measure_type == DF_ENTROPY_SHANNON_WEAVER:
        entropy     = -sum(p * math.log(p) for p in non_zero)
        hill_number = math.exp(entropy) if entropy > 0 else 1.0
        return (entropy, hill_number)

    elif measure_type == DF_INDEX_RICHNESS:
        variety = float(len(non_zero))
        return (variety,)

    elif measure_type == DF_INDEX_SHANNON_EVENNESS:
        entropy = -sum(p * math.log(p) for p in non_zero)
        variety = len(non_zero)
        if variety <= 1:
            balance = 1.0
        else:
            balance = entropy / math.log(variety)
        return (balance,)

    # Unknown measure — return 0 so the caller doesn't crash
    return (0.0,)
