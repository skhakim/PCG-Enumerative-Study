# Exact PCG Recognition via the Four-Point Condition

This project implements an exact SMT-based recognizer for Pairwise Compatibility Graphs (PCGs) using the classical Four-Point Condition (4PC) characterization of tree metrics.

## Overview

Given a graph (G=(V,E)), the solver searches for:

* A distance matrix (D=(d_{ij})),
* A PCG interval ([d_{\min}, d_{\max}]),

such that:

1. (D) is a tree metric.
2. Two vertices are adjacent in (G) if and only if their distance lies within ([d_{\min}, d_{\max}]).

The existence of such a distance matrix is equivalent to the graph being a PCG.

## Method

The implementation uses Z3 over quantifier-free linear real arithmetic (QF_LRA).

The model includes:

* One distance variable for every unordered vertex pair.
* Triangle inequality constraints.
* PCG interval constraints.
* Exact Four-Point Condition constraints for every vertex quadruple.

The Four-Point Condition states that for every four vertices (a,b,c,d), among

[
d(a,b)+d(c,d),
\quad
d(a,c)+d(b,d),
\quad
d(a,d)+d(b,c),
]

the two largest values must be equal.

This characterizes finite tree metrics exactly.

## Output

The solver returns one of:

* **SAT**: the graph is a PCG and a witness distance matrix is produced.
* **UNSAT**: the graph is not a PCG.
* **UNKNOWN**: the solver timed out or could not determine satisfiability.

## Requirements

```bash
pip install z3-solver networkx tqdm
```

## Running

```bash
python grid333-check.py
```

An SMT-LIB2 file is also exported for experimentation with alternative SMT solvers such as Z3, cvc5, Yices, or MathSAT.

## Notes

This is an exact formulation. No tree topology is assumed or enumerated. The search is performed directly in the space of tree metrics using the Four-Point Condition.

For the (3\times3\times3) grid graph, the model contains:

* 351 distance variables,
* 351 PCG constraints,
* 17,550 Four-Point constraints,

making it a challenging but fully exact recognition instance.
