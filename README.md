# ExtrinsicThermodynamicGeometry

Code supporting *Extrinsic thermodynamic geometry and the cost of constraint*. 

## Files

`symmetry.py` : edge-coloured Markov graphs, their automorphism groups, the isotypic decomposition of $\mathbb{R}^n$ under a subgroup, and the induced representation of a group on a control set

`models.py`   : Markov models built on graph structure - colours assigned by rate laws, rate matrix $\mathbb{W}$, Laplacian $L$, friction metric $g = L^+$, control sets as submanifolds of the simplex with inherited symmetry, and spin-system builders.

`geometry.py` :

## Dependencies

Python ≥ 3.11

Numerics and plotting:

    pip install numpy scipy networkx matplotlib jax

Group theory (automorphism groups, stabilizers, orbits, character tables) is delegated to GAP through the SageMath interface, installed from the modular [passagemath](https://passagemath.org) distribution; no full Sage install is needed:

    pip install passagemath-gap passagemath-graphs passagemath-modules passagemath-flint passagemath-pari

Binary wheels are available for Linux and macOS; Windows requires WSL.

## References
- The Sage Developers, *SageMath, the Sage Mathematics Software System (Version 10.x)*, https://www.sagemath.org, DOI 10.5281/zenodo.8042260.
- The GAP Group, *GAP – Groups, Algorithms, and Programming, Version 4.14.0*, 2024, https://www.gap-system.org.
