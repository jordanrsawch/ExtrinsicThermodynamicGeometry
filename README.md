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

Group theory (automorphism groups, stabilizers, orbits, character tables) carried out by GAP through the SageMath interface, installed from the modular [passagemath](https://passagemath.org) distribution; no full Sage install is needed:

    pip install passagemath-gap passagemath-graphs passagemath-modules passagemath-flint passagemath-pari

Binary wheels are available for Linux and macOS; Windows requires WSL.


## Example: nested field-control submanifolds


```python
from symmetry import *
from models import *
from geometry import *
import numpy as np
import matplotlib.pyplot as plt
```

Define the 2x2 spin system, plot G, and retrieve Aut(G).


```python
n_spins = 4

M = spin_system(n_spins, glauber(1))
plot_graph(M.G);

AutG = M.G.aut
print("|Aut(G)|: ", AutG.order())
```

    |Aut(G)|:  384



    
![png](README_files/README_4_1.png)
    


Define the field-only control sets of varying resolution with $J_{\rm NN} = 1$ fixed, and define also the full-control set. Check total geodesity by symmetries.


```python
# 2x2 grid 0 1 / 2 3 in normal reading order; fixed ferromegnetic nearest-neighbour couplings at the endpoints
Bh, h_labels = field_basis(n_spins)
BJ, _ = coupling_basis(n_spins, [(0,1), (0,2), (1,3), (2,3)]);  J_NN = 1;  theta0 = BJ @ (J_NN * np.ones(4)) 

# nested field control sets, as explicit charts
Mh  = ControlSet(Bh @ np.array([[1, 1, 1, 1]]).T, theta0=theta0, name="M_h")    # one field
Mx  = ControlSet(Bh @ np.array([[1, 0, 0, 1], [0, 1, 1, 0]]).T, theta0=theta0, name="M_x")    # diagonals share a field
Mpar = ControlSet(Bh @ np.array([[1, 1, 0, 0], [0, 0, 1, 1]]).T, theta0=theta0, name="M_||")   # rows share a field
Mhi = ControlSet(Bh, theta0=theta0, name='M_hi') # Full field control
Full, _ = spin_full_control(4)


S, HB = Full.symmetry_group(M.G.aut)                   # induced representation dict and group H_B
# Check total geodesity
for Mk in (Mh, Mx, Mpar, Mhi):
    for amb in (Mhi, Full):
        if Mk.name != amb.name:
            bool, K = is_totally_geodesic(Mk, amb, HB)
            print(f"{Mk.name} in {amb.name}: {bool},  K = {K.structure_description()}")
```

    M_h in M_hi: True,  K = D4
    M_h in full control (4 spins): False,  K = D4
    M_x in M_hi: True,  K = C2 x C2
    M_x in full control (4 spins): False,  K = C2 x C2
    M_|| in M_hi: True,  K = C2
    M_|| in full control (4 spins): False,  K = C2
    M_hi in full control (4 spins): False,  K = 1


Set the endpoints $\pi_0 = \pi(h{=}{-1})$ and $\pi_1 = \pi(h{=}{+1})$ in $M_h$.

`stabilizer(AutG, pi0, pi1)` is the subgroup of $\mathrm{Aut}(G)$ fixing both endpoints, and `fixed_manifold(H)` is its fixed set $M_H$, the distributions constant on the orbits of $H$ (dimension = number of orbits − 1; the lumped chain). $M_H$ is totally geodesic in the simplex.


```python
pi0, pi1 = Mh.pi(-1), Mh.pi(1)

H_full = stabilizer(AutG, pi0, pi1)
MH = fixed_manifold(H_full)
print("stabilizer in Aut(G):", H_full.structure_description(), " orbits:", len(orbits(H_full)), " dim M_H =", MH.dim)
print("M_H totally geodesic in the simplex:", is_totally_geodesic(MH, Full, AutG)[0])
print("M_h totally geodesic in M_H:        ", is_totally_geodesic(Mh, MH, AutG)[0])
```

    stabilizer in Aut(G): D4  orbits: 6  dim M_H = 5
    M_H totally geodesic in the simplex: True
    M_h totally geodesic in M_H:         False


`Mhi.symmetry_group(AutG)` returns $\mathrm{Aut}(G, M_{h_i})$, the symmetries mapping $M_{h_i}$ to itself, with their induced action $S_\sigma$ on the field coordinates ($P_\sigma B = B S_\sigma$); `stabilizer(HB, pi0, pi1)` returns the subgroup $H \leq \mathrm{Aut}(G, M_{h_i})$ that fixes the endpoints. (Here `H` and `H_full` are the same group). `Mhi.fixed(H)` gives the fixed set of $H$ within $M_{h_i}$, or $M_H \cap M_{h_i}$, recovering $M_h$. `Mhi.sectors(H)` decomposes the field space $\mathbb{R}^4$ into isotypic sectors under $H$: the invariant sector $\mathcal{U}_0$ is the tangent space of $M_h$, the others are the symmetry-breaking directions, and an irrep of dimension 0 here is a symmetry that field control cannot break.


```python
S, HB = Mhi.symmetry_group(AutG)
H = stabilizer(HB, pi0, pi1)
print("H = stabilizer in Aut(G, M_hi):", H.structure_description())

Mfix, A = Mhi.fixed(H)
print("fixed set of H in M_hi: dim", Mfix.dim, " direction", np.round(A.T / np.abs(A).max(), 2), " (= M_h)")

Sc = Mhi.sectors(H)
for r in range(len(Sc.Pi)):
    kind = "invariant (tangent to M_h)" if r == Sc.rho0 else ("breaking" if Sc.dim[r] else "inaccessible")
    print(f"rho={r}: dim {Sc.dim[r]}  d={Sc.d[r]} m={Sc.m[r]}  {kind}")
    for col in Sc.U[r].T:
        print("        ", np.round(col / np.abs(col).max(), 2))
```

    H = stabilizer in Aut(G, M_hi): D4
    fixed set of H in M_hi: dim 1  direction [[1. 1. 1. 1.]]  (= M_h)
    rho=0: dim 1  d=1 m=1  invariant (tangent to M_h)
             [1. 1. 1. 1.]
    rho=1: dim 1  d=1 m=1  breaking
             [ 1. -1. -1.  1.]
    rho=2: dim 0  d=1 m=0  inaccessible
    rho=3: dim 0  d=1 m=0  inaccessible
    rho=4: dim 2  d=2 m=1  breaking
             [-0.86 -1.    1.    0.86]
             [-1.    0.86 -0.86  1.  ]


The inherited metric $\zeta(h_i) = J^\mathsf{T} L^+ J$ on $M_{h_i}$ is $H$-invariant, so at every point of $M_h$ it is block-diagonal in the sector basis (Schur's lemma), while off $M_h$ the symmetry is broken and the blocks couple. Blocks are labelled by sector: $\rho = 0$ is tangent to $M_h$, and only $\rho = 1, 4$ are accessible symmetry-breaking sectors here. Sector $\rho = 4$ carries a single 2-dimensional irrep ($d_\rho = 2$, $m_\rho = 1$), on which $\zeta$ must act as a multiple of the identity, hence the two equal entries.


```python
geo_hi = ControlGeometry(M, Mhi)
N, labels = Sc.normal_basis()
Q = np.hstack([A, N])                                   # (tangent to M_h | normal sectors)
sec = [Sc.rho0] + [int(l) for l in labels]              # sector index of each column, e.g. [0, 1, 4, 4]

points = [("h = -1", -np.ones(4)), ("h = 0", np.zeros(4)),
          ("h = +0.7", 0.7 * np.ones(4)), ("off M_h", np.array([0.3, -0.5, 0.2, 0.1]))]

fig, axes = plt.subplots(2, len(points), figsize=(3.2 * len(points), 6.2))
for j, (title, u) in enumerate(points):
    G = np.asarray(geo_hi.metric(u))
    for i, (Mat, ylab) in enumerate([(G, "field basis"), (Q.T @ G @ Q, "sector basis")]):
        ax = axes[i, j]
        rel = np.abs(Mat) / np.abs(Mat).max()
        ax.imshow(rel, cmap="Blues", vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(Mat):
            ax.text(c, r, f"{v:.3g}", ha="center", va="center", fontsize=7,
                    color="white" if rel[r, c] > 0.6 else "black")
        ticks = [f"$h_{k}$" for k in range(4)] if i == 0 else [rf"$\rho={s}$" for s in sec]
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(ticks); ax.set_yticklabels(ticks)
        if i == 0: ax.set_title(title)
        if j == 0: ax.set_ylabel(ylab)
fig.suptitle("inherited metric $\\zeta$ on $M_{h_i}$: block-diagonal in the sectors along $M_h$, coupled off it")
plt.tight_layout()
```


    
![png](README_files/README_12_0.png)
    


## References
- The Sage Developers, *SageMath, the Sage Mathematics Software System (Version 10.x)*, https://www.sagemath.org, DOI 10.5281/zenodo.8042260.
- The GAP Group, *GAP – Groups, Algorithms, and Programming, Version 4.14.0*, 2024, https://www.gap-system.org.


