"""symmetry.py — edge-coloured Markov graphs, their automorphism groups, and
the isotypic (sector) decomposition of R^n under a subgroup.

Objects and conventions
  * G = (Omega, E, C): states 0..n-1, edges, edge colouring. Colours group
    edges that share a rate law; colour 0 marks a non-edge.
  * Aut(G): permutations s of states with C[s(u), s(v)] == C[u, v]. Held as a
    Sage PermutationGroup whose domain is the state set, so elements are
    0-indexed; as_tuple(g,n) gives s with s[u] the image of u.
  * P_s: permutation matrix acting on distributions, (P_s pi)_{s(u)} = pi_u.
  * H <= Aut(G): the subgroup fixing the endpoint distributions. M_H is the
    fixed manifold, of dimension (#orbits of H) - 1.
  * Sectors: R^n = U_0 (+) sum_{rho != 0} U_rho, the isotypic decomposition
    of the permutation representation of H; U_0 = tangent directions of M_H
    (plus the normalisation), the U_rho are the symmetry-breaking sectors.

Group theory is delegated to Sage/GAP 
"""

from dataclasses import dataclass, field
from functools import cached_property
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sage.all__sagemath_graphs  
from sage.graphs.graph import Graph as SageGraph
from sage.groups.perm_gps.permgroup import PermutationGroup_generic


# ============================================================ coloured graphs

@dataclass(frozen=True)
class ColouredGraph:
    """Edge-coloured Markov graph.

    Attributes
    ----------
    n : int
        Number of states |Omega|.
    colour : np.ndarray, shape (n, n), dtype int
        Symmetric colour matrix; colour[u, v] == 0 is a non-edge, c >= 1 is an
        edge of colour c. Zero diagonal.
    labels : n-tuple of str, optional
        State labels for plotting, e.g. spin configurations.
    """
    n: int
    colour: np.ndarray
    labels: tuple = None

    def __post_init__(self):
        C = self.colour
        assert C.shape == (self.n, self.n), "colour matrix has wrong shape"
        assert (C == C.T).all(), "colour matrix must be symmetric"
        assert (np.diag(C) == 0).all(), "no self-loops"
        assert (C >= 0).all() and np.issubdtype(C.dtype, np.integer)
        assert self.labels is None or len(self.labels) == self.n, "labels must match n"

    @property
    def edges(self):
        """List of (u, v, colour) with u < v."""
        return [(u, v, int(self.colour[u, v]))
                for u in range(self.n) for v in range(u + 1, self.n)
                if self.colour[u, v] > 0]

    # Memoized group data
    @cached_property
    def aut(self):
        """Aut(G) as a Sage PermutationGroup on {0, ..., n-1}."""
        return automorphisms(self)

    @cached_property
    def orbits(self):
        """Vertex orbits of the full automorphism group."""
        return orbits(self.aut)

    def to_nx(self):
        """nx.Graph with edge attribute 'colour' and node attribute 'label' (plotting only)."""
        H = nx.Graph()
        for v in range(self.n):
            H.add_node(v, label=self.labels[v] if self.labels else str(v))
        for u, v, c in self.edges:
            H.add_edge(u, v, colour=c)
        return H


def from_edges(n, edges, colours=None):
    """ColouredGraph from an edge list; equal colour labels -> same colour.

    Colour ids are assigned 1, 2, ... in order of first appearance. Default
    None gives the trivial colouring (all colour 1).
    """
    if colours is None:
        colours = [0] * len(edges)
    assert len(colours) == len(edges), "need one colour label per edge"
    ids = {}
    C = np.zeros((n, n), dtype=int)
    for (u, v), lab in zip(edges, colours):
        assert u != v, "no self-loops"
        C[u, v] = C[v, u] = ids.setdefault(lab, len(ids) + 1)
    return ColouredGraph(n=n, colour=C)


def cycle(n, colour_pattern=None):
    """n-cycle 0-1-...-(n-1)-0; edge i = (i, i+1 mod n) has colour_pattern[i % len].

    cycle(6, ['chem', 'mech']) is the alternating six-cycle ring example
    """
    edges = [(i, (i + 1) % n) for i in range(n)]
    pat = colour_pattern or [0]
    return from_edges(n, edges, [pat[i % len(pat)] for i in range(n)])


# ================================================================ group functions

def perm_matrix(s):
    """Permutation matrix P_s for permutation s, acting as (P_s pi)_{s(u)} = pi_u"""
    s = list(s); n = len(s)
    P = np.zeros((n, n))
    P[s, range(n)] = 1.0
    return P


def as_tuple(g, n):
    """Convert Sage permutation to tuple s, with s[u] the image of u."""
    return tuple(int(g(u)) for u in range(n))


def elements(H):
    """Elements of H as 0-indexed tuples, sorted (identity first)."""
    n = H.degree()
    return sorted(as_tuple(g, n) for g in H)

def permutation_matrices(H):
    """Permutation matrices P_s for all s in H, sorted (identity first)."""
    return [perm_matrix(s) for s in elements(H)]


def automorphisms(G):
    """Aut(G): colour-preserving automorphisms as a Sage PermutationGroup.

    Use Sage's built-in functions: A.order(), A.orbits(), A.stabilizer(...), A.character_table(), ...
    """
    S = SageGraph(G.n)                        # vertices 0..n-1 first, so the domain is the state set
    S.add_edges(G.edges)                       # (u, v, colour): colour becomes the edge label
    return S.automorphism_group(edge_labels=True)


def stabilizer(A, *pis, tol=1e-12):
    """Largest subgroup of A fixing every distribution in pis.

    P_s pi = pi iff s preserves each level set {u : pi_u = c}, so the joint stabilizer is the intersection of the 
    set-wise stabilizers of all level sets of all the pis.
    """
    H = A
    for pi in pis:
        pi = np.asarray(pi, float)
        assert pi.shape == (A.degree(),), "distribution has wrong length"
        for c in np.unique(np.round(pi / tol)) * tol:            # distinct values of pi
            level_set = [int(u) for u in np.flatnonzero(np.abs(pi - c) <= tol)]
            H = H.stabilizer(level_set, "OnSets")
    return A.subgroup(H.gens())


def orbits(H):
    """Vertex orbits of H: sorted lists, ordered by smallest member."""
    return sorted(sorted(int(i) for i in o) for o in H.orbits())


def lumping_map(H):
    """Lambda : R^n -> R^r, (Lambda pi)_a = sum_{u in O_a} pi_u; rows are orbit indicators.

    Lambda^+ = Lambda^T diag(1/|O_a|) and Lambda^+ Lambda = Pi_0 (eq. Pi0);
    ker Lambda = sum_{rho != 0} U_rho (eq. conormal). Returns (Lambda, orbits).
    """
    orbs = orbits(H)
    Lam = np.zeros((len(orbs), H.degree()))
    for a, o in enumerate(orbs):
        Lam[a, o] = 1.0
    return Lam, orbs


# =================================================================== sectors

@dataclass
class Sectors:
    """Isotypic decomposition R^n = (+)_rho U_rho of a representation of H.

        Pi_rho = (d_rho / |H|) sum_h conj(chi_rho(h)) rep(h)      (Serre, Thm. 8)
        m_rho  = (1 / |H|)     sum_h conj(chi_rho(h)) tr rep(h)

    with chi_rho from H's character table. Irreps with complex characters come in conjugate pairs; 
    each pair is merged into one real sector of twice the dimension, so every Pi_rho is a real matrix.

    Construct with Sectors(H) for the permutation representation on states, or Sectors(H, rep) with 
    rep a dict {tuple: matrix} or callable tuple -> matrix, e.g. the induced representation on control coordinates. 
    A sector with dim 0 is an irrep of H absent from rep (an inaccessible sector).
    """
    # Called
    H: PermutationGroup_generic               
    rep: object = None                          # None -> perm_matrix
    tol: float = 1e-8

    # derived
    n: int = field(init=False)                  # dimension of the represented space
    Pi: list = field(init=False)                # Pi[r]: projector onto U_r
    U: list = field(init=False)                 # U[r]: orthonormal basis of U_r, shape (n, dim[r])
    dim: np.ndarray = field(init=False)         # dim[r] = rank Pi[r] = d[r] * m[r]
    d: np.ndarray = field(init=False)           # real irrep dimension
    m: np.ndarray = field(init=False)           # multiplicity
    chi: list = field(init=False)               # chi[r]: character on H.conjugacy_classes() (summed over a merged pair)
    rho0: int = field(init=False)               # index of the invariant sector

    def __post_init__(self):
        H, tol = self.H, self.tol
        rep = self.rep
        if rep is None:
            rep = perm_matrix
        elif isinstance(rep, dict):
            rep = rep.__getitem__

        order, n = int(H.order()), H.degree()
        classes = H.conjugacy_classes()
        chi = np.array(H.character_table()).astype(complex)    # rows irreps, cols classes
        d = chi[:, 0].real.round().astype(int)                 # first class is the identity
        sizes = np.array([int(c.cardinality()) for c in classes])

        # Build class sums and compute traces of rep(h) for h in each class
        R = [sum(rep(as_tuple(g, n)) for g in c) for c in classes]
        traces = np.array([np.trace(Ra) for Ra in R]) / sizes  # tr rep(h) for h in class a

        # Merge conjugate pairs of irreps into single real sectors.
        sector_irreps, seen = [], set()
        for r in range(len(chi)):
            if r in seen:
                continue
            partner = [q for q in range(len(chi)) if q != r and np.allclose(chi[q], chi[r].conj())]
            grp = [r] + partner[:1]
            seen.update(grp)
            sector_irreps.append(grp)

        # Build projectors and bases for each sector
        self.n = R[0].shape[0]
        self.Pi, self.U, dim, dd, mm, self.chi = [], [], [], [], [], []

        for irrep in sector_irreps:
            # Build projector from the character table (Serre, Thm. 8), check that it is real and idempotent
            P = sum((d[r] / order) * sum(chi[r, a].conjugate() * R[a] for a in range(len(classes)))
                    for r in irrep)
            assert np.abs(P.imag).max() < tol, "projector not real after pairing conjugates"
            P = P.real
            assert np.allclose(P @ P, P, atol=tol), "not a projector"
            assert np.allclose(P, P.T, atol=tol)
            P = 0.5 * (P + P.T)  # symmetrize to avoid numerical issues

            # The sector is the 1-eigenspace of P 
            w, V = np.linalg.eigh(P)
            Ub = V[:, w > 0.5]

            # Compute the multiplicity of the irrep in the representation from the character table (Serre, Thm. 4).
            mult = sum(np.sum(sizes * chi[r].conj() * traces) / order for r in irrep).real

            # Populate the fields
            self.Pi.append(P); self.U.append(Ub); dim.append(Ub.shape[1])
            dd.append(int(d[irrep[0]]) * len(irrep)); mm.append(int(round(mult / len(irrep))))
            self.chi.append(chi[irrep].sum(axis=0))

        self.dim, self.d, self.m = np.array(dim), np.array(dd), np.array(mm)
        assert self.dim.sum() == self.n, "sectors do not span the space"

        # Identify the invariant sector rho0 (the one with character 1 on all classes)
        rho0 = [i for i, g in enumerate(sector_irreps) if np.allclose(chi[g[0]], 1)]
        assert len(rho0) == 1
        self.rho0 = rho0[0]

    @property
    def breaking(self):
        """Indices of the symmetry-breaking sectors rho != 0."""
        return [r for r in range(len(self.Pi)) if r != self.rho0]

    @property
    def accessible(self):
        return [r for r in self.breaking if self.dim[r] > 0]

    @property
    def inaccessible(self):
        return [r for r in self.breaking if self.dim[r] == 0]

    def normal_projector(self):
        """Projector onto N_pi M_H = sum_{rho != 0} U_rho; constant in pi."""
        return np.eye(self.n) - self.Pi[self.rho0]

    def normal_basis(self):
        """Orthonormal basis of N_pi M_H split into sectors: (B, sector index per column)."""
        cols = [self.U[r] for r in self.accessible]
        labels = np.concatenate([np.full(self.dim[r], r) for r in self.accessible])
        return np.hstack(cols), labels


def induced_representation(H, B, tol=1e-9):
    """Restriction of the permutation representation of H to the subspace col(B).

    Use: the columns of B are the state-energy perturbations conjugate to the control coordinates
    u of a control set M = {pi(u) : pi_mu ~ exp(-V_mu(u))}, V(u) = V_0 + Bu. B is assumed to have full
    column rank (gauge-fixed). Sectors(H_B, rep=S) then gives the accessible and inaccessible sectors of M.

    Parameters
    ----------
    H : Sage PermutationGroup on {0, ..., n-1}
    B : (n, m) array of full column rank
    tol : residual tolerance for membership of P_sigma B in col(B)

    Returns
    -------
    S : dict {as_tuple(sigma): (m, m) array}
    H_B : Sage PermutationGroup, the elements of H preserving col(B)
    """
    B = np.asarray(B, float)
    n, m = B.shape
    assert n == H.degree(), "B has wrong number of rows"
    assert np.linalg.matrix_rank(B) == m, "B must have full column rank"

    # S_sigma = B^+ P_sigma B, then those that preserve col(B) form the subgroup H_B
    pinvB = np.linalg.pinv(B)
    S, kept = {}, []
    for g in H:
        s = as_tuple(g, n)
        PB = perm_matrix(s) @ B
        Ss = pinvB @ PB
        if np.abs(PB - B @ Ss).max() <= tol:          # col(B) preserved
            S[s] = Ss
            kept.append(g)
    H_B = H if len(kept) == H.order() else H.subgroup(kept)

    # Consistency: S_e = I, every S_sigma invertible, and S a homomorphism.
    # Sage multiplies permutations left to right, (g*h)(u) = h(g(u)), while the
    # matrices satisfy P_{g after h} = P_g P_h; so S[g] @ S[h] must equal S[h*g].
    assert np.allclose(S[tuple(range(n))], np.eye(m), atol=tol), "S_e != I"
    assert min(abs(np.linalg.det(M)) for M in S.values()) > tol, "singular S_sigma"
    worst = max(np.abs(S[as_tuple(h * g, n)] - S[as_tuple(g, n)] @ S[as_tuple(h, n)]).max()
                for g in H_B for h in H_B)
    assert worst < tol, f"S is not a homomorphism (residual {worst:.2e}); is B gauge-fixed?"
    return S, H_B


# ================================================================== plotting


def plot_graph(G, pos=None, ax=None, cmap=plt.cm.tab10):
    """Draw G with edges coloured by colour id. Returns (ax, pos) for reuse."""
    Gnx = G.to_nx()
    pos = pos or nx.kamada_kawai_layout(Gnx)
    ax = ax or plt.gca()
    edge_cols = [cmap((Gnx.edges[e]['colour'] - 1) % cmap.N) for e in Gnx.edges]
    nx.draw_networkx_nodes(Gnx, pos, ax=ax, node_color='white', edgecolors='black', node_size=600)
    nx.draw_networkx_labels(Gnx, pos, ax=ax, labels=nx.get_node_attributes(Gnx, 'label'))
    nx.draw_networkx_edges(Gnx, pos, ax=ax, edge_color=edge_cols, width=2.5)
    ax.set_axis_off()
    return ax, pos
