"""symmetry.py - edge-coloured Markov graphs, their automorphim groups, and 
the isotypic (sector) decomposition under a subgroup.

Group theory delegated to Sage/GAP. 
"""

from dataclasses import dataclass, field
from functools import cached_property
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sage.all__sagemath_graphs
from sage.graphs.digraph import DiGraph as SageDiGraph
from sage.groups.perm_gps.permgroup import PermutationGroup_generic
from copy import copy

# ============ coloured graphs =============
@dataclass(frozen = True)
class ColouredGraph:
    """ Edge-coloured Markov digraph. 
    
    Each ordered pair of adjacent states (mu, nu) carries a colour colour[mu, nu]
    that stands for the rate law of the transition nu -> mu. 

    Under detailed balance, each edge carries only a single independent colour, as the
    rate law of the reverse transition is determined by the forward transition and the
    stationary distribution. Each edge is therefore one of two kinds:
        * undirected edges, with colour[mu, nu] = colour[nu, mu] (conductance is symmetric)
        * directed edges, with colour[mu, nu] != colour[nu, mu] (conductance is asymmetric)
    
    Attributes
    ----------
    n : int
        Number of states (vertices) in the graph.
    colour : np.ndarray, shape (n, n), dtype int
        Colour matrix, zero on the diagonal and for non-adjacent states. 
    labels : n-tuple of str, optional
        Labels for the states (vertices) of the graph. If not provided, defaults to 
        string representations of the integers 0, 1, ..., n-1.
    
    """
    n: int
    colour: np.ndarray
    labels: tuple = None
    laws: dict = None       # {colour: rate law w(y)}, set by models2.markov_graph; None for a bare graph

    # Conditions on colour matrix and labels
    def __post_init__(self):
        C = self.colour
        assert C.shape == (self.n, self.n), "Colour matrix must be square with shape (n, n)"
        assert np.issubdtype(C.dtype, np.integer) and (C >= 0).all(), "Colour matrix must be non-negative integers"
        assert (np.diag(C) == 0).all(), "No self-loops allowed: diagonal of colour matrix must be zero"
        assert ((C > 0) == (C > 0).T).all(), "All transitions must be reversible" 
        for c in np.unique(C[C > 0]):
            assert len(np.unique(C.T[C == c])) == 1, f"Transitions of colour {c} have reverses of different colours"
        assert self.labels is None or len(self.labels) == self.n, "Wrong number of labels provided"

    # Basic properties
    @property
    def is_directed(self):
        """Check if the graph is directed. True if some rate lawdepends on the direction of its transition."""
        return bool((self.colour != self.colour.T).any())
    @property
    def arcs(self):
        """ List of directed edges (arcs) together with their colours.""" 
        C = self.colour 
        return [(mu, nu, int(C[mu,nu])) for mu in range(self.n) for nu in range(self.n) if C[mu, nu] > 0]
    @property
    def edges(self):
        """ Every adjacent pair of states, with the colour of directed edges taken to be the ordering (mu, nu)
            with the smaller colour."""
        C = self.colour
        def listed(mu, nu):
            return C[mu, nu] < C[nu, mu] or (C[mu, nu] == C[nu, mu] and mu < nu)
        return [(mu, nu, int(C[mu, nu])) for mu in range(self.n) for nu in range(self.n) 
                if C[mu, nu] > 0 and listed(mu, nu)]

    # Memoized group data
    @cached_property
    def aut(self):
        """Aut(G) as a Sage Permutation group on {0, 1, ..., n-1}."""
        return automorphisms(self)
    @cached_property
    def orbits(self):
        """Orbits of the full automorphism group."""
        return orbits(self.aut)

    # Convert to networkx graph
    def to_nx(self):
        """nx.DiGraph with one arc per entry of `edges'. Used for plotting."""
        graph = nx.DiGraph()
        for mu in range(self.n):
            graph.add_node(mu, label = self.labels[mu] if self.labels is not None else str(mu))
        for mu, nu, c in self.edges:
            graph.add_edge(nu, mu, colour = c, directed = bool(self.colour[nu, mu] != c))
        return graph


# Simple constructor functions for ColouredGraph objects
def from_edges(n, edges, colours = None, labels = None):
    """Undirected ColouredGraph from an edge list.
    
    Colour ids are assigned 1, 2, ... in order of first appearance. Default None
    gives trivial colouring (C[mu, nu] = 1 for all edges)."""
    if colours is None:
        colours = [1] * len(edges)
    assert len(colours) == len(edges), "Number of colours must match number of edges"
    ids = {}
    C = np.zeros((n, n), dtype=int)
    for (mu, nu), lab in zip(edges, colours):
        assert mu != nu, "No self-loops allowed"
        C[mu, nu] = C[nu, mu] = ids.setdefault(lab, len(ids) + 1)
    return ColouredGraph(n=n, colour=C, labels=labels)

# ============ basic group functions =============

def as_tuple(g, n):
    """Convert Sage permutation g to a tuples s, with s[u] the image of u."""
    return tuple(int(g(u)) for u in range(n))

def elements(H):
    """Elements of Sage group H as 0-indexed tuples."""
    n = H.degree()
    return sorted(as_tuple(g, n) for g in H)

def orbits(H):
    """Orbits of sage group H as a list of lists of integers."""
    return sorted(sorted(int(i) for i in orb) for orb in H.orbits())

def perm_matrix(s):
    """Permutation matrixfor permutation s."""
    s = list(s); n = len(s)
    P = np.zeros((n, n))
    P[s, range(n)] = 1.0
    return P

def perm_matrices(H):
    """Permutation matrices for all elements of Sage group H."""
    return [perm_matrix(s) for s in elements(H)]

def cycle_notation(s):
    """Permutation tuple s (s[mu] = image of mu) in cycle notation."""
    cycles, seen = [], set()
    for start in range(len(s)):
        if start in seen or s[start] == start:
            continue
        cycle, mu = [], start
        while mu not in seen:
            seen.add(mu); cycle.append(mu); mu = s[mu]
        cycles.append("(" + " ".join(str(mu) for mu in cycle) + ")")
    return "".join(cycles) or "id"

def automorphisms(G):
    """Automorphism group of ColouredGraph G as a Sage PermutationGroup using Sage's built-in
    functions. Computed as the automorphism group of the digraph with edges (mu, nu) labelelled 
    by colour."""
    digraph = SageDiGraph(G.n) 
    digraph.add_edges(G.arcs) 
    return digraph.automorphism_group(edge_labels=True)

def stabilizer(H, *pis, tol = 1E-9):
    """Largest subgroup of H fixing every distribution in pis."""
    H_out = H
    for pi in pis:
        pi = np.asarray(pi, float)
        assert pi.shape == (H.degree(),), "Distribution has wrong shape"
        assert (pi > 0).all(), "Distribution must lie in the interior of the simplex"
        V = -np.log(pi); breaks = np.flatnonzero(np.diff(V[np.argsort(V)]) > tol) + 1
        level_sets = [sorted(int(u) for u in block) for block in np.split(np.argsort(V), breaks)] # sets of equal occupation
        for block in level_sets:
            # Check which elements of H_out fix the block setwise, rebuild group from those elements
            H_out = H.subgroup(H_out.stabilizer(block, "OnSets").gens())
    return H_out

# ============ Character theory =============

def real_character_table(H, show=False):
    """Real irreducible characters of Sage permutation group H, computed from its complex characters.
    
    Returns a dict with keys 'order', 'classes', 'chi', 'd', 'e', 'kind, 'chi_complex'."""
    n, order = H.degree(), int(H.order())

    # Conjugacy classes aslists of tuples, starting with identity class
    classes = sorted(sorted(as_tuple(g, n) for g in classes) for classes in H.conjugacy_classes())
    class_of = {s: a for a, cls in enumerate(classes) for s in cls}
    sizes = np.array([len(cls) for cls in classes], dtype=int)

    # Complex irreducible characters from GAP, evaluated on one element for each conjugacy class. 
    representatives = {class_of[as_tuple(g,n)]: g for g in H}
    chi_complex = np.array([[complex(character(representatives[a])) for a in range(len(classes))]
                             for character in H.irreducible_characters()])

    # Class containing sigma^2, for sigma in each class.         
    square_class = [class_of[tuple(s[s[mu]] for mu in range(n))] for s in (cls[0] for cls in classes)]

    rows, used = [], set()
    for r, chi in enumerate(chi_complex):
        if r in used:
            continue
        indicator = int(round((sizes * chi[square_class]).sum().real / order))
        if indicator == 1:
            chi_real, kind = chi, "real"
        elif indicator == -1:
            chi_real, kind = 2 * chi, "quaternionic"
        else:
            partner = next(q for q, other in enumerate(chi_complex) if q != r and np.allclose(other, chi.conj()))
            used.add(partner)
            chi_real, kind = chi + chi_complex[partner], "complex"
        assert np.abs(chi_real.imag).max() < 1E-9, "Real character has nonzero imaginary part. That shouldn't happen..."
        rows.append((chi_real.real, kind))

    # Sort irreps: First trivial, then sort by dimension, then by character values.
    rows.sort(key = lambda row: (not np.allclose(row[0], 1), row[0][0], tuple(-row[0])))
    chi_real = np.array([row[0] for row in rows])
    d = np.rint(chi_real[:,0]).astype(int)
    e = np.rint((sizes * chi_real **2).sum(axis = 1) / order).astype(int)
    assert (d ** 2 // e).sum() == order, "Sum of d^2 / e over the real irreps does not equal |H|. That shouldn't happen..."
    table = {"order": order,                    # |H|
             "classes": classes,                # conjugacy classes
             "chi": chi_real,                   # real characters 
             "d": d,                            # d[rho] = dimension of real irrep rho
             "e": e,                            # e[rho] = 1, 2, or 4 dep on type of irrep
             "kind": [row[1] for row in rows],  # 'real', 'complex', or 'quaternionic' 
             "chi_complex": chi_complex         # complex characters from GAP
             }

    if show:
        print_character_table(table)
    return table

def print_character_table(table):
    """Print the conjugacy classes, the complex character table, and the real character table."""
    classes, chi_complex = table["classes"], table["chi_complex"]
    fmt = lambda x: f"{x.real:g}" if abs(x.imag) < 1E-9 else f"{x.real:.3g}{x.imag:+.3g}i"
    width = max(6, max(len(fmt(x)) for x in chi_complex.ravel()) + 2)
    header = "".join(f"{'C' + str(a):>{width}}" for a in range(len(classes)))
    print(f"|H| = {table['order']}\n\nConjugacy classes")
    for a, cls in enumerate(classes):
        print(f"  C{a}: size {len(cls)}, representative {cycle_notation(cls[0])}")
    print(f"\nComplex irreducible characters\n{'':>4}{header}")
    for r, chi in enumerate(chi_complex):
        print(f"{r:>4}" + "".join(f"{fmt(x):>{width}}" for x in chi))
    print(f"\nReal irreducible characters\n{'rho':>4}{header}    d  e  type")
    for rho, chi in enumerate(table["chi"]):
        print(f"{rho:>4}" + "".join(f"{x:>{width}g}" for x in chi)
              + f"    {table['d'][rho]}  {table['e'][rho]}  {table['kind'][rho]}")

# ============ Sectors =============

def sector_projectors(table, rep=None, tol=1e-9):
    """Multiplicities m_rho and projectors Pi_rho of the sectors of a real representation of H.

    table : dict from real_character_table(H).
    rep   : dict {sigma: matrix}, with sigma a tuple as in table['classes'], giving the matrix that represents
            each element of H. None gives the permutation matrices P_sigma (the representation on states).

    Returns (m, Pi): m[rho] is the multiplicity of real irrep rho in the representation, and Pi[rho] the projector
    onto its sector, of rank d_rho m_rho.
    """
    if rep is None:
        rep = {sigma: perm_matrix(sigma) for cls in table["classes"] for sigma in cls}
    order, chi, d, e = table["order"], table["chi"], table["d"], table["e"]

    m, Pi = [], []
    for rho in range(len(chi)):
        character_sum = sum(chi[rho, a] * rep[sigma] for a, cls in enumerate(table["classes"]) for sigma in cls)
        m.append(np.trace(character_sum) / (e[rho] * order))
        Pi.append(d[rho] / (e[rho] * order) * character_sum)

    return np.rint(m).astype(int), Pi

def sector_basis(projector, tol=1e-8):
    """Orthonormal basis of the sector onto which `projector` projects, as the columns of a matrix Q.

    Gram-Schmidt on the projected coordinate vectors Pi e_0, Pi e_1, ...: each is made orthogonal to the basis
    vectors found so far and is kept, normalized, if not annihilated. Loop stops when rank Pi = tr Pi basis 
    vectors are found. 
    """
    dimension = len(projector)
    rank = int(round(np.trace(projector)))
    basis = []
    for i in range(dimension):
        if len(basis) == rank:
            break
        vector = projector[:, i].copy()                      # Pi e_i
        for q in basis:
            vector -= (q @ vector) * q                       
        if np.linalg.norm(vector) > tol:
            basis.append(vector / np.linalg.norm(vector))

    assert len(basis) == rank, "could not find a full basis of the sector"
    return np.column_stack(basis) if basis else np.zeros((dimension, 0))

def induced_representation(H, B, tol=1e-9):
    """Action of the elements of H on the coordinates of a control set.

    B[mu, i] = -d V_mu / d u_i is the (n, m) matrix of conjugate forces.

    Returns S = {sigma: S_digma} a dictionary of matrices, H_B the subgroup of H whose elemnts preserve the
    column space of B. 
    """
    B = np.asarray(B, float)
    n, m = B.shape
    assert n == H.degree(), "B needs one row per state"

    # A uniform shift of all state energies does not change pi, so only the zero-mean part of each column matters.
    B = B - B.mean(axis=0)
    assert np.linalg.matrix_rank(B) == m, "the control coordinates are redundant: B does not have full column rank"

    B_pinv = np.linalg.pinv(B)
    S, preserving = {}, []
    for g in H:
        sigma = as_tuple(g, n)
        permuted_B = perm_matrix(sigma) @ B
        S_sigma = B_pinv @ permuted_B
        if np.abs(permuted_B - B @ S_sigma).max() <= tol:        # P_sigma B = B S_sigma: column space preserved
            S[sigma] = S_sigma
            preserving.append(g)
    return S, H.subgroup(preserving)


def sectors(H, rep=None):
    """Sectors of a real representation of H: one dict for each real irrep rho, with rho = 0 the symmetric sector.

    rep is a dict {sigma: matrix} as in sector_projectors; None gives the permutation matrices on the states.
    Each dict has the irrep dimension 'd', its type 'e' (1, 2 or 4), the multiplicity 'm' of the irrep in the
    representation, the 'projector' onto the sector, and an orthonormal 'basis' of it (d m columns).
    """
    table = real_character_table(H)
    m, Pi = sector_projectors(table, rep)
    return [{"d": int(table["d"][rho]), "e": int(table["e"][rho]), "m": int(m[rho]),
             "projector": Pi[rho], "basis": sector_basis(Pi[rho])} for rho in range(len(m))]