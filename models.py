"""models.py — Markov models on edge-coloured graphs: rate laws, conductances,
friction metric, and control sets on the simplex.

A model is a graph G with one rate law per colour. Every edge (mu, nu) carries a
symmetric conductance c_e(pi) = law_e(pi_mu, pi_nu), defining the weighted graph
Laplacian matrix L. Control sets are submanifolds of the simplex given by charts
pi(u). Linear coupling to state energies gives the exponential-family chart.
"""

from dataclasses import dataclass, field
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from symmetry import *
from itertools import combinations


# ------------------------------------------------------------- rate laws
# c(u, v) for endpoint populations u = pi_mu, v = pi_nu. Symmetric in (u, v).

def power(alpha, Gamma=1.0):
    """c = Gamma (u v)^alpha; alpha = 1/2 is the geometric-mean (SQRA) law."""
    def law(u, v):
        return Gamma * (u * v) ** alpha
    law.name = f"power(alpha={alpha},Gamma={Gamma})"
    return law

def glauber(Gamma=1.0):
    """c = Gamma u v / (u + v): Glauber (single-spin-flip) conductance."""
    def law(u, v):
        return Gamma * u * v / (u + v)
    law.name = f"glauber(Gamma={Gamma})"
    return law

def constant(Gamma=1.0):
    """c = Gamma: a fixed-barrier mechanism that does not track the populations."""
    def law(u, v):
        return Gamma + 0.0 * (u + v) # autodiff compatible
    law.name = f"constant(Gamma={Gamma})"
    return law

def hill(n=1.0, K=1.0, Gamma=1.0):
    """c = Gamma x^n / (K^n + x^n), x = sqrt(u v): saturating conductance with a scale K."""
    def law(u, v):
        xn = (u * v) ** (n / 2.0)
        return Gamma * xn / (K ** n + xn)
    law.name = f"hill(n={n},K={K},Gamma={Gamma})"
    return law

def cooperative_glauber(n=1.0, Gamma=1.0):
    """c = Gamma (u v)^n / (u^n + v^n): Glauber with cooperativity exponent, scale-free."""
    def law(u, v):
        return Gamma * (u * v) ** n / (u ** n + v ** n)
    law.name = f"coop_glauber(n={n},Gamma={Gamma})"
    return law


# ----------------------------------------------------------------- model

@dataclass(frozen=True)
class Model:
    """Markov model: an edge-coloured graph with one rate law per colour.

    Construct either with Model.from_edges(n, [(u, v, law), ...]), with colouring derived from
    the laws, or directly as Model(G, laws) with laws a dict {colour: law} and G a ColouredGraph.
    """

    G: ColouredGraph
    laws: dict                                  # {colour id: law}
    xp: object = np                             # numpy or jax.numpy

    # derived
    n: int = field(init=False)                  # number of vertices
    m: int = field(init=False)                  # number of edges
    iu: np.ndarray = field(init=False)          # edge endpoints, iu[e] < iv[e]
    iv: np.ndarray = field(init=False)
    colour: np.ndarray = field(init=False)      # colour of each edge
    incidence: np.ndarray = field(init=False)   # signed incidence, (n, m): L = D diag(c) D^T

    def __post_init__(self):
        E = self.G.edges
        set_ = object.__setattr__                 # frozen dataclass: assign via object.__setattr__
        set_(self, "n", self.G.n); set_(self, "m", len(E))
        set_(self, "iu", np.array([e[0] for e in E], int))
        set_(self, "iv", np.array([e[1] for e in E], int))
        set_(self, "colour", np.array([e[2] for e in E], int))
        missing = set(self.colour.tolist()) - set(self.laws)
        assert not missing, f"no rate law for colours {missing}"
        # Build signed incidence matrix D: D[mu, e] = +1 if mu = iu[e], -1 if mu = iv[e], 0 otherwise.
        D = np.zeros((self.n, self.m))
        D[self.iu, range(self.m)] = 1.0
        D[self.iv, range(self.m)] = -1.0
        set_(self, "incidence", D)

    @classmethod
    def from_edges(cls, n, edges, xp=np, labels=None):
        """Build the coloured graph from (u, v, law) triples.

        Edges get the same colour iff their laws have the same .name (same law and same parameters); colour ids 
        are 1, 2, ... in order of first appearance.
        """
        ids, laws = {}, {}
        C = np.zeros((n, n), dtype=int)
        for u, v, law in edges:
            assert u != v, "no self-loops"
            assert C[u, v] == 0, f"duplicate edge ({u}, {v})"
            c = ids.setdefault(law.name, len(ids) + 1)
            laws[c] = law
            C[u, v] = C[v, u] = c
        return cls(ColouredGraph(n=n, colour=C, labels=labels), laws, xp)

    def conductances(self, pi):
        """c_e(pi) = law_{colour(e)}(pi_u, pi_v) for every edge."""
        u, v = pi[self.iu], pi[self.iv]
        c = 0.0
        for col, law in self.laws.items():
            c = c + (self.colour == col) * law(u, v)
        return c

    def L(self, pi):
        """Conductance Laplacian L = D diag(c) D^T: PSD, kernel span{1}."""
        return (self.incidence * self.conductances(pi)) @ self.incidence.T

    def Lplus(self, pi):
        """Friction metric g = L^+ = (L + J)^{-1} - J with J = 11^T/n (exact since ker L = span{1})."""
        J = self.xp.ones((self.n, self.n)) / self.n
        return self.xp.linalg.inv(self.L(pi) + J) - J

    def W(self, pi):
        """Detailed-balance generator with stationary distribution pi: w_{mu nu} = c_{mu nu} / pi_nu 
        numpy only; for jax use .at[].add."""
        c = self.conductances(pi)
        R = np.zeros((self.n, self.n))
        np.add.at(R, (self.iu, self.iv), c / pi[self.iv])
        np.add.at(R, (self.iv, self.iu), c / pi[self.iu])
        R[np.diag_indices(self.n)] = -R.sum(axis=0)
        return R

    # ---- symmetry ----------------------------------------------------------
    def check_symmetry(self, pi=None, tol=1e-10, H=None):
        """Verify that the graph automorphisms are dynamical symmetries:
        P_s L(pi) P_s^T == L(P_s pi) for the generators of H (default Aut(G)),
        at a generic pi (default: random). Returns the worst residual."""
        H = self.G.aut if H is None else H
        if pi is None:
            rng = np.random.default_rng(0)
            pi = rng.random(self.n); pi /= pi.sum()
        pi = np.asarray(pi, float)
        worst = 0.0
        for g in H.gens():
            P = perm_matrix(as_tuple(g, self.n))
            worst = max(worst, np.abs(P @ self.L(pi) @ P.T - self.L(P @ pi)).max())
        assert worst < tol, f"colouring is not a dynamical symmetry (residual {worst:.2e})"
        return worst


# ------------------------------------------------------------ control sets

@dataclass(frozen=True)
class ControlSet:
    """A control set M' = { pi(u) : u in R^m } in the exponential-family chart

            pi(u) = softmax(theta0 + B u),

    i.e. control coordinates coupled linearly to the state energies in a system in contact with
    a single heat bath. Column j of B is the force conjugate to u^j; theta0 is a fixed
    background. The tangent space at pi(u) is (diag pi - pi pi^T) col(B).

    Symmetry: sectors(H) restricts the permutation representation of H to col(B) and decomposes 
    the control space; fixed(H) is the sub-control-set M' cap M_H.
    """
    B: np.ndarray
    theta0: np.ndarray = None
    name: str = ""

    # derived
    n: int = field(init=False)                  # number of states
    dim: int = field(init=False)                # number of control coordinates m

    def __post_init__(self):
        set_ = object.__setattr__
        B = np.asarray(self.B, float)
        n, m = B.shape
        B = B - B.mean(axis=0, keepdims=True)                    # gauge fix
        s = np.linalg.svd(B, compute_uv=False)
        rank = int((s > 1e-10 * max(1.0, s[0])).sum())
        assert rank == m, f"control basis has rank {rank} < {m} after gauge fixing (redundant coordinates)"
        set_(self, "B", B)
        set_(self, "theta0", np.zeros(n) if self.theta0 is None else np.asarray(self.theta0, float))
        set_(self, "n", n); set_(self, "dim", m)

    # ---- chart -------------------------------------------------------------
    def pi(self, u):
        """pi(u) = softmax(theta0 + B u); u may be a scalar when dim == 1. jax-traceable."""
        th = self.theta0 + self.B @ jnp.atleast_1d(u)
        w = jnp.exp(th - th.max())
        return w / w.sum()

    def jacobian(self, u):
        """d pi / d u = (diag pi - pi pi^T) B, shape (n, m); columns lie in 1^perp."""
        p = self.pi(u)
        return (jnp.diag(p) - jnp.outer(p, p)) @ self.B

    def coordinates(self, pi, atol=1e-10):
        """u with pi(u) = pi, by least squares on log pi. Raises if pi is not in M'."""
        A = np.hstack([self.B, np.ones((self.n, 1))])
        u = np.linalg.lstsq(A, np.log(pi) - self.theta0, rcond=None)[0][:self.dim]
        err = float(np.max(np.abs(np.asarray(self.pi(u)) - pi)))
        if err > atol:
            raise ValueError(f"pi not in control set '{self.name}' (residual {err:.2e})")
        return u

    # ---- symmetry ----------------------------------------------------------
    def symmetry_group(self, H, tol=1e-9):
        """Aut(G, M'): the elements of H mapping M' to itself, with their action S on control coordinates.

        Asserts H-invariant theta0 by checking that (P_s - I) theta0 is in col(B) + span{1} for all s in H_B.
        """
        S, H_B = induced_representation(H, self.B, tol)
        if np.any(self.theta0):
            A1 = np.hstack([self.B, np.ones((self.n, 1))]) # Columns span col(B) + span{1}
            kept = []
            for g in H_B:
                P = perm_matrix(as_tuple(g, self.n))
                shift = P @ self.theta0 - self.theta0
                coef = np.linalg.lstsq(A1, shift, rcond=None)[0]
                if np.abs(A1 @ coef - shift).max() <= tol:
                    assert np.abs(coef[:self.dim]).max() <= tol, \
                        "theta0 is not H-invariant: the induced action is affine"
                    kept.append(g)
            if len(kept) < H_B.order():
                H_B = H.subgroup(kept)
                S = {s: M for s, M in S.items() if s in {as_tuple(g, self.n) for g in kept}}
        return S, H_B

    def sectors(self, H, tol=1e-9):
        """Isotypic decomposition of the control space R^m under Aut(G, M') <= H."""
        S, H_B = self.symmetry_group(H, tol)
        return Sectors(H_B, rep=S)

    def fixed(self, H, tol=1e-9):
        """The sub-control-set M = M' cap M_H, and the lift A of its coordinates
        to control coordinates: u = A y, pi = softmax(theta0 + B A y).
        A is an orthonormal basis of the invariant sector U'_0."""
        Sc = self.sectors(H, tol)
        A = Sc.U[Sc.rho0]
        assert A.shape[1] > 0, "M' cap M_H is a point"
        return ControlSet(self.B @ A, self.theta0, name=f"{self.name} ∩ M_H"), A


def simplex(n):
    """Full control: the open simplex, coordinates = log-weights of all n states (one gauge direction removed)."""
    return ControlSet(np.eye(n)[:, :-1], name="simplex")


def fixed_manifold(H):
    """M_H = {pi : P_s pi = pi for all s in H} as a control set; columns are
    orbit indicators, so pi is constant on the orbits of H. dim = #orbits - 1."""
    orbs = orbits(H)
    B = np.zeros((H.degree(), len(orbs)))
    for a, o in enumerate(orbs):
        B[o, a] = 1.0
    return ControlSet(B[:, :-1] if len(orbs) > 1 else B, name=f"M_H (r={len(orbs)})")


def is_totally_geodesic(M, Mprime, A, tol=1e-9):
    """
    """
    n = M.n
    assert n == Mprime.n and np.allclose(M.theta0, Mprime.theta0), "M and Mprime differ in theta0"
    coef = np.linalg.lstsq(Mprime.B, M.B, rcond=None)[0]
    assert np.abs(Mprime.B @ coef - M.B).max() < tol, "M is not a subset of Mprime"

    S, H_amb = induced_representation(A, Mprime.B, tol)           # sigma with P_sigma col(B_amb) = col(B_amb)
    # sigma fixes M pointwise iff P_sigma (theta0 + M.B y) = theta0 + M.B y mod constants for all y:
    # (P_sigma - I) theta0 in span(1) and P_sigma M.B = M.B.
    fixing = []
    for g in H_amb:
        P = perm_matrix(as_tuple(g, n))
        shift = P @ M.theta0 - M.theta0
        if np.abs(shift - shift.mean()).max() < tol and np.abs(P @ M.B - M.B).max() < tol:
            fixing.append(g)

    K = H_amb.subgroup(fixing)
    if K.order() == 1:
        return False, K
    
    # Above determined that M \subseteq Fix(K). Remains to check that Fix(K) \subseteq M.
    # Proceeds by determining an orthonormal basis F if the invariant sector under K and then
    # checking whether F and M.B have the same column space.
    SK = {s: Ss for s, Ss in S.items() if s in {as_tuple(g, n) for g in K}}  # Induced representation of Aut(G,M') in coods of M'
    Sec = Sectors(K, rep=SK)
    F = Mprime.B @ Sec.U[Sec.rho0] 
    same = (F.shape[1] == M.dim and np.abs(F @ np.linalg.lstsq(F, M.B, rcond=None)[0] - M.B).max() < tol) 

    return bool(same), K

def is_totally_geodesic(M, Mprime, A, tol=1e-9):
    """Check if the control set M is totally geodesic in Mprime.

    Sufficient condition: M is the fixed-point set within Mprime of a subgroup K of Aut(G, Mprime) whose elements 
    are isometries of the inherited metric. K is taken as large as possible: every element of A preserving 
    Mprime that fixes M pointwise. Returns (verdict, K).

    Containment M in Mprime is checked set-wise. 
    
    A False verdict means "not certified by symmetry", not "not totally geodesic".
    """
    n = M.n
    assert n == Mprime.n

    # M in Mprime: theta0 - theta0' in col(B') + span(1), and col(B) in col(B').
    A1 = np.hstack([Mprime.B, np.ones((n, 1))])
    base = np.linalg.lstsq(A1, M.theta0 - Mprime.theta0, rcond=None)[0]
    assert np.abs(A1 @ base - (M.theta0 - Mprime.theta0)).max() < tol, "M's background is not in Mprime"
    direc = np.linalg.lstsq(Mprime.B, M.B, rcond=None)[0]              # M's directions in Mprime coordinates
    assert np.abs(Mprime.B @ direc - M.B).max() < tol, "M's directions are not in Mprime"

    S, H_amb = induced_representation(A, Mprime.B, tol)                # sigma with P_sigma col(B') = col(B')
    # sigma fixes M pointwise iff (P_sigma - I) theta0 is constant and P_sigma B = B.
    fixing = []
    for g in H_amb:
        P = perm_matrix(as_tuple(g, n))
        shift = P @ M.theta0 - M.theta0
        if np.abs(shift - shift.mean()).max() < tol and np.abs(P @ M.B - M.B).max() < tol:
            fixing.append(g)
    K = H_amb.subgroup(fixing)
    if K.order() == 1:
        return False, K
    # Fix(K) in Mprime is (M's base point) + (invariant sector of K on Mprime's coordinates);
    # M is that set iff its direction space equals the invariant sector.
    SK = {s: Ss for s, Ss in S.items() if s in {as_tuple(g, n) for g in K}}
    Sec = Sectors(K, rep=SK)
    F = Sec.U[Sec.rho0]
    same = (F.shape[1] == M.dim and np.abs(F @ (F.T @ direc) - direc).max() < tol)
    return bool(same), K


# ============================================================ spin systems
# State index s in 0..2**k - 1 for k spins; bit j of s is 1 iff spin j is up, sigma_j(s) = +1. 

def spins(n_spins):
    """sigma[s, j] = +1/-1 for spin j in state s; shape (2**n_spins, n_spins)."""
    s = np.arange(2 ** n_spins)
    return np.array([[1 if (x >> j) & 1 else -1 for j in range(n_spins)] for x in s], float)

def spin_labels(n_spins):
    """State labels like '$0$$1$' read left to right as spin 0, 1, ..."""
    return tuple(''.join(r'1' if (s >> j) & 1 else r'0' for j in range(n_spins))
                 for s in range(2 ** n_spins))

def spin_system(n_spins, laws, xp=np):
    """Model of n_spins spins under single-spin-flip dynamics.

    Edges connect states differing by one spin flip; the edge (s, s ^ 2**j) carries laws[j], 
    the flip law of spin j, so its colour is spin j's mechanism class.
    """
    if callable(laws):
        laws = [laws] * n_spins
    assert len(laws) == n_spins, "need one flip law per spin"

    n = 2 ** n_spins
    edges = [(s, s ^ (1 << j), laws[j]) for s in range(n) for j in range(n_spins) if s < s ^ (1 << j)]
    return Model.from_edges(n, edges, xp=xp, labels=spin_labels(n_spins))


def spin_monomials(n_spins, subsets):
    """Control basis of spin monomials: column j is prod_{k in subsets[j]} sigma_k.

    Returns (B, names). subsets like [(0,), (1,), (0, 1)] gives the two fields
    and the coupling of a two-spin system. Nonempty monomials have zero mean
    over states, so gauge fixing leaves them unchanged.
    """
    sig = spins(n_spins)
    B = np.stack([np.prod(sig[:, list(sub)], axis=1) for sub in subsets], axis=1)
    names = ["".join(f"s{k}" for k in sub) for sub in subsets]
    return B, names

def field_basis(n_spins, on=None):
    """Local fields h_j on the spins in `on` (default all): columns sigma_j."""
    on = list(range(n_spins)) if on is None else list(on)
    B, _ = spin_monomials(n_spins, [(j,) for j in on])
    return B, [f"h{j}" for j in on]


def coupling_basis(n_spins, pairs):
    """Pair couplings J_ij for (i, j) in pairs: columns sigma_i sigma_j."""
    B, _ = spin_monomials(n_spins, [tuple(p) for p in pairs])
    return B, [f"J{i}{j}" for i, j in pairs]

def spin_full_control(n_spins, name=None):
    """Full control of a spin system in Ising coordinates.

    Columns are all 2**n_spins - 1 nonempty spin monomials, ordered by increasing order (fields, then pair couplings, 
    then triples, ...). This is full control with u = (h, J, K, ...) the coefficients of the general Ising Hamiltonian.

    Returns (ControlSet, names).
    """
    subsets = [c for r in range(1, n_spins + 1) for c in combinations(range(n_spins), r)]
    B, names = spin_monomials(n_spins, subsets)
    return ControlSet(B, name=name or f"full control ({n_spins} spins)"), names

