"""models.py - Markov models on edge-coloured graphs. Defines rate laws, conductances, the
friction metric, and control sets on the simplex.
"""

from dataclasses import dataclass, field 
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from symmetry import *
from itertools import combinations 

# ============ rate laws =============
# A rate law is a function w(y) of y = pi_target / pi_source = exp[-(V_target - V_source)]

_laws = {}      # (name, parameters) -> law. For setting colours in model.

def glauber(w0=1.0):
    """Glauber rates, w(y) = w0 y / (1 + y)."""
    key = ("glauber", float(w0))
    if key not in _laws:
        def w(y):
            return w0 * y / (1 + y)
        _laws[key] = w
    return _laws[key]

def power_law(delta, w0=1.0):
    """Power-law rates with splitting factor delta, w(y) = w0 y^delta. delta = 1/2 gives square-root rates,
    edge is directed for delta != 1/2."""
    key = ("power_law", float(delta), float(w0))
    if key not in _laws:
        def w(y):
            return w0 * y ** delta
        _laws[key] = w
    return _laws[key]

def reverse_law(w):
    """Rate law of the reverse transition, fixed by detailed balance: w_reverse(y) = y w(1/y)."""
    def w_reverse(y):
        return y * w(1 / y)
    return w_reverse

def is_symmetric(w):
    """True if the rate law is the same in both directions, w(y) = y w(1/y)."""
    y = np.array([0.3, 1.7, 5.0])
    return bool(np.allclose(w(y), reverse_law(w)(y), rtol=1e-12, atol=0))


# ============ Markov graph with rate laws =============

def markov_graph(n, transitions, labels=None):
    """ColouredGraph on n states that carries its rate laws, from a list of transitions (source, target, w).

    Each adjacent pair of states is listed once, in either direction: w is the rate law of source -> target, and
    the law of target -> source follows from detailed balance (reverse_law). Two transitions get the same colour
    if and only if they are given the same law object, e.g. the same glauber(1.0). The edge is undirected if
    w is the same in both directions (is_symmetric) and directed otherwise.

    The laws are stored in G.laws = {colour: w}.
    """
    forward_colour, reverse_colour, laws = {}, {}, {}
    C = np.zeros((n, n), dtype=int)
    for source, target, w in transitions:
        assert source != target, "No self-loops allowed"
        assert C[source, target] == 0, f"States {source} and {target} are listed twice"
        if w not in forward_colour:                              # first appearance of this law: assign a colour
            forward_colour[w] = len(laws) + 1
            laws[forward_colour[w]] = w
            if is_symmetric(w):
                reverse_colour[w] = forward_colour[w]
            else:                                                # the reverse law gets the next colour
                reverse_colour[w] = len(laws) + 1
                laws[reverse_colour[w]] = reverse_law(w)
        C[target, source] = forward_colour[w]                    # colour[mu, nu] is the colour of nu -> mu
        C[source, target] = reverse_colour[w]
    return ColouredGraph(n=n, colour=C, labels=labels, laws=laws)

def rate_matrix(G, pi):
    """Rate matrix W(pi): W[mu, nu] = w_{mu nu}(pi_mu / pi_nu) is the rate of the transition nu -> mu."""
    assert G.laws is not None, "Graph carries no rate laws - build it with markov_graph."
    mu, nu, colour = np.array(G.arcs).T   
    rates = jnp.stack([G.laws[c](pi[m] / pi[n]) for m, n, c in G.arcs])
    W = jnp.zeros((G.n, G.n)).at[mu, nu].set(rates)
    return W - jnp.diag(W.sum(axis=0))

def laplacian(G, pi):
    """Weighted graph Laplacian L(pi) = -W(pi) D_pi. 
    Off-diagonal entries are minus the conductances, c_{mu nu} = w_{mu nu} pi_nu = -L[mu, nu].
    """
    return -rate_matrix(G, pi) * pi                           

# ============ control sets =============

def equilibrium(u, B, theta0=None):
    """Equilibrium distribution pi(u) propto exp(theta0 + B u)."""
    theta = B @ u if theta0 is None else theta0 + B @ u
    weights = jnp.exp(theta - theta.max()) 
    return weights / weights.sum()

def chart_jacobian(u, B, theta0=None):
    """Jacobian d pi / d u = (D_pi - pi pi^T) B, shape (n, m). Columns are tangent vectors."""
    pi = equilibrium(u, B, theta0)
    return (jnp.diag(pi) - jnp.outer(pi, pi)) @ B

def coordinates(pi, B, theta0=None, tol=1e-10):
    """Control coordinates u of a distribution pi in the control set, pi = equilibrium(u, B, theta0).

    ln pi - theta0 = B u + constant is solved for u by least squares. Raises ValueError if pi is not in the set.
    """
    pi = np.asarray(pi, float)
    target = np.log(pi) if theta0 is None else np.log(pi) - theta0
    u = np.linalg.lstsq(np.column_stack([B, np.ones(len(pi))]), target, rcond=None)[0][:-1]
    error = np.abs(np.asarray(equilibrium(u, B, theta0)) - pi).max()
    if error > tol:
        raise ValueError(f"The distribution is not in the control set (largest discrepancy {error:.1e})")
    return u

def control_set_symmetries(H, B, theta0=None, tol=1e-9):
    """The elements of H that map the control set to itself, and their action on the control coordinates.

    Returns (S, H_M): H_M is the subgroup of such elements, Aut(G, M') when H = Aut(G), and S = {sigma: S_sigma}.
    """
    S, H_B = induced_representation(H, B, tol)
    if theta0 is None:
        return S, H_B
    theta0 = np.asarray(theta0, float)
    centred_B = B - B.mean(axis=0)
    assert np.abs(centred_B.T @ theta0).max() <= tol * (1 + np.abs(theta0).max()), \
        "theta0 has a component along the columns of B; move it into the control coordinates"

    n = len(theta0)
    preserving = []
    for g in H_B:
        shift = perm_matrix(as_tuple(g, n)) @ theta0 - theta0
        if np.ptp(shift) <= tol:
            preserving.append(g)
    H_M = H_B.subgroup(preserving)
    return {as_tuple(g, n): S[as_tuple(g, n)] for g in H_M}, H_M

def control_set_sectors(H, B, theta0=None, tol=1e-9):
    """Sectors of the control coordinates of (B, theta0) under H: sectors(H, S), with S the action of H on the
    coordinates. H must be a subgroup of Aut(G, M'), with M' = span{pi(Bu + theta_0)}."""
    S, H_M = control_set_symmetries(H, B, theta0, tol)
    assert H_M.order() == H.order(), "Some elements of H do not map the control set to itself"
    return sectors(H, S)

# ============ symmetric control sets =============

def symmetric_subset(H, B, theta0=None, tol=1e-9):
    """The symmetric control set M = M_H ∩ M' inside the control set M' given by (B, theta0), from S_sigma u = u.

    Returns (B_M, Q_0). B_M = B Q_0 holds the control directions of M (given same theta0 as M'), and the
    columns of Q_0 are an orthonormal basis of the symmetric sector, so u = Q_0 y converts coordinates y on M
    to coordinates u on M'. 
    """
    S, H_M = control_set_symmetries(H, B, theta0, tol)
    assert H_M.order() == H.order(), "Some elements of H do not map the control set to itself"
    m, Pi = sector_projectors(real_character_table(H), S)
    Q_0 = sector_basis(Pi[0])
    B_M = B @ Q_0
    return B_M,  Q_0

def orbit_control_subset(H):
    """Conjugate force matrix B along all distributions fixed by H: its columns are the indicators of (r-1) of the r 
    orbits of H. Last orbit excluded since it is not required due tonormalization. Corresponding theta0 = None. 
    """
    orbit_list = orbits(H)
    B = np.zeros((H.degree(), len(orbit_list)))
    for column, orbit in enumerate(orbit_list):
        B[orbit, column] = 1.0
    return B[:, :-1]

def same_column_space(X, Y, tol=1e-9):
    """True if X and Y span the same space once the mean of each column is removed (uniform shifts ignored)."""
    X, Y = X - X.mean(axis=0), Y - Y.mean(axis=0)
    rank = lambda M: np.linalg.matrix_rank(M, tol=tol)
    return rank(X) == rank(Y) == rank(np.column_stack([X, Y]))

def is_totally_geodesic(H, B_M, B, theta0=None, tol=1e-9):
    """Test by symmetry whether the control set M = (B_M, theta0) is totally geodesic in M' = (B, theta0). Input group
    H is usually Aut(G).

    Returns (certified, K). certified = False means only that symmetry does not settle the question; the second
        fundamental form does. K is the largest subgroup of H whose elements map M' to itself and fix every 
        distribution of M. 
    """
    assert same_column_space(np.column_stack([B_M, B]),  B, tol), "M is not contained in M'"
    S, H_M = control_set_symmetries(H, B, theta0, tol)                # Aut(G, M') within H
    centred_B_M = B_M - B_M.mean(axis=0)
    n = len(B)
    fixing = [g for g in H_M                                       
              if np.abs(perm_matrix(as_tuple(g, n)) @ centred_B_M - centred_B_M).max() <= tol]
    K = H_M.subgroup(fixing)
    B_fixed, Q_0 = symmetric_subset(K, B, theta0, tol)
    return same_column_space(B_fixed, B_M, tol), K

def coordinate_symmetry_constraints(H, B, theta0=None, names=None, tol=1e-9):
    """Returns the relations among the control coordinates of (B, theta0) that hold on its H-symmetric subset. 
    
    names : coordinate labels (default u0, u1, ...)

    As written, only applies to cases where the induced representation of H on the coordinates are signed permutations.
    symmetric_subset still works in the general case.

    Returns a list of strings describing the constraints.
    """
    S, H_M = control_set_symmetries(H, B, theta0, tol)
    assert H_M.order() == H.order(), "Some elements of H do not map the control set to itself"
    m = B.shape[1]
    names = [f"u{i}" for i in range(m)] if names is None else list(names)

    relations, seen = [], set()
    for i in range(m):
        if i in seen:
            continue
        sign_of, vanishes = {i: 1}, False                        # coordinates tied to u_i, as u_j = sign_of[j] * u_i
        for S_sigma in S.values():
            image = S_sigma[:, i]                                # S_sigma maps coordinate i to +/- coordinate j
            j = int(np.argmax(np.abs(image)))
            sign = int(np.sign(image[j]))
            is_signed_unit_vector = abs(abs(image[j]) - 1) <= tol and np.abs(np.delete(image, j)).max(initial=0) <= tol
            if not is_signed_unit_vector:
                raise ValueError("A symmetry mixes the coordinates instead of permuting them.")
            if sign_of.setdefault(j, sign) != sign:              # u_j = u_i and u_j = -u_i: both are zero
                vanishes = True
        seen.update(sign_of)
        terms = [("" if sign_of[j] > 0 else "-") + names[j] for j in sorted(sign_of)]
        if vanishes:
            relations.append(" = ".join(names[j] for j in sorted(sign_of)) + " = 0")
        elif len(terms) == 1:
            relations.append(terms[0] + " free")
        else:
            relations.append(" = ".join(terms))
    return relations

# ============ spin systems =============
# n_spins Ising spins, state s in 0, ..., 2^n_spins - 1 has spin j up (+1) if bit j of s is 1 and down (-1) if 0

def spin_values(n_spins):
    """Array of shape (2^n_spins, n_spins) whose entry [s, j] is the value, +1 or -1, of spin j in state s."""
    return np.array([[1.0 if (s >> j) & 1 else -1.0 for j in range(n_spins)] for s in range(2 ** n_spins)])

def spin_labels(n_spins):
    """State labels such as '↑↓↓↑', listing spins 0, 1, ... from left to right."""
    return tuple("".join("↑" if value > 0 else "↓" for value in state) for state in spin_values(n_spins))

def spin_system(n_spins, laws=None):
    """Markov graph of n_spins spins with single-spin-flip dynamics. laws is one rate law for all spins or a list
    of rate laws for each spin separately. Default law is Glauber. 
    """
    if laws is None:
        laws = glauber()
    if callable(laws):
        laws = [laws] * n_spins
    assert len(laws) == n_spins, "Need one rate law per spin"
    transitions = [(s, s ^ (1 << j), laws[j])                    # s ^ (1 << j) is state s with spin j flipped
                   for s in range(2 ** n_spins) for j in range(n_spins) if s < s ^ (1 << j)]
    return markov_graph(2 ** n_spins, transitions, labels=spin_labels(n_spins))

def spin_monomials(n_spins, subsets):
    """Conjugate force matrix B whose columns are products of spins: column i is the product of the spins in 
    subsets[i]. The coordinate conjugate to (j,) is the field h_j, to (j, k) the coupling J_jk, and so on.
    """
    values = spin_values(n_spins)
    return np.column_stack([np.prod(values[:, list(subset)], axis=1) for subset in subsets])

def field_basis(n_spins):
    """Conjugate force matrix B for control of the n_spins local fields h_j: column j is the value of spin j."""
    return spin_monomials(n_spins, [(j,) for j in range(n_spins)])

def coupling_basis(n_spins, pairs):
    """Conjugate force matrix B for control of the pair couplings J_jk, one column per pair (j, k) in pairs."""
    return spin_monomials(n_spins, pairs)

def full_control_basis(n_spins):
    """Conjugate force matrix B for full control: all 2^n_spins - 1 products of distinct spins, ordered as fields, 
    pair couplings, three-spin couplings, and so on. Its coordinates reach every distribution of the simplex."""
    subsets = [subset for size in range(1, n_spins + 1) for subset in combinations(range(n_spins), size)]
    return spin_monomials(n_spins, subsets)