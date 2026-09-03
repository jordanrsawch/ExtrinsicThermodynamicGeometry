"""geometry.py — friction geometry of the simplex and of control sets.

The friction metric on the open simplex is g = L^+. Geometric invariants are insensitive to extension of L^+ off the simplex,
so no charts or constraints are required, and L itself serves as the inverse metric (L L^+ = I on 1^perp), and curvature 
tensors are projected onto 1^perp in every slot.

A control set M' = {pi(u)} inherits the metric h(u) = J^T g J with J = dpi/du, nondegenerate.

Index conventions (all derivatives by jax forward-mode):

    Gamma_{lmn} = 1/2 (d_m g_{nl} + d_n g_{lm} - d_l g_{mn})
    Gamma^r_{mn} = g^{rl} Gamma_{lmn}
    R^r_{smn}   = d_m Gamma^r_{ns} - d_n Gamma^r_{ms} + Gamma^r_{ml} Gamma^l_{ns} - Gamma^r_{nl} Gamma^l_{ms}
    R_{wsmn}    = g_{wr} R^r_{smn},   R(e_a, e_b, e_c, e_d) = R_{abcd}

"""

import numpy as np
import jax, jax.numpy as jnp
from jax import jacfwd
jax.config.update("jax_enable_x64", True)


class Geometry:
    """Levi-Civita geometry from a metric and its inverse.

    Subclasses supply _metric(x) and _inv_metric(x), and _project(R), which the degenerate simplex case 
    uses to drop the spurious content along 1. Public methods are jitted versions of the private ones.
    """

    def __init__(self):
        self.metric      = jax.jit(self._metric)
        self.inv_metric  = jax.jit(self._inv_metric)
        self.christoffel = jax.jit(self._christoffel_up)
        self.riemann     = jax.jit(self._riemann_low)
        self.tidal       = jax.jit(self._tidal)

    # ---- supplied by subclasses -------------------------------------------
    def _metric(self, x):
        raise NotImplementedError

    def _inv_metric(self, x):
        raise NotImplementedError

    def _project(self, R):
        return R

    # ---- Levi-Civita ---------------------------------------------------------
    def _dg(self, x):
        """d_l g_{mn}, indexed [l, m, n]."""
        return jnp.moveaxis(jacfwd(self._metric)(x), -1, 0)

    def _christoffel_low(self, x):
        dg = self._dg(x)
        return 0.5 * (jnp.einsum('mnl->lmn', dg) + jnp.einsum('nlm->lmn', dg) - dg)

    def _christoffel_up(self, x):
        return jnp.einsum('rl,lmn->rmn', self._inv_metric(x), self._christoffel_low(x))

    def _riemann_low(self, x):
        """R_{wsmn} with the first index lowered by the metric, then projected."""
        Gu  = self._christoffel_up(x)
        dGu = jnp.moveaxis(jacfwd(self._christoffel_up)(x), -1, 0)     # [d, r, m, n]
        R = (jnp.einsum('mrns->rsmn', dGu) - jnp.einsum('nrms->rsmn', dGu)
             + jnp.einsum('rml,lns->rsmn', Gu, Gu)
             - jnp.einsum('rnl,lms->rsmn', Gu, Gu))
        return self._project(jnp.einsum('rw,rsmn->wsmn', self._metric(x), R))

    def _tidal(self, x, v):
        """A_{ad} = R(e_a, v, e_d, v) / ||v||^2_g: the normalized tidal form.
        Its diagonal entries in a g-orthonormal frame are sectional curvatures
        of the planes spanned with v; depends on v only through its direction."""
        Rd = self._riemann_low(x)
        return jnp.einsum('abcd,b,d->ac', Rd, v, v) / (v @ self._metric(x) @ v)

    def sectional(self, x, u, v):
        """K(span{u, v}) = R(u, v, u, v) / (|u|^2 |v|^2 - <u, v>^2)."""
        Rd, g = self._riemann_low(x), self._metric(x)
        num = jnp.einsum('abcd,a,b,c,d->', Rd, u, v, u, v)
        return num / ((u @ g @ u) * (v @ g @ v) - (u @ g @ v) ** 2)


class SimplexGeometry(Geometry):
    """The open simplex with g = L^+, in the coordinates pi themselves."""

    def __init__(self, model):
        self.model, self.n = model, model.n
        self.Jn = jnp.ones((self.n, self.n)) / self.n
        super().__init__()

    def _metric(self, pi):
        return jnp.linalg.inv(self.model.L(pi) + self.Jn) - self.Jn     # L^+

    def _inv_metric(self, pi):
        return self.model.L(pi)                                        # inverts L^+ on 1^perp

    def _project(self, R):
        for ax in range(4):                                            # P_{1^perp} on each slot
            R = R - R.mean(axis=ax, keepdims=True)
        return R


class ControlGeometry(Geometry):
    """A control set M' with the inherited metric G(u) = J^T L^+ J, J = dpi/du.

    The columns of J lie in 1^perp, so J^T L^+ J = J^T (L + Jn)^{-1} J exactly,
    which avoids forming L^+. G is nondegenerate (B gauge-fixed), so no
    projection is needed.
    """

    def __init__(self, model, control):
        self.model, self.control = model, control
        self.n = control.dim                                           # manifold dimension
        self.Jn = jnp.ones((model.n, model.n)) / model.n
        self.pi  = jax.jit(control.pi)
        self.jac = jax.jit(control.jacobian)
        super().__init__()

    def _metric(self, u):
        p, Jm = self.control.pi(u), self.control.jacobian(u)
        return Jm.T @ jnp.linalg.solve(self.model.L(p) + self.Jn, Jm)

    def _inv_metric(self, u):
        return jnp.linalg.inv(self._metric(u))


# ------------------------------------------------------------- frames

def g_orthonormal(U, g):
    """g-orthonormal basis of span(U) from a Euclidean-orthonormal sector basis U (n, d).

    M = U^T g U restricts the metric to the sector. We require transformation C such that B = UC is a basis
    of span{U} such that B^T g B = C^T M C = I. Symmetric choice is C = M^{-1/2}, computed here via eigh. 
    """
    M = U.T @ g @ U
    w, V = jnp.linalg.eigh(M)
    C = (V @ jnp.diag(w ** -0.5) @ V.T)
    return U @ C


def K_blocks(geo, x, v, S):
    """The sector blocks K_rho of the tidal form at a point.

    geo is the ambient geometry, x is a point, v is a tangent vector, and S is a sector decomposition of the 
    ambient tangent space under the appropriate symmetry group.

    Returns {rho: (K_rho, eigenvalues)} for every accessible symmetry-breaking sector of S, with K_rho the 
    tidal form A(x, v) restricted to a g-orthonormal frame of U_rho. K_rho is symmetric of size dim U_rho, 
    and its spectrum comes in d_rho-fold degenerate groups.
    """
    g = geo.metric(x)
    A = geo.tidal(x, v)
    out = {}
    for r in S.accessible:
        B = g_orthonormal(jnp.asarray(S.U[r]), g) 
        K = B.T @ A @ B
        K = 0.5 * (K + K.T)
        out[r] = (K, jnp.linalg.eigvalsh(K))
    return out


def path_velocity(path, t):
    """d x/dt along a discretised path by second-order finite differences on a non-uniform grid (three-point interior 
    formula, one-sided at the ends). Only the direction feeds the tidal form, which is homogeneous in v."""
    p, t = np.asarray(path, float), np.asarray(t, float)
    assert p.shape[0] == t.shape[0] and p.shape[0] >= 3
    v = np.empty_like(p)
    hm, hp = (t[1:-1] - t[:-2])[:, None], (t[2:] - t[1:-1])[:, None]
    v[1:-1] = (hm**2 * p[2:] + (hp**2 - hm**2) * p[1:-1] - hp**2 * p[:-2]) / (hm * hp * (hm + hp))
    h1, h2 = t[1] - t[0], t[2] - t[1]
    v[0] = (-(2*h1 + h2)/(h1*(h1 + h2)) * p[0] + (h1 + h2)/(h1*h2) * p[1] - h1/(h2*(h1 + h2)) * p[2])
    g1, g2 = t[-1] - t[-2], t[-2] - t[-3]
    v[-1] = ((2*g1 + g2)/(g1*(g1 + g2)) * p[-1] - (g1 + g2)/(g1*g2) * p[-2] + g1/(g2*(g1 + g2)) * p[-3])
    return v


def parallel_transport(geo, path, t, frame0):
    """Transport a frame (n, k) along a path: D_t X = 0, i.e. Xdot = -Gamma(x)[:, m, :] xdot^m X, integrated by RK4 
    with the path linearly interpolated. Preserves g-inner products and each sector U_rho when the path lies in M_H.

    D_t X = 0 written as a matrix ODE by defining A^r_n(t) = -\Gamma^r_mn(x(t))\dot{x}^m(t), giving \dot{X} - A(t)X = 0.
    """
    p, t = np.asarray(path), np.asarray(t)
    v = path_velocity(p, t)

    def A_at(x, vel):
        return -jnp.einsum('rmn,m->rn', geo.christoffel(x), vel)

    # RK4 integration
    X, out = jnp.asarray(frame0), [jnp.asarray(frame0)]
    for k in range(len(t) - 1):
        h = t[k + 1] - t[k]
        pk, pk1, vk, vk1 = map(jnp.asarray, (p[k], p[k + 1], v[k], v[k + 1]))
        pm, vm = 0.5 * (pk + pk1), 0.5 * (vk + vk1)
        A0, Am, A1 = A_at(pk, vk), A_at(pm, vm), A_at(pk1, vk1)
        k1 = A0 @ X
        k2 = Am @ (X + 0.5 * h * k1)
        k3 = Am @ (X + 0.5 * h * k2)
        k4 = A1 @ (X + h * k3)
        X = X + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(X)
    return jnp.stack(out)


# ------------------------------------------------------------- submanifolds

def second_fundamental_form(sub, amb, embed, u):
    """II^a_{ij} of a submanifold in an ambient manifold, at coordinates u of the submanifold.

    embed : callable, submanifold coordinates -> ambient coordinates (jax-traceable).
    Gauss formula in coordinates, with J = d embed / du:
            II^a_ij = d_i d_j embed^a + Gamma^a_{bc} J^b_i J^c_j - Gamma'^k_{ij} J^a_k,
    Gamma the ambient Christoffels, Gamma' the submanifold's Christoffels. For the simplex as ambient, 
    embed = control.pi and the ambient Christoffels are extension-independent because the columns of J 
    lie in 1^perp. II == 0 everywhere iff the submanifold is totally geodesic.
    """
    x = embed(u)
    J = jacfwd(embed)(u)                                     # (dim amb, dim sub)
    H2 = jacfwd(jacfwd(embed))(u)                            # (dim amb, dim sub, dim sub)
    return (H2 + jnp.einsum('abc,bi,cj->aij', amb.christoffel(x), J, J)
               - jnp.einsum('kij,ak->aij', sub.christoffel(u), J))