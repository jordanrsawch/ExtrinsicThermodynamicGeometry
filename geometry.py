"""geometry.py - Riemannian geometry of the thermodynamic manifold and of control sets.

A geometry is specified by metric(x), one of

    simplex_metric(G)                 pi -> L^+(pi)                   the simplex, coordinates pi
    control_set_metric(G, B, theta0)  u  -> J^T L^+ J                 a control set, coordinates u

All other functions are derived from the metric. Index conventions:

    Gamma_{lam mu nu}    = 1/2 (d_mu g_{nu lam} + d_nu g_{lam mu} - d_lam g_{mu nu})
    Gamma^kap_{mu nu}    = g^{kap lam} Gamma_{lam mu nu}
    R_{lam rho mu nu}    = d_mu Gamma_{lam nu rho} - d_nu Gamma_{lam mu rho}
                           + g^{kap sig} (Gamma_{kap nu lam} Gamma_{sig mu rho} - Gamma_{kap mu lam} Gamma_{sig nu rho})
    R(X, Y, Z, W)        = R_{lam rho mu nu} W^lam Z^rho X^mu Y^nu

g^{kap lam} is the the Laplacian L for the simplex metric L^+ is singular and the metric inverse otherwise.

All derivatives by forward-mode automatic differentiation in JAX.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd
from models import *
from functools import partial
from scipy.interpolate import CubicSpline

jax.config.update("jax_enable_x64", True)


# ============ metrics =============

def friction_metric(G, pi):
    """Friction metric L^+(pi) on the simplex, the Moore-Penrose pseudoinverse of the Laplacian."""
    L = laplacian(G, pi)
    uniform = jnp.ones(L.shape) / L.shape[0]
    return jnp.linalg.inv(L + uniform) - uniform

def simplex_metric(G):
    """pi -> L^+(pi): metric of the full-control thermodynamic manifold in the coordinates pi."""
    def metric(pi):
        return friction_metric(G, pi)
    return metric

def control_set_metric(G, B, theta0=None):
    """u -> J^T L^+ J, with J = d pi / d u: metric inherited by the control set (B, theta0)."""
    def metric(u):
        J = chart_jacobian(u, B, theta0)
        return J.T @ friction_metric(G, equilibrium(u, B, theta0)) @ J
    return metric

# ============ connection and curvature =============

def christoffel_lowered(metric, x):
    """Christoffel symbols with all indices down, Gamma_{lam mu nu}, indexed [lam, mu, nu]."""
    dg = jacfwd(metric)(x)                                       # dg[mu, nu, lam] = d_lam g_{mu nu}
    return 0.5 * (jnp.einsum('nlm->lmn', dg)                     #  d_mu  g_{nu lam}
                  + jnp.einsum('lmn->lmn', dg)                   #  d_nu  g_{lam mu}
                  - jnp.einsum('mnl->lmn', dg))                  #  d_lam g_{mu nu}

def christoffel(metric, x):
    """Christoffel symbols Gamma^kap_{mu nu} = g^{kap lam} Gamma_{lam mu nu}, indexed [kap, mu, nu]."""
    inverse_metric = jnp.linalg.pinv(metric(x), hermitian=True)
    return jnp.einsum('kl,lmn->kmn', inverse_metric, christoffel_lowered(metric, x))

def riemann(metric, x):
    """Riemann curvature tensor with all indices down, R_{lam rho mu nu}, indexed [lam, rho, mu, nu]."""
    inverse_metric = jnp.linalg.pinv(metric(x), hermitian=True)
    Gamma = christoffel_lowered(metric, x)                       # Gamma[lam, mu, nu]
    dGamma = jacfwd(christoffel_lowered, argnums=1)(metric, x)   # dGamma[lam, mu, nu, alpha] = d_alpha Gamma_{lam mu nu}
    return (jnp.einsum('lnrm->lrmn', dGamma)                                      #  d_mu Gamma_{lam nu rho}
            - jnp.einsum('lmrn->lrmn', dGamma)                                    #  d_nu Gamma_{lam mu rho}
            + jnp.einsum('ks,knl,smr->lrmn', inverse_metric, Gamma, Gamma)        #  g^{kap sig} Gamma_{kap nu lam} Gamma_{sig mu rho}
            - jnp.einsum('ks,kml,snr->lrmn', inverse_metric, Gamma, Gamma))       #  g^{kap sig} Gamma_{kap mu lam} Gamma_{sig nu rho}

def sectional_curvature(metric, x, X, Y):
    """Sectional curvature K = R(X, Y, Y, X) / (|X|^2 |Y|^2 - g(X, Y)^2) of the plane spanned by X and Y at x."""
    g = metric(x)
    R_XYYX = jnp.einsum('lrmn,l,r,m,n->', riemann(metric, x), X, Y, X, Y)         # W = X, Z = Y, X = X, Y = Y
    return R_XYYX / ((X @ g @ X) * (Y @ g @ Y) - (X @ g @ Y) ** 2)

def curvature_form(metric, x, velocity):
    """The matrix K with X^T K Y = R(X, v, v, Y) / |v|^2 for velocity v at x.
    """
    R_vv = jnp.einsum('lrmn,r,n->ml', riemann(metric, x), velocity, velocity)     # R_{lam rho mu nu} v^rho v^nu
    return R_vv / (velocity @ metric(x) @ velocity)

# ============ submanifolds =============

def second_fundamental_form(metric_M, metric_ambient, embedding, u):
    """Second fundamental form of a submanifold M at its coordinates u, indexed [a, i, j]:

        II^a_{ij} = d_i d_j v^a + Gamma'^a_{bc} d_i v^b d_j v^c - Gamma^k_{ij} d_k v^a,

    with v = embedding(u) the coordinates of the same point in the ambient manifold, Gamma' the ambient
    Christoffel symbols and Gamma those of M. II = 0 at every point if and only if M is totally geodesic.
    """
    v = embedding(u)
    dv = jacfwd(embedding)(u)                                    # dv[a, i]     = d_i v^a
    ddv = jacfwd(jacfwd(embedding))(u)                           # ddv[a, i, j] = d_i d_j v^a
    return (ddv
            + jnp.einsum('abc,bi,cj->aij', christoffel(metric_ambient, v), dv, dv)
            - jnp.einsum('kij,ak->aij', christoffel(metric_M, u), dv))

# ============ evaluation along a protocol =============

@partial(jax.jit, static_argnums=0)
def metric_at(metric, points):
    """metric(x) at every x in points, shape (number of points, dim, dim)."""
    return jax.vmap(metric)(points)

@partial(jax.jit, static_argnums=0)
def transport_generator_at(metric, points, velocities):
    """The matrix A = -Gamma^i_{jk} v^j, indexed [i, k], at every point: parallel transport is dn/dt = A n."""
    return jax.vmap(lambda x, v: -jnp.einsum('ijk,j->ik', christoffel(metric, x), v))(points, velocities)

@partial(jax.jit, static_argnums=0)
def curvature_form_at(metric, points, velocities):
    """curvature_form(metric, x, v) at every point."""
    return jax.vmap(lambda x, v: curvature_form(metric, x, v))(points, velocities)

def excess_work(metric, times, path):
    """Excess work of a protocol, the integral over time of the squared speed v^T g v (trapezoid rule)."""
    times, path = np.asarray(times, float), np.asarray(path, float)
    velocity = CubicSpline(times, path).derivative()(times)
    g = np.asarray(metric_at(metric, path))
    squared_speed = np.einsum('ai,aij,aj->a', velocity, g, velocity)
    return np.trapezoid(squared_speed, times)

def orthonormal_frame(Q, g):
    """Frame N = Q (Q^T g Q)^{-1/2}."""
    eigenvalues, eigenvectors = np.linalg.eigh(Q.T @ g @ Q)
    return Q @ eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T

def parallel_frames(metric, times, path, bases):
    """Parallel orthonormal frames along a protocol, one for each sector.

    bases is a list of matrices (e.g. from sector_basis). Each is g-orthonormalized and then parallel 
    transported by fourth-order Runge-Kutta. The frame is reorthonormalized after each step to remove the
    integration error.

    Returns list of arrays, one per sector, of shape (N+1, dim, sector dimension) giving the frame at each time.
    """
    times, path = np.asarray(times, float), np.asarray(path, float)
    spline = CubicSpline(times, path)
    velocity = spline.derivative()
    midpoints = (times[:-1] + times[1:]) / 2
    A_grid = np.asarray(transport_generator_at(metric, path, velocity(times)))
    A_mid = np.asarray(transport_generator_at(metric, spline(midpoints), velocity(midpoints)))
    g = np.asarray(metric_at(metric, path))

    frames = []
    for Q in bases:
        N = orthonormal_frame(Q, g[0])
        frame = [N]
        for a in range(len(times) - 1):
            dt = times[a + 1] - times[a]
            k1 = A_grid[a] @ N
            k2 = A_mid[a] @ (N + dt / 2 * k1)
            k3 = A_mid[a] @ (N + dt / 2 * k2)
            k4 = A_grid[a + 1] @ (N + dt * k3)
            N = N + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            N = orthonormal_frame(N, g[a + 1])
            frame.append(N)
        frames.append(np.stack(frame))
    return frames

def sector_curvature(metric, times, path, frames):
    """Sector curvature operators K_rho(t) along a protocol, in the given frames.

    Returns a list of arrays of shape (N+1, k, k), with k the dimension of the sector.
    """
    times, path = np.asarray(times, float), np.asarray(path, float)
    velocity = CubicSpline(times, path).derivative()(times)
    A = np.asarray(curvature_form_at(metric, path, velocity))
    return [np.einsum('aik,aij,ajl->akl', N, A, N) for N in frames]

def scaled_curvature(K, W):
    """Eigenvalues of K_rho(t) multiplied by W / pi^2, shape (N+1, k), in increasing order at each time.

    With W the excess work of the protocol, the thresholds of the thermodynamic comparison theorem sit at 1.
    """
    return W * np.linalg.eigvalsh(K) / np.pi ** 2

# ============ Jacobi fields and conjugate points =============

def reduce_sector_curvature(K, d, m):
    """Sector curvature K_rho(t), shape (N+1, d m, d m), reduced to the block to be integrated.

    Sectors with both d > 1 and m > 1 need a basis adapted to the irrep and are not treated here.
    """
    if d == 1:
        return K
    if m == 1:
        return (np.trace(K, axis1=1, axis2=2) / d)[:, None, None]
    raise NotImplementedError("Sectors with both d > 1 and m > 1 are not treated.")

def jacobi_determinant(times, K, W, n_steps=5000, orthonormalize_every=10):
    """Determinant of the matrix x(t) of Jacobi fields of one sector along a protocol. K, shape (N+1, k, k), 
    is the sector curvature at the given times (in a parallel frame), and W is the excess work of the protocol. 
    The equation is integrated as a first-order system in (x, y = dx/dt) by fourth-order Runge-Kutta with n_steps 
    equal steps.

    Every `orthonormalize_every` steps the columns of the stacked matrix (x; y) are replaced by orthonormal
    combinations of themselves, (x; y) = Q R -> Q. The new columns solve the same equation, and det x is divided
    by det R, which is recorded, so the determinant returned is that of the original x. This keeps x well
    conditioned when the Jacobi fields grow at very different rates.

    Returns (t, sign, log_abs), each of length n_steps + 1, with det x(t) = sign * exp(log_abs).
    """
    times, K = np.asarray(times, float), np.asarray(K, float)
    K_spline = CubicSpline(times, K)
    k = K.shape[1]

    def derivative(s, x, y):                                     # d/dt (x, y) = (y, -W K x)
        return y, -W * K_spline(s) @ x

    x, y = np.zeros((k, k)), np.eye(k)
    dt = (times[-1] - times[0]) / n_steps
    sign_R, log_R = 1.0, 0.0                                     # sign and log |.| of the product of all det R so far
    t, sign, log_abs = [times[0]], [0.0], [-np.inf]              # det x(0) = 0

    for step in range(n_steps):
        s = times[0] + step * dt
        k1x, k1y = derivative(s,          x,                 y)
        k2x, k2y = derivative(s + dt / 2, x + dt / 2 * k1x,  y + dt / 2 * k1y)
        k3x, k3y = derivative(s + dt / 2, x + dt / 2 * k2x,  y + dt / 2 * k2y)
        k4x, k4y = derivative(s + dt,     x + dt * k3x,      y + dt * k3y)
        x = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        y = y + dt / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)

        sign_x, log_x = np.linalg.slogdet(x)
        t.append(s + dt); sign.append(sign_R * sign_x); log_abs.append(log_R + log_x)

        if (step + 1) % orthonormalize_every == 0:
            Q, R = np.linalg.qr(np.vstack([x, y]))
            x, y = Q[:k], Q[k:]
            sign_R *= np.prod(np.sign(np.diag(R)))
            log_R += np.sum(np.log(np.abs(np.diag(R))))

    return np.array(t), np.array(sign), np.array(log_abs)

def conjugate_points(t, sign, log_abs):
    """Times at which det x changes sign, from the output of jacobi_determinant.

    Each crossing is placed by linear interpolation of det x between the two steps on either side of it.
    """
    points = []
    for i in range(2, len(t)):                                   # det x = 0 at t[0], so start from the second step
        if sign[i - 1] * sign[i] < 0:
            scale = max(log_abs[i - 1], log_abs[i])
            before = sign[i - 1] * np.exp(log_abs[i - 1] - scale)
            after = sign[i] * np.exp(log_abs[i] - scale)
            points.append(t[i - 1] + (t[i] - t[i - 1]) * before / (before - after))
    return points