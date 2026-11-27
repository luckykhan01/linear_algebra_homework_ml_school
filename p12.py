import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)	# random number generator for test data


def rref(A, tol=1e-10):
    R = np.array(A, dtype=float)
    m, n = R.shape
    pivots = []
    r = 0
    for c in range(n):
        if r == m:
            break
        best = r
        for i in range(r + 1, m):
            if abs(R[i, c]) > abs(R[best, c]):
                best = i
        if abs(R[best, c]) < tol:
            continue
        R[[r, best]] = R[[best, r]]
        R[r] = R[r] / R[r, c]
        for i in range(m):
            if i != r:
                R[i] = R[i] - R[i, c] * R[r]
        pivots.append(c)
        r = r + 1
    return R, pivots


def dual_basis(P):
    n = len(P)
    M = np.zeros((n, 2 * n))
    M[:, :n] = P
    M[:, n:] = np.eye(n)
    R, pivots = rref(M)
    return R[:, n:]


P = rng.normal(size=(5, 5))
rows = dual_basis(P)
print("r_i b_j is the identity:", np.allclose(rows @ P, np.eye(5)))

x = rng.normal(size=5)
total = np.zeros(5)
for i in range(5):
    total = total + (rows[i] @ x) * P[:, i]
print("x = sum (r_i x) b_i:", np.allclose(total, x))


def vandermonde(t):
    n = len(t)
    V = np.zeros((n, n))
    for k in range(n):
        for j in range(n):
            V[k, j] = t[k] ** j
    return V


# the columns of V^-1 are the lagrange polynomials: row k of V evaluates a polynomial
# at t_k, so V @ c = (p(t_0), ..., p(t_d)) for the coefficient vector c of p. the j-th
# column c_j of V^-1 satisfies V @ c_j = e_j, so its polynomial is 1 at t_j and 0 at
# all other nodes, which is exactly L_j
Vinv = dual_basis(vandermonde([-1, 0, 1]))
for j in range(3):
    print(f"L_{j} coefficients (1, t, t^2):", np.round(Vinv[:, j], 8))
# for the nodes -1, 0, 1: L_0 = (t^2 - t)/2, L_1 = 1 - t^2, L_2 = (t^2 + t)/2,
# the same as in example 8.9


def interpolate(f, nodes):
    L = dual_basis(vandermonde(nodes))
    c = np.zeros(len(nodes))
    for k in range(len(nodes)):
        c = c + f(nodes[k]) * L[:, k]
    return c


def poly_value(c, x):
    y = 0 * x
    for j in range(len(c)):
        y = y + c[j] * x ** j
    return y


def f1(t):
    return np.sin(3 * t)


def f2(t):
    return 1 / (1 + t ** 2)


plt.figure(figsize=(12, 4))
cases = [(f1, -1, 1, 9, "sin(3t), 9 nodes"), (f2, -5, 5, 21, "1/(1+t^2), 21 nodes")]
for i in range(2):
    f, a, b, n, name = cases[i]
    nodes = np.linspace(a, b, n)
    c = interpolate(f, nodes)
    xs = np.linspace(a, b, 1000)
    print(name, "max error:", np.max(np.abs(poly_value(c, xs) - f(xs))))
    plt.subplot(1, 2, i + 1)
    plt.plot(xs, f(xs), label="f")
    plt.plot(xs, poly_value(c, xs), "--", label="p")
    plt.plot(nodes, f(nodes), "o", label="nodes")
    plt.title(name)
    plt.legend()
plt.savefig("p12.png")
plt.show()

# for sin(3t) the interpolation is accurate. for 1/(1+t^2) the polynomial is
# accurate in the middle but oscillates strongly near the ends of [-5, 5] (error
# about 60). adding more equally spaced nodes makes it worse. this is runge's phenomen
