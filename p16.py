import time
import numpy as np
rng = np.random.default_rng(0)	# random number generator for test data


# return L, U with A = L @ U (no row swaps)
def lu(A):
    S = np.array(A, dtype=float)
    n = len(S)
    L = np.eye(n)
    U = np.zeros((n, n))
    for k in range(n):
        U[k, k:] = S[k, k:]
        L[k + 1:, k] = S[k + 1:, k] / S[k, k]
        S[k + 1:, k + 1:] = S[k + 1:, k + 1:] - np.outer(L[k + 1:, k], U[k, k + 1:])
    return L, U


# solve L U x = b by forward and back substitution
def solve_lu(L, U, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


# example 6.107
A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]])
L, U = lu(A)
print("L =")
print(L)
print("U =")
print(U)
print("solution of A x = (5, 13, 31):", solve_lu(L, U, np.array([5, 13, 31])))

lu_ok = True
solve_ok = True
for k in range(10):
    A = rng.normal(size=(6, 6))
    b = rng.normal(size=6)
    L, U = lu(A)
    if not np.allclose(L @ U, A):
        lu_ok = False
    if not np.allclose(solve_lu(L, U, b), np.linalg.solve(A, b)):
        solve_ok = False
print("L @ U == A:", lu_ok)
print("solve_lu agrees with np.linalg.solve:", solve_ok)

A = np.array([[1e-17, 1], [1, 1]])
b = np.array([1, 2])
L, U = lu(A)
print("solve_lu:", solve_lu(L, U, b))
print("np.linalg.solve:", np.linalg.solve(A, b))
A2 = np.array([[1, 1], [1e-17, 1]])
b2 = np.array([2, 1])
L, U = lu(A2)
print("with row swap:", solve_lu(L, U, b2))
# the solver returns (0, 1) instead of (1, 1). the pivot 1e-17 gives the multiplier
# 1e17, and in floating point 1 - 1e17 and 2 - 1e17 are both rounded to -1e17, so the
# information is lost. after swapping the rows the pivot is 1, no huge numbers appear,
# and the answer is correct


def inverse(A):
    n = len(A)
    L, U = lu(A)
    Ainv = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        Ainv[:, i] = solve_lu(L, U, e)
    return Ainv


X = rng.normal(size=(200, 50))
new_rows = rng.normal(size=(100, 50))

t0 = time.perf_counter()
Ainv = inverse(X.T @ X)
for x in new_rows:
    z = Ainv @ x
    Ainv = Ainv - np.outer(z, z) / (1 + x @ z)
t_sm = time.perf_counter() - t0

t0 = time.perf_counter()
Xc = X
for x in new_rows:
    Xc = np.vstack([Xc, x])
    Ainv_scratch = inverse(Xc.T @ Xc)
t_scratch = time.perf_counter() - t0

print("same result:", np.allclose(Ainv, Ainv_scratch))
print(f"sherman-morrison: {t_sm * 1000:.1f} ms")
print(f"from scratch each time: {t_scratch * 1000:.1f} ms")
# both approaches give the same inverse up to rounding errors. sherman-morrison is
# faster: each update costs about d^2 operations, while recomputing costs about
# N * d^2 + d^3
