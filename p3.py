import numpy as np
rng = np.random.default_rng(0)	# random number generator for test data


# return (R, pivots): the rref of A and the list of pivot columns
def rref(A, tol=1e-10):
    R = np.array(A, dtype=float)	# work on a copy of A
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


def rank(A):
    R, pivots = rref(A)
    return len(pivots)


def null_space_basis(A):
    R, pivots = rref(A)
    n = R.shape[1]
    basis = []
    for j in range(n):
        if j not in pivots:
            v = np.zeros(n)
            v[j] = 1
            for i in range(len(pivots)):
                v[pivots[i]] = -R[i, j]
            basis.append(v)
    return basis


def col_space_basis(A):
    R, pivots = rref(A)
    A = np.array(A, dtype=float)
    basis = []
    for j in pivots:
        basis.append(A[:, j])
    return basis


A = np.array([[1, 2, -1], [3, -1, 4]])
R, pivots = rref(A)
print(np.round(R, 8) + 0)
print("pivots:", pivots)

# example 6.31
F = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
R, pivots = rref(F)
print(np.round(R, 8) + 0)
print("pivots:", pivots)
print("null space of F:", null_space_basis(F))
print("column space of F:", col_space_basis(F))

all_ok = True
for k in range(20):
    A = rng.normal(size=(5, 7))
    N = null_space_basis(A)
    for v in N:
        if not np.allclose(A @ v, 0):
            all_ok = False
    if rank(A) + len(N) != 7:
        all_ok = False
    if rank(A) != np.linalg.matrix_rank(A):
        all_ok = False
print("20 random 5x7 matrices:", all_ok)

A = rng.normal(size=(6, 4)) @ rng.normal(size=(4, 8))
print("rank:", rank(A))
print("dim null space:", len(null_space_basis(A)))
