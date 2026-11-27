import numpy as np
rng = np.random.default_rng(0)	# random number generator for test data

T = np.arange(24).reshape(2, 3, 4)
flat = T.reshape(-1)
same = True
for i in range(2):
    for j in range(3):
        for k in range(4):
            if T[i, j, k] != flat[12 * i + 4 * j + k]:
                same = False
print("T[i, j, k] == T.reshape(-1)[12i + 4j + k]:", same)

U1 = T.reshape(2, 12)
U2 = T.transpose(1, 0, 2).reshape(3, 8)
print(U1)
print("rank:", np.linalg.matrix_rank(U1))
print(U2)
print("rank:", np.linalg.matrix_rank(U2))
# both unfoldings have rank 2: in each of them, every row is a combination of two
# fixed vectors

# all operations of table 1
A = rng.normal(size=(3, 4))
B = rng.normal(size=(4, 5))
S = rng.normal(size=(4, 4))
x = rng.normal(size=4)
y = rng.normal(size=4)
print("matrix product:", np.allclose(np.einsum("ij,jk->ik", A, B), A @ B))
print("trace:", np.allclose(np.einsum("ii->", S), np.trace(S)))
print("dot product:", np.allclose(np.einsum("i,i->", x, y), x @ y))
print("outer product:", np.allclose(np.einsum("i,j->ij", x, y), np.outer(x, y)))
print("transpose:", np.allclose(np.einsum("ij->ji", A), A.T))
print("bilinear form:", np.allclose(np.einsum("i,ij,j->", x, S, y), x @ S @ y))

batch_A = rng.normal(size=(100, 4, 4))
batch_B = rng.normal(size=(100, 4, 4))
traces = np.einsum("bii->b", batch_A)
products = np.einsum("bij,bjk->bik", batch_A, batch_B)
traces_loop = np.zeros(100)
products_loop = np.zeros((100, 4, 4))
for b in range(100):
    traces_loop[b] = np.trace(batch_A[b])
    products_loop[b] = batch_A[b] @ batch_B[b]
print("batch of traces:", np.allclose(traces, traces_loop))
print("batch of products:", np.allclose(products, products_loop))

A = rng.normal(size=(3, 4))
B = rng.normal(size=(2, 5))
X = rng.normal(size=(4, 5))
print("kron:", np.allclose((A @ X @ B.T).reshape(-1), np.kron(A, B) @ X.reshape(-1)))
# for 100 x 100 matrices, np.kron(A, B) is a 10^4 x 10^4 matrix, so it stores 10^8
# numbers (about 800 megabytes). computing A @ X @ B.T directly needs only about
# 2 * 10^6 multiplications and never builds this matrix


def rank_one(u, v, w):
    return np.einsum("i,j,k->ijk", u, v, w)


u = rng.normal(size=3)
v = rng.normal(size=4)
w = rng.normal(size=5)
T = rank_one(u, v, w)
uv = np.outer(u, v)
for k in range(5):
    print("slice", k, "is", round(w[k], 4), "* u v^T:", np.allclose(T[:, :, k], w[k] * uv))
