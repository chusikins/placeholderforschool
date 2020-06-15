import numpy as np


SIZE = 3
A = np.array(np.random.uniform(0, 10, (SIZE, SIZE)))


def swap_columns(input_array, col1, col2):
    temp = np.copy(input_array[:][col1])
    input_array[:][col1] = input_array[:][col2]
    input_array[:][col2] = temp


def swap_rows(input_array, row1, row2):
    temp = np.copy(input_array[row1][:])
    input_array[row1][:] = input_array[row2][:]
    input_array[row2][:] = temp


def get_PU_LU(matrix):
    C = np.copy(matrix)
    P = np.identity(SIZE)
    n = 0

    for i in range(SIZE):
        pivot_value = 0
        pivot_row = -1
        temp = np.copy(C)
        for row in range(SIZE)[i:SIZE]:
            if np.fabs(C[row][i]) > pivot_value:
                pivot_value = np.fabs(C[row][i])
                pivot_row = row
        swap_rows(P, pivot_row, i)
        swap_rows(C, pivot_row, i)
        if not np.array_equal(temp, C):
            n += 1
        for j in range(SIZE)[i + 1:SIZE]:
            C[j][i] /= C[i][i]
            for s in range(SIZE)[i + 1:SIZE]:
                C[j][s] -= C[j][i] * C[i][s]
    L = np.tril(C, -1) + np.identity(SIZE)
    U = np.triu(C)

    return L, U, P, n


L, U, P, n = get_PU_LU(A)


print("Starting matrix", "\n",
      A, "\n", "\n",
      "Upper", "\n",
      U, "\n", "\n",
      "Lower", "\n",
      L, "\n" "\n",
      "Pivot", "\n",
      P, "\n", "\n")

print("PA = LU, so Pivot dot Starting matrix = to Upper dot Lower", "\n",
      "PA:", "\n",
      P.dot(A), "\n", "\n",
      "LU:", "\n", "\n",
      L.dot(U))

# det, detA = det(P^-1)*det(L)*det(U) = (-1)^n * det(U)
print("Lets check the left, the middle and the right", "\n",
      "Left",  "\n",
      np.linalg.det(A), "\n", "\n",
      "Middle", "\n",
      np.linalg.det(np.linalg.inv(P)) * np.linalg.det(L) * np.linalg.det(U), "\n", "\n",
      "Right", "\n",
      ((-1) ** n) * np.linalg.det(U))


# Ax = b
def eq_LUx_b(L, U, P, b):
    b = list(np.array(P).dot(np.array(b).transpose()))
    y = []
    for i in range(SIZE):
        k = b[i]
        for j in range(SIZE)[0:i]:
            k -= L[i][j] * y[j]
        y.append(k)
    x = []
    for i in range(SIZE):
        k = y[SIZE - i - 1]
        for j in range(SIZE)[0:i]:
            k -= U[SIZE - i - 1][SIZE - j - 1] * x[j]
        x.append(k / U[SIZE - i - 1][SIZE - i - 1])
    x.reverse()
    return x


print("A*x = L*U*x = b")
print("b")
b = list(np.random.uniform(0, 10, SIZE))
print(b)
print("x")
x = eq_LUx_b(L, U, P, b)
print(x)
print("check")
print(np.array(A).dot(np.transpose(np.array(x))))


# c) inverse matrix
def inverse(L, U, P):
    invA = []
    eye = np.eye(SIZE)
    for i in range(SIZE):
        invA.append(eq_LUx_b(L, U, P, list(eye)[i]))
    return np.transpose(invA)


print("Inversed A:")
invA = inverse(L, U, P)
print(np.array(invA))
print("invA*A")
print(np.array(invA).dot(np.array(A)))
print("A*invA")
print(np.array(A).dot(np.array(invA)))

# d) Число обусловленности A
print("mu")
print(np.linalg.norm(np.array(invA)) * np.linalg.norm(np.array(A)))


