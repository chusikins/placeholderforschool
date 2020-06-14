import numpy as np

SIZE = int(input("Input matrix size : "))


# Генератор матрицы с диагональным преобладанием
def gen_matrix_diag_pred(size):
    a1 = np.random.uniform(0, 10, (size, size)) * (np.ones((size, size)) - np.eye(size))
    a2 = np.random.uniform(10 * size, 10 * size + 10, (size, size)) * np.eye(size)
    return a1 + a2


# Генератор положительно определенной матрицы
def gen_matrix_positive_def(size):
    a = np.random.uniform(0, 10, (size, size))
    matrix = np.transpose(a).dot(a)
    return matrix


# Метод Якоби
# Алгоритм в матричной форме
def jacobi(matrix, b):
    EPS = 0.0001
    n = 0
    D = matrix * np.eye(SIZE)
    # invD = np.linalg.inv(D)
    invD = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        if D[i, i] != 0:
            invD[i, i] = 1 / D[i, i]
    B = np.eye(SIZE) - invD.dot(matrix)
    g = invD.dot(b)
    x = np.zeros(SIZE)
    # Если |B|>1 то оценка не работает
    print("Норма матрицы B:")
    print(np.linalg.norm(B))
    q = 1
    if np.linalg.norm(B) < 1:
        k_aprior = 1 + int((np.log(EPS) + np.log(1 - np.linalg.norm(B)) - np.log(np.linalg.norm(g))) / np.log(np.linalg.norm(B)))
        print("Априорная оценка колиества итераций метода Якоби")
        print(k_aprior)
        q = np.linalg.norm(B) / (1 - np.linalg.norm(B))
    else:
        print("Норма больше 1.")
    norm = 1000
    while q * norm > EPS:
        n += 1
        x_new = B.dot(x) + g
        norm = np.linalg.norm(x - x_new)
        if norm > 100000:
            print("Система расходится(Метод Якоби)")
            return None, None
        x = x_new
    return x, n


# Метод Зейделя
# Алгоритм в матричной форме
def seidel(matrix, b):
    EPS = 0.0001
    n = 0
    LD = np.tril(matrix)
    invLD = np.linalg.inv(LD)
    U = np.triu(np.array(matrix), 1)
    B = -invLD.dot(U)
    g = invLD.dot(b)
    x = np.zeros(SIZE)
    # Если |B|>1 то оценка не работает
    print("Норма матрицы B:")
    print(np.linalg.norm(B))
    q = 1
    if np.linalg.norm(B) < 1:
        k_aprior = 1 + int((np.log(EPS) + np.log(1 - np.linalg.norm(B)) - np.log(np.linalg.norm(g))) / np.log(np.linalg.norm(B)))
        print("Априорная оценка колиества итераций метода Зейделя")
        print(k_aprior)
        q = np.linalg.norm(B) / (1 - np.linalg.norm(B))
    else:
        print("Норма больше 1.")
    norm = 1000
    while q * norm > EPS:
        n += 1
        x_new = B.dot(x) + g
        norm = np.linalg.norm(x - x_new)
        if norm > 100000:
            print("Система расходится(Метод Зейделя)")
            return None, None
        x = x_new
    return x, n


print("Случай с матрицей с диагональным преобладанием:")
print("A*x = b")
print("A")
A = gen_matrix_diag_pred(SIZE)
print(A)
b = np.random.uniform(0, 10, SIZE)
print("b")
print(b)
print("Метод Якоби")
x, n = jacobi(A, b)
if x is not None:
    print("iter Jacobi")
    print(n)
    print("x by Jacobi")
    print(x)
print("Метод Зейделя")
x, n = seidel(A, b)
print("iter Seidel")
print(n)
print("x by Seidel")
print(x)
print("x = invA*b")
print(np.linalg.inv(A).dot(b))

print()
print("Случай с положительно определенной матрицей:")
print("A*x = b")
print("A")
A = gen_matrix_positive_def(SIZE)
print(A)
b = np.random.uniform(0, 10, SIZE)
print("b")
print(b)
print("Метод Якоби")
x, n = jacobi(A, b)
if x is not None:
    print("iter Jacobi")
    print(n)
    print("x by Jacobi")
    print(x)
print("Метод Зейделя")
x, n = seidel(A, b)
print("iter Seidel")
print(n)
print("x by Seidel")
print(x)
print("x = invA*b")
print(np.linalg.inv(A).dot(b))
