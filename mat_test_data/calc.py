import numpy as np

def read_matrice(filename):
    return np.loadtxt(filename, dtype=int)

A = read_matrice("matA.txt")
B = read_matrice("matB.txt")
result = np.matmul(A, B)

# np.savetxt("result_A_B.txt", result, fmt="%d")
np.save("result_A_B.npy", result)
