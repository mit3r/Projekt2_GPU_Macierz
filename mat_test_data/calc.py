import numpy as np

def read_matrice(filename):
    return np.fromfile(filename, dtype=np.int8).reshape(3200, 3200)

A = read_matrice("matA.bin")
B = read_matrice("matB.bin")

print("Matrix A:")
print(A)  
print(A.shape)
print("Matrix B:")
print(B)
print(B.shape)  


A = A.astype(np.int32)
B = B.astype(np.int32)

result = np.matmul(A, B)
result.tofile("result_A_B.bin")


# np.savetxt("result_A_B.txt", result, fmt="%d",encoding="binary")
# np.save("result_A_B.npy", result)
