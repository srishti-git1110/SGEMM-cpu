import numpy as np
import timeit

N = 4096
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
int3info = np.iinfo(np.int32)
numpy_max_int = int3info.max

for i in range(N):
    for j in range(N):
        A[i][j] = (float) (i + j) / (float) (numpy_max_int)
        B[i][j] = (float) (i * j) / (float) (numpy_max_int)

start = timeit.default_timer()
C = np.matmul(A, B)
end = timeit.default_timer()
print("time taken with NumPy: ", end - start) # 0.1042s