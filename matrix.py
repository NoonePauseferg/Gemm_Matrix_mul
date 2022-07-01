import numpy as np
import time
import os

N = 1024
if __name__ == '__main__':
  A = np.random.randn(N, N).astype(np.float32);
  B = np.random.randn(N, N).astype(np.float32);

  s = time.monotonic()

  C = A @ B;

  e = time.monotonic()

  print(f"{N*N*2*N / (e - s) * 1e-9:.2f} GFLOPS")
  print(A[0])
  with open("data", "wb") as f:
    f.write(A.data)
    f.write(B.data)
    f.write(C.data)
