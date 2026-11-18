import time
import numpy as np

a = np.ones(256 * 1024 * 1024 // 4, dtype=np.float32)   # 256 MB of float32 ones, far bigger than any CPU cache
a.sum()                                                   # warm up: make sure every page is mapped
t0 = time.time()                                          # start the timer
for _ in range(10):                                       # read the whole array 10 times
    a.sum()                                               # summing forces every byte through the CPU
dt = time.time() - t0                                     # elapsed seconds
print(f"single-thread read bandwidth: {10 * a.nbytes / dt / 1e9:.1f} GB/s")   # bytes read divided by time