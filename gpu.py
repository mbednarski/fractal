import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import math

@cuda.jit()
def kernel(data):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    data[idx] = 1

# grid: 1 blok o rozmiarze 256
arr = np.ones((4096)) * -1
d_arr = cuda.to_device(arr)

threads_per_block = 1024
blocks_per_grid = int(math.ceil(arr.shape[0] / threads_per_block))

kernel[blocks_per_grid, threads_per_block](d_arr)

print(d_arr.copy_to_host())
