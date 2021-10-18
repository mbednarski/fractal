import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import math


"""
Cuda tutorial 01

In this ex we are going to perform a few most fundamental operations.

1. Allocate a 1D array with size 4096 elements on the device (not in RAM, but in the GPU memory)
2. Set block size
3. Compute the required mesh size
4. Execute the kernel
5. Transfer results back to the host.

"""

# the most basic cuda kernel - let the thread identify itself and mark position as visited
@cuda.jit()
def kernel(data):
    # this is VERY common!
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    # write the data
    data[idx] = 1

def print_devices():
    for d in cuda.list_devices():
        print(d._device)

if __name__ == "__main__":
    print_devices()

    # create the data on CPU
    array = np.zeros(4096)

    # copy to the device
    d_array = cuda.to_device(array)
    print(d_array)

    # set number of threads per block
    threads_per_block = 32

    # compute correct number of blocks in order to handle all data
    blocks_per_grid = int(math.ceil(array.shape[0] / threads_per_block))

    print('Threads per block: ', threads_per_block)
    print('Blocks per grid: ', blocks_per_grid)

    # run the kernel!
    kernel[blocks_per_grid, threads_per_block](d_array)

    # copy results back
    result = d_array.copy_to_host()
    print(result)
