#include "cutools.hpp"

#define CUDA_MAJOR 2
#define CUDA_MINOR 0

int __DEVICE_COUNT__(-1); // -1 means to be done
bool is_cuda_available_()
{
    // The number of compatible devices is counted only one time
    // and stored into __DEVICE_COUNT__
    if (__DEVICE_COUNT__ == -1)
    {
        __DEVICE_COUNT__ = 0;
        
        int deviceCount;
        cudaError_t e = cudaGetDeviceCount(&deviceCount);
        if (e == cudaSuccess)
        {
            // for each GPU check if it has the required compute capability
            for (int i(0) ; i != deviceCount ; i++)
            {
                cudaDeviceProp prop;
                e = cudaGetDeviceProperties(&prop, i);
                if (e == cudaSuccess && (prop.major > CUDA_MAJOR || (prop.major == CUDA_MAJOR && prop.minor >= CUDA_MINOR)))
                {
                    cudaSetDevice(i);
                    __DEVICE_COUNT__++;
                }
            }
        }
    }
    return __DEVICE_COUNT__ != 0;
}

