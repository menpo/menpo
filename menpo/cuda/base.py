def is_cuda_available():
    """
    Check whether or not the system can launch a CUDA version of the code
    
    It checks:
    1. Did the user compile menpo using nvcc?
       In that case, 'from menpo.cuda.cutools import is_cuda_available'
       should work
    2. Does the user have setup CUDA on its machine?
       In that case, 'from menpo.cuda.cutools import is_cuda_available'
       should work
       Otherwise, an exception will be raised by the module:
        missing library libcudart.so
    3. Does the user have a CUDA-capable GPU? Does its GPU has the required
       'Compute Capability'?
    """
    
    try: #1 #2
        from menpo.cuda import cutools
    except ImportError:
        return False
    
    return cutools.is_cuda_available() #3

