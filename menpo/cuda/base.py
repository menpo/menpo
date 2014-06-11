def is_cuda_available():
    r"""
    Check if this system is capable of running CUDA Menpo code.

    Returns
    -------
    is_cuda_available : `bool`
        True iff the user has CUDA *and* a suitable driver installed.
    """
    
    # The cutools module is only available if the user compiled with nvcc.
    # Even if Menpo was compiled with nvcc, the user may not have the CUDA
    # driver/lib available on its machine. In that case,
    # 'from menpo.cuda.cutools import is_cuda_available' will fail due to
    # missing dynamic libraries
    try:
        from menpo.cuda import cutools
    except ImportError:
        return False
    
    # Check if the user has a CUDA-capable GPU which has at least the
    # minimum required 'Compute Capability'
    return cutools.is_cuda_available()

