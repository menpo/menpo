#ifndef __CHECK_CUDA_ERRORS_HPP__
#define __CHECK_CUDA_ERRORS_HPP__

// define printf for kernels
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
    # error printf is only supported on devices of compute capability 2.0 and higher, please compile with -arch=sm_20 or higher    
#endif

#ifdef _TIME
    #define __CLOG__ \
            cudaEvent_t start, stop;\
            float _time_;
    #define __START__ \
            cudaEventCreate(&start); \
            cudaEventCreate(&stop); \
            cudaEventRecord(start, 0);
    #define __STOP__ \
            cudaEventRecord(stop, 0); \
            cudaEventSynchronize(stop); \
            cudaEventElapsedTime(&_time_, start, stop); \
            cudaEventDestroy(start); \
            cudaEventDestroy(stop); \
            printf("In %s at line %d\t%s\tTOOK: %fms\n", \
                   __FILE__, __LINE__, __PRETTY_FUNCTION__, _time_);
#else
    #define __CLOG__
    #define __START__
    #define __STOP__
#endif

// raise an excpetion Python-side, does not leave C/C++ code
#define cudaErrorCheck(call) { cudaAssert(call,__FILE__,__LINE__); }

// goto onfailure, on failure
#define cudaErrorCheck_goto(call) { if (!cudaAssert(call,__FILE__,__LINE__)) {goto onfailure;} }

bool cudaAssert(const cudaError err, const char *file, const int line);
void throwRuntimeError(const char *what);

#endif
