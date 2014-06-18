#ifndef __CHECK_CUDA_ERRORS_HPP__
#define __CHECK_CUDA_ERRORS_HPP__

#define cudaErrorCheck(call) { cudaAssert(call,__FILE__,__LINE__); }

// Return on failure
#define cudaErrorCheck_void(call) { if (!cudaAssert(call,__FILE__,__LINE__)) {return;} }
#define throwRuntimeError_void(what) { throwRuntimeError(what); return; }

bool cudaAssert(const cudaError err, const char *file, const int line);
void throwRuntimeError(const char *what);

#endif
