#ifndef __CHECK_CUDA_ERRORS_HPP__
#define __CHECK_CUDA_ERRORS_HPP__

// raise an excpetion Python-side, does not leave C/C++ code
#define cudaErrorCheck(call) { cudaAssert(call,__FILE__,__LINE__); }

// goto onfailure, on failure
#define cudaErrorCheck_goto(call) { if (!cudaAssert(call,__FILE__,__LINE__)) {goto onfailure;} }

bool cudaAssert(const cudaError err, const char *file, const int line);
void throwRuntimeError(const char *what);

#endif
