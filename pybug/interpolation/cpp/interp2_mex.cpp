#include <mex.h>
#include <string.h>
#include "interp2.h"

/*
Build using:

    mex -largeArrayDims interp2_mex.cpp interp2.cpp -DIS_MEX
*/

void perform_interpolation(mxArray *plhs[], const mxArray *prhs[],
                           const std::string method)
{
    const NDARRAY F(mxGetNumberOfDimensions(prhs[0]),
                    mxGetDimensions(prhs[0]),
                    mxGetNumberOfElements(prhs[0]),
                    (double*)mxGetData(prhs[0]));
    const NDARRAY X(mxGetNumberOfDimensions(prhs[1]),
                    mxGetDimensions(prhs[1]),
                    mxGetNumberOfElements(prhs[1]),
                    (double*)mxGetData(prhs[1]));
    const NDARRAY Y(mxGetNumberOfDimensions(prhs[2]),
                    mxGetDimensions(prhs[2]),
                    mxGetNumberOfElements(prhs[2]),
                    (double*)mxGetData(prhs[2]));

    // This is a bit ugly - but it allows us to have a variable number of channels
    // Since we have to allocate the memory and pass it in to the function, the
    // function is sanity checked INSIDE the generic code
    mwSize outDims[3];
    mwSize n_outDims;
    for (n_outDims = 0; n_outDims < X.n_dims; n_outDims++)
    {
        outDims[n_outDims] = X.dims[n_outDims];
    }
    if (F.n_dims >= 3)
    {
        outDims[n_outDims++] = F.dims[2];
    }
    // Assumes that the output will have the same shape as the input
    // with an optional number of channels
    mxArray *out_data = mxCreateNumericArray(n_outDims, outDims,
                                             mxDOUBLE_CLASS, mxREAL);

    interpolate(&F, &Y, &X, method, (double*)mxGetData(out_data));

    plhs[0] = out_data;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if ((nrhs != 3) && (nrhs != 4))
    {
        mexErrMsgTxt("Wrong number of input arguments for Z = interp2_mex(F, X, Y, [method])");
    }

    if (nlhs > 1)
    {
        mexErrMsgTxt("Wrong number of output arguments for Z = interp2_mex(F, X, Y, [method])");
    }

    // Get interpolation type, if it exists, else pass an empty string!
    char *buff;
    if (nrhs == 4) {
        size_t buflen = (mxGetM(prhs[3]) * mxGetN(prhs[3])) + 1;
        buff = (char*)mxMalloc(buflen);
        mxGetString(prhs[3], buff, buflen);
    }
    else
    {
        // Set the default to be bilinear interpolation
        buff = (char*)mxMalloc(9);
        strncpy(buff, "bilinear", 9);
    }
    // Get const string to pass
    const std::string type(buff);

    if (mxIsDouble(prhs[0]) && mxIsDouble(prhs[1]) && mxIsDouble(prhs[2]))
    {
        perform_interpolation(plhs, prhs, type);
    }
    else
    {
        mexErrMsgTxt("interp2_mex only supports double arguments for F, X, Y");
    }

    // Free up dynamically allocated string
    mxFree(buff);
}
