#include "HOG.h"
#include <mex.h>

using namespace std;

// MAIN FUNCTION
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    struct wins info;
    const float pi = 3.1415926536;
    double *inputImage, *options, *windowImage, *descriptorVector, *descriptorMatrix, *WindowsMatrixDescriptorsMatrix, *WindowsCentersMatrix;
    mxArray *mxWindowsMatrixDescriptorsMatrix, *mxWindowsCentersMatrix, *windowImageTemp, *descriptorVectorTemp, *descriptorMatrixTemp;
    int hist1, hist2, imageSize[2], imageWidth,imageHeight,out[3];
    unsigned int inputImageIsGrayscale;
    double binsSize;
    mwSize dims3[3],dims5[5];
    
    // Load input image, find its size and return error messages if necessary
    if (nrhs==0)
        mexErrMsgTxt("ERROR:ComputeHOGdescriptor:No input image given.");
    if (mxGetClassID(prhs[0])!=mxDOUBLE_CLASS)
        mexErrMsgTxt("ERROR:ComputeHOGdescriptor:Input image must be of type double.");
    inputImage = mxGetPr(prhs[0]);
    imageSize[0] = mxGetM(prhs[0]);
    imageSize[1] = mxGetN(prhs[0]);
    inputImageIsGrayscale = 1;
    if (mxGetNumberOfDimensions(prhs[0])==3)
    {
        imageSize[1] /= 3;
        inputImageIsGrayscale = 0;
    }
    imageWidth = imageSize[1];
    imageHeight = imageSize[0];
    
    // Load options and return error messages if necessary
    if (nlhs!=1)
        mexErrMsgTxt("ERROR:ComputeHOGdescriptor:No output argument.");
    if (nrhs>1)
    {
        options = mxGetPr(prhs[1]);
        if (options[0]!=1 && options[0]!=2)
            mexErrMsgTxt("ERROR:ComputeHOGdescriptor:Set SparseOrDense=1 for sparse and 2 for dense.");
        if (options[0]==2)
        {
            if (options[1]<0)
                mexErrMsgTxt("ERROR:ComputeHOGdescriptor:Window height must be positive.");
            if (options[2]<0)
                mexErrMsgTxt("ERROR:ComputeHOGdescriptor:Window width must be positive.");
            if (options[3]!=1 && options[3]!=2)
                mexErrMsgTxt("ERROR:ComputeHOGdescriptor:Set WindowSizeMetricUnit=1 for blocks and 2 for pixels.");
            if (options[4]<=0)
                mexErrMsgTxt("ERROR:ComputeHOGdescriptor:Horizontal window step must be positive.");
            if (options[5]<=0)
                mexErrMsgTxt("ERROR:ComputeHOGdescriptor:Vertical window step must be positive.");
            if (options[6]!=1 && options[6]!=2)
                mexErrMsgTxt("ERROR:ComputeHOGdescriptor:Set WindowStepMetricUnit=1 for cells and 2 for pixels.");
            if (options[7]!=0 && options[7]!=1)
                mexErrMsgTxt("ERROR:ComputeHOGdescriptor:Set ReturnOnlyWindowsWithinImageLimits={0,1} depending if you want to pad edge windows or not.");
        }
        if (options[8]!=1 && options[8]!=2)
            mexErrMsgTxt("ERROR:ComputeHOGdescriptor:Set Method=1 for DalalTriggs and 2 for ZhuRamanan.");
        if (options[9]<=0)
            mexErrMsgTxt("ERROR:ComputeHOGdescriptor:Number of orientation bins must be positive.");
        if (options[10]<=0)
            mexErrMsgTxt("ERROR:ComputeHOGdescriptor:Cell size (in pixels) must be positive.");
        if (options[11]<=0)
            mexErrMsgTxt("ERROR:ComputeHOGdescriptor:Block size (in cells) must be positive.");
        if (options[12]!=0 && options[12]!=1)
            mexErrMsgTxt("ERROR:ComputeHOGdescriptor:Set SignedOrUnsignedGradients={0,1} depending if you want signed or unsigned gradients.");
        if (options[13]<=0)
            mexErrMsgTxt("ERROR:ComputeHOGdescriptor:Value for L2-norm clipping must be positive.");
        if (options[14]!=0 && options[14]!=1)
            mexErrMsgTxt("ERROR:ComputeHOGdescriptor:Set PrintInformation=1 to enable it and 0 to disable.");
    }
    else
    {
        options = new double[18];
        options[0] = 1; //sparseOrDense
        options[1] = 0; //windowHeight
        options[2] = 0; //windowWidth
        options[3] = 0; //windowSizeMetricUnit
        options[4] = 0; //windowStepHorizontal
        options[5] = 0; //windowStepVertical
        options[6] = 0; //windowStepMetricUnit
        options[7] = 0; //ReturnOnlyWindowsWithinImageLimits
        options[8] = 1; //Method
        options[9] = 9; //NumberOfOrientationBins
        options[10] = 8; //CellHeightAndWidthInPixels
        options[11] = 2; //BlockHeightAndWidthInCells
        options[12] = 0; //SignedOrUnsignedGradients
        options[13] = 0.2; //L2normClipping
        options[14] = 0; //PrintInformation
    }
    
    // Compute windows information
    info = WindowsInformation(options, imageHeight, imageWidth, inputImageIsGrayscale);

    // Initialize Window Image
    if (inputImageIsGrayscale==1)
        windowImageTemp = mxCreateDoubleMatrix(info.windowHeight,info.windowWidth,mxREAL);
    else
    {
        mwSize dims[3];
        dims[0] = info.windowHeight;
        dims[1] = info.windowWidth;
        dims[2] = 3;
        windowImageTemp = mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL);
    }
    windowImage = mxGetPr(windowImageTemp);
    
    
    // Initialize descriptor vector/matrix
    binsSize = (1+(options[12]==1))*pi/options[9];
    if (options[8]==1)
    {
        info.descriptorLengthPerBlock = options[11]*options[11]*options[9];
        hist1 = 2 + ceil(-0.5 + info.windowHeight/options[10]);
        hist2 = 2 + ceil(-0.5 + info.windowWidth/options[10]);
        info.descriptorLengthPerWindow = (hist1-2-(options[11]-1))*(hist2-2-(options[11]-1))*info.descriptorLengthPerBlock;
        descriptorVectorTemp = mxCreateDoubleMatrix(info.descriptorLengthPerWindow,1,mxREAL);
        descriptorVector = mxGetPr(descriptorVectorTemp);
        // both ways of calculating number of blocks are equal
        //numberOfBlocksPerWindowVertically = 1+floor((info.windowHeight-options[11]*options[10])/options[10]);
        //numberOfBlocksPerWindowHorizontally = 1+floor((info.windowWidth-options[11]*options[10])/options[10]);
        info.numberOfBlocksPerWindowVertically = hist1-2-(options[11]-1);
        info.numberOfBlocksPerWindowHorizontally = hist2-2-(options[11]-1);
        dims3[0] = info.numberOfBlocksPerWindowVertically;
        dims3[1] = info.numberOfBlocksPerWindowHorizontally;
        dims3[2] = info.descriptorLengthPerBlock;
        descriptorMatrixTemp = mxCreateNumericArray(3,dims3,mxDOUBLE_CLASS,mxREAL);
        descriptorMatrix = mxGetPr(descriptorMatrixTemp);
    }
    else if (options[8]==2)
    {
        hist1 = (int)round((double)info.windowHeight/(double)options[10]);
        hist2 = (int)round((double)info.windowWidth/(double)options[10]);
        out[0] = max(hist1-2,0); //You can change this to out[0] = max(hist1-1,0); and out[1] = max(hist2-1,0), in order to return the same output size as dalaltriggs
        out[1] = max(hist2-2,0); //You can do the same in lines 1361,1362
        out[2] = 27+4;
        descriptorMatrixTemp = mxCreateNumericArray(3,out,mxDOUBLE_CLASS,mxREAL);
        descriptorMatrix = mxGetPr(descriptorMatrixTemp);
        info.numberOfBlocksPerWindowHorizontally = out[1];
        info.numberOfBlocksPerWindowVertically = out[0];
        info.descriptorLengthPerBlock = out[2];
        info.descriptorLengthPerWindow = info.numberOfBlocksPerWindowHorizontally*info.numberOfBlocksPerWindowVertically*info.descriptorLengthPerBlock;
    }

    // Initialize output matrices
    dims5[0] = info.numberOfWindowsVertically;
    dims5[1] = info.numberOfWindowsHorizontally;
    dims5[2] = info.numberOfBlocksPerWindowVertically;
    dims5[3] = info.numberOfBlocksPerWindowHorizontally;
    dims5[4] = info.descriptorLengthPerBlock;
    mxWindowsMatrixDescriptorsMatrix = mxCreateNumericArray(5,dims5,mxDOUBLE_CLASS,mxREAL);
    WindowsMatrixDescriptorsMatrix = mxGetPr(mxWindowsMatrixDescriptorsMatrix);
    
    dims3[0] = info.numberOfWindowsVertically;
    dims3[1] = info.numberOfWindowsHorizontally;
    dims3[2] = 2;
    mxWindowsCentersMatrix = mxCreateNumericArray(3,dims3,mxDOUBLE_CLASS,mxREAL);
    WindowsCentersMatrix = mxGetPr(mxWindowsCentersMatrix);
    
    const char *fieldnames[] = {"Descriptor","WindowsCentersCoordinates"};
    plhs[0] = mxCreateStructMatrix(1,1,2,fieldnames);
    
    // Print information if asked
    PrintInformation(options,info);

    // Processing
    MainLoop(options, info, windowImage, descriptorMatrix, descriptorVector, inputImage, WindowsMatrixDescriptorsMatrix, WindowsCentersMatrix);
    mxSetFieldByNumber(plhs[0],0,0,mxWindowsMatrixDescriptorsMatrix);
    mxSetFieldByNumber(plhs[0],0,1,mxWindowsCentersMatrix);
    
    // Destroy arrays
    mxDestroyArray(windowImageTemp);
    mxDestroyArray(descriptorMatrixTemp);
    if (options[8]==1)
        mxDestroyArray(descriptorVectorTemp);
}
