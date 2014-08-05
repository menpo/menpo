#include "HOG.h"
#include "check_cuda_errors.hpp"
#include "Python.h"

#define MAX_THREADS_1D      256
#define MAX_THREADS_1D_SMALL 64
#define MAX_THREADS_2D       16

#define MAX_THREADS_3DX       4
#define MAX_THREADS_3DY       4
#define MAX_THREADS_3DZ      16

HOG::HOG(unsigned int windowHeight, unsigned int windowWidth,
         unsigned int numberOfChannels, unsigned int method,
         unsigned int numberOfOrientationBins,
         unsigned int cellHeightAndWidthInPixels,
         unsigned int blockHeightAndWidthInCells, bool enableSignedGradients,
         double l2normClipping) {
    unsigned int descriptorLengthPerBlock = 0,
                 numberOfBlocksPerWindowVertically = 0,
                 numberOfBlocksPerWindowHorizontally = 0;

    if (method == 1) {
        descriptorLengthPerBlock = blockHeightAndWidthInCells *
                                   blockHeightAndWidthInCells *
                                   numberOfOrientationBins;
        numberOfBlocksPerWindowVertically = 1 +
        (windowHeight - blockHeightAndWidthInCells*cellHeightAndWidthInPixels)
        / cellHeightAndWidthInPixels;
        numberOfBlocksPerWindowHorizontally = 1 +
        (windowWidth - blockHeightAndWidthInCells * cellHeightAndWidthInPixels)
        / cellHeightAndWidthInPixels;
    }
    else if (method==2) {
        descriptorLengthPerBlock = 27 + 4;
        numberOfBlocksPerWindowVertically =
        (unsigned int)round((double)windowHeight /
                            (double)cellHeightAndWidthInPixels) - 2;
        numberOfBlocksPerWindowHorizontally =
        (unsigned int)round((double)windowWidth /
                            (double)cellHeightAndWidthInPixels) - 2;
    }
    this->method = method;
    this->numberOfOrientationBins = numberOfOrientationBins;
    this->cellHeightAndWidthInPixels = cellHeightAndWidthInPixels;
    this->blockHeightAndWidthInCells = blockHeightAndWidthInCells;
    this->enableSignedGradients = enableSignedGradients;
    this->l2normClipping = l2normClipping;
    this->numberOfBlocksPerWindowHorizontally =
                    numberOfBlocksPerWindowHorizontally;
    this->numberOfBlocksPerWindowVertically =
                    numberOfBlocksPerWindowVertically;
    this->descriptorLengthPerBlock = descriptorLengthPerBlock;
    this->descriptorLengthPerWindow = numberOfBlocksPerWindowHorizontally *
                                      numberOfBlocksPerWindowVertically *
                                      descriptorLengthPerBlock;
    this->windowHeight = windowHeight;
    this->windowWidth = windowWidth;
    this->numberOfChannels = numberOfChannels;
}

HOG::~HOG() {
}


void HOG::applyOnChunk(double *windowImage, double *descriptorVector) {
    if (this->method == 1)
        PyErr_SetString(PyExc_RuntimeError,
                        "HOG::applyOnChunk is not implemented for DalalTriggs");
    else
        ZhuRamananHOGdescriptor(windowImage, this->cellHeightAndWidthInPixels,
                                this->windowHeight, this->windowWidth,
                                this->numberOfChannels, descriptorVector);
}

void HOG::applyOnImage(const ImageWindowIterator &iwi, const double *image,
                       double *outputImage, int *windowsCenters) {
    __CLOG__
    double *d_image = 0;
    if (this->method == 1) {
        const unsigned int imageHeight = iwi._imageHeight;
        const unsigned int imageWidth = iwi._imageWidth;
        const unsigned int numberOfChannels = iwi._numberOfChannels;
        
        __START__
        cudaErrorCheck_goto(cudaMalloc(&d_image, imageHeight * imageWidth * numberOfChannels * sizeof(double)));
        cudaErrorCheck_goto(cudaMemcpy(d_image, image, imageHeight * imageWidth * numberOfChannels * sizeof(double), cudaMemcpyHostToDevice));
        __STOP("@ Malloc & Memcpy for <image> @")
        this->DalalTriggsHOGdescriptorOnImage(iwi, d_image, outputImage, windowsCenters);
        __START__
        cudaErrorCheck_goto(cudaFree(d_image));
        d_image = 0;
        __STOP("@ Free for <image> @")
    } else
        PyErr_SetString(PyExc_RuntimeError,
                        "HOG::applyOnImage is not implemented for ZhuRamanan");
    return;

onfailure:
    cudaFree(d_image);
    return;
}

bool HOG::isApplyOnImage() {
    if (this->method == 1) // easier to read this way
        return true;
    else
        return false;
}

// ZHU & RAMANAN: Face Detection, Pose Estimation and Landmark Localization
//                in the Wild
void ZhuRamananHOGdescriptor(double *inputImage,
                             int cellHeightAndWidthInPixels,
                             unsigned int imageHeight, unsigned int imageWidth,
                             unsigned int numberOfChannels,
                             double *descriptorMatrix) {
    // unit vectors used to compute gradient orientation
    double uu[9] = {1.0000, 0.9397, 0.7660, 0.500, 0.1736, -0.1736, -0.5000,
                    -0.7660, -0.9397};
    double vv[9] = {0.0000, 0.3420, 0.6428, 0.8660, 0.9848, 0.9848, 0.8660,
                    0.6428, 0.3420};
    int x, y, o;

    // memory for caching orientation histograms & their norms
    int blocks[2];
    blocks[0] = (int)round((double)imageHeight /
                           (double)cellHeightAndWidthInPixels);
    blocks[1] = (int)round((double)imageWidth /
                           (double)cellHeightAndWidthInPixels);
    double *hist = (double *)calloc(blocks[0] * blocks[1] * 18, sizeof(double));
    double *norm = (double *)calloc(blocks[0] * blocks[1], sizeof(double));

    // memory for HOG features
    int out[3];
    out[0] = max(blocks[0]-2, 0);
    out[1] = max(blocks[1]-2, 0);
    out[2] = 27+4;

    int visible[2];
    visible[0] = blocks[0] * cellHeightAndWidthInPixels;
    visible[1] = blocks[1] * cellHeightAndWidthInPixels;

    for (x = 1; x < visible[1] - 1; x++) {
        for (y = 1; y < visible[0] - 1; y++) {
            // compute gradient
            // first channel
            double *s = inputImage + min(x, imageWidth-2) * imageHeight +
                        min(y, imageHeight-2);
            double dy = *(s + 1) - *(s - 1);
            double dx = *(s + imageHeight) - *(s - imageHeight);
            double v = dx * dx + dy * dy;
            // rest of channels
            for (unsigned int z = 1; z < numberOfChannels; z++) {
                s += imageHeight * imageWidth;
                double dy2 = *(s + 1) - *(s - 1);
                double dx2 = *(s + imageHeight) - *(s - imageHeight);
                double v2 = dx2 * dx2 + dy2 * dy2;
                // pick channel with strongest gradient
                if (v2 > v) {
                    v = v2;
                    dx = dx2;
                    dy = dy2;
                }
            }

            // snap to one of 18 orientations
            double best_dot = 0;
            int best_o = 0;
            for (o = 0; o < 9; o++) {
                double dot = uu[o] * dx + vv[o] * dy;
                if (dot > best_dot) {
                    best_dot = dot;
                    best_o = o;
                }
                else if (-dot > best_dot) {
                    best_dot = - dot;
                    best_o = o + 9;
                }
            }

            // add to 4 histograms around pixel using linear interpolation
            double xp = ((double)x + 0.5) /
                        (double)cellHeightAndWidthInPixels - 0.5;
            double yp = ((double)y + 0.5) /
                        (double)cellHeightAndWidthInPixels - 0.5;
            int ixp = (int)floor(xp);
            int iyp = (int)floor(yp);
            double vx0 = xp - ixp;
            double vy0 = yp - iyp;
            double vx1 = 1.0 - vx0;
            double vy1 = 1.0 - vy0;
            v = sqrt(v);

            if (ixp >= 0 && iyp >= 0)
                *(hist + ixp*blocks[0] + iyp + best_o*blocks[0]*blocks[1])
                    += vx1 * vy1 * v;

            if (ixp+1 < blocks[1] && iyp >= 0)
                *(hist + (ixp+1)*blocks[0] + iyp + best_o*blocks[0]*blocks[1])
                    += vx0 * vy1 * v;

            if (ixp >= 0 && iyp+1 < blocks[0])
                *(hist + ixp*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1])
                    += vx1 * vy0 * v;

            if (ixp+1 < blocks[1] && iyp+1 < blocks[0])
                *(hist + (ixp+1)*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1])
                    += vx0 * vy0 * v;
        }
    }

    // compute energy in each block by summing over orientations
    for (int o = 0; o < 9; o++) {
        double *src1 = hist + o * blocks[0] * blocks[1];
        double *src2 = hist + (o + 9) * blocks[0] * blocks[1];
        double *dst = norm;
        double *end = norm + blocks[1] * blocks[0];
        while (dst < end) {
            *(dst++) += (*src1 + *src2) * (*src1 + *src2);
            src1++;
            src2++;
        }
    }

    // compute features
    for (x = 0; x < out[1]; x++) {
        for (y = 0; y < out[0]; y++) {
            double *dst = descriptorMatrix + x * out[0] + y;
            double *src, *p, n1, n2, n3, n4;

            p = norm + (x + 1) * blocks[0] + y + 1;
            n1 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) +
                            *(p + blocks[0] + 1) + eps);
            p = norm + (x + 1) * blocks[0] + y;
            n2 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) +
                            *(p + blocks[0] + 1) + eps);
            p = norm + x * blocks[0] + y + 1;
            n3 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) +
                            *(p + blocks[0] + 1) + eps);
            p = norm + x * blocks[0] + y;
            n4 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) +
                            *(p + blocks[0] + 1) + eps);

            double t1 = 0;
            double t2 = 0;
            double t3 = 0;
            double t4 = 0;

            // contrast-sensitive features
            src = hist + (x + 1) * blocks[0] + (y + 1);
            for (int o = 0; o < 18; o++) {
                double h1 = min(*src * n1, 0.2);
                double h2 = min(*src * n2, 0.2);
                double h3 = min(*src * n3, 0.2);
                double h4 = min(*src * n4, 0.2);
                *dst = 0.5 * (h1 + h2 + h3 + h4);
                t1 += h1;
                t2 += h2;
                t3 += h3;
                t4 += h4;
                dst += out[0] * out[1];
                src += blocks[0] * blocks[1];
            }

            // contrast-insensitive features
            src = hist + (x + 1) * blocks[0] + (y + 1);
            for (int o = 0; o < 9; o++) {
                double sum = *src + *(src + 9 * blocks[0] * blocks[1]);
                double h1 = min(sum * n1, 0.2);
                double h2 = min(sum * n2, 0.2);
                double h3 = min(sum * n3, 0.2);
                double h4 = min(sum * n4, 0.2);
                *dst = 0.5 * (h1 + h2 + h3 + h4);
                dst += out[0] * out[1];
                src += blocks[0] * blocks[1];
            }

            // texture features
            *dst = 0.2357 * t1;
            dst += out[0] * out[1];
            *dst = 0.2357 * t2;
            dst += out[0] * out[1];
            *dst = 0.2357 * t3;
            dst += out[0] * out[1];
            *dst = 0.2357 * t4;
        }
    }
    free(hist);
    free(norm);
}

void HOG::DalalTriggsHOGdescriptorOnImage(const ImageWindowIterator &iwi,
                                          double *d_image,
                                          double *outputImage,
                                          int *windowsCenters) {
    __CLOG__
    int rowCenter, columnCenter;
    
    // Define useful variables
    
    const unsigned int numWindows = iwi._numberOfWindowsVertically*iwi._numberOfWindowsHorizontally;
    const int hist1 = 2 + (this->windowHeight / this->cellHeightAndWidthInPixels);
    const int hist2 = 2 + (this->windowWidth / this->cellHeightAndWidthInPixels);
    
    double *d_blockNorm = 0, *d_block = 0, *d_outputImage = 0;
    
    // Each block has to compute its blockNorm[x][y]
    // stored into d_blockNorm[x-1 + blockNorm_dims.x*(y-1)]
    // Computation done with reduce-pattern
    //   x: unsigned int x = 1; x < hist2 - blockHeightAndWidthInCells; x++
    //      => requires (hist2 - blockHeightAndWidthInCells -1) Blocks
    //   y: unsigned int y = 1; y < hist1 - blockHeightAndWidthInCells; y++
    //      => requires (hist1 - blockHeightAndWidthInCells -1) Blocks
    // Each block works on 3D-Threads
    //   i: unsigned int i = 0; i < blockHeightAndWidthInCells; i++
    //   j: unsigned int j = 0; j < blockHeightAndWidthInCells; j++
    //      => usually: blockHeightAndWidthInCells=2
    //   k: unsigned int k = 0; k < numberOfOrientationBins; k++
    //      => usually: larger than blockHeightAndWidthInCells
    const dim3 blockNorm_dims(hist2 - blockHeightAndWidthInCells -1,
                              hist1 - blockHeightAndWidthInCells -1);
    const dim3 dimBlock_norm(MAX_THREADS_3DX, MAX_THREADS_3DY, MAX_THREADS_3DZ);
    const dim3 dimGrid_norm(blockNorm_dims.x*iwi._numberOfWindowsHorizontally,
                            blockNorm_dims.y*iwi._numberOfWindowsVertically, 1);
    
    // Each thread has to compute a single value of d_block for a given window
    // - block[i,j,k,x,y]
    //
    // Idea of the size of block vector for a given window:
    //    0 <= i < 2 (=blockHeightAndWidthInCells)
    //    0 <= j < 2 (=blockHeightAndWidthInCells)
    //    0 <= k < 9 (=numberOfOrientationBins)
    //    0 <= x < 1 (=blockNorm_dims.x)
    //    0 <= y < 1 (=blockNorm_dims.y)
    // = 2x2x9 = 36 elements to compute << 256 (=MAX_THREADS_1D)
    //(= block_size)
    //
    // For some configuration, it might be greater than MAX_THREADS_1D,
    // the kernel handles this case
    //
    // A kernel's block corresponds to a window
    // each kernel's block has to compute its elements of d_block
    const dim3 dimBlock_block(MAX_THREADS_1D_SMALL, 1, 1);
    const dim3 dimGrid_block(iwi._numberOfWindowsHorizontally,
                             iwi._numberOfWindowsVertically, 1);
    const unsigned int block_size = blockHeightAndWidthInCells
                                    * blockHeightAndWidthInCells
                                    * numberOfOrientationBins
                                    * blockNorm_dims.x
                                    * blockNorm_dims.y;
    
    // Each thread has to compute one element of outputImage
    // The kernel block is equivalent to a window, each block computes its own
    // "descriptorVector" - which is immediately written into outputImage 
    const dim3 dimBlock_desc(dimBlock_block);
    const dim3 dimGrid_desc(dimGrid_block);
    
    const unsigned long long int d_outputImage_size_t = iwi._numberOfWindowsVertically
            * iwi._numberOfWindowsHorizontally
            * this->descriptorLengthPerWindow * sizeof(double);
    
    const dim3 h_dims(hist1, hist2, this->numberOfOrientationBins);
    const unsigned long int h_size = h_dims.x * h_dims.y * h_dims.z * numWindows;
    const unsigned long long int d_h_size_t = h_size * sizeof(double);
    double *d_h = 0; // contains all the histograms
    
    const dim3 dimBlock(MAX_THREADS_2D, MAX_THREADS_2D, 1);
    const dim3 dimGrid((this->windowWidth * iwi._numberOfWindowsHorizontally + dimBlock.x -1)/dimBlock.x, (this->windowHeight * iwi._numberOfWindowsVertically + dimBlock.y -1)/dimBlock.y, 1);
    
    
    // Pre-allocate CUDA memory for DalalTriggsHOGdescriptor
    //
    // Allocating/Deleting memory takes lots of time for small vectors
    // Allocating/Deleting vectors before remove the cost of this operation
    
    __START__
    cudaErrorCheck_goto(cudaMalloc(&d_blockNorm, blockNorm_dims.x
                                                 * blockNorm_dims.y
                                                 * numWindows
                                                 * sizeof(double)));
    cudaErrorCheck_goto(cudaMalloc(&d_block, cellHeightAndWidthInPixels
                                             * cellHeightAndWidthInPixels
                                             * numberOfOrientationBins
                                             * blockNorm_dims.x
                                             * blockNorm_dims.y
                                             * numWindows
                                             * sizeof(double)));
    
    // Compute all the histograms together using CUDA
    //   h_dims: dimension of one histogram
    //   numWindows: number of histograms to compute
    //
    //       +--+--+  +--+
    // d_h = |h0|h1|..|hn|
    //       +--+--+  +--+
    //
    // where hx is a histogram
    //   and n = numWindows
    
    cudaErrorCheck_goto(cudaMalloc(&d_h, d_h_size_t));
    cudaErrorCheck_goto(cudaMemset(d_h, 0., d_h_size_t));
    
    // Allocate memory for the CUDA version of outputImage
    
    cudaErrorCheck_goto(cudaMalloc(&d_outputImage, d_outputImage_size_t));
    __STOP("@ Malloc & Memset for <output, d_h..> @")
    
    // Compute values for histograms
    __START__
    DalalTriggsHOGdescriptor_compute_histograms<<<dimGrid, dimBlock>>>(d_h, h_dims,
                                                                       d_image, iwi._imageHeight, iwi._imageWidth,
                                                                       this->windowHeight, this->windowWidth, this->numberOfChannels,
                                                                       this->numberOfOrientationBins, this->cellHeightAndWidthInPixels,
                                                                       this->enableSignedGradients ? 1 : 0 /*signedOrUnsignedGradients*/,
                                                                       (1 + (this->enableSignedGradients ? 1 : 0)) * pi / this->numberOfOrientationBins /*binsSize*/,
                                                                       numWindows, iwi._numberOfWindowsVertically,
                                                                       iwi._numberOfWindowsHorizontally,
                                                                       iwi._enablePadding, iwi._windowStepVertical, iwi._windowStepHorizontal);
    cudaErrorCheck_goto(cudaThreadSynchronize()); // block until the device is finished
    __STOP("@ Kernel: compute_histograms @")
    
    // Histogram normalization
    // Evaluate blockNorm based on d_h
    __START__
    cudaErrorCheck_goto(cudaMemset(d_blockNorm, 0., blockNorm_dims.x * blockNorm_dims.y * numWindows * sizeof(double)));
    DalalTriggsHOGdescriptor_compute_blocknorm<<<dimGrid_norm,
                                                 dimBlock_norm,
                                                 MAX_THREADS_3DX
                                                 * MAX_THREADS_3DY
                                                 * MAX_THREADS_3DZ
                                                 * sizeof(double)>>>
                                                    (d_blockNorm, blockNorm_dims,
                                                     d_h, h_dims,
                                                     numberOfOrientationBins,
                                                     blockHeightAndWidthInCells,
                                                     iwi._numberOfWindowsVertically);
    cudaErrorCheck_goto(cudaThreadSynchronize()); // block until the device is finished
    __STOP("@ Kernel: compute_blocknorm @")
    
    // Compute block
    __START__
    DalalTriggsHOGdescriptor_compute_block<<<dimGrid_block,
                                             dimBlock_block>>>
                                                (d_block,
                                                 d_blockNorm, blockNorm_dims,
                                                 d_h, h_dims,
                                                 numberOfOrientationBins,
                                                 blockHeightAndWidthInCells,
                                                 l2normClipping,
                                                 iwi._numberOfWindowsVertically,
                                                 block_size);
    cudaErrorCheck_goto(cudaThreadSynchronize()); // block until the device is finished
    __STOP("@ Kernel: compute_block @")
    
    // Evaluate blockNorm based on d_block
    __START__
    DalalTriggsHOGdescriptor_compute_blocknorm2<<<dimGrid_norm,
                                                  dimBlock_norm,
                                                  MAX_THREADS_3DX
                                                  * MAX_THREADS_3DY
                                                  * MAX_THREADS_3DZ
                                                  * sizeof(double)>>>
                                                    (d_blockNorm, blockNorm_dims,
                                                     d_block,
                                                     numberOfOrientationBins,
                                                     blockHeightAndWidthInCells,
                                                     iwi._numberOfWindowsVertically);
    cudaErrorCheck_goto(cudaThreadSynchronize()); // block until the device is finished
    __STOP("@ Kernel: compute_blocknorm2 @")
    
    // Compute outputImage
    __START__
    DalalTriggsHOGdescriptor_compute_outputImage<<<dimGrid_desc,
                                                   dimBlock_desc>>>
                                                    (d_outputImage,
                                                     d_block,
                                                     d_blockNorm,
                                                     blockNorm_dims,
                                                     numberOfOrientationBins,
                                                     blockHeightAndWidthInCells,
                                                     numWindows,
                                                     iwi._numberOfWindowsVertically,
                                                     block_size);
    cudaErrorCheck_goto(cudaThreadSynchronize()); // block until the device is finished
    __STOP("@ Kernel: compute_outputImage @")
    
    // Histogram normalization
    // & windowsCenters initialization
    //
    // Everything is done with native-C code (ie. without any CUDA implementation)
    __START__
    for (unsigned int windowIndexVertical = 0; windowIndexVertical < iwi._numberOfWindowsVertically; windowIndexVertical++) {
        for (unsigned int windowIndexHorizontal = 0; windowIndexHorizontal < iwi._numberOfWindowsHorizontally; windowIndexHorizontal++) {
            // Find window limits
            if (!iwi._enablePadding) {
                rowCenter = windowIndexVertical*iwi._windowStepVertical
                            + (int)round((double)iwi._windowHeight / 2.0) - 1;
                columnCenter = windowIndexHorizontal*iwi._windowStepHorizontal
                            + (int)round((double)iwi._windowWidth / 2.0) - 1;
            } else {
                rowCenter = windowIndexVertical*iwi._windowStepVertical;
                columnCenter = windowIndexHorizontal*iwi._windowStepHorizontal;
            }
            
            // Store results
            windowsCenters[windowIndexVertical+iwi._numberOfWindowsVertically*windowIndexHorizontal] = rowCenter;
            windowsCenters[windowIndexVertical+iwi._numberOfWindowsVertically*(windowIndexHorizontal+iwi._numberOfWindowsHorizontally)] = columnCenter;
        }
    }
    __STOP("@ Histogram Normalization @")
    
    __START__
    cudaErrorCheck_goto(cudaMemcpy(
            outputImage, d_outputImage,
            d_outputImage_size_t, cudaMemcpyDeviceToHost));
    cudaErrorCheck_goto(cudaFree(d_outputImage));
    d_outputImage = 0;
    
    cudaErrorCheck_goto(cudaFree(d_h));
    d_h = 0;
    cudaErrorCheck_goto(cudaFree(d_block));
    d_block = 0;
    cudaErrorCheck_goto(cudaFree(d_blockNorm));
    d_blockNorm = 0;
    __STOP("@ Memcpy & Free for <output, d_h..> @")
    return;
    
onfailure:
    cudaFree(d_h);
    cudaFree(d_outputImage);
    cudaFree(d_block);
    cudaFree(d_blockNorm);
    return;
}

/* Kernels */

__device__ double atomicAdd(double* address, double val) {
    // http://stackoverflow.com/questions/16882253/cuda-atomicadd-produces-wrong-result
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#define getInImage(i,j,k) ((i+rowFrom<0 || i+rowFrom>imageHeight-1 || j+columnFrom<0 || j+columnFrom>imageWidth-1) ? 0. : d_inputImage[(i+rowFrom) + imageHeight*((j+columnFrom) + imageWidth*k)])
__global__ void DalalTriggsHOGdescriptor_compute_histograms(double *d_h,
                                                               const dim3 h_dims,
                                                               const double *d_inputImage,
                                                               const unsigned int imageHeight,
                                                               const unsigned int imageWidth,
                                                               const unsigned int windowHeight,
                                                               const unsigned int windowWidth,
                                                               const unsigned int numberOfChannels,
                                                               const unsigned int numberOfOrientationBins,
                                                               const unsigned int cellHeightAndWidthInPixels,
                                                               const unsigned signedOrUnsignedGradients,
                                                               const double binsSize,
                                                               const int numHistograms,
                                                               const int numberOfWindowsVertically,
                                                               const int numberOfWindowsHorizontally,
                                                               const bool enablePadding,
                                                               const int windowStepVertical, const int windowStepHorizontal) {
    // Compute histograms values
    
    // Retrieve pixel position
    int x_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_ >= numberOfWindowsHorizontally * windowWidth)
        return;
    int y_ = blockIdx.y * blockDim.y + threadIdx.y;
    if (y_ >= numberOfWindowsVertically * windowHeight)
        return;
    
    int x = x_ % windowWidth;
    int windowIndexHorizontal = x_ / windowWidth;
    
    int y = y_ % windowHeight;
    int windowIndexVertical = y_ / windowHeight;
    
    unsigned int factor_y_dim = h_dims.x;
    unsigned int factor_z_dim = factor_y_dim * h_dims.y;
    unsigned int factor_o_dim = factor_z_dim * h_dims.z;
    
    int offsetWindow = factor_o_dim * (windowIndexVertical + numberOfWindowsVertically * windowIndexHorizontal);
    int rowFrom, columnFrom;
    if (!enablePadding) {
        rowFrom = windowIndexVertical*windowStepVertical;
        columnFrom = windowIndexHorizontal*windowStepHorizontal;
    } else {
        rowFrom = windowIndexVertical*windowStepVertical - (int)round((double)windowHeight / 2.0) + 1;
        columnFrom = windowIndexHorizontal*windowStepHorizontal - (int)ceil((double)windowWidth / 2.0) + 1;
    }
     
    // Compute deltas
    double dx[3], dy[3];
    
    if (x == 0) {
        for (unsigned int z = 0; z < numberOfChannels; z++)
            dx[z] = getInImage(y, x+1, z);
    } else {
        if (x == windowWidth - 1) {
            for (unsigned int z = 0; z < numberOfChannels; z++)
                dx[z] = -getInImage(y, x-1, z);
        } else {
            for (unsigned int z = 0; z < numberOfChannels; z++)
                dx[z] = getInImage(y, x+1, z) - getInImage(y, x-1, z);
        }
    }

    if(y == 0) {
        for (unsigned int z = 0; z < numberOfChannels; z++)
            dy[z] = -getInImage(y+1, x, z);
    } else {
        if (y == windowHeight - 1) {
            for (unsigned int z = 0; z < numberOfChannels; z++)
                dy[z] = getInImage(y-1, x, z);
        } else {
            for (unsigned int z = 0; z < numberOfChannels; z++)
                dy[z] = -getInImage(y+1, x, z) + getInImage(y-1, x, z);
        }
    }

    // Choose dominant channel based on magnitude
    double gradientMagnitude = sqrt(dx[0] * dx[0] + dy[0] * dy[0]);
    double gradientOrientation = atan2(dy[0], dx[0]);
    if (numberOfChannels > 1) {
        double tempMagnitude = gradientMagnitude;
        for (unsigned int cli = 1 ; cli < numberOfChannels ; ++cli) {
            tempMagnitude= sqrt(dx[cli] * dx[cli] + dy[cli] * dy[cli]);
            if (tempMagnitude > gradientMagnitude) {
                gradientMagnitude = tempMagnitude;
                gradientOrientation = atan2(dy[cli], dx[cli]);
            }
        }
    }

    if (gradientOrientation < 0)
        gradientOrientation += pi + (signedOrUnsignedGradients == 1) * pi;

    // Trilinear interpolation
    int bin1 = (gradientOrientation / binsSize) - 1;
    unsigned int bin2 = bin1 + 1;
    int x1   = x / cellHeightAndWidthInPixels;
    int x2   = x1 + 1;
    int y1   = y / cellHeightAndWidthInPixels;
    int y2   = y1 + 1;
    
    double Xc = (x1 + 1 - 1.5) * cellHeightAndWidthInPixels + 0.5;
    double Yc = (y1 + 1 - 1.5) * cellHeightAndWidthInPixels + 0.5;
    double Oc = (bin1 + 1 + 1 - 1.5) * binsSize;
    
    if (bin2 == numberOfOrientationBins)
        bin2 = 0;
    
    if (bin1 < 0)
        bin1 = numberOfOrientationBins - 1;
    
    // Compute histograms
    //  using reduce-pattern
    //
    // d_h needs to be set to 0.
    
    atomicAdd(
            &d_h[offsetWindow + y1 + x1*factor_y_dim + bin1*factor_z_dim],
            gradientMagnitude *
                (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                (1-((gradientOrientation-Oc)/binsSize)));
    atomicAdd(
            &d_h[offsetWindow + y1 + x1*factor_y_dim + bin2*factor_z_dim],
            gradientMagnitude *
                (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                (((gradientOrientation-Oc)/binsSize)));
    atomicAdd(
            &d_h[offsetWindow + y2 + x1*factor_y_dim + bin1*factor_z_dim],
            gradientMagnitude *
                (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                (1-((gradientOrientation-Oc)/binsSize)));
    atomicAdd(
            &d_h[offsetWindow + y2 + x1*factor_y_dim + bin2*factor_z_dim],
            gradientMagnitude *
                (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                (((gradientOrientation-Oc)/binsSize)));
    atomicAdd(
            &d_h[offsetWindow + y1 + x2*factor_y_dim + bin1*factor_z_dim],
            gradientMagnitude *
                (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                (1-((gradientOrientation-Oc)/binsSize)));
    atomicAdd(
            &d_h[offsetWindow + y1 + x2*factor_y_dim + bin2*factor_z_dim],
            gradientMagnitude *
                (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                (((gradientOrientation-Oc)/binsSize)));
    atomicAdd(
            &d_h[offsetWindow + y2 + x2*factor_y_dim + bin1*factor_z_dim],
            gradientMagnitude *
                (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                (1-((gradientOrientation-Oc)/binsSize)));
    atomicAdd(
            &d_h[offsetWindow + y2 + x2*factor_y_dim + bin2*factor_z_dim],
            gradientMagnitude *
                (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                (((gradientOrientation-Oc)/binsSize)));
}

__global__ void DalalTriggsHOGdescriptor_compute_blocknorm(double *d_blockNorm,
                                                           const dim3 blockNorm_dims,
                                                           const double *d_h,
                                                           const dim3 h_dims,
                                                           const unsigned int numberOfOrientationBins,
                                                           const unsigned int blockHeightAndWidthInCells,
                                                           const unsigned int numberOfWindowsVertically) {
    // 2D-reduce to compute d_blockNorm for every (x,y)
    
    // Size of shared memory must be blockDim.x*blockDim.y*blockDim.z
    // and a power of 2
    extern __shared__ double cache[];
    
    // Compute factors
    unsigned int factor_z_dim = h_dims.x * h_dims.y;
    unsigned int factor_y_dim = h_dims.x;
    
    // Retrieve indice of the element
    unsigned int x = blockIdx.x;
    unsigned int y = blockIdx.y;
    unsigned int z = (y/blockNorm_dims.y) + numberOfWindowsVertically * (x/blockNorm_dims.x);
    x = (x % blockNorm_dims.x) +1;
    y = (y % blockNorm_dims.y) +1;
    unsigned int offsetH = h_dims.x * h_dims.y * h_dims.z * z;
    
    unsigned int i = threadIdx.x;
    unsigned int j = threadIdx.y;
    unsigned int k = threadIdx.z;
    unsigned int current_id = i + j*blockDim.x
                              + k*blockDim.x*blockDim.y;
    
    cache[current_id] = 0.;
    for (unsigned int i_=i ; i_ < blockHeightAndWidthInCells ; i_+=blockDim.x)
        for (unsigned int j_=j ; j_ < blockHeightAndWidthInCells ; j_+=blockDim.y)
            for (unsigned int k_=k ; k_ < numberOfOrientationBins ; k_+=blockDim.z)
                cache[current_id] += d_h[y+i_ + (x+j_) * factor_y_dim + k_ * factor_z_dim + offsetH]
                                     * d_h[y+i_ + (x+j_) * factor_y_dim + k_ * factor_z_dim + offsetH];
    __syncthreads();
    
    // Reduce operation
    // all threads in the current block have to compute d_blockNorm[x + hist2*y]
    int padding = blockDim.x*blockDim.y*blockDim.z/2;
    while (padding != 0) {
        if (current_id < padding)
            cache[current_id] += cache[current_id + padding];
        __syncthreads();
        padding /= 2;
    }
    
    if (i == 0 && j == 0 && k == 0)
        d_blockNorm[x-1 + blockNorm_dims.x*(y-1)
                    + blockNorm_dims.x*blockNorm_dims.y*z] = cache[0];
}

__global__ void DalalTriggsHOGdescriptor_compute_block(double *d_block,
                                                       const double *d_blockNorm,
                                                       const dim3 blockNorm_dims,
                                                       const double *d_h,
                                                       const dim3 h_dims,
                                                       const unsigned int numberOfOrientationBins,
                                                       const unsigned int blockHeightAndWidthInCells,
                                                       const double l2normClipping,
                                                       const unsigned int numberOfWindowsVertically,
                                                       const unsigned int block_size) {
    // Each thread has to compute one value of block[i,j,k,x,y]
    // for a given windows (blockIdx.x, blockIdx.y)
    
    // Compute window's index
    //unsigned int windowIndexHorizontal = blockIdx.x;
    //unsigned int windowIndexVertical = blockIdx.y;
    unsigned windowIndex = (blockIdx.y + numberOfWindowsVertically * blockIdx.x);
    
    // Compute factors
    unsigned int factor_z_dim = h_dims.x * h_dims.y;
    unsigned int factor_y_dim = h_dims.x;
    
    // Retrieve ids of elements to compute during this thread
    // In most of the cases, the loop should be called only once
    for (unsigned int elementIndex(threadIdx.x) ; elementIndex < block_size ; elementIndex += blockDim.x) {
        //elementIndex = i + j*blockHeightAndWidthInCells
        //               + k*blockHeightAndWidthInCells*blockHeightAndWidthInCells
        //               + (x-1)*blockHeightAndWidthInCells*blockHeightAndWidthInCells
        //                 *numberOfOrientationBins
        //               + (y-1)*blockHeightAndWidthInCells*blockHeightAndWidthInCells
        //                 *numberOfOrientationBins*blockNorm_dims.x
        unsigned int i = elementIndex;
        unsigned int j = i / blockHeightAndWidthInCells;
        i %= blockHeightAndWidthInCells;
        unsigned int k = j / blockHeightAndWidthInCells;
        j %= blockHeightAndWidthInCells;
        unsigned int x = k / numberOfOrientationBins;
        k %= numberOfOrientationBins;
        unsigned int y = (x / blockNorm_dims.x) +1;
        x = (x % blockNorm_dims.x) +1;
        
        unsigned int current_id = elementIndex
                                  + windowIndex * blockHeightAndWidthInCells
                                    * blockHeightAndWidthInCells
                                    * numberOfOrientationBins
                                    * blockNorm_dims.x * blockNorm_dims.y;
        
        double blockNorm = sqrt(d_blockNorm[x-1 + blockNorm_dims.x*(y-1)
                                            + blockNorm_dims.x*blockNorm_dims.y*windowIndex]);
        if (blockNorm > 0) {
            unsigned int offsetH = h_dims.x * h_dims.y * h_dims.z * windowIndex;
            double tmpValue = d_h[y+i + (x+j) * factor_y_dim
                                  + k * factor_z_dim + offsetH] / blockNorm;
            if (tmpValue > l2normClipping)
                d_block[current_id] = l2normClipping;
            else
                d_block[current_id] = tmpValue;
        } else
            d_block[current_id] = 0.;
    }
}

__global__ void DalalTriggsHOGdescriptor_compute_blocknorm2(double *d_blockNorm,
                                                            const dim3 blockNorm_dims,
                                                            const double *d_block,
                                                            const unsigned int numberOfOrientationBins,
                                                            const unsigned int blockHeightAndWidthInCells,
                                                            const unsigned int numberOfWindowsVertically) {
    // 2D-reduce to compute d_blockNorm for every (x,y)
    
    // Size of shared memory must be blockDim.x*blockDim.y*blockDim.z
    // and a power of 2
    extern __shared__ double cache[];
    
    // Retrieve indice of the element
    unsigned int x = blockIdx.x;
    unsigned int y = blockIdx.y;
    unsigned int z = (y/blockNorm_dims.y) + numberOfWindowsVertically * (x/blockNorm_dims.x);
    x = (x % blockNorm_dims.x) +1;
    y = (y % blockNorm_dims.y) +1;
    
    unsigned int i = threadIdx.x;
    unsigned int j = threadIdx.y;
    unsigned int k = threadIdx.z;
    unsigned int current_id = i + j*blockDim.x
                              + k*blockDim.x*blockDim.y;
    
    cache[current_id] = 0.;
    for (unsigned int i_=i ; i_ < blockHeightAndWidthInCells ; i_+=blockDim.x) {
        for (unsigned int j_=j ; j_ < blockHeightAndWidthInCells ; j_+=blockDim.y) {
            for (unsigned int k_=k ; k_ < numberOfOrientationBins ; k_+=blockDim.z) {
                unsigned int current_id_norm = i_ + j_*blockHeightAndWidthInCells
                              + k_*blockHeightAndWidthInCells*blockHeightAndWidthInCells
                              + (x-1)*blockHeightAndWidthInCells*blockHeightAndWidthInCells
                                *numberOfOrientationBins
                              + (y-1)*blockHeightAndWidthInCells*blockHeightAndWidthInCells
                                *numberOfOrientationBins*blockNorm_dims.x
                              + z*blockHeightAndWidthInCells*blockHeightAndWidthInCells
                                *numberOfOrientationBins*blockNorm_dims.x*blockNorm_dims.y;
                cache[current_id] += d_block[current_id_norm] * d_block[current_id_norm];
            }
        }
    }
    
    // Reduce operation
    // all threads in the current block have to compute d_blockNorm[x + hist2*y]
    int padding = blockDim.x*blockDim.y*blockDim.z/2;
    while (padding != 0) {
        if (current_id < padding)
            cache[current_id] += cache[current_id + padding];
        __syncthreads();
        padding /= 2;
    }
    
    // Several k values will have to participate to this value
    if (i == 0 && j == 0)
        d_blockNorm[x-1 + blockNorm_dims.x*(y-1)
                    + blockNorm_dims.x*blockNorm_dims.y*z] = cache[0];
}

__global__ void DalalTriggsHOGdescriptor_compute_outputImage(double *d_outputImage,
                                                             const double *d_block,
                                                             const double *d_blockNorm,
                                                             const dim3 blockNorm_dims,
                                                             const unsigned int numberOfOrientationBins,
                                                             const unsigned int blockHeightAndWidthInCells,
                                                             const unsigned int numWindows,
                                                             const unsigned int numberOfWindowsVertically,
                                                             const unsigned int block_size) {
    // Each thread has to compute one value of outputImage
    // for a given windows (blockIdx.x, blockIdx.y)
    
    // Compute window's index
    //unsigned int windowIndexHorizontal = blockIdx.x;
    //unsigned int windowIndexVertical = blockIdx.y;
    unsigned windowIndex = (blockIdx.y + numberOfWindowsVertically * blockIdx.x);
    
    // Retrieve ids of elements to compute during this thread
    // In most of the cases, the loop should be called only once
    for (unsigned int elementIndex(threadIdx.x) ; elementIndex < block_size ; elementIndex += blockDim.x) {
        //elementIndex = i + j*blockHeightAndWidthInCells
        //               + k*blockHeightAndWidthInCells*blockHeightAndWidthInCells
        //               + (x-1)*blockHeightAndWidthInCells*blockHeightAndWidthInCells
        //                 *numberOfOrientationBins
        //               + (y-1)*blockHeightAndWidthInCells*blockHeightAndWidthInCells
        //                 *numberOfOrientationBins*blockNorm_dims.x
        unsigned int i = elementIndex;
        unsigned int j = i / blockHeightAndWidthInCells;
        i %= blockHeightAndWidthInCells;
        unsigned int k = j / blockHeightAndWidthInCells;
        j %= blockHeightAndWidthInCells;
        unsigned int x = k / numberOfOrientationBins;
        k %= numberOfOrientationBins;
        unsigned int y = (x / blockNorm_dims.x) +1;
        x = (x % blockNorm_dims.x) +1;
        
        
        unsigned int descriptorIndex = k + numberOfOrientationBins * (
                j + blockHeightAndWidthInCells * (
                i + blockHeightAndWidthInCells * (
                y-1 + blockNorm_dims.y * (x-1))));
        
        double blockNorm = d_blockNorm[x-1 + blockNorm_dims.x*(y-1)
                           + blockNorm_dims.x*blockNorm_dims.y*windowIndex];
        if (blockNorm > 0) {
            blockNorm = sqrt(blockNorm);
            unsigned int current_id = elementIndex
                                      + windowIndex * blockHeightAndWidthInCells
                                        * blockHeightAndWidthInCells
                                        * numberOfOrientationBins
                                        * blockNorm_dims.x * blockNorm_dims.y;
            d_outputImage[windowIndex + numWindows*descriptorIndex]
                        = d_block[current_id] / blockNorm;
        } else
            d_outputImage[windowIndex + numWindows*descriptorIndex] = 0.;
    }
}
