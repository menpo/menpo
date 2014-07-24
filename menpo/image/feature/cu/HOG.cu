#include "HOG.h"
#include "check_cuda_errors.hpp"
#include "Python.h"

#define MAX_THREADS_1D 256
#define MAX_THREADS_2D  16

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
    double *d_image = NULL;
    if (this->method == 1) {
        const unsigned int imageHeight = iwi._imageHeight;
        const unsigned int imageWidth = iwi._imageWidth;
        const unsigned int numberOfChannels = iwi._numberOfChannels;
        
        cudaErrorCheck_goto(cudaMalloc(&d_image, imageHeight * imageWidth * numberOfChannels * sizeof(double)));
        cudaErrorCheck_goto(cudaMemcpy(d_image, image, imageHeight * imageWidth * numberOfChannels * sizeof(double), cudaMemcpyHostToDevice));
        
        this->DalalTriggsHOGdescriptorOnImage(iwi, d_image, outputImage, windowsCenters);
        
        cudaErrorCheck_goto(cudaFree(d_image));
        d_image = NULL;
    } else
        PyErr_SetString(PyExc_RuntimeError,
                        "HOG::applyOnImage is not implemented for ZhuRamanan");
    return;

onfailure:
    if (d_image != NULL)
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
__global__ void DalalTriggsHOGdescriptor_precompute_histograms(double *d_h,
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
                                                               const int numHistograms, const int numberOfWindowsVertically,
                                                               const bool enablePadding,
                                                               const int windowStepVertical, const int windowStepHorizontal) {
    // Pre-compute histograms values
    // The array that contains "d_h" needs to be
    //  2*cellHeightAndWidthInPixels * 2*cellHeightAndWidthInPixels larger
    // Reduce kernel needs to be call in order to retrieve the expected
    // histograms
    
    // Retrieve pixel position
    int x_ = blockIdx.x * blockDim.x + threadIdx.x;
    int x = x_ % windowWidth;
    int windowIndexHorizontal = x_ / windowWidth;
    
    int y_ = blockIdx.y * blockDim.y + threadIdx.y;
    int y = y_ % windowHeight;
    int windowIndexVertical = y_ / windowHeight;
    
    unsigned int factor_y_dim = h_dims.x;
    unsigned int factor_z_dim = factor_y_dim * h_dims.y;
    unsigned int factor_o_dim = factor_z_dim * h_dims.z;
    unsigned int factor_a_dim = factor_o_dim * numHistograms;
    
    int offsetWindow = factor_o_dim * (windowIndexVertical + numberOfWindowsVertically * windowIndexHorizontal);
    int rowFrom, columnFrom;
    if (enablePadding) {
        rowFrom = windowIndexVertical*windowStepVertical;
        columnFrom = windowIndexHorizontal*windowStepHorizontal;
    } else {
        rowFrom = windowIndexVertical*windowStepVertical - (int)round((double)windowHeight / 2.0) + 1;
        columnFrom = windowIndexHorizontal*windowStepHorizontal - (int)ceil((double)windowWidth / 2.0) + 1;
    }
     
    // Check if position is inside the image
    if (x >= windowWidth || y >= windowHeight)
        return;
    
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
    int a    = (x % (2*cellHeightAndWidthInPixels)) * (2*cellHeightAndWidthInPixels)
               + (y % (2*cellHeightAndWidthInPixels));
    
    double Xc = (x1 + 1 - 1.5) * cellHeightAndWidthInPixels + 0.5;
    double Yc = (y1 + 1 - 1.5) * cellHeightAndWidthInPixels + 0.5;
    double Oc = (bin1 + 1 + 1 - 1.5) * binsSize;
    
    if (bin2 == numberOfOrientationBins)
        bin2 = 0;
    
    if (bin1 < 0)
        bin1 = numberOfOrientationBins - 1;
    
    // Compute histograms
    //  using reduce-pattern
    
    // d_h contains (2*cellHeightAndWidthInPixels)*(2*cellHeightAndWidthInPixels)
    //  times required d_h
    //  which is equal to: 2*8*2*8 = 256
    // d_h needs to be set to 0.
    
    d_h[offsetWindow + y1 + x1*factor_y_dim + bin1*factor_z_dim + a*factor_a_dim] = gradientMagnitude *
                                                        (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                        (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                        (1-((gradientOrientation-Oc)/binsSize));
    d_h[offsetWindow + y1 + x1*factor_y_dim + bin2*factor_z_dim + a*factor_a_dim] = gradientMagnitude *
                                                        (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                        (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                        (((gradientOrientation-Oc)/binsSize));
    d_h[offsetWindow + y2 + x1*factor_y_dim + bin1*factor_z_dim + a*factor_a_dim] = gradientMagnitude *
                                                        (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                        (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                        (1-((gradientOrientation-Oc)/binsSize));
    d_h[offsetWindow + y2 + x1*factor_y_dim + bin2*factor_z_dim + a*factor_a_dim] = gradientMagnitude *
                                                        (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                        (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                        (((gradientOrientation-Oc)/binsSize));
    d_h[offsetWindow + y1 + x2*factor_y_dim + bin1*factor_z_dim + a*factor_a_dim] = gradientMagnitude *
                                                        (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                        (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                        (1-((gradientOrientation-Oc)/binsSize));
    d_h[offsetWindow + y1 + x2*factor_y_dim + bin2*factor_z_dim + a*factor_a_dim] = gradientMagnitude *
                                                        (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                        (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                        (((gradientOrientation-Oc)/binsSize));
    d_h[offsetWindow + y2 + x2*factor_y_dim + bin1*factor_z_dim + a*factor_a_dim] = gradientMagnitude *
                                                        (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                        (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                        (1-((gradientOrientation-Oc)/binsSize));
    d_h[offsetWindow + y2 + x2*factor_y_dim + bin2*factor_z_dim + a*factor_a_dim] = gradientMagnitude *
                                                        (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                        (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                        (((gradientOrientation-Oc)/binsSize));
}

__global__ void DalalTriggsHOGdescriptor_reduce_histograms(double *d_h,
                                                           const dim3 h_dims,
                                                           const unsigned int cellHeightAndWidthInPixels,
                                                           const int numberOfWindowsVertically,
                                                           const int numberOfWindowsHorizontally) {
    // cache size has to be a power of 2
    // usually set to MAX_THREADS_1D
    extern __shared__ double cache[];
    
    // Compute factors
    unsigned int factor_y_dim = h_dims.x * numberOfWindowsHorizontally;
    unsigned int factor_z_dim = factor_y_dim * h_dims.y * numberOfWindowsVertically;
    unsigned int factor_a_dim = factor_z_dim * h_dims.z;
    
    // Retrieve indice of the element
    // Another idea would have been to retrieve h_element_id directly from
    //   blockIdx..
    // Unfortunately the number of blocks per dimension of the grid
    //   is limited (65535). We may have run out of blocks.
    //   On my test image: 16x135x16x137x9 would have been required
    unsigned int x = blockIdx.y;
    unsigned int y = blockIdx.x;
    unsigned int bin = blockIdx.z;
    unsigned int a = threadIdx.x;
    unsigned int h_element_id = y + x*factor_y_dim + bin*factor_z_dim;
    unsigned int numElements = (2*cellHeightAndWidthInPixels) * (2*cellHeightAndWidthInPixels);
    
    // Copy to cache
    // The forloop make it possible to deal with too many data:
    //  in that case, a thread could have to deal with more than one element
    // /!\ Not Coalesced Memory access - can be slow
    if (a < numElements)
    {
        cache[a] = d_h[h_element_id + a*factor_a_dim];
        for (unsigned int a_(a+blockDim.x) ; a_ < numElements ; a_ += blockDim.x)
            cache[a] += d_h[h_element_id + a_*factor_a_dim];
    }
    else
        cache[a] = 0.;
    __syncthreads();
    
    // Reduce operation
    // all threads in the current block have to compute d_h[h_element_id]
    int padding = blockDim.x/2;
    while (padding != 0) {
        if (a < padding)
            cache[a] += cache[a + padding];
        __syncthreads();
        padding /= 2;
    }
    
    // Copy to d_h[h_element_id]
    if (a == 0)
        d_h[h_element_id] = cache[0];
}

void HOG::DalalTriggsHOGdescriptorOnImage(const ImageWindowIterator &iwi,
                                          double *d_image,
                                          double *outputImage,
                                          int *windowsCenters) {
    int rowCenter, columnCenter;
    unsigned int offsetH;
    double* descriptorVector = new double[this->descriptorLengthPerWindow];
    
    // Compute all the histograms together using CUDA
    //   h_dims: dimension of one histogram
    //   numHistograms_d_h: number of histograms to compute
    //   numCopies_d_h: number of copies - used for reduce pattern
    //
    //       +-------------+-------------+..+-------------+
    //       |+--+--+  +--+|+--+--+  +--+|  |+--+--+  +--+|
    // d_h = ||h0|h1|..|hn|||h0|h1|..|hn||..||h0|h1|..|hn||
    //       |+--+--+  +--+|+--+--+  +--+|  |+--+--+  +--+|
    //       +-------------+-------------+..+-------------+
    //        #0            #1               #m
    //
    // where hx is a histogram
    //   and n = numHistograms_d_h, m = numCopies_d_h
    // after a call to reduce kernel, #0 will contain the histograms
    //    #x contains temporary values that must be sum up in order to have the histogram value
    
    const int hist1 = 2 + (this->windowHeight / this->cellHeightAndWidthInPixels);
    const int hist2 = 2 + (this->windowWidth / this->cellHeightAndWidthInPixels);
    const dim3 h_dims(hist1, hist2, this->numberOfOrientationBins);
    const unsigned int numHistograms_d_h = iwi._numberOfWindowsVertically*iwi._numberOfWindowsHorizontally;
    const unsigned int numCopies_d_h = (2*this->cellHeightAndWidthInPixels) * (2*this->cellHeightAndWidthInPixels);
    double *d_h = NULL;
    double h[h_dims.x * h_dims.y * h_dims.z * numHistograms_d_h]; // contains all the histograms
    
    // For pre-computation
    const dim3 dimBlock(MAX_THREADS_2D, MAX_THREADS_2D, 1);
    const dim3 dimGrid((this->windowWidth * iwi._numberOfWindowsHorizontally + dimBlock.x -1)/dimBlock.x, (this->windowHeight * iwi._numberOfWindowsVertically + dimBlock.y -1)/dimBlock.y, 1);
     
    // For reduce:
    //    each block is responsible to evaluate the value of
    //      d_h[historamId][blockIdx.x][blockIdx.y][blockIdx.z]
    //    the reduce operation concerns (2*cellHeightAndWidthInPixels) * (2*cellHeightAndWidthInPixels) elements
    const dim3 dimGrid_reduce(h_dims.x * iwi._numberOfWindowsHorizontally, h_dims.y * iwi._numberOfWindowsVertically, h_dims.z);
    
    cudaErrorCheck_goto(cudaMalloc(&d_h, h_dims.x * h_dims.y * h_dims.z * numCopies_d_h * numHistograms_d_h * sizeof(double)));
    cudaErrorCheck_goto(cudaMemset(d_h, 0., h_dims.x * h_dims.y * h_dims.z * numCopies_d_h * numHistograms_d_h * sizeof(double)));
    
    // Pre-compute values for histograms
    DalalTriggsHOGdescriptor_precompute_histograms<<<dimGrid, dimBlock>>>(d_h, h_dims,
                                                                          d_image, iwi._imageHeight, iwi._imageWidth,
                                                                          this->windowHeight, this->windowWidth, this->numberOfChannels,
                                                                          this->numberOfOrientationBins, this->cellHeightAndWidthInPixels,
                                                                          this->enableSignedGradients ? 1 : 0 /*signedOrUnsignedGradients*/,
                                                                          (1 + (this->enableSignedGradients ? 1 : 0)) * pi / this->numberOfOrientationBins /*binsSize*/,
                                                                          numHistograms_d_h, iwi._numberOfWindowsVertically,
                                                                          iwi._enablePadding, iwi._windowStepVertical, iwi._windowStepHorizontal);
    cudaErrorCheck_goto(cudaThreadSynchronize()); // block until the device is finished
    
    // Compute histograms (reduce pattern)
    DalalTriggsHOGdescriptor_reduce_histograms<<<dimGrid_reduce,
                                                 MAX_THREADS_1D,
                                                 MAX_THREADS_1D*sizeof(double)>>>
                                                    (d_h, h_dims,
                                                     this->cellHeightAndWidthInPixels,
                                                     iwi._numberOfWindowsVertically,
                                                     iwi._numberOfWindowsHorizontally);
    cudaErrorCheck_goto(cudaThreadSynchronize()); // block until the device is finished
    
    cudaErrorCheck_goto(cudaMemcpy(h, d_h, h_dims.x * h_dims.y * h_dims.z * numHistograms_d_h * sizeof(double), cudaMemcpyDeviceToHost));
    cudaErrorCheck_goto(cudaFree(d_h));
    d_h = NULL;
    
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
            offsetH = h_dims.x * h_dims.y * h_dims.z * (windowIndexVertical + iwi._numberOfWindowsVertically * windowIndexHorizontal);

            // Compute descriptor of window
            DalalTriggsHOGdescriptor(h, offsetH, this->numberOfOrientationBins,
                             this->cellHeightAndWidthInPixels,
                             this->blockHeightAndWidthInCells,
                             this->enableSignedGradients,
                             this->l2normClipping,
                             iwi._imageHeight, iwi._imageWidth,
                             this->windowHeight, this->windowWidth,
                             this->numberOfChannels,
                             descriptorVector);

            // Store results
            for (unsigned int d = 0; d < this->descriptorLengthPerWindow; d++)
                outputImage[windowIndexVertical+iwi._numberOfWindowsVertically*(windowIndexHorizontal+iwi._numberOfWindowsHorizontally*d)] = descriptorVector[d];
            windowsCenters[windowIndexVertical+iwi._numberOfWindowsVertically*windowIndexHorizontal] = rowCenter;
            windowsCenters[windowIndexVertical+iwi._numberOfWindowsVertically*(windowIndexHorizontal+iwi._numberOfWindowsHorizontally)] = columnCenter;
        }
    }
    
    
    // Free temporary matrices
    delete[] descriptorVector;

onfailure:
    if (d_h != NULL)
        cudaFree(d_h);
}

// DALAL & TRIGGS: Histograms of Oriented Gradients for Human Detection
void DalalTriggsHOGdescriptor(double *h,
                              unsigned int offsetH,
                              unsigned int numberOfOrientationBins,
                              unsigned int cellHeightAndWidthInPixels,
                              unsigned int blockHeightAndWidthInCells,
                              bool signedOrUnsignedGradientsBool,
                              double l2normClipping,
                              unsigned int imageHeight, unsigned int imageWidth,
                              unsigned int windowHeight, unsigned int windowWidth,
                              unsigned int numberOfChannels,
                              double *descriptorVector) {
   
    // Variables
    //  * Compute gradients & Compute histograms
    
    const int hist1 = 2 + (windowHeight / cellHeightAndWidthInPixels);
    const int hist2 = 2 + (windowWidth / cellHeightAndWidthInPixels);
    const dim3 h_dims(hist1, hist2, numberOfOrientationBins);
    const unsigned int factor_z_dim = h_dims.x * h_dims.y;
    const unsigned int factor_y_dim = h_dims.x;
    
    //  * Block normalization
    
    int descriptorIndex(0);
    vector<vector<vector<double> > > block(blockHeightAndWidthInCells, vector<vector<double> >
                                           (blockHeightAndWidthInCells, vector<double>
                                            (numberOfOrientationBins, 0.0) ) );
     
    // Block normalization
    
    for (unsigned int x = 1; x < hist2 - blockHeightAndWidthInCells; x++) {
        for (unsigned int y = 1; y < hist1 - blockHeightAndWidthInCells; y++) {
            float blockNorm(0);
            for (unsigned int i = 0; i < blockHeightAndWidthInCells; i++)
                for (unsigned int j = 0; j < blockHeightAndWidthInCells; j++)
                    for (unsigned int k = 0; k < numberOfOrientationBins; k++)
                        blockNorm += h[offsetH + y+i + (x+j) * factor_y_dim + k * factor_z_dim] * h[offsetH + y+i + (x+j) * factor_y_dim + k * factor_z_dim];

            blockNorm = sqrt(blockNorm);
            for (unsigned int i = 0; i < blockHeightAndWidthInCells; i++) {
                for (unsigned int j = 0; j < blockHeightAndWidthInCells; j++) {
                    for (unsigned int k = 0; k < numberOfOrientationBins; k++) {
                        if (blockNorm > 0) {
                            block[i][j][k] = h[offsetH + y+i + (x+j) * factor_y_dim + k * factor_z_dim] / blockNorm;
                            if (block[i][j][k] > l2normClipping)
                                block[i][j][k] = l2normClipping;
                        }
                    }
                }
            }

            blockNorm = 0;
            for (unsigned int i = 0; i < blockHeightAndWidthInCells; i++)
                for (unsigned int j = 0; j < blockHeightAndWidthInCells; j++)
                    for (unsigned int k = 0; k < numberOfOrientationBins; k++)
                        blockNorm += block[i][j][k] * block[i][j][k];

            blockNorm = sqrt(blockNorm);
            for (unsigned int i = 0; i < blockHeightAndWidthInCells; i++) {
                for (unsigned int j = 0; j < blockHeightAndWidthInCells; j++) {
                    for (unsigned int k = 0; k < numberOfOrientationBins; k++) {
                        if (blockNorm > 0)
                            descriptorVector[descriptorIndex] =
                                block[i][j][k] / blockNorm;
                        else
                            descriptorVector[descriptorIndex] = 0.0;
                        descriptorIndex++;
                    }
                }
            }
        }
    }
}
