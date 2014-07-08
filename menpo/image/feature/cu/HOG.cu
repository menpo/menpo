#include "HOG.h"
#include "check_cuda_errors.hpp"

#define MAX_THREADS_1D 256
#define MAX_THREADS_2D  16

#define MAX_THREADS_3DX  4
#define MAX_THREADS_3DY  4
#define MAX_THREADS_3DZ 16

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


void HOG::apply(double *windowImage, double *descriptorVector) {
    if (this->method == 1)
        DalalTriggsHOGdescriptor(windowImage, this->numberOfOrientationBins,
                                 this->cellHeightAndWidthInPixels,
                                 this->blockHeightAndWidthInCells,
                                 this->enableSignedGradients,
                                 this->l2normClipping, this->windowHeight,
                                 this->windowWidth, this->numberOfChannels,
                                 descriptorVector);
    else
        ZhuRamananHOGdescriptor(windowImage, this->cellHeightAndWidthInPixels,
                                this->windowHeight, this->windowWidth,
                                this->numberOfChannels, descriptorVector);
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

__global__ void DalalTriggsHOGdescriptor_precompute_histograms(double *d_h,
                                                               const dim3 h_dims,
                                                               const double *d_inputImage,
                                                               const unsigned int imageHeight,
                                                               const unsigned int imageWidth,
                                                               const unsigned int numberOfChannels,
                                                               const unsigned int numberOfOrientationBins,
                                                               const unsigned int cellHeightAndWidthInPixels,
                                                               const unsigned signedOrUnsignedGradients,
                                                               const double binsSize) {
    // Pre-compute histograms values
    // The array that contains "d_h" needs to be
    //  2*cellHeightAndWidthInPixels * 2*cellHeightAndWidthInPixels larger
    // Reduce kernel needs to be call in order to retrieve the expected
    // histogram
    
    // Retrieve pixel position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int factor_a_dim = h_dims.x * h_dims.y * h_dims.z;
    unsigned int factor_z_dim = h_dims.x * h_dims.y;
    unsigned int factor_y_dim = h_dims.x;
     
    // Check if position is inside the image
    if (x >= imageWidth || y >= imageHeight)
        return;
    
    // Compute deltas
    double dx[3], dy[3];
    
    if (x == 0) {
        for (unsigned int z = 0; z < numberOfChannels; z++)
            dx[z] = d_inputImage[y + (x + 1) * imageHeight + z * imageHeight * imageWidth];
    } else {
        if (x == imageWidth - 1) {
            for (unsigned int z = 0; z < numberOfChannels; z++)
                dx[z] = -d_inputImage[y + (x - 1) * imageHeight + z * imageHeight * imageWidth];
        } else {
            for (unsigned int z = 0; z < numberOfChannels; z++)
                dx[z] = d_inputImage[y + (x + 1) * imageHeight + z * imageHeight * imageWidth] - d_inputImage[y + (x - 1) * imageHeight + z * imageHeight * imageWidth];
        }
    }

    if(y == 0) {
        for (unsigned int z = 0; z < numberOfChannels; z++)
            dy[z] = -d_inputImage[y + 1 + x * imageHeight + z * imageHeight * imageWidth];
    } else {
        if (y == imageHeight - 1) {
            for (unsigned int z = 0; z < numberOfChannels; z++)
                dy[z] = d_inputImage[y - 1 + x * imageHeight + z * imageHeight * imageWidth];
        } else {
            for (unsigned int z = 0; z < numberOfChannels; z++)
                dy[z] = -d_inputImage[y + 1 + x * imageHeight + z * imageHeight * imageWidth] + d_inputImage[y - 1 + x * imageHeight + z * imageHeight * imageWidth];
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
    
    d_h[y1 + x1*factor_y_dim + bin1*factor_z_dim + a*factor_a_dim] = gradientMagnitude *
                                                        (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                        (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                        (1-((gradientOrientation-Oc)/binsSize));
    d_h[y1 + x1*factor_y_dim + bin2*factor_z_dim + a*factor_a_dim] = gradientMagnitude *
                                                        (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                        (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                        (((gradientOrientation-Oc)/binsSize));
    d_h[y2 + x1*factor_y_dim + bin1*factor_z_dim + a*factor_a_dim] = gradientMagnitude *
                                                        (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                        (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                        (1-((gradientOrientation-Oc)/binsSize));
    d_h[y2 + x1*factor_y_dim + bin2*factor_z_dim + a*factor_a_dim] = gradientMagnitude *
                                                        (1-((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                        (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                        (((gradientOrientation-Oc)/binsSize));
    d_h[y1 + x2*factor_y_dim + bin1*factor_z_dim + a*factor_a_dim] = gradientMagnitude *
                                                        (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                        (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                        (1-((gradientOrientation-Oc)/binsSize));
    d_h[y1 + x2*factor_y_dim + bin2*factor_z_dim + a*factor_a_dim] = gradientMagnitude *
                                                        (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                        (1-((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                        (((gradientOrientation-Oc)/binsSize));
    d_h[y2 + x2*factor_y_dim + bin1*factor_z_dim + a*factor_a_dim] = gradientMagnitude *
                                                        (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                        (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                        (1-((gradientOrientation-Oc)/binsSize));
    d_h[y2 + x2*factor_y_dim + bin2*factor_z_dim + a*factor_a_dim] = gradientMagnitude *
                                                        (((x+1-Xc)/cellHeightAndWidthInPixels)) *
                                                        (((y+1-Yc)/cellHeightAndWidthInPixels)) *
                                                        (((gradientOrientation-Oc)/binsSize));
}

__global__ void DalalTriggsHOGdescriptor_reduce_histograms(double *d_h,
                                                           const dim3 h_dims,
                                                           const unsigned int cellHeightAndWidthInPixels) {
    // cache size has to be a power of 2
    // usually set to MAX_THREADS_1D
    extern __shared__ double cache[];
    
    // Compute factors
    unsigned int factor_a_dim = h_dims.x * h_dims.y * h_dims.z;
    unsigned int factor_z_dim = h_dims.x * h_dims.y;
    unsigned int factor_y_dim = h_dims.x;
    
    // Retrieve indice of the element
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

__global__ void DalalTriggsHOGdescriptor_compute_blocknorm(double *d_blockNorm,
                                                           const dim3 blockNorm_dims,
                                                           const double *d_h,
                                                           const dim3 h_dims,
                                                           const unsigned int numberOfOrientationBins,
                                                           const unsigned int blockHeightAndWidthInCells) {
    // 2D-reduce to compute d_blockNorm for every (x,y)
    
    // Size of shared memory must be blockDim.x*blockDim.y*blockDim.z
    // and a power of 2
    extern __shared__ double cache[];
    
    // Compute factors
    unsigned int factor_z_dim = h_dims.x * h_dims.y;
    unsigned int factor_y_dim = h_dims.x;
    
    // Retrieve indice of the element
    unsigned int x = blockIdx.x +1;
    unsigned int y = blockIdx.y +1;
    
    unsigned int i = threadIdx.x;
    unsigned int j = threadIdx.y;
    unsigned int k = threadIdx.z;
    unsigned int current_id = i + j*blockDim.x
                              + k*blockDim.x*blockDim.y;
    
    cache[current_id] = 0.;
    for (unsigned int i_=i ; i_ < blockHeightAndWidthInCells ; i_+=blockDim.x)
        for (unsigned int j_=j ; j_ < blockHeightAndWidthInCells ; j_+=blockDim.y)
            for (unsigned int k_=k ; k_ < numberOfOrientationBins ; k_+=blockDim.z)
                cache[current_id] += d_h[y+i_ + (x+j_) * factor_y_dim + k_ * factor_z_dim]
                                     * d_h[y+i_ + (x+j_) * factor_y_dim + k_ * factor_z_dim];
    
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
        d_blockNorm[x-1 + blockNorm_dims.x*(y-1)] = cache[0];
}

__global__ void DalalTriggsHOGdescriptor_compute_block(double *d_block,
                                                       const double *d_blockNorm,
                                                       const dim3 blockNorm_dims,
                                                       const double *d_h,
                                                       const dim3 h_dims,
                                                       const unsigned int numberOfOrientationBins,
                                                       const unsigned int blockHeightAndWidthInCells,
                                                       const double l2normClipping) {
    // Each thread has to compute one value of block[i,j,k,x,y]
    
    // Compute factors
    unsigned int factor_z_dim = h_dims.x * h_dims.y;
    unsigned int factor_y_dim = h_dims.x;
    
    // Retrieve indice of the element
    unsigned int x = blockIdx.x;
    unsigned int y = blockIdx.y;
    
    unsigned int i = threadIdx.x;
    if (x >= blockNorm_dims.x) {
        i += ((unsigned int) (x/blockNorm_dims.x)) * blockDim.x;
        x = x % blockNorm_dims.x;
    }
    x += 1;
    unsigned int j = threadIdx.y;
    if (y >= blockNorm_dims.y) {
        j += ((unsigned int) (y/blockNorm_dims.y)) * blockDim.y;
        y = y % blockNorm_dims.y;
    }
    y += 1;
    unsigned int k = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (i >= blockHeightAndWidthInCells || j >= blockHeightAndWidthInCells
        || k >= numberOfOrientationBins)
        return;
    
    unsigned int current_id = i + j*blockHeightAndWidthInCells
                              + k*blockHeightAndWidthInCells*blockHeightAndWidthInCells
                              + (x-1)*blockHeightAndWidthInCells*blockHeightAndWidthInCells
                                *numberOfOrientationBins
                              + (y-1)*blockHeightAndWidthInCells*blockHeightAndWidthInCells
                                *numberOfOrientationBins*blockNorm_dims.x;
    
    
    double blockNorm = sqrt(d_blockNorm[x-1 + blockNorm_dims.x*(y-1)]);
    if (blockNorm > 0) {
        double tmpValue = d_h[y+i + (x+j) * factor_y_dim + k * factor_z_dim] / blockNorm;
        if (tmpValue > l2normClipping)
            d_block[current_id] = l2normClipping;
        else
            d_block[current_id] = tmpValue;
    } else
        d_block[current_id] = 0.;
}

__global__ void DalalTriggsHOGdescriptor_compute_blocknorm2(double *d_blockNorm,
                                                            const dim3 blockNorm_dims,
                                                            const double *d_block,
                                                            const unsigned int numberOfOrientationBins,
                                                            const unsigned int blockHeightAndWidthInCells) {
    // 2D-reduce to compute d_blockNorm for every (x,y)
    
    // Size of shared memory must be blockDim.x*blockDim.y*blockDim.z
    // and a power of 2
    extern __shared__ double cache[];
    
    // Retrieve indice of the element
    unsigned int x = blockIdx.x +1;
    unsigned int y = blockIdx.y +1;
    
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
                                *numberOfOrientationBins*blockNorm_dims.x;
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
        d_blockNorm[x-1 + blockNorm_dims.x*(y-1)] = cache[0];
}

// DALAL & TRIGGS: Histograms of Oriented Gradients for Human Detection
void DalalTriggsHOGdescriptor(double *inputImage,
                              unsigned int numberOfOrientationBins,
                              unsigned int cellHeightAndWidthInPixels,
                              unsigned int blockHeightAndWidthInCells,
                              bool signedOrUnsignedGradientsBool,
                              double l2normClipping, unsigned int imageHeight,
                              unsigned int imageWidth,
                              unsigned int numberOfChannels,
                              double *descriptorVector) {
   
    // ** VARIABLES **
    //  * Compute gradients & Compute histograms
    
    const int hist1 = 2 + (imageHeight / cellHeightAndWidthInPixels);
    const int hist2 = 2 + (imageWidth / cellHeightAndWidthInPixels);
    const dim3 h_dims(hist1, hist2, numberOfOrientationBins);
    double *d_h = NULL, *d_inputImage = NULL;
    const dim3 dimBlock(MAX_THREADS_2D, MAX_THREADS_2D, 1);
    const dim3 dimGrid((imageWidth + dimBlock.x -1)/dimBlock.x, (imageHeight + dimBlock.y -1)/dimBlock.y, 1);
    const unsigned int numCopies_d_h = (2*cellHeightAndWidthInPixels) * (2*cellHeightAndWidthInPixels);
     
    // each block is responsible to evaluate the value of
    // d_h[blockIdx.x][blockIdx.y][blockIdx.z]
    // the reduce operation concerns (2*cellHeightAndWidthInPixels) * (2*cellHeightAndWidthInPixels)
    // elements
    const dim3 dimGrid_reduce(h_dims.x, h_dims.y, h_dims.z);
    
    //  * Block normalization
    
    int descriptorIndex(0);
    
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
    double *d_blockNorm = NULL;
    const dim3 blockNorm_dims(hist2 - blockHeightAndWidthInCells -1,
                              hist1 - blockHeightAndWidthInCells -1);
    const dim3 dimBlock_norm(MAX_THREADS_3DX, MAX_THREADS_3DY, MAX_THREADS_3DZ);
    const dim3 dimGrid_norm(blockNorm_dims.x, blockNorm_dims.y, 1);
    double h_blockNorm[blockNorm_dims.x * blockNorm_dims.y];
    
    // Each thread has to compute one value of block[i,j,k,x,y]
    // The corresponding is stored into
    //   unsigned int current_id = i + j*blockHeightAndWidthInCells
    //      + k*blockHeightAndWidthInCells*blockHeightAndWidthInCells
    //      + (x-1)*blockHeightAndWidthInCells*blockHeightAndWidthInCells
    //        *numberOfOrientationBins
    //      + (y-1)*blockHeightAndWidthInCells*blockHeightAndWidthInCells
    //        *numberOfOrientationBins*blockNorm_dims.x;
    //   d_block[current_id]
    // The number of blocks in the grid depends on blockNorm_dims,
    // numberOfOrientationBins and blockHeightAndWidthInCells
    double *d_block = NULL;
    double h_block[cellHeightAndWidthInPixels * cellHeightAndWidthInPixels
                   * numberOfOrientationBins * blockNorm_dims.x
                   * blockNorm_dims.y];
    const dim3 dimBlock_block(MAX_THREADS_3DX, MAX_THREADS_3DY, MAX_THREADS_3DZ);
    const dim3 dimGrid_block(
            blockNorm_dims.x
            * ((blockHeightAndWidthInCells+MAX_THREADS_3DX-1)/MAX_THREADS_3DX),
            blockNorm_dims.y
            * ((blockHeightAndWidthInCells+MAX_THREADS_3DY-1)/MAX_THREADS_3DY),
            (numberOfOrientationBins + MAX_THREADS_3DZ -1)/MAX_THREADS_3DZ);
    
    // ** GRADIENTS and HISTOGRAMS **
    // Compute gradients (zero padding)
    // Compute histograms
    //  using CUDA
    
    // d_h is set numCopies_d_h times larger in order to benefit from reduce pattern
    cudaErrorCheck_goto(cudaMalloc(&d_h, h_dims.x * h_dims.y * h_dims.z * numCopies_d_h * sizeof(double)));
    cudaErrorCheck_goto(cudaMemset(d_h, 0., h_dims.x * h_dims.y * h_dims.z * numCopies_d_h * sizeof(double)));
    
    cudaErrorCheck_goto(cudaMalloc(&d_inputImage, imageHeight * imageWidth * numberOfChannels * sizeof(double)));
    cudaErrorCheck_goto(cudaMemcpy(d_inputImage, inputImage, imageHeight * imageWidth * numberOfChannels * sizeof(double), cudaMemcpyHostToDevice));
    
    DalalTriggsHOGdescriptor_precompute_histograms<<<dimGrid, dimBlock>>>(d_h, h_dims,
                                                                          d_inputImage, imageHeight, imageWidth, numberOfChannels,
                                                                          numberOfOrientationBins, cellHeightAndWidthInPixels,
                                                                          signedOrUnsignedGradientsBool ? 1 : 0 /*signedOrUnsignedGradients*/,
                                                                          (1 + (signedOrUnsignedGradientsBool ? 1 : 0)) * pi / numberOfOrientationBins /*binsSize*/);
    cudaErrorCheck_goto(cudaThreadSynchronize()); // block until the device is finished
    
    DalalTriggsHOGdescriptor_reduce_histograms<<<dimGrid_reduce,
                                                 MAX_THREADS_1D,
                                                 MAX_THREADS_1D*sizeof(double)>>>
                                                    (d_h, h_dims,
                                                     cellHeightAndWidthInPixels);
    cudaErrorCheck_goto(cudaThreadSynchronize()); // block until the device is finished
    
    cudaErrorCheck_goto(cudaFree(d_inputImage));
    d_inputImage = NULL;
    
    // ** BLOCK NORMALIZATION **
    
    // Evaluate blockNorm based on d_h
    cudaErrorCheck_goto(cudaMalloc(&d_blockNorm, blockNorm_dims.x * blockNorm_dims.y * sizeof(double)));
    cudaErrorCheck_goto(cudaMemset(d_blockNorm, 0., blockNorm_dims.x * blockNorm_dims.y * sizeof(double)));
    
    DalalTriggsHOGdescriptor_compute_blocknorm<<<dimGrid_norm,
                                                 dimBlock_norm,
                                                 MAX_THREADS_3DX
                                                 * MAX_THREADS_3DY
                                                 * MAX_THREADS_3DZ
                                                 * sizeof(double)>>>
                                                    (d_blockNorm, blockNorm_dims,
                                                     d_h, h_dims,
                                                     numberOfOrientationBins,
                                                     cellHeightAndWidthInPixels);
    cudaErrorCheck_goto(cudaThreadSynchronize()); // block until the device is finished
    
    // Compute block[i,j,k,x,y]
    cudaErrorCheck_goto(cudaMalloc(&d_block, cellHeightAndWidthInPixels
                                             * cellHeightAndWidthInPixels
                                             * numberOfOrientationBins
                                             * blockNorm_dims.x
                                             * blockNorm_dims.y
                                             * sizeof(double)));
    
    DalalTriggsHOGdescriptor_compute_block<<<dimGrid_block,
                                             dimBlock_block>>>
                                                (d_block,
                                                 d_blockNorm, blockNorm_dims,
                                                 d_h, h_dims,
                                                 numberOfOrientationBins,
                                                 blockHeightAndWidthInCells,
                                                 l2normClipping);
    cudaErrorCheck_goto(cudaThreadSynchronize());
    
    cudaErrorCheck_goto(cudaFree(d_h));
    d_h = NULL;
    
    // Evaluate blockNorm based on d_block
    DalalTriggsHOGdescriptor_compute_blocknorm2<<<dimGrid_norm,
                                                  dimBlock_norm,
                                                  MAX_THREADS_3DX
                                                  * MAX_THREADS_3DY
                                                  * MAX_THREADS_3DZ
                                                  * sizeof(double)>>>
                                                    (d_blockNorm, blockNorm_dims,
                                                     d_block,
                                                     numberOfOrientationBins,
                                                     blockHeightAndWidthInCells);
    cudaErrorCheck_goto(cudaThreadSynchronize()); // block until the device is finished
    
    // Copy to CPU
    cudaErrorCheck_goto(cudaMemcpy(h_blockNorm, d_blockNorm,
                                   blockNorm_dims.x * blockNorm_dims.y * sizeof(double),
                                   cudaMemcpyDeviceToHost));
    
    cudaErrorCheck_goto(cudaMemcpy(h_block, d_block,
                                   cellHeightAndWidthInPixels
                                   * cellHeightAndWidthInPixels
                                   * numberOfOrientationBins
                                   * blockNorm_dims.x * blockNorm_dims.y
                                   * sizeof(double),
                                   cudaMemcpyDeviceToHost));
    
    cudaErrorCheck_goto(cudaFree(d_blockNorm));
    d_blockNorm = NULL;
    cudaErrorCheck_goto(cudaFree(d_block));
    d_block = NULL;
    
    for (unsigned int x = 1; x < hist2 - blockHeightAndWidthInCells; x++) {
        for (unsigned int y = 1; y < hist1 - blockHeightAndWidthInCells; y++) {
            float blockNorm(sqrt(h_blockNorm[x-1 + blockNorm_dims.x*(y-1)]));
            for (unsigned int i = 0; i < blockHeightAndWidthInCells; i++) {
                for (unsigned int j = 0; j < blockHeightAndWidthInCells; j++) {
                    for (unsigned int k = 0; k < numberOfOrientationBins; k++) {
                        unsigned int current_id = i + j*blockHeightAndWidthInCells
                            + k*blockHeightAndWidthInCells*blockHeightAndWidthInCells
                            + (x-1)*blockHeightAndWidthInCells*blockHeightAndWidthInCells
                                *numberOfOrientationBins
                            + (y-1)*blockHeightAndWidthInCells*blockHeightAndWidthInCells
                                *numberOfOrientationBins*blockNorm_dims.x;
                        if (blockNorm > 0)
                            descriptorVector[descriptorIndex] =
                                h_block[current_id] / blockNorm;
                        else
                            descriptorVector[descriptorIndex] = 0.0;
                        descriptorIndex++;
                    }
                }
            }
        }
    }
    return;

onfailure:
    if (d_h != NULL)
        cudaFree(d_h);
    if (d_blockNorm != NULL)
        cudaFree(d_blockNorm);
    if (d_block != NULL)
        cudaFree(d_block);
    if (d_inputImage != NULL)
        cudaFree(d_inputImage);
    
    return;
}
