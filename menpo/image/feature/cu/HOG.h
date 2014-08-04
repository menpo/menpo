#pragma once
#include "../cpp/WindowFeature.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <vector>
#include <string.h>

const float pi = 3.1415926536;

#define eps 0.0001 // small value, used to avoid division by zero

using namespace std;

class HOG: public WindowFeature {
public:
	HOG(unsigned int windowHeight, unsigned int windowWidth,
	    unsigned int numberOfChannels, unsigned int method,
	    unsigned int numberOfOrientationBins,
	    unsigned int cellHeightAndWidthInPixels,
	    unsigned int blockHeightAndWidthInCells, bool enableSignedGradients,
	    double l2normClipping);
	virtual ~HOG();
	void applyOnChunk(double *windowImage, double *descriptorVector);
    void applyOnImage(const ImageWindowIterator &iwi, const double *image,
                      double *outputImage, int *windowsCenters);
    bool isApplyOnImage();
    void DalalTriggsHOGdescriptorOnImage(const ImageWindowIterator &iwi,
                                         double *d_image,
                                         double *outputImage,
                                         int *windowsCenters);
	unsigned int descriptorLengthPerBlock, numberOfBlocksPerWindowHorizontally,
	             numberOfBlocksPerWindowVertically;
private:
    unsigned int method, numberOfOrientationBins, cellHeightAndWidthInPixels,
                 blockHeightAndWidthInCells, windowHeight, windowWidth,
                 numberOfChannels;
    bool enableSignedGradients;
    double l2normClipping;
};

void ZhuRamananHOGdescriptor(double *inputImage,
                             int cellHeightAndWidthInPixels,
                             unsigned int imageHeight, unsigned int imageWidth,
                             unsigned int numberOfChannels,
                             double *descriptorMatrix);

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
                              double *d_blockNorm, double *block,
                              double *d_outputImage,
                              unsigned int offsetOutputImage,
                              unsigned int factorOutputImage,
                              unsigned int windowId);

/* Kernels' signature declaration */

#ifdef __global__

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
                                                            const int windowStepVertical, const int windowStepHorizontal);

__global__ void DalalTriggsHOGdescriptor_compute_blocknorm(double *d_blockNorm,
                                                           const dim3 blockNorm_dims,
                                                           const double *d_h,
                                                           const dim3 h_dims,
                                                           const unsigned int numberOfOrientationBins,
                                                           const unsigned int blockHeightAndWidthInCells,
                                                           const unsigned int numberOfWindowsVertically);

__global__ void DalalTriggsHOGdescriptor_compute_block(double *d_block,
                                                       const double *d_blockNorm,
                                                       const dim3 blockNorm_dims,
                                                       const double *d_h,
                                                       const dim3 h_dims,
                                                       const unsigned int numberOfOrientationBins,
                                                       const unsigned int blockHeightAndWidthInCells,
                                                       const double l2normClipping,
                                                       const unsigned int numberOfWindowsVertically,
                                                       const unsigned int block_size);

__global__ void DalalTriggsHOGdescriptor_compute_blocknorm2(double *d_blockNorm,
                                                            const dim3 blockNorm_dims,
                                                            const double *d_block,
                                                            const unsigned int numberOfOrientationBins,
                                                            const unsigned int blockHeightAndWidthInCells,
                                                            const unsigned int numberOfWindowsVertically);

__global__ void DalalTriggsHOGdescriptor_compute_outputImage(double *d_outputImage,
                                                             const double *d_block,
                                                             const double *d_blockNorm,
                                                             const dim3 blockNorm_dims,
                                                             const unsigned int numberOfOrientationBins,
                                                             const unsigned int blockHeightAndWidthInCells,
                                                             const unsigned int numWindows,
                                                             const unsigned int numberOfWindowsVertically,
                                                             const unsigned int block_size);

#endif
