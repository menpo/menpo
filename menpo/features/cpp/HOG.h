#pragma once
#include "WindowFeature.h"
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

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }
static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }

class HOG: public WindowFeature {
public:
	HOG(unsigned int windowHeight, unsigned int windowWidth,
	    unsigned int numberOfChannels, unsigned int method,
	    unsigned int numberOfOrientationBins,
	    unsigned int cellHeightAndWidthInPixels,
	    unsigned int blockHeightAndWidthInCells, bool enableSignedGradients,
	    double l2normClipping);
	virtual ~HOG();
	void apply(double *windowImage, double *descriptorVector);
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
void DalalTriggsHOGdescriptor(double *inputImage,
                              unsigned int numberOfOrientationBins,
                              unsigned int cellHeightAndWidthInPixels,
                              unsigned int blockHeightAndWidthInCells,
                              bool signedOrUnsignedGradientsBool,
                              double l2normClipping, unsigned int imageHeight,
                              unsigned int imageWidth,
                              unsigned int numberOfChannels,
                              double *descriptorVector);
