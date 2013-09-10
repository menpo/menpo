#pragma once
#include "WindowFeature.h"

class HOG: public WindowFeature {
public:
	HOG(unsigned int windowHeight, unsigned int windowWidth, unsigned int method, unsigned int numberOfOrientationBins, unsigned int cellHeightAndWidthInPixels,
			unsigned int blockHeightAndWidthInCells, bool enableSignedGradients, double l2normClipping);
	virtual ~HOG();
	void apply(double *windowImage, unsigned int windowHeight, unsigned int windowWidth, bool imageIsGrayscale);
	void print_information();
	unsigned int descriptorLengthPerWindow;
private:
    unsigned int method, numberOfOrientationBins, cellHeightAndWidthInPixels, blockHeightAndWidthInCells;
    bool enableSignedGradients;
    double l2normClipping;
    unsigned int numberOfBlocksPerWindowHorizontally, numberOfBlocksPerWindowVertically, descriptorLengthPerBlock;
};
