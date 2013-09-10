#include "HOG.h"
#include <iostream>
#include <cmath>
#define _USE_MATH_DEFINES

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }
static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }

HOG::HOG(unsigned int windowHeight, unsigned int windowWidth, unsigned int method, unsigned int numberOfOrientationBins, unsigned int cellHeightAndWidthInPixels,
		unsigned int blockHeightAndWidthInCells, bool enableSignedGradients, double l2normClipping) {
	double binsSize;
	unsigned int descriptorLengthPerBlock, hist1, hist2, descriptorLengthPerWindow, numberOfBlocksPerWindowVertically, numberOfBlocksPerWindowHorizontally;

	// Initialize descriptor vector/matrix
	if (enableSignedGradients == true)
		binsSize = 2*M_PI/numberOfOrientationBins;
	else
		binsSize = M_PI/numberOfOrientationBins;

    if (method==1)
    {
        descriptorLengthPerBlock = blockHeightAndWidthInCells*blockHeightAndWidthInCells*numberOfOrientationBins;
        hist1 = 2 + ceil(-0.5 + windowHeight/cellHeightAndWidthInPixels);
        hist2 = 2 + ceil(-0.5 + windowWidth/cellHeightAndWidthInPixels);
        descriptorLengthPerWindow = (hist1-2-(blockHeightAndWidthInCells-1))*(hist2-2-(blockHeightAndWidthInCells-1))*descriptorLengthPerBlock;
        // both ways of calculating number of blocks are equal
        //numberOfBlocksPerWindowVertically = 1+floor((windowHeight-blockHeightAndWidthInCells*cellHeightAndWidthInPixels)/cellHeightAndWidthInPixels);
        //numberOfBlocksPerWindowHorizontally = 1+floor((windowWidth-blockHeightAndWidthInCells*cellHeightAndWidthInPixels)/cellHeightAndWidthInPixels);
        numberOfBlocksPerWindowVertically = hist1-2-(blockHeightAndWidthInCells-1);
        numberOfBlocksPerWindowHorizontally = hist2-2-(blockHeightAndWidthInCells-1);
    }
    else if (method==2)
    {
        hist1 = (unsigned int)round((double)windowHeight/(double)cellHeightAndWidthInPixels);
        hist2 = (unsigned int)round((double)windowWidth/(double)cellHeightAndWidthInPixels);
        numberOfBlocksPerWindowVertically = max(hist1-2,0); //You can change this to out[0] = max(hist1-1,0); and out[1] = max(hist2-1,0), in order to return the same output size as dalaltriggs
        numberOfBlocksPerWindowHorizontally = max(hist2-2,0); //You can do the same in lines 1361,1362
        descriptorLengthPerBlock = 27+4;
        descriptorLengthPerWindow = numberOfBlocksPerWindowHorizontally*numberOfBlocksPerWindowVertically*descriptorLengthPerBlock;
    }

    this->method = method;
    this->numberOfOrientationBins = numberOfOrientationBins;
    this->cellHeightAndWidthInPixels = cellHeightAndWidthInPixels;
    this->blockHeightAndWidthInCells = blockHeightAndWidthInCells;
    this->enableSignedGradients = enableSignedGradients;
    this->l2normClipping = l2normClipping;
    this->numberOfBlocksPerWindowHorizontally = numberOfBlocksPerWindowHorizontally;
    this->numberOfBlocksPerWindowVertically = numberOfBlocksPerWindowVertically;
    this->descriptorLengthPerBlock = descriptorLengthPerBlock;
    this->descriptorLengthPerWindow = descriptorLengthPerWindow;
}

HOG::~HOG() {

}

void HOG::print_information() {
	std::cout << std::endl << "HOG options" << std::endl;
	if (this->method==1) {
		std::cout << "Method of Dalal & Triggs" << std::endl;
		std::cout << "Cell = " << this->cellHeightAndWidthInPixels << "x" << this->cellHeightAndWidthInPixels << " pixels" << std::endl;
		std::cout << "Block = " << this->blockHeightAndWidthInCells << "x" << this->blockHeightAndWidthInCells << " cells" << std::endl;
		if (this->enableSignedGradients == true)
			std::cout << this->numberOfOrientationBins << " orientation bins and signed gradients" << std::endl;
		else
			std::cout << this->numberOfOrientationBins << " orientation bins and unsigned gradients" << std::endl;
		std::cout << "L2-norm clipped at " << this->l2normClipping << std::endl;
		std::cout << "Number of blocks per window = " << this->numberOfBlocksPerWindowVertically << "x" << this->numberOfBlocksPerWindowHorizontally << std::endl;
		std::cout << "Descriptor length per window = " << this->numberOfBlocksPerWindowVertically << "x" << this->numberOfBlocksPerWindowHorizontally << "x" << this->descriptorLengthPerBlock << " = " << this->descriptorLengthPerWindow << std::endl;
	}
	else {
		std::cout << "Method of Zhu & Ramanan" << std::endl;
		std::cout << "Cell = " << this->cellHeightAndWidthInPixels << "x" << this->cellHeightAndWidthInPixels << " pixels" << std::endl;
		std::cout << "Number of blocks per window = " << this->numberOfBlocksPerWindowVertically << "x" << this->numberOfBlocksPerWindowHorizontally << std::endl;
		std::cout << "Descriptor length per window = " << this->numberOfBlocksPerWindowVertically << "x" << this->numberOfBlocksPerWindowHorizontally << "x" << this->descriptorLengthPerBlock << " = " << this->descriptorLengthPerWindow << std::endl;
	}
}

void HOG::apply(double *windowImage, unsigned int windowHeight, unsigned int windowWidth, bool imageIsGrayscale)
{
	int i, j, k;
	double s = 0;

	std::cout << "Hello from HOG's apply" << std::endl;
	if (imageIsGrayscale == false)
	{
		for (i = 0; i < windowHeight; i++)
			for (j = 0; j < windowHeight; j++)
				for (k = 0; k < 3; k++)
					s = s + windowImage[i+windowHeight*(j+windowWidth*k)];
		std::cout << "SUM = " << s << std::endl;
	}
	else {
		for (i = 0; i < windowHeight; i++)
			for (j = 0; j < windowHeight; j++)
				s = s + windowImage[i+windowHeight*j];
		std::cout << "SUM = " << s << std::endl;
	}

}
