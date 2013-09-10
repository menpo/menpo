#include <iostream>
#include "HOG.h"
#include "LBP.h"
#include "ImageWindowIterator.h"
#include <stdlib.h>
#include <stdio.h>

int main()
{
	// DEFINE WINDOWS-RELATED OPTIONS
	unsigned int imageWidth = 100, imageHeight = 100;
	unsigned int windowWidth = 16, windowHeight = 16;
	unsigned int windowStepHorizontal = 1, windowStepVertical = 1;
	bool enablePadding = false;
	bool imageIsGrayscale = true;

	// DEFINE FEATURE-RELATED OPTIONS
	unsigned int sparseOrDense = 2; // 1 for sparse, 2 for dense
	unsigned int method = 2; // 1 dalaltriggs, 2 for zhuramanan
	unsigned int numberOfOrientationBins = 9;
	unsigned int cellHeightAndWidthInPixels = 4;
	unsigned int blockHeightAndWidthInCells = 2;
	bool enableSignedGradients = true;
    double l2normClipping = 0.2;

    // VERBOSE
    bool verbose = true;

    ///////////////////////////////////////////////////////////////////////////

	// INITIALIZE INPUT IMAGE WITH RAND
    unsigned int i1, i2, k;
	double *image;
	if (imageIsGrayscale==false) {
		image = (double *) malloc(imageHeight*imageWidth*3*sizeof(double));
		for (i1 = 0; i1 < imageHeight; i1++)
			for (i2 = 0; i2 < imageWidth; i2++)
				for (k = 0; k < 3; k++)
					image[i1+imageHeight*(i2+imageWidth*k)] = rand()% 100 + 1;
	}
	else {
		image = (double *) malloc(imageHeight*imageWidth*sizeof(double));
		for (i1 = 0; i1 < imageHeight; i1++)
			for (i2 = 0; i2 < imageWidth; i2++)
					image[i1+imageHeight*i2] = rand()% 100 + 1;
	}

	// CREATE WINDOW FEATURE
	HOG windowFeature = HOG(windowHeight, windowWidth, method, numberOfOrientationBins, cellHeightAndWidthInPixels, blockHeightAndWidthInCells, enableSignedGradients, l2normClipping);
	//LBP windowFeature = LBP();

	// CREATE ITERATOR
	ImageWindowIterator iter = ImageWindowIterator(image, imageHeight, imageWidth, windowHeight, windowWidth, windowStepHorizontal, windowStepVertical, enablePadding, imageIsGrayscale, &windowFeature);

	// CREATE OUTPUT IMAGE
	double *outputImage;
	outputImage = (double *) malloc(iter.numberOfWindowsVertically*iter.numberOfWindowsHorizontally*windowFeature.descriptorLengthPerWindow*sizeof(double));

	// CREATE WINDOWS CENTERS
	int *windowsCenters;
	windowsCenters = (int *) malloc(iter.numberOfWindowsVertically*iter.numberOfWindowsHorizontally*2*sizeof(int));

	// CALL ITERATOR
	iter.apply(outputImage, windowsCenters);

	// VERBOSE
	if (verbose==true)
		windowFeature.print_information();
	if (verbose==true)
		iter.print_information();

	return 0;
}
