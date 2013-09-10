#include <iostream>
#include "HOG.h"
#include "LBP.h"
#include "ImageWindowIterator.h"
#include <stdlib.h>

int main()
{
	unsigned int imageWidth = 10, imageHeight = 10;
	unsigned int windowWidth = 3, windowHeight = 3;
	unsigned int windowStepHorizontal = 1, windowStepVertical = 1;
	bool enablePadding = false;
	bool imageIsGrayscale = true;

	double *image;
	if (imageIsGrayscale==false) {
		unsigned int i1, i2, k;
		image = (double *) malloc(imageHeight*imageWidth*3*sizeof(double));
		for (i1 = 0; i1 < imageHeight; i1++)
			for (i2 = 0; i2 < imageWidth; i2++)
				for (k = 0; k < 3; k++)
					image[i1+imageHeight*(i2+imageWidth*k)] = rand()% 100 + 1;
	}
	else {
		unsigned int i1, i2;
		image = (double *) malloc(imageHeight*imageWidth*sizeof(double));
		for (i1 = 0; i1 < imageHeight; i1++)
			for (i2 = 0; i2 < imageWidth; i2++)
					image[i1+imageHeight*i2] = rand()% 100 + 1;
	}

	HOG windowFeature = HOG();
	//LBP windowFeature = LBP();

	/*HOG windowFeature = HOG();
	double a = 6;
	windowFeature.apply(&a, 4, 2);

	LBP windowFeature2 = LBP();
	double aa = 6;
	windowFeature2.apply(&aa, 4, 2);*/

	ImageWindowIterator iter = ImageWindowIterator(image, imageHeight, imageWidth, windowHeight, windowWidth, windowStepHorizontal, windowStepVertical, enablePadding, imageIsGrayscale, &windowFeature);
	iter.print_information();
	iter.apply();

	return 0;
}
