#include <iostream>
#include "HOG.h"
#include "LBP.h"
#include "ImageWindowIterator.h"
#include <stdlib.h>

int main(int argc, char** argv)
{
	unsigned int imageWidth = 640, imageHeight = 480;
	unsigned int windowWidth = 16, windowHeight = 16;
	unsigned int windowStepHorizontal = 1, windowStepVertical = 1;
	bool enablePadding = true;
	bool imageIsGrayscale = true;

	double image = 6;
	//double image[imageWidth][imageHeight];
	//unsigned int i1, i2;

	//for (i1 = 0; i1 < imageWidth; i1++) {
	//	for (i2 = 0; i2 < imageHeight; i2++) {
	//		image[i1][i2] = rand()% 100 + 1;
	//		}
	//	}

	HOG windowFeature = HOG();
	//LBP windowFeature = LBP();

	/*HOG windowFeature = HOG();
	double a = 6;
	windowFeature.apply(&a, 4, 2);

	LBP windowFeature2 = LBP();
	double aa = 6;
	windowFeature2.apply(&aa, 4, 2);*/

	ImageWindowIterator iter = ImageWindowIterator(&image, imageWidth, imageHeight, windowHeight, windowWidth, windowStepHorizontal, windowStepVertical, enablePadding, imageIsGrayscale, &windowFeature);
	iter.apply();

	return 0;
}
