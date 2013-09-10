#pragma once
#include "WindowFeature.h"

class ImageWindowIterator {
public:
	unsigned int numberOfWindowsHorizontally, numberOfWindowsVertically, numberOfWindows;
	ImageWindowIterator(double *image, unsigned int imageHeight, unsigned int imageWidth,
			unsigned int windowHeight, unsigned int windowWidth, unsigned int windowStepHorizontal,
			unsigned int windowStepVertical, bool enablePadding, bool imageIsGrayscale, WindowFeature *windowFeature);
	virtual ~ImageWindowIterator();
	void apply(double *outputImage, int *windowsCenters);
	void print_information();
private:
	double *image;
	unsigned int imageWidth, imageHeight;
	WindowFeature *windowFeature;
    unsigned int windowHeight, windowWidth;
    unsigned int windowStepHorizontal, windowStepVertical;
    bool enablePadding, imageIsGrayscale;
};
