#pragma once
#include "WindowFeature.h"

class ImageWindowIterator {
public:
	ImageWindowIterator(double *image, unsigned int imageWidth, unsigned int imageHeight,
			unsigned int windowHeight, unsigned int windowWidth, unsigned int windowStepHorizontal,
			unsigned int windowStepVertical, bool enablePadding, bool imageIsGrayscale, WindowFeature *windowFeature);
	virtual ~ImageWindowIterator();
	void apply();
private:
	double *image;
	unsigned int imageWidth, imageHeight;
	WindowFeature *windowFeature;
    unsigned int numberOfWindowsHorizontally, numberOfWindowsVertically, numberOfWindows;
    unsigned int windowHeight, windowWidth;
    unsigned int windowStepHorizontal, windowStepVertical;
    bool enablePadding, imageIsGrayscale;
};
