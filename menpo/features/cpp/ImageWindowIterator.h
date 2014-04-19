#pragma once
#include "WindowFeature.h"

#if defined(_WIN32)
//    inline int round(float x) { return floor(x + 0.5); }
//    inline int round(double x) { return floor(x + 0.5); }
#endif

class ImageWindowIterator {
public:
	unsigned int _numberOfWindowsHorizontally, _numberOfWindowsVertically, _numberOfWindows;
	unsigned int _imageWidth, _imageHeight, _numberOfChannels;
    unsigned int _windowHeight, _windowWidth;
    unsigned int _windowStepHorizontal, _windowStepVertical;
    bool _enablePadding;
	ImageWindowIterator(double *image, unsigned int imageHeight, unsigned int imageWidth, unsigned int numberOfChannels,
	        unsigned int windowHeight, unsigned int windowWidth, unsigned int windowStepHorizontal,
			unsigned int windowStepVertical, bool enablePadding);
	virtual ~ImageWindowIterator();
	void apply(double *outputImage, int *windowsCenters, WindowFeature *windowFeature);
private:
	double *_image;
};
