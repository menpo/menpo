#pragma once
#include "WindowFeature.h"

class ImageWindowIterator {
public:
	ImageWindowIterator(double *image, unsigned int imageWidth, unsigned int imageHeight, WindowFeature *windowFeature);
	virtual ~ImageWindowIterator();
	void apply();
private:
	double *image;
	unsigned int imageWidth, imageHeight;
	WindowFeature *windowFeature;
};
