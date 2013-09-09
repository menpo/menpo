#include "ImageWindowIterator.h"
#include <iostream>

ImageWindowIterator::ImageWindowIterator(double *image, unsigned int imageWidth,
		unsigned int imageHeight, WindowFeature *windowFeature)
{
	this->image = image;
	this->imageHeight = imageHeight;
	this->imageWidth = imageWidth;
	this->windowFeature = windowFeature;

}

ImageWindowIterator::~ImageWindowIterator() {
	// TODO Auto-generated destructor stub
}

void ImageWindowIterator::apply()
{
	std::cout << "Hello from iterator's apply" << std::endl;
	this->windowFeature->apply(this->image, this->imageWidth, this->imageHeight);
}
