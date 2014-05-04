#include "ImageWindowIterator.h"
#include <iostream>
#include <math.h>
#include <stdlib.h>

ImageWindowIterator::ImageWindowIterator(double *image, unsigned int imageHeight, unsigned int imageWidth, unsigned int numberOfChannels,
		unsigned int windowHeight, unsigned int windowWidth, unsigned int windowStepHorizontal,
		unsigned int windowStepVertical, bool enablePadding) {
    unsigned int numberOfWindowsHorizontally, numberOfWindowsVertically;

    // Find number of windows
    if (!enablePadding) {
        numberOfWindowsHorizontally = 1 + (imageWidth - windowWidth) / windowStepHorizontal;
        numberOfWindowsVertically = 1 + (imageHeight - windowHeight) / windowStepVertical;
    }
    else {
        numberOfWindowsHorizontally = 1 + ((imageWidth - 1) / windowStepHorizontal);
        numberOfWindowsVertically = 1 + ((imageHeight - 1) / windowStepVertical);
    }

	this->_image = image;
	this->_imageHeight = imageHeight;
	this->_imageWidth = imageWidth;
	this->_numberOfChannels = numberOfChannels;
	this->_windowHeight = windowHeight;
	this->_windowWidth = windowWidth;
	this->_windowStepHorizontal = windowStepHorizontal;
	this->_windowStepVertical = windowStepVertical;
	this->_enablePadding = enablePadding;
	this->_numberOfWindowsHorizontally = numberOfWindowsHorizontally;
	this->_numberOfWindowsVertically = numberOfWindowsVertically;
}

ImageWindowIterator::~ImageWindowIterator() {
}


void ImageWindowIterator::apply(double *outputImage, int *windowsCenters, WindowFeature *windowFeature) {
	int rowCenter, rowFrom, rowTo, columnCenter, columnFrom, columnTo, i, j, k;
	unsigned int windowIndexHorizontal, windowIndexVertical, d;
	int imageHeight = (int)_imageHeight;
	int imageWidth = (int)_imageWidth;
	int numberOfChannels = (int)_numberOfChannels;

    // Initialize temporary matrices
	double* windowImage = new double[_windowHeight*_windowWidth*_numberOfChannels];
	double* descriptorVector = new double[windowFeature->descriptorLengthPerWindow];

    // Main loop
    for (windowIndexVertical = 0; windowIndexVertical < _numberOfWindowsVertically; windowIndexVertical++) {
        for (windowIndexHorizontal = 0; windowIndexHorizontal < _numberOfWindowsHorizontally; windowIndexHorizontal++) {
            // Find window limits
            if (!_enablePadding) {
                rowFrom = windowIndexVertical*_windowStepVertical;
                rowTo = rowFrom + _windowHeight - 1;
                rowCenter = rowFrom + (int)round((double)_windowHeight / 2.0) - 1;
                columnFrom = windowIndexHorizontal*_windowStepHorizontal;
                columnTo = columnFrom + _windowWidth - 1;
                columnCenter = columnFrom + (int)round((double)_windowWidth / 2.0) - 1;
            }
            else {
                rowCenter = windowIndexVertical*_windowStepVertical;
                rowFrom = rowCenter - (int)round((double)_windowHeight / 2.0) + 1;
                rowTo = rowFrom + _windowHeight - 1;
                columnCenter = windowIndexHorizontal*_windowStepHorizontal;
                columnFrom = columnCenter - (int)ceil((double)_windowWidth / 2.0) + 1;
                columnTo = columnFrom + _windowWidth - 1;
            }

            // Copy window image
			for (i = rowFrom; i <= rowTo; i++) {
				for (j = columnFrom; j <= columnTo; j++) {
					if (i < 0 || i > imageHeight-1 || j < 0 || j > imageWidth-1)
						for (k = 0; k < numberOfChannels; k++)
							windowImage[(i-rowFrom)+_windowHeight*((j-columnFrom)+_windowWidth*k)] = 0;
					else
						for (k=0; k < numberOfChannels; k++)
							windowImage[(i-rowFrom)+_windowHeight*((j-columnFrom)+_windowWidth*k)] = _image[i+imageHeight*(j+imageWidth*k)];
				}
			}

            // Compute descriptor of window
            windowFeature->apply(windowImage, descriptorVector);

            // Store results
            for (d = 0; d < windowFeature->descriptorLengthPerWindow; d++)
            	outputImage[windowIndexVertical+_numberOfWindowsVertically*(windowIndexHorizontal+_numberOfWindowsHorizontally*d)] = descriptorVector[d];
            windowsCenters[windowIndexVertical+_numberOfWindowsVertically*windowIndexHorizontal] = rowCenter;
            windowsCenters[windowIndexVertical+_numberOfWindowsVertically*(windowIndexHorizontal+_numberOfWindowsHorizontally)] = columnCenter;
        }
    }

    // Free temporary matrices
    delete[] windowImage;
    delete[] descriptorVector;
}

