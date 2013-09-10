#include "ImageWindowIterator.h"
#include <iostream>
#include <cmath>
#include <stdlib.h>

ImageWindowIterator::ImageWindowIterator(double *image, unsigned int imageHeight, unsigned int imageWidth,
		unsigned int windowHeight, unsigned int windowWidth, unsigned int windowStepHorizontal,
		unsigned int windowStepVertical, bool enablePadding, bool imageIsGrayscale, WindowFeature *windowFeature) {
	unsigned int numberOfWindowsHorizontally, numberOfWindowsVertically, numberOfWindows, i;

    // Find number of windows
    if (enablePadding==false) {
        numberOfWindowsHorizontally = 1+floor((imageWidth-windowWidth)/windowStepHorizontal);
        numberOfWindowsVertically = 1+floor((imageHeight-windowHeight)/windowStepVertical);
    }
    else {
        numberOfWindowsHorizontally = ceil(imageWidth/windowStepHorizontal);
        numberOfWindowsVertically = ceil(imageHeight/windowStepVertical);
    }
    numberOfWindows = numberOfWindowsHorizontally*numberOfWindowsVertically;

	this->image = image;
	this->imageHeight = imageHeight;
	this->imageWidth = imageWidth;
	this->windowHeight = windowHeight;
	this->windowWidth = windowWidth;
	this->windowStepHorizontal = windowStepHorizontal;
	this->windowStepVertical = windowStepVertical;
	this->enablePadding = enablePadding;
	this->imageIsGrayscale = imageIsGrayscale;
	this->numberOfWindowsHorizontally = numberOfWindowsHorizontally;
	this->numberOfWindowsVertically = numberOfWindowsVertically;
	this->numberOfWindows = numberOfWindows;
	this->windowFeature = windowFeature;
}

ImageWindowIterator::~ImageWindowIterator() {
	// TODO Auto-generated destructor stub
}

void ImageWindowIterator::print_information() {
	std::cout << std::endl << "Image Window Iterator" << std::endl << "---------------------" << std::endl;
    if (this->imageIsGrayscale==true)
    	std::cout << "Input image is GRAY with size " << this->imageHeight << "x" << this->imageWidth << std::endl;
    else
    	std::cout << std::endl << "Image is RGB with size " << this->imageHeight << "x" << this->imageWidth << std::endl;
    std::cout << "Window of size " << this->windowHeight << "x" << this->windowWidth << " and step (" << this->windowStepHorizontal << "," << this->windowStepVertical << ")" << std::endl;
    if (this->enablePadding==true)
    	std::cout << "Padding is enabled" << std::endl;
    else
    	std::cout << "Padding is disabled" << std::endl;
    std::cout << "Number of windows is " << this->numberOfWindowsVertically << "x" << this->numberOfWindowsHorizontally << " = " << this->numberOfWindows << std::endl;
    std::cout << "Output image has size " << this->numberOfWindowsVertically << "x" << this->numberOfWindowsHorizontally << "x" << windowFeature->descriptorLengthPerWindow << std::endl << std::endl;
}

void ImageWindowIterator::apply(double *outputImage, int *windowsCenters) {
	unsigned int windowHeight, windowWidth, windowStepHorizontal, windowStepVertical, imageHeight, imageWidth;
	bool enablePadding, imageIsGrayscale;
	int rowCenter, rowFrom, rowTo, columnCenter, columnFrom, columnTo, i, j, k, d;
	unsigned int windowIndexHorizontal, windowIndexVertical;
	unsigned int numberOfWindowsHorizontally, numberOfWindowsVertically, numberOfWindows;

    // Load window size, image size, window step, padding
    windowHeight = this->windowHeight;
    windowWidth = this->windowWidth;
    imageHeight = this->imageHeight;
    imageWidth = this->imageWidth;
    windowStepHorizontal = this->windowStepHorizontal;
    windowStepVertical = this->windowStepVertical;
    numberOfWindowsHorizontally = this->numberOfWindowsHorizontally;
    numberOfWindowsVertically = this->numberOfWindowsVertically;
    numberOfWindows = this->numberOfWindows;
    enablePadding = this->enablePadding;
    imageIsGrayscale = this->imageIsGrayscale;

    // Initialize windowImage
    double* windowImage;
    if (imageIsGrayscale == true)
    	windowImage = (double *) malloc(imageHeight*imageWidth*sizeof(double));
    else
    	windowImage = (double *) malloc(imageHeight*imageWidth*3*sizeof(double));

    // Initialize descriptorVector
    double* descriptorVector;
    descriptorVector = (double *) malloc(windowFeature->descriptorLengthPerWindow*sizeof(double));

    // Main loop
    for (windowIndexVertical = 0; windowIndexVertical < numberOfWindowsVertically; windowIndexVertical++) {
        for (windowIndexHorizontal = 0; windowIndexHorizontal < numberOfWindowsHorizontally; windowIndexHorizontal++)
        {
            // Find window limits
            if (enablePadding == false) {
                rowFrom = windowIndexVertical*windowStepVertical;
                rowTo = rowFrom + windowHeight - 1;
                rowCenter = rowFrom + (int)round((double)windowHeight/2) - 1;
                columnFrom = windowIndexHorizontal*windowStepHorizontal;
                columnTo = columnFrom + windowWidth - 1;
                columnCenter = columnFrom + (int)round((double)windowWidth/2) - 1;
            }
            else {
                rowCenter = windowIndexVertical*windowStepVertical;
                rowFrom = rowCenter - (int)round((double)windowHeight/2) + 1;
                rowTo = rowFrom + windowHeight - 1;
                columnCenter = windowIndexHorizontal*windowStepHorizontal;
                columnFrom = columnCenter - (int)ceil((double)windowWidth/2) + 1;
                columnTo = columnFrom + windowWidth - 1;
            }

            // Copy window image
            if (imageIsGrayscale == true) {
                for (i=rowFrom; i<=rowTo; i++) {
                    for (j=columnFrom; j<=columnTo; j++) {
                        if (i<0 || i>imageHeight-1 || j<0 || j>imageWidth-1)
                        	windowImage[(i-rowFrom)+windowHeight*(j-columnFrom)] = 0;
                        else
                        	windowImage[(i-rowFrom)+windowHeight*(j-columnFrom)] = image[i+imageHeight*j];
                    }
                }
            }
            else {
                for (i=rowFrom; i<=rowTo; i++) {
                    for (j=columnFrom; j<=columnTo; j++) {
                        if (i<0 || i>imageHeight-1 || j<0 || j>imageWidth-1)
                            for (k=0; k<3; k++)
                            	windowImage[(i-rowFrom)+windowHeight*((j-columnFrom)+windowWidth*k)] = 0;
                        else
                            for (k=0; k<3; k++)
                            	windowImage[(i-rowFrom)+windowHeight*((j-columnFrom)+windowWidth*k)] = image[i+imageHeight*(j+imageWidth*k)];
                    }
                }
            }

            // Compute descriptor of window
            this->windowFeature->apply(windowImage, imageIsGrayscale, descriptorVector);

            // Store results
            //std::cout << windowIndexVertical << "," << windowIndexHorizontal << " STORING" << std::endl;
            for (d = 0; d < windowFeature->descriptorLengthPerWindow; d++)
            	outputImage[windowIndexVertical+numberOfWindowsVertically*(windowIndexHorizontal+numberOfWindowsHorizontally*d)] = descriptorVector[d];
            windowsCenters[windowIndexVertical+numberOfWindowsVertically*windowIndexHorizontal] = rowCenter;
            windowsCenters[windowIndexVertical+numberOfWindowsVertically*(windowIndexHorizontal+numberOfWindowsHorizontally)] = columnCenter;
        }
    }

    // Free windowImage
    free(windowImage);
    free(descriptorVector);



	//this->windowFeature->apply(this->image, this->imageWidth, this->imageHeight);
}

