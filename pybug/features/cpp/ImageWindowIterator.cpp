#include "ImageWindowIterator.h"
#include <iostream>
#include <cmath>

ImageWindowIterator::ImageWindowIterator(double *image, unsigned int imageWidth, unsigned int imageHeight,
		unsigned int windowHeight, unsigned int windowWidth, unsigned int windowStepHorizontal,
		unsigned int windowStepVertical, bool enablePadding, bool imageIsGrayscale, WindowFeature *windowFeature) {
	this->image = image;
	this->imageHeight = imageHeight;
	this->imageWidth = imageWidth;
	this->windowHeight = windowHeight;
	this->windowWidth = windowWidth;
	this->windowStepHorizontal = windowStepHorizontal;
	this->windowStepVertical = windowStepVertical;
	this->enablePadding = enablePadding;
	this->imageIsGrayscale = imageIsGrayscale;
	this->windowFeature = windowFeature;
}

ImageWindowIterator::~ImageWindowIterator() {
	// TODO Auto-generated destructor stub
}

void ImageWindowIterator::apply() {
	unsigned int windowHeight, windowWidth, windowStepHorizontal, windowStepVertical, imageHeight, imageWidth;
	bool enablePadding, imageIsGrayscale;
	int rowCenter, rowFrom, rowTo, columnCenter, columnFrom, columnTo, i, j, k;
	unsigned int windowIndexHorizontal, windowIndexVertical;

	std::cout << "Hello from iterator's apply" << std::endl;

    // Load window size, image size, window step, padding
    windowHeight = this->windowHeight;
    windowWidth = this->windowWidth;
    imageHeight = this->imageHeight;
    imageWidth = this->imageWidth;
    windowStepHorizontal = this->windowStepHorizontal;
    windowStepVertical = this->windowStepVertical;
    enablePadding = this->enablePadding;
    imageIsGrayscale = this->imageIsGrayscale;

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

    // Print information
    std::cout << std::endl << "Image size = " << imageHeight << "x" << imageWidth << std::endl;
    if (imageIsGrayscale==true)
    	std::cout << "Image is GRAY" << std::endl;
    else
    	std::cout << "Image is RGB" << std::endl;
    std::cout << "Window size = " << windowHeight << "x" << windowWidth << std::endl;
    std::cout << "Window step = (" << windowStepHorizontal << "," << windowStepVertical << ")" << std::endl;
    if (enablePadding==true)
    	std::cout << "Padding enabled" << std::endl;
    else
    	std::cout << "Padding disabled" << std::endl;
    std::cout << "Number of windows = " << numberOfWindowsVertically << "x" << numberOfWindowsHorizontally << " = " << numberOfWindows << std::endl;


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

            // Create window image
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
                        if (i<0 || i>imageHeight-1 || j<0 || j>imageWidth-1) {
                            for (k=0; k<3; k++)
                               windowImage[(i-rowFrom)+windowHeight*((j-columnFrom)+windowWidth*k)] = 0;
                        }
                        else {
                            for (k=0; k<3; k++)
                               windowImage[(i-rowFrom)+windowHeight*((j-columnFrom)+windowWidth*k)] = image[i+imageHeight*(j+imageWidth*k)];
                        }
                    }
                }
            }

            /*// Compute descriptor of window
            DalalTriggsHOGdescriptor(windowImage,params,info.windowSize,descriptorVector,info.inputImageIsGrayscale);
            d=0;
            for (d2=0; d2<info.numberOfBlocksPerWindowHorizontally; d2++) {
                for (d1=0; d1<info.numberOfBlocksPerWindowVertically; d1++) {
                    for (d3=0; d3<info.descriptorLengthPerBlock; d3++) {
                        descriptorMatrix[d1+info.numberOfBlocksPerWindowVertically*(d2+info.numberOfBlocksPerWindowHorizontally*d3)] = descriptorVector[d];
                        d = d + 1; }}}

            // Store results
            for (d2=0; d2<info.numberOfBlocksPerWindowHorizontally; d2++)
                for (d1=0; d1<info.numberOfBlocksPerWindowVertically; d1++)
                    for (d3=0; d3<info.descriptorLengthPerBlock; d3++)
                        WindowsMatrixDescriptorsMatrix[windowIndexVertical+info.numberOfWindowsVertically*(windowIndexHorizontal+info.numberOfWindowsHorizontally*(d1+info.numberOfBlocksPerWindowVertically*(d2+info.numberOfBlocksPerWindowHorizontally*d3)))] = descriptorMatrix[d1+info.numberOfBlocksPerWindowVertically*(d2+info.numberOfBlocksPerWindowHorizontally*d3)];
            WindowsCentersMatrix[windowIndexVertical+info.numberOfWindowsVertically*windowIndexHorizontal] = rowCenter + 1;
            WindowsCentersMatrix[windowIndexVertical+info.numberOfWindowsVertically*(windowIndexHorizontal+info.numberOfWindowsHorizontally)] = columnCenter + 1;*/
        }
    }



	//this->windowFeature->apply(this->image, this->imageWidth, this->imageHeight);
}
