#include "LBP.h"

LBP::LBP(unsigned int windowHeight, unsigned int windowWidth, unsigned int numberOfChannels, unsigned int radius[], unsigned int samples[]) {
	unsigned int descriptorLengthPerWindow = 0;

    this->radius = radius;
    this->samples = samples;
    this->descriptorLengthPerWindow = descriptorLengthPerWindow;
    this->windowHeight = windowHeight;
    this->windowWidth = windowWidth;
    this->numberOfChannels = numberOfChannels;
}

LBP::~LBP() {
}


void LBP::apply(double *windowImage, double *descriptorVector) {
    LBPdescriptor(windowImage, this->radius, this->samples, this->windowHeight, this->windowWidth, this->numberOfChannels, descriptorVector);
}


void LBPdescriptor(double *inputImage, unsigned int radius, unsigned int samples, unsigned int imageHeight, unsigned int imageWidth, unsigned int numberOfChannels, double *descriptorMatrix) {
int a = 0;
}