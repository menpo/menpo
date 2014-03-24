#pragma once
#include "WindowFeature.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define small_val 1e-6 //used to check if interpolation is needed

using namespace std;

class LBP: public WindowFeature {
public:
	LBP(unsigned int windowHeight, unsigned int windowWidth, unsigned int numberOfChannels,
	        unsigned int radius, unsigned int samples);
	virtual ~LBP();
	void apply(double *windowImage, double *descriptorVector);
private:
    unsigned int radius, samples, windowHeight, windowWidth, numberOfChannels;
};

void LBPdescriptor(double *inputImage, unsigned int radius, unsigned int samples, unsigned int imageHeight, unsigned int imageWidth, unsigned int numberOfChannels, double *descriptorMatrix);
double roundn(double x, int n);
