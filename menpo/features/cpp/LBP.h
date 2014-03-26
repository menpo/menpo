#pragma once
#include "WindowFeature.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define small_val 1e-6 //used to check if interpolation is needed

const double PI = 3.141592653589793238463;

using namespace std;

class LBP: public WindowFeature {
public:
	LBP(unsigned int windowHeight, unsigned int windowWidth, unsigned int numberOfChannels,
	        unsigned int *radius, unsigned int *samples, unsigned int numberOfRadiusSamplesCombinations, unsigned int mapping_type);
	virtual ~LBP();
	void apply(double *windowImage, double *descriptorVector);
private:
    unsigned int *radius, *samples;
    unsigned int numberOfRadiusSamplesCombinations, windowHeight, windowWidth, numberOfChannels, mapping_type;
};

void LBPdescriptor(double *inputImage, unsigned int *radius, unsigned int *samples, unsigned int numberOfRadiusSamplesCombinations, unsigned int mapping_type, unsigned int imageHeight, unsigned int imageWidth, unsigned int numberOfChannels, double *descriptorVector);
//double roundn(double x, int n);
int power2(int index);
void generate_codes_mapping_table(unsigned int *mapping_table, unsigned int mapping_type, unsigned int n_samples);
int count_bit_transitions(int a, unsigned int n_samples);
int count_bits(int n);

