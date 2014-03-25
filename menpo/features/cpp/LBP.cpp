#include "LBP.h"

LBP::LBP(unsigned int windowHeight, unsigned int windowWidth, unsigned int numberOfChannels, unsigned int *radius, unsigned int *samples, unsigned int numberOfRadiusSamplesCombinations) {
	unsigned int descriptorLengthPerWindow = numberOfRadiusSamplesCombinations * numberOfChannels;
    this->radius = radius;
    this->samples = samples;
    this->numberOfRadiusSamplesCombinations = numberOfRadiusSamplesCombinations;
    this->descriptorLengthPerWindow = descriptorLengthPerWindow;
    this->windowHeight = windowHeight;
    this->windowWidth = windowWidth;
    this->numberOfChannels = numberOfChannels;
}

LBP::~LBP() {
}


void LBP::apply(double *windowImage, double *descriptorVector) {
    LBPdescriptor(windowImage, this->radius, this->samples, this->numberOfRadiusSamplesCombinations, this->windowHeight, this->windowWidth, this->numberOfChannels, descriptorVector);
}


void LBPdescriptor(double *inputImage, unsigned int *radius, unsigned int *samples, unsigned int numberOfRadiusSamplesCombinations, unsigned int imageHeight, unsigned int imageWidth, unsigned int numberOfChannels, double *descriptorVector) {
    unsigned int i, s, ch;
    double angle_step, min_x, max_x, min_y, max_y, origin_y, origin_x;
    double *samples_coords_x, *samples_coords_y;
//    bool flag;
//    double angle_step, min_x, max_x, min_y, max_y, block_size_x, block_size_y, origin_y, origin_x, origin_val, sample_val, rx, ry;
//    double tx, ty, w1, w2, w3, w4;
//    int lbp_code=0;

    for (i=0; i<numberOfRadiusSamplesCombinations; i++) {
        // find coordinates of sampling points
        samples_coords_x = new double[samples[i]];
        samples_coords_y = new double[samples[i]];
        angle_step = 2*PI/samples[i];
        for (s=0; s<samples[i]; s++) {
            samples_coords_x[s] = radius[i] * cos(s * angle_step);
            samples_coords_y[s] = -radius[i] * sin(s * angle_step);
        }

        // find max and min coordinates of sampling points
        min_x = samples_coords_x[0];
        max_x = samples_coords_x[0];
        min_y = samples_coords_y[0];
        max_y = samples_coords_y[0];
        for (s=1; s<samples[i]; s++) {
            if (samples_coords_x[s] < min_x)
                min_x = samples_coords_x[s];
            if (samples_coords_x[s] > max_x)
                max_x = samples_coords_x[s];
            if (samples_coords_y[s] < min_y)
                min_y = samples_coords_y[s];
            if (samples_coords_y[s] > max_y)
                max_y = samples_coords_y[s];
        }

        // find coordinates of the origin/centre (0,0) in the block reference frame
        origin_y = -floor(min(min_y,0.0));
        origin_x = -floor(min(min_x,0.0));

        for (ch=0; ch<numberOfChannels; ch++) {
        }
    }

/*
	// find block size
	block_size_x = ceil(max(max_x,0.0)) - floor(min(min_x,0.0)) + 1;
    block_size_y = ceil(max(max_y,0.0)) - floor(min(min_y,0.0)) + 1;

    // compute the lbp code
    for (i=0; i<samples; i++) {
        // coordinates of sampling point in the block reference frame
        sample_x = origin_x + samples_coords_x[i];
        sample_y = origin_y + samples_coords_y[i];

        // value of origin
        origin_val = inputImage[origin_y + origin_x*imageHeight + 0*imageHeight*imageWidth];

        // check if interpolation is needed
        rx = round(sample_x);
        ry = round(sample_y);
        if ( (fabs(sample_x - rx) < small_val) && (fabs(sample_y - ry) < small_val) )
            sample_val = inputImage[ry + rx*imageHeight + 0*imageHeight*imageWidth];
        else {
            tx = sample_x - floor(sample_x);
            ty = sample_y - floor(sample_y);
            w1 = roundn((1 - tx) * (1 - ty), -6);
            w2 = roundn(tx * (1 - ty), -6);
            w3 = roundn((1 - tx) * ty, -6);
            // w4 = roundn(tx * ty, -6);
            w4 = roundn(1 - w1 - w2 - w3, -6);

            sample_val = inputImage[ry + rx*imageHeight + 0*imageHeight*imageWidth];
        }

        // update the lbp code
        if (sample_val >= origin_val)
            lbp_code = lbp_code + 2^i;

        descriptorVector[0] = lbp_code;
*/

        descriptorVector[0] = inputImage[0];

        // Empty memory
        delete [] samples_coords_x;
        delete [] samples_coords_y;
//    }
}

double roundn(double x, int n) {
    double p;

    if (n < 0) {
        p = 10 ^ -n;
        x = round(p * x) / p;
    }
    else if (n > 0) {
        p = 10 ^ n;
        x = p * round(x / p);
    }
    else
        x = round(x);
    return x;
}
