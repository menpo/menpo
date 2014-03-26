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
    int centre_y, centre_x, rx, ry, fx, fy, cx, cy;
    double angle_step, min_x, min_y, sample_x, sample_y, centre_val, sample_val;
    double *samples_coords_x;
    double *samples_coords_y;
    double tx, ty, w1, w2, w3, w4;
    int lbp_code;

    for (i=0; i<numberOfRadiusSamplesCombinations; i++) {
        // find coordinates of sampling points with the axes origin (0,0) on the window centre
        samples_coords_x = new double[samples[i]];
        samples_coords_y = new double[samples[i]];
        angle_step = 2*PI/samples[i];
        for (s=0; s<samples[i]; s++) {
            samples_coords_x[s] = radius[i] * cos(s * angle_step);
            samples_coords_y[s] = - (radius[i] * sin(s * angle_step));
        }

        // find min coordinates of sampling points with the axes origin (0,0) on the window centre
        min_x = samples_coords_x[0];
        min_y = samples_coords_y[0];
        for (s=1; s<samples[i]; s++) {
            if (samples_coords_x[s] < min_x)
                min_x = samples_coords_x[s];
            if (samples_coords_y[s] < min_y)
                min_y = samples_coords_y[s];
        }

        // find coordinates of the window centre in the window reference frame (axes origin in bottom left corner)
        centre_y = (int)-floor(min(min_y,0.0));
        centre_x = (int)-floor(min(min_x,0.0));

        // value of centre
        centre_val = inputImage[centre_y + centre_x*imageHeight + ch*imageHeight*imageWidth];

        // for each channel, compute the lbp code
        for (ch=0; ch<numberOfChannels; ch++) {
            lbp_code = 0;
            for (s=0; s<samples[i]; s++) {
                // coordinates of sampling point in the window reference frame (axes origin in bottom left corner)
                sample_x = centre_x + samples_coords_x[s];
                sample_y = centre_y + samples_coords_y[s];

                // check if interpolation is needed
                rx = (int)round(sample_x);
                ry = (int)round(sample_y);
                if ( (fabs(sample_x - rx) < small_val) && (fabs(sample_y - ry) < small_val) )
                    sample_val = inputImage[ry + rx*imageHeight + ch*imageHeight*imageWidth];
                else {
                    fx = (int)floor(sample_x);
                    fy = (int)floor(sample_y);
                    cx = (int)ceil(sample_x);
                    cy = (int)ceil(sample_y);
                    tx = sample_x - fx;
                    ty = sample_y - fy;
                    w1 = roundn((1 - tx) * (1 - ty), -6);
                    w2 = roundn(tx * (1 - ty), -6);
                    w3 = roundn((1 - tx) * ty, -6);
                    // w4 = roundn(tx * ty, -6);
                    w4 = roundn(1 - w1 - w2 - w3, -6);
                    sample_val = w1*inputImage[fy + fx*imageHeight + ch*imageHeight*imageWidth] + w2*inputImage[fy + cx*imageHeight + ch*imageHeight*imageWidth] + w3*inputImage[cy + fx*imageHeight + ch*imageHeight*imageWidth] + w4*inputImage[cy + cx*imageHeight + ch*imageHeight*imageWidth];
                    sample_val = roundn(sample_val, -4);
                }

                // update the lbp code
                if (sample_val >= centre_val)
                    lbp_code = lbp_code + 2^s;
            }
            descriptorVector[i + ch*numberOfRadiusSamplesCombinations] = lbp_code;
        }
    }
    // Empty memory
    delete [] samples_coords_x;
    delete [] samples_coords_y;
}

double roundn(double x, int n) {
    if (n < 0) {
        double p = 10 ^ -n;
        x = round(p * x) / p;
    }
    else if (n > 0) {
        double p = 10 ^ n;
        x = p * round(x / p);
    }
    else
        x = round(x);
    return x;
}
