#include "LBP.h"

LBP::LBP(unsigned int windowHeight, unsigned int windowWidth, unsigned int numberOfChannels, unsigned int *radius, unsigned int *samples, unsigned int numberOfRadiusSamplesCombinations, unsigned int mapping_type) {
	unsigned int descriptorLengthPerWindow = numberOfRadiusSamplesCombinations * numberOfChannels;
    this->radius = radius;
    this->samples = samples;
    this->numberOfRadiusSamplesCombinations = numberOfRadiusSamplesCombinations;
    this->mapping_type = mapping_type;
    this->descriptorLengthPerWindow = descriptorLengthPerWindow;
    this->windowHeight = windowHeight;
    this->windowWidth = windowWidth;
    this->numberOfChannels = numberOfChannels;
}

LBP::~LBP() {
}


void LBP::apply(double *windowImage, double *descriptorVector) {
    LBPdescriptor(windowImage, this->radius, this->samples, this->numberOfRadiusSamplesCombinations, this->mapping_type, this->windowHeight, this->windowWidth, this->numberOfChannels, descriptorVector);
}


void LBPdescriptor(double *inputImage, unsigned int *radius, unsigned int *samples, unsigned int numberOfRadiusSamplesCombinations, unsigned int mapping_type, unsigned int imageHeight, unsigned int imageWidth, unsigned int numberOfChannels, double *descriptorVector) {
    unsigned int i, s, ch, max_samples;
    int centre_y, centre_x, rx, ry, fx, fy, cx, cy;
    double angle_step, centre_val, sample_val;
    double *samples_x;
    double *samples_y;
    double tx, ty, w1, w2, w3, w4;
    int lbp_code;

    // initialize samples coordinates arrays
    max_samples = samples[0];
    for (i=1; i<numberOfRadiusSamplesCombinations; i++) {
        if (samples[i] > max_samples)
            max_samples = samples[i];
    }
    samples_x = new double[max_samples];
    samples_y = new double[max_samples];

    // find coordinates of the window centre in the window reference frame (axes origin in bottom left corner)
    centre_y = (int)((imageHeight-1)/2);
    centre_x = (int)((imageWidth-1)/2);

    for (i=0; i<numberOfRadiusSamplesCombinations; i++) {
        // find coordinates of sampling point in the window reference frame (axes origin in bottom left corner)
        angle_step = 2*PI/samples[i];
        for (s=0; s<samples[i]; s++) {
            samples_x[s] = centre_x + radius[i] * cos(s * angle_step);
            samples_y[s] = centre_y - radius[i] * sin(s * angle_step);
        }

        // for each channel, compute the lbp code
        for (ch=0; ch<numberOfChannels; ch++) {
            // value of centre
            centre_val = inputImage[centre_y + centre_x*imageHeight + ch*imageHeight*imageWidth];
            lbp_code = 0;
            for (s=0; s<samples[i]; s++) {
                // check if interpolation is needed
                rx = (int)round(samples_x[s]);
                ry = (int)round(samples_y[s]);
                if ( (fabs(samples_x[s] - rx) < small_val) && (fabs(samples_y[s] - ry) < small_val) )
                    sample_val = inputImage[ry + rx*imageHeight + ch*imageHeight*imageWidth];
                else {
                    fx = (int)floor(samples_x[s]);
                    fy = (int)floor(samples_y[s]);
                    cx = (int)ceil(samples_x[s]);
                    cy = (int)ceil(samples_y[s]);
                    tx = samples_x[s] - fx;
                    ty = samples_y[s] - fy;
                    // compute interpolation weights
                    /*w1 = roundn((1 - tx) * (1 - ty), -6);
                    w2 = roundn(tx * (1 - ty), -6);
                    w3 = roundn((1 - tx) * ty, -6);
                    // w4 = roundn(tx * ty, -6);
                    w4 = roundn(1 - w1 - w2 - w3, -6);*/
                    w1 = (1 - tx) * (1 - ty);
                    w2 =      tx  * (1 - ty);
                    w3 = (1 - tx) *      ty ;
                    w4 =      tx  *      ty ;
                    sample_val = w1*inputImage[fy + fx*imageHeight + ch*imageHeight*imageWidth] +
                                 w2*inputImage[fy + cx*imageHeight + ch*imageHeight*imageWidth] +
                                 w3*inputImage[cy + fx*imageHeight + ch*imageHeight*imageWidth] +
                                 w4*inputImage[cy + cx*imageHeight + ch*imageHeight*imageWidth];
                    //sample_val = roundn(sample_val, -4);
                }

                // update the lbp code
                if (sample_val >= centre_val)
                    lbp_code = lbp_code + power2(s);
            }
            descriptorVector[i + ch*numberOfRadiusSamplesCombinations] = lbp_code;
        }
    }
    // Empty memory
    delete [] samples_x;
    delete [] samples_y;
}

/*double roundn(double x, int n) {
    if (n < 0) {
        double p = pow(10, -n);
        x = round(p * x) / p;
    }
    else if (n > 0) {
        double p = pow(10, n);
        x = p * round(x / p);
    }
    else
        x = round(x);
    return x;
}*/

void generate_codes_mapping_table(double *mapping_table, int *mapping_table_size, int mapping_type) {
double a=1;
}

int power2(int index) {
    if (index == 0)
        return 1;
    int number = 2;
    for (int i = 1; i < index; i++)
        number = number * 2;
    return number;
}