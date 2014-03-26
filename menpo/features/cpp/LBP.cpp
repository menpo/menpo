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
    unsigned int *mapping_table;
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
    mapping_table = new unsigned int[power2(max_samples)];

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

        if (i==0) {
            generate_codes_mapping_table(mapping_table, mapping_type, samples[i]);
            printf("\n%d, %d:[",mapping_type,power2(samples[i]));
            for (int kk=0; kk<power2(samples[i]); kk++)
                printf("%d, ",mapping_table[kk]);
            printf("]\n");
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
                    lbp_code += power2(s);
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

int power2(int index) {
    if (index == 0)
        return 1;
    int number = 2;
    for (int i = 1; i < index; i++)
        number = number * 2;
    return number;
}

void generate_codes_mapping_table(unsigned int *mapping_table, unsigned int mapping_type, unsigned int n_samples) {
    int index, c, num_trans, rm, r, j;
    unsigned int newMax = 0;
    // new_max --> mapping_table_size
    // n_samples --> n_samples
    // table --> mapping_table
    if (mapping_type == 1) {
        // uniform-2
        newMax = n_samples * (n_samples - 1) + 3;
        index = 0;
        for (c=0; c<power2(n_samples); c++) {
            // number of 1->0 and 0->1 transitions in a binary string x is equal
            // to the number of 1-bits in XOR(x, rotate_left(x))
            num_trans = count_bit_transitions(c, n_samples);
            if (num_trans <= 2) {
                mapping_table[c] = index;
                index += 1;
            }
            else
                mapping_table[c] = newMax - 1;
        }
    }
    else if (mapping_type == 2) {
        // rotation invariant
        int *tmp_map = new int[power2(n_samples)];
        for (c=0; c<power2(n_samples); c++)
            tmp_map[c] = -1;
        newMax = 0;
        for (c=0; c<power2(n_samples); c++) {
            rm = c;
            r = c;
            for (j=1; j<n_samples; j++) {
                r = leftRotate(r, n_samples, 1);
                if (r < rm)
                    rm = r;
            }
            if (tmp_map[rm] < 0) {
                tmp_map[rm] = newMax;
                newMax += 1;
            }
            mapping_table[c] = tmp_map[rm];
        }
    }
    else if (mapping_type == 3) {
        // rotation invariant and uniform-2
        newMax = n_samples + 2;
        for (c=0; c<power2(n_samples); c++) {
            // number of 1->0 and 0->1 transitions in a binary string x is equal
            // to the number of 1-bits in XOR(x, rotate_left(x))
            num_trans = count_bit_transitions(c, n_samples);
            if (num_trans <= 2)
                mapping_table[c] = count_bits(c);
            else
                mapping_table[c] = n_samples + 1;
        }
    }
}

/*int count_bit_transitions(int a)
{
    assert((-1 >> 1) < 0); // check for arithmetic shift
    int count = 0;
    for(a ^= (a >> 1); a; a &= a - 1)
    	++count;
    return count;
}*/

int count_bit_transitions(int a, unsigned int n_samples) {
    int b = a >> 1; // sign-extending shift properly counts bits at the ends
    int c = a ^ b;  // xor marks bits that are not the same as their neighbors on the left
    if (a >= power2(n_samples-1))
        return count_bits(c)-1; // count number of set bits in c
    else
        return count_bits(c); // count number of set bits in c
}

int count_bits(int n) {
    unsigned int c; // c accumulates the total bits set in v
    for (c = 0; n; c++)
        n &= n - 1; // clear the least significant bit set
    return c;
}

int leftRotate(int num, unsigned int len_bits, unsigned int move_bits) {
   /* In num<<move_bits, last move_bits bits are 0. To put first 3 bits of num at
     last, do bitwise or of num<<move_bits with num >>(len_bits - move_bits) */
   return (num << move_bits % len_bits) & (power2(len_bits) - 1) | ((num & (power2(len_bits) - 1)) >> (len_bits - (move_bits % len_bits)));
}
