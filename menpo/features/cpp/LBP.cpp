#include "LBP.h"

LBP::LBP(unsigned int windowHeight, unsigned int windowWidth, unsigned int numberOfChannels,
         unsigned int *radius, unsigned int *samples, unsigned int numberOfRadiusSamplesCombinations,
         unsigned int mapping_type, unsigned int *uniqueSamples, unsigned int *whichMappingTable, unsigned int numberOfUniqueSamples) {
	unsigned int descriptorLengthPerWindow = numberOfRadiusSamplesCombinations * numberOfChannels;
    this->radius = radius;
    this->samples = samples;
    this->whichMappingTable = whichMappingTable;
    this->numberOfRadiusSamplesCombinations = numberOfRadiusSamplesCombinations;
    this->descriptorLengthPerWindow = descriptorLengthPerWindow;
    this->windowHeight = windowHeight;
    this->windowWidth = windowWidth;
    this->numberOfChannels = numberOfChannels;

    // find mapping table for each unique samples value
    unsigned int **mapping_tables, i;
    mapping_tables = new unsigned int*[numberOfUniqueSamples];
    for (i=0; i<numberOfUniqueSamples; i++) {
	    mapping_tables[i] = new unsigned int[power2(uniqueSamples[i])];
	    if (mapping_type != 0)
    	    generate_codes_mapping_table(mapping_tables[i], mapping_type, uniqueSamples[i]);
    	else {
    	    for (int j=0; j<power2(uniqueSamples[i]); j++)
        	    mapping_tables[i][j] = j;
        }
	}
    this->mapping_tables = mapping_tables;
}

LBP::~LBP() {
}


void LBP::apply(double *windowImage, double *descriptorVector) {
    LBPdescriptor(windowImage, this->radius, this->samples, this->numberOfRadiusSamplesCombinations, this->whichMappingTable,
                  this->mapping_tables, this->windowHeight, this->windowWidth, this->numberOfChannels, descriptorVector);
}


void LBPdescriptor(double *inputImage, unsigned int *radius, unsigned int *samples, unsigned int numberOfRadiusSamplesCombinations,
                   unsigned int *whichMappingTable, unsigned int **mapping_tables, unsigned int imageHeight, unsigned int imageWidth,
                   unsigned int numberOfChannels, double *descriptorVector) {
    unsigned int i, s, ch, max_samples;
    int centre_y, centre_x, rx, ry, fx, fy, cx, cy;
    double angle_step, centre_val, sample_val;
    double tx, ty, w1, w2, w3, w4;
    int lbp_code;

    // initialize samples coordinates arrays and mapping array
    max_samples = samples[0];
    for (i=1; i<numberOfRadiusSamplesCombinations; i++) {
        if (samples[i] > max_samples)
            max_samples = samples[i];
    }
    double *samples_x = new double[max_samples];
    double *samples_y = new double[max_samples];

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
                    // compute interpolation weights and value
                    w1 = (1 - tx) * (1 - ty);
                    w2 =      tx  * (1 - ty);
                    w3 = (1 - tx) *      ty ;
                    w4 =      tx  *      ty ;
                    sample_val = w1*inputImage[fy + fx*imageHeight + ch*imageHeight*imageWidth] +
                                 w2*inputImage[fy + cx*imageHeight + ch*imageHeight*imageWidth] +
                                 w3*inputImage[cy + fx*imageHeight + ch*imageHeight*imageWidth] +
                                 w4*inputImage[cy + cx*imageHeight + ch*imageHeight*imageWidth];
                }

                // update the lbp code
                if (sample_val >= centre_val)
                    lbp_code += power2(s);
            }

            // store lbp code with mapping
            descriptorVector[i + ch*numberOfRadiusSamplesCombinations] = mapping_tables[whichMappingTable[i]][lbp_code];
        }
    }
    // Empty memory
    delete [] samples_x;
    delete [] samples_y;
}

int power2(int index) {
    if (index == 0)
        return 1;
    int number = 2;
    for (int i = 1; i < index; i++)
        number = number * 2;
    return number;
}

void generate_codes_mapping_table(unsigned int *mapping_table, unsigned int mapping_type, unsigned int n_samples) {
    int c;
    unsigned int newMax = 0;
    if (mapping_type == 1) {
        // uniform-2
        newMax = n_samples * (n_samples - 1) + 3;
        int index = 0, num_trans;
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
        int rm, r;
        int *tmp_map = new int[power2(n_samples)];
        for (c=0; c<power2(n_samples); c++)
            tmp_map[c] = -1;
        newMax = 0;
        for (c=0; c<power2(n_samples); c++) {
            rm = c;
            r = c;
            for (int j=1; j<n_samples; j++) {
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
        int num_trans;
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
