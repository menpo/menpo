#include <stdio.h>
#include "central_difference.h"

static inline long long SUB2IND(const long long j, const long long i, const long long k,
                                const long long row_size, const long long col_size,
                                const long long n_channels) {
    //return ((i + col_size * j) * n_channels) + k;
    return (i + col_size * j) + (row_size * col_size * k);
}

void central_difference(const double* in, const long long rows,
                        const long long cols, const long long n_channels,
                        double* out) {
    const long long n_output_channels = n_channels * 2;
    long long output_index = 0;

    #pragma omp for
    for (long long k = 0; k < n_channels; k += 1) {
        // row-derivative
        for (long long i = 0; i < rows; i += 1) {
            for (long long j = 0; j < cols; j += 1) {
                if (j == 0) {
                    output_index = SUB2IND(0, i, k, rows, cols, n_output_channels);
                    out[output_index] = in[SUB2IND(1, i, k, rows, cols, n_channels)] - in[SUB2IND(0, i, k, rows, cols, n_channels)];
                }
                else if (j == rows - 1) {
                    output_index = SUB2IND(j, i, k, rows, cols, n_output_channels);
                    out[output_index] = in[SUB2IND(rows - 1, i, k, rows, cols, n_channels)] - in[SUB2IND(rows - 2, i, k, rows, cols, n_channels)];
                }
                else {
                    output_index = SUB2IND(j, i, k, rows, cols, n_output_channels);
                    out[output_index] = (in[SUB2IND(j + 1, i, k, rows, cols, n_channels)] - in[SUB2IND(j - 1, i, k, rows, cols, n_channels)]) / 2.0;
                }
            }
        }

        // column-derivative
        for (long long j = 0; j < rows; j += 1) {
            for (long long i = 0; i < cols; i += 1) {
                if (i == 0) {
                    output_index = SUB2IND(j, 0, n_channels + k , rows, cols, n_output_channels);
                    out[output_index] = in[SUB2IND(j, 1, k, rows, cols, n_channels)] - in[SUB2IND(j, 0, k, rows, cols, n_channels)];
                }
                else if (i == cols - 1) {
                    output_index = SUB2IND(j, cols - 1, n_channels + k, rows, cols, n_output_channels);
                    out[output_index] = in[SUB2IND(j, cols - 1, k, rows, cols, n_channels)] - in[SUB2IND(j, cols - 2, k, rows, cols, n_channels)];
                }
                else {
                    output_index = SUB2IND(j, i, n_channels + k, rows, cols, n_output_channels);
                    out[output_index] = (in[SUB2IND(j, i + 1, k, rows, cols, n_channels)] - in[SUB2IND(j, i - 1, k, rows, cols, n_channels)]) / 2.0;
                }
            }
        }
    }
}
