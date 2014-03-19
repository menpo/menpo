#include "interp2.h"

inline double safe_round(const double number)
{
    return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}

inline size_t index_array(const size_t N_ROWS, const size_t N_COLS, const size_t N_CHANNELS,
                          size_t row_index, size_t col_index)
{
    if (row_index < 0)
    {
        row_index = 0;
    }
    else if (row_index >= N_ROWS)
    {
        row_index = N_ROWS - 1;
    }

    if (col_index < 0)
    {
        col_index = 0;
    }
    else if (col_index >= N_COLS)
    {
        col_index = N_COLS - 1;
    }

// MATLAB is column major, so we index in differently
#if defined(IS_MEX)
    return N_ROWS * col_index + row_index;
#else
    return N_CHANNELS * N_COLS * row_index + N_CHANNELS * col_index;
#endif
}

void indices_linear(size_t &f_index_00, size_t &f_index_10, size_t &f_index_01, size_t &f_index_11,
                    const size_t row_index, const size_t col_index,
                    const size_t N_ROWS, const size_t N_COLS, const size_t N_CHANNELS)
{
    f_index_00 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index, col_index);
    f_index_10 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index + 1, col_index);

    f_index_01 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index, col_index + 1);
    f_index_11 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index + 1, col_index + 1);
}

void indices_cubic(size_t &f_index_00, size_t &f_index_10, size_t &f_index_20, size_t &f_index_30,
                   size_t &f_index_01, size_t &f_index_11, size_t &f_index_21, size_t &f_index_31,
                   size_t &f_index_02, size_t &f_index_12, size_t &f_index_22, size_t &f_index_32,
                   size_t &f_index_03, size_t &f_index_13, size_t &f_index_23, size_t &f_index_33,
                   const size_t row_index, const size_t col_index,
                   const size_t N_ROWS, const size_t N_COLS, const size_t N_CHANNELS)
{
    f_index_00 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index - 1, col_index - 1);
    f_index_10 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index, col_index - 1);
    f_index_20 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index + 1, col_index - 1);
    f_index_30 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index + 2, col_index - 1);

    f_index_01 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index - 1, col_index);
    f_index_11 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index, col_index);
    f_index_21 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index + 1, col_index);
    f_index_31 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index + 2, col_index);

    f_index_02 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index - 1, col_index + 1);
    f_index_12 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index, col_index + 1);
    f_index_22 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index + 1, col_index + 1);
    f_index_32 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index + 2, col_index + 1);

    f_index_03 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index - 1, col_index + 2);
    f_index_13 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index, col_index + 2);
    f_index_23 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index + 1, col_index + 2);
    f_index_33 = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index + 2, col_index + 2);
}

void interpolate_nearest(const double *F, const double *row_vector, const double *col_vector,
                         const size_t N_ELEMS, const size_t N_ROWS, const size_t N_COLS, const size_t N_CHANNELS,
                         double *out)
{
    const size_t F_MAX = N_ROWS * N_COLS;
    for (size_t i = 0; i < N_ELEMS; i++)
    {
        const double &row_index = row_vector[i];
        const double &col_index = col_vector[i];

        const size_t row_index_round = SAFE_INDEX(int(safe_round(row_index)));
        const size_t col_index_round = SAFE_INDEX(int(safe_round(col_index)));

        const size_t f_index = index_array(N_ROWS, N_COLS, N_CHANNELS, row_index_round, col_index_round);

        for (size_t j = 0; j < N_CHANNELS; j++)
        {
            out[OUT_INDEX(i, j)] = F[F_INDEX(f_index, j)];
        }
    }

}

void interpolate_bilinear(const double *F, const double *row_vector, const double *col_vector,
                         const size_t N_ELEMS, const size_t N_ROWS, const size_t N_COLS, const size_t N_CHANNELS,
                         double *out)
{
    const size_t F_MAX = N_ROWS * N_COLS;
    for (size_t i = 0; i < N_ELEMS; i++)
    {
        const double &row_index = row_vector[i];
        const double &col_index = col_vector[i];

        const double row_floor = floor(row_index);
        const double col_floor = floor(col_index);

        const double drow = row_index - row_floor;
        const double dcol = col_index - col_floor;

        const double wrow0 = 1.0 - drow;
        const double wrow1 = drow;

        const double wcol0 = 1.0 - dcol;
        const double wcol1 = dcol;

        size_t f_index_00, f_index_10, f_index_01, f_index_11;

        indices_linear(f_index_00, f_index_10, f_index_01, f_index_11,
                       SAFE_INDEX(int(row_floor)), SAFE_INDEX(int(col_floor)),
                       N_ROWS, N_COLS, N_CHANNELS);

        for (size_t j = 0; j < N_CHANNELS; j++)
        {
            out[OUT_INDEX(i, j)] =
                wcol0 * (wrow0 * F[F_INDEX(f_index_00, j)] + wrow1 * F[F_INDEX(f_index_10, j)]) +
                wcol1 * (wrow0 * F[F_INDEX(f_index_01, j)] + wrow1 * F[F_INDEX(f_index_11, j)]);
        }
    }
}

void interpolate_bicubic(const double *F, const double *row_vector, const double *col_vector,
                         const size_t N_ELEMS, const size_t N_ROWS, const size_t N_COLS, const size_t N_CHANNELS,
                         double *out)
{
    const size_t F_MAX = N_ROWS * N_COLS;
    for (size_t i = 0; i < N_ELEMS; i++)
    {
        const double &row_index = row_vector[i];
        const double &col_index = col_vector[i];

        const double row_floor = floor(row_index);
        const double col_floor = floor(col_index);

        const double drow = row_index - row_floor;
        const double dcol = col_index - col_floor;

        const double drow2 = drow * drow;
        const double drow3 = drow2 * drow;

        const double dcol2 = dcol * dcol;
        const double dcol3 = dcol2 * dcol;

        const double wrow0 = 0.5 * (-drow + 2.0 * drow2 - drow3);
        const double wrow1 = 0.5 * (2.0 - 5.0 * drow2 + 3.0 * drow3);
        const double wrow2 = 0.5 * (drow + 4.0 * drow2 - 3.0 * drow3);
        const double wrow3 = 0.5 * (-drow2 + drow3);


        const double wcol0 = 0.5 * (-dcol + 2.0 * dcol2 - dcol3);
        const double wcol1 = 0.5 * (2.0 - 5.0 * dcol2 + 3.0 * dcol3);
        const double wcol2 = 0.5 * (dcol + 4.0 * dcol2 - 3.0 * dcol3);
        const double wcol3 = 0.5 * (-dcol2 + dcol3);

        size_t f_index_00, f_index_10, f_index_20, f_index_30,
               f_index_01, f_index_11, f_index_21, f_index_31,
               f_index_02, f_index_12, f_index_22, f_index_32,
               f_index_03, f_index_13, f_index_23, f_index_33;

        indices_cubic(f_index_00, f_index_10, f_index_20, f_index_30,
                      f_index_01, f_index_11, f_index_21, f_index_31,
                      f_index_02, f_index_12, f_index_22, f_index_32,
                      f_index_03, f_index_13, f_index_23, f_index_33,
                      SAFE_INDEX(int(row_floor)), SAFE_INDEX(int(col_floor)),
                      N_ROWS, N_COLS, N_CHANNELS);

        for (size_t j = 0; j < N_CHANNELS; j++)
        {
            out[OUT_INDEX(i, j)] =
                wcol0 * (wrow0 * F[F_INDEX(f_index_00, j)] + wrow1 * F[F_INDEX(f_index_10, j)] +
                         wrow2 * F[F_INDEX(f_index_20, j)] + wrow3 * F[F_INDEX(f_index_30, j)]) +
                wcol1 * (wrow0 * F[F_INDEX(f_index_01, j)] + wrow1 * F[F_INDEX(f_index_11, j)] +
                         wrow2 * F[F_INDEX(f_index_21, j)] + wrow3 * F[F_INDEX(f_index_31, j)]) +
                wcol2 * (wrow0 * F[F_INDEX(f_index_02, j)] + wrow1 * F[F_INDEX(f_index_12, j)] +
                         wrow2 * F[F_INDEX(f_index_22, j)] + wrow3 * F[F_INDEX(f_index_32, j)]) +
                wcol3 * (wrow0 * F[F_INDEX(f_index_03, j)] + wrow1 * F[F_INDEX(f_index_13, j)] +
                         wrow2 * F[F_INDEX(f_index_23, j)] + wrow3 * F[F_INDEX(f_index_33, j)]);
        }
    }
}

InterpolationMethod parseInterpolationMethod(const std::string method_str)
{
    InterpolationMethod method;

    if (method_str.compare("nearest") == 0)
    {
        method = Nearest;
    }
    else if (method_str.compare("bilinear") == 0)
    {
        method = Bilinear;
    }
    else if (method_str.compare("bicubic") == 0)
    {
        method = Bicubic;
    }
    else
    {
        RAISE_ERROR("Valid interpolation methods are bicubic, bilinear, nearest");
    }

    return method;
}

void interpolate(const NDARRAY *F, const NDARRAY *row_vector, const NDARRAY *col_vector,
                 const std::string type, double *out_data)
{
    // Sanity checks
    if (row_vector->n_dims != col_vector->n_dims)
    {
        RAISE_ERROR("Row indices vector and column indices vector must have the same number of dimensions");
    }

    if (F->n_dims > 3)
    {
        RAISE_ERROR("F must be an [ROWS x COLS x [CHANNELS]] array");
    }

    for (size_t i = 0; i < row_vector->n_dims; i++)
    {
        if (row_vector->dims[i] != col_vector->dims[i])
        {
            RAISE_ERROR("Row indices vector and column indices vector must match in the size of each dimension");
        }
    }

    const size_t N_ROWS = F->dims[0];
    const size_t N_COLS = F->dims[1];
    const size_t N_CHANNELS = F->n_dims == 3 ? F->dims[2] : 1;
    const size_t N_ELEMS = row_vector->n_elems;

    switch(parseInterpolationMethod(type))
    {
        case Nearest:
            interpolate_nearest(F->data, row_vector->data, col_vector->data, N_ELEMS, N_ROWS, N_COLS, N_CHANNELS, out_data);
            break;
        case Bilinear:
            interpolate_bilinear(F->data, row_vector->data, col_vector->data, N_ELEMS, N_ROWS, N_COLS, N_CHANNELS, out_data);
            break;
        case Bicubic:
            interpolate_bicubic(F->data, row_vector->data, col_vector->data, N_ELEMS, N_ROWS, N_COLS, N_CHANNELS, out_data);
            break;
        default:
            RAISE_ERROR("Interpolation method not supported");
    }
}
