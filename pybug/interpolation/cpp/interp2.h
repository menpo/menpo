#ifndef INTERn_channels2_H_
#define INTERn_channels2_H_

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <stdexcept>

#if defined(IS_MEX)
    #include <mex.h>
    #define RAISE_ERROR(msg) mexErrMsgTxt(msg);
    // MATLAB is 1-based so we need to subtract 1 each time to index properly
    #define SAFE_INDEX(x) x - 1
    // MATLAB is column major, so to keep the indexing fast, we define indexing macros
    #define OUT_INDEX_OFFSET N_ELEMS
    #define F_INDEX_OFFSET F_MAX
    // Convenience macros to properly index in to array
    #define OUT_INDEX(i, j) i + j * OUT_INDEX_OFFSET
    #define F_INDEX(index, j) index + j * F_INDEX_OFFSET
#else
    #define RAISE_ERROR(msg) throw std::invalid_argument(msg);
    #define SAFE_INDEX(x) x
    // Python is row major, so to keep the indexing fast, we define indexing macros
    #define OUT_INDEX_OFFSET N_CHANNELS
    #define F_INDEX_OFFSET 1
    // Convenience macros to properly index in to array
    #define OUT_INDEX(i, j) i * OUT_INDEX_OFFSET + j
    #define F_INDEX(index, j) index + j
#endif

enum InterpolationMethod { Nearest, Bilinear, Bicubic };

struct NDARRAY
{
    const size_t n_dims;
    const size_t *dims;
    const size_t n_elems;
    double *data;
    
    NDARRAY(const size_t nd, const size_t *ds, const size_t ne, double *d):
        n_dims(nd), dims(ds), n_elems(ne), data(d)
    {
    }
};

inline double safe_round(const double number);

inline size_t index_array(const size_t N_ROWS, const size_t N_COLS, const size_t N_CHANNELS,
                       size_t row_index, size_t col_index);

void indices_linear(size_t &f_index_00, size_t &f_index_10, size_t &f_index_01, size_t &f_index_11,
                    const size_t row_index, const size_t col_index,
                    const size_t N_ROWS, const size_t N_COLS, const size_t N_CHANNELS);

void indices_cubic(size_t &f_index_00, size_t &f_index_10, size_t &f_index_20, size_t &f_index_30,
                   size_t &f_index_01, size_t &f_index_11, size_t &f_index_21, size_t &f_index_31,
                   size_t &f_index_02, size_t &f_index_12, size_t &f_index_22, size_t &f_index_32,
                   size_t &f_index_03, size_t &f_index_13, size_t &f_index_23, size_t &f_index_33,
                   const size_t row_index, const size_t col_index,
                   const size_t N_ROWS, const size_t N_COLS, const size_t N_CHANNELS);

void interpolate_nearest(const double *F, const double *row_vector, const double *col_vector,
                         const size_t N_ELEMS, const size_t N_ROWS, const size_t N_COLS, const size_t N_CHANNELS,
                         double *out);

void interpolate_bilinear(const double *F, const double *row_vector, const double *col_vector,
                         const size_t N_ELEMS, const size_t N_ROWS, const size_t N_COLS, const size_t N_CHANNELS,
                         double *out);

void interpolate_bicubic(const double *F, const double *row_vector, const double *col_vector,
                         const size_t N_ELEMS, const size_t N_ROWS, const size_t N_COLS, const size_t N_CHANNELS,
                         double *out);

InterpolationMethod parseInterpolationMethod(const std::string method_str);

void interpolate(const NDARRAY *F, const NDARRAY *row_vector, const NDARRAY *col_vector,
                 const std::string type, double *out_data);

#endif
