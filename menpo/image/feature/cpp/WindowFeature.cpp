#include "WindowFeature.h"
#include <iostream>
#include <Python.h>

WindowFeature::WindowFeature() {
}

WindowFeature::~WindowFeature() {
}

bool WindowFeature::isApplyOnImage() {
    return false;
}

void WindowFeature::applyOnImage(const ImageWindowIterator &iwi, const double *image, double *outputImage, int *windowsCenters) {
    PyErr_SetString(PyExc_RuntimeError, "WindowFeature::applyOnImage is not implemented");
}
