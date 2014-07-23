#pragma once
#include "ImageWindowIterator.h"

#if defined(_MSC_VER)
    #define round(x) (x >= 0 ? (x + 0.5) : (x - 0.5))
#endif

class ImageWindowIterator;
class WindowFeature {
public:
	WindowFeature();
	virtual ~WindowFeature();
	virtual void applyOnChunk(double *windowImage, double *descriptorVector) = 0;
    virtual void applyOnImage(const ImageWindowIterator &iwi, const double *image, double *outputImage, int *windowsCenters);
	unsigned int descriptorLengthPerWindow;
    virtual bool isApplyOnImage(); // Otherwise it applies on precomputed chunks
};
