#pragma once

#if defined(_MSC_VER)
    #define round(x) ((x) >= 0 ? (long)((x)+0.5) : (long)((x)-0.5))
#endif

class WindowFeature {
public:
	WindowFeature();
	virtual ~WindowFeature();
	virtual void apply(double *windowImage, double *descriptorVector) = 0;
	unsigned int descriptorLengthPerWindow;
};
