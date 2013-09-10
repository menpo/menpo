#pragma once

class WindowFeature {
public:
	WindowFeature();
	virtual ~WindowFeature();
	virtual void apply(double *windowImage, bool imageIsGrayscale, double *descriptorVector) = 0;
	virtual void print_information() = 0;
	unsigned int descriptorLengthPerWindow;
};
