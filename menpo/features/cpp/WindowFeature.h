#pragma once

class WindowFeature {
public:
	WindowFeature();
	virtual ~WindowFeature();
	virtual void apply(double *windowImage, unsigned int numberOfChannels, bool imageIsGrayscale, double *descriptorVector) = 0;
	unsigned int descriptorLengthPerWindow;
};
