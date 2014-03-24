#pragma once

class WindowFeature {
public:
	WindowFeature();
	virtual ~WindowFeature();
	virtual void apply(double *windowImage, unsigned int numberOfChannels, double *descriptorVector) = 0;
	unsigned int descriptorLengthPerWindow;
};
