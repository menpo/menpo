#pragma once

class WindowFeature {
public:
	WindowFeature();
	virtual ~WindowFeature();
	virtual void apply(double *windowImage, unsigned int windowHeight, unsigned int windowWidth, bool imageIsGrayscale) = 0;
};
