#pragma once
#include "WindowFeature.h"

class HOG: public WindowFeature {
public:
	HOG();
	virtual ~HOG();
	void apply(double *windowImage, unsigned int windowHeight, unsigned int windowWidth, bool imageIsGrayscale);
};
