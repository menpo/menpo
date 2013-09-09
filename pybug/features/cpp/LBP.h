#pragma once
#include "WindowFeature.h"

class LBP : public WindowFeature {
public:
	LBP();
	virtual ~LBP();
	void apply(double *windowImage, unsigned int windowHeight, unsigned int windowWidth);
};
