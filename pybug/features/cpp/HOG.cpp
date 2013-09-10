#include "HOG.h"
#include <iostream>

HOG::HOG() {

}

HOG::~HOG() {

}

void HOG::apply(double *windowImage, unsigned int windowHeight, unsigned int windowWidth, bool imageIsGrayscale)
{
	int i, j, k;
	double s = 0;

	std::cout << "Hello from HOG's apply" << std::endl;
	if (imageIsGrayscale == false)
	{
		for (i = 0; i < windowHeight; i++)
			for (j = 0; j < windowHeight; j++)
				for (k = 0; k < 3; k++)
					s = s + windowImage[i+windowHeight*(j+windowWidth*k)];
		std::cout << "SUM = " << s << std::endl;
	}
	else {
		for (i = 0; i < windowHeight; i++)
			for (j = 0; j < windowHeight; j++)
				s = s + windowImage[i+windowHeight*j];
		std::cout << "SUM = " << s << std::endl;
	}
}
