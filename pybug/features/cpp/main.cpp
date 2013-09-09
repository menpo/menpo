#include <iostream>
#include "HOG.h"
#include "LBP.h"
#include "ImageWindowIterator.h"

int main(int argc, char** argv)
{
	HOG windowFeature = HOG();
	double a = 6;
	windowFeature.apply(&a, 4, 2);

	LBP windowFeature2 = LBP();
	double aa = 6;
	windowFeature2.apply(&aa, 4, 2);

	ImageWindowIterator iter = ImageWindowIterator(&a, 4, 2, &windowFeature);
	iter.apply();

	std::cout << "Hello from main function" << std::endl;
	return 0;
}
