#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <string.h>

#define eps 0.0001 // small value, used to avoid division by zero

// struct for information variables
struct wins {
   int numberOfWindowsHorizontally, numberOfWindowsVertically, numberOfWindows, windowHeight, windowWidth, windowStepHorizontal, windowStepVertical, windowSize[2];
   int numberOfBlocksPerWindowHorizontally, numberOfBlocksPerWindowVertically, descriptorLengthPerBlock, descriptorLengthPerWindow;
   int imageSize[2], imageHeight, imageWidth;
   unsigned int returnOnlyWindowsWithinImageLimits, inputImageIsGrayscale;
};

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }
static inline int min(int x, int y) { return (x <= y ? x : y); }
static inline int max(int x, int y) { return (x <= y ? y : x); }

// Function definition
struct wins WindowsInformation(double *options, int imageHeight, int imageWidth, unsigned int inputImageIsGrayscale);
void PrintInformation(double *options, struct wins info);
void MainLoop(double *options, struct wins info, double *windowImage, double *descriptorMatrix, double *descriptorVector, double *inputImage, double *WindowsMatrixDescriptorsMatrix, double *WindowsCentersMatrix);
void ZhuRamananHOGdescriptor(double *inputImage, int cellHeightAndWidthInPixels, int *imageSize, double *descriptorMatrix);
void DalalTriggsHOGdescriptor(double *inputImage, double *params, int *imageSize, double *descriptorVector, unsigned int grayscale);
