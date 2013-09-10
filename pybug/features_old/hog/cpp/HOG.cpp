#include "HOG.h"

using namespace std;

// MAIN LOOP PROCESSING
void MainLoop(double *options, struct wins info, double *windowImage, double *descriptorMatrix, double *descriptorVector, double *inputImage, double *WindowsMatrixDescriptorsMatrix, double *WindowsCentersMatrix)
{
    double *params;
    int rowCenter,rowFrom,rowTo,columnCenter,columnFrom,columnTo,windowCounter;
    unsigned int i,j,k,d,d1,d2,d3,d4,d5,windowIndexHorizontal,windowIndexVertical;
    double binsSize;

    if (options[8]==1)
    {
        params = new double[5];
        params[0] = options[9];
        params[1] = options[10];
        params[2] = options[11];
        params[3] = options[12];
        params[4] = options[13];
        windowCounter = -1;
        for (windowIndexVertical=0; windowIndexVertical<info.numberOfWindowsVertically; windowIndexVertical++)
        {
            for (windowIndexHorizontal=0; windowIndexHorizontal<info.numberOfWindowsHorizontally; windowIndexHorizontal++)
            {
                // Update window counter
                windowCounter = windowCounter + 1;

                // Find window edges coordinates
                //if (windowHeight==imageHeight && windowWidth==imageWidth)
                //{
                //    windowIndexVertical = windowIndexVertical + 1;
                //    windowIndexHorizontal = windowIndexHorizontal + 1;
                //    windowStepVertical = floor(imageHeight/2);
                //    windowStepHorizontal = floor(imageWidth/2);
                //}
                //rowCenter = windowIndexVertical*windowStepVertical;
                //rowFrom = rowCenter - (int)floor(windowHeight/2);
                //rowTo = rowCenter + (int)ceil(windowHeight/2) - 1;

                //columnCenter = windowIndexHorizontal*windowStepHorizontal;
                //columnFrom = columnCenter - (int)floor(windowWidth/2);
                //columnTo = columnCenter + (int)ceil(windowWidth/2) - 1;
                ////rowFrom = windowIndexVertical*windowStepVertical;
                ////rowTo = windowIndexVertical*windowStepVertical + windowHeight - 1;
                ////columnFrom = windowIndexHorizontal*windowStepHorizontal;
                ////columnTo = windowIndexHorizontal*windowStepHorizontal + windowWidth - 1;

                if (info.returnOnlyWindowsWithinImageLimits==1)
                {
                    rowFrom = windowIndexVertical*info.windowStepVertical;
                    rowTo = rowFrom + info.windowHeight - 1;
                    rowCenter = rowFrom + (int)round((double)info.windowHeight/2) - 1;
                    columnFrom = windowIndexHorizontal*info.windowStepHorizontal;
                    columnTo = columnFrom + info.windowWidth - 1;
                    columnCenter = columnFrom + (int)round((double)info.windowWidth/2) - 1;
                }
                else
                {
                    rowCenter = windowIndexVertical*info.windowStepVertical;
                    rowFrom = rowCenter - (int)round((double)info.windowHeight/2) + 1;
                    rowTo = rowFrom + info.windowHeight - 1;
                    columnCenter = windowIndexHorizontal*info.windowStepHorizontal;
                    columnFrom = columnCenter - (int)ceil((double)info.windowWidth/2) + 1;
                    columnTo = columnFrom + info.windowWidth - 1;
                }

                //mexPrintf("Rows(%d)=[%d | %d | %d]   Columns(%d)=[%d | %d | %d]\n",windowIndexVertical,rowFrom,rowCenter,rowTo,windowIndexHorizontal,columnFrom,columnCenter,columnTo);

                // Create window image
                if (info.inputImageIsGrayscale==1)
                {
                    for (i=rowFrom; i<=rowTo; i++)
                    {
                        for (j=columnFrom; j<=columnTo; j++)
                        {
                            if (i<0 || i>info.imageHeight-1 || j<0 || j>info.imageWidth-1)
                                windowImage[(i-rowFrom)+info.windowHeight*(j-columnFrom)] = 0;
                            else
                                windowImage[(i-rowFrom)+info.windowHeight*(j-columnFrom)] = inputImage[i+info.imageHeight*j];
                        }
                    }
                }
                else
                {
                    for (i=rowFrom; i<=rowTo; i++)
                    {
                        for (j=columnFrom; j<=columnTo; j++)
                        {
                            if (i<0 || i>info.imageHeight-1 || j<0 || j>info.imageWidth-1)
                            {
                                for (k=0; k<3; k++)
                                    windowImage[(i-rowFrom)+info.windowHeight*((j-columnFrom)+info.windowWidth*k)] = 0;
                            }
                            else
                            {
                                for (k=0; k<3; k++)
                                    windowImage[(i-rowFrom)+info.windowHeight*((j-columnFrom)+info.windowWidth*k)] = inputImage[i+info.imageHeight*(j+info.imageWidth*k)];
                            }
                        }
                    }
                }

                // Compute descriptor of window
                DalalTriggsHOGdescriptor(windowImage,params,info.windowSize,descriptorVector,info.inputImageIsGrayscale);
                d=0;
                for (d2=0; d2<info.numberOfBlocksPerWindowHorizontally; d2++) {
                    for (d1=0; d1<info.numberOfBlocksPerWindowVertically; d1++) {
                        for (d3=0; d3<info.descriptorLengthPerBlock; d3++) {
                            descriptorMatrix[d1+info.numberOfBlocksPerWindowVertically*(d2+info.numberOfBlocksPerWindowHorizontally*d3)] = descriptorVector[d];
                            d = d + 1; }}}

                // Store results
                for (d2=0; d2<info.numberOfBlocksPerWindowHorizontally; d2++)
                    for (d1=0; d1<info.numberOfBlocksPerWindowVertically; d1++)
                        for (d3=0; d3<info.descriptorLengthPerBlock; d3++)
                            WindowsMatrixDescriptorsMatrix[windowIndexVertical+info.numberOfWindowsVertically*(windowIndexHorizontal+info.numberOfWindowsHorizontally*(d1+info.numberOfBlocksPerWindowVertically*(d2+info.numberOfBlocksPerWindowHorizontally*d3)))] = descriptorMatrix[d1+info.numberOfBlocksPerWindowVertically*(d2+info.numberOfBlocksPerWindowHorizontally*d3)];
                WindowsCentersMatrix[windowIndexVertical+info.numberOfWindowsVertically*windowIndexHorizontal] = rowCenter + 1;
                WindowsCentersMatrix[windowIndexVertical+info.numberOfWindowsVertically*(windowIndexHorizontal+info.numberOfWindowsHorizontally)] = columnCenter + 1;
            }
        }
    }
    else if (options[8]==2)
    {
        windowCounter = -1;
        for (windowIndexVertical=0; windowIndexVertical<info.numberOfWindowsVertically; windowIndexVertical++)
        {
            for (windowIndexHorizontal=0; windowIndexHorizontal<info.numberOfWindowsHorizontally; windowIndexHorizontal++)
            {
                // Update window counter
                windowCounter = windowCounter + 1;

                // Find window edges coordinates
                //if (windowHeight==imageHeight && windowWidth==imageWidth)
                //{
                //    windowIndexVertical = windowIndexVertical + 1;
                //    windowIndexHorizontal = windowIndexHorizontal + 1;
                //    windowStepVertical = floor(imageHeight/2);
                //    windowStepHorizontal = floor(imageWidth/2);
                //}
                //rowCenter = windowIndexVertical*windowStepVertical;
                //rowFrom = rowCenter - (int)floor(windowHeight/2);
                //rowTo = rowCenter + (int)ceil(windowHeight/2) - 1;

                //columnCenter = windowIndexHorizontal*windowStepHorizontal;
                //columnFrom = columnCenter - (int)floor(windowWidth/2);
                //columnTo = columnCenter + (int)ceil(windowWidth/2) - 1;
                ////rowFrom = windowIndexVertical*windowStepVertical;
                ////rowTo = windowIndexVertical*windowStepVertical + windowHeight - 1;
                ////columnFrom = windowIndexHorizontal*windowStepHorizontal;
                ////columnTo = windowIndexHorizontal*windowStepHorizontal + windowWidth - 1;

                if (info.returnOnlyWindowsWithinImageLimits==1)
                {
                    rowFrom = windowIndexVertical*info.windowStepVertical;
                    rowTo = rowFrom + info.windowHeight - 1;
                    rowCenter = rowFrom + (int)round((double)info.windowHeight/2) - 1;
                    columnFrom = windowIndexHorizontal*info.windowStepHorizontal;
                    columnTo = columnFrom + info.windowWidth - 1;
                    columnCenter = columnFrom + (int)round((double)info.windowWidth/2) - 1;
                }
                else
                {
                    rowCenter = windowIndexVertical*info.windowStepVertical;
                    rowFrom = rowCenter - (int)round((double)info.windowHeight/2) + 1;
                    rowTo = rowFrom + info.windowHeight - 1;
                    columnCenter = windowIndexHorizontal*info.windowStepHorizontal;
                    columnFrom = columnCenter - (int)ceil((double)info.windowWidth/2) + 1;
                    columnTo = columnFrom + info.windowWidth - 1;
                }

                // Create window image
                if (info.inputImageIsGrayscale==1)
                {
                    for (i=rowFrom; i<=rowTo; i++)
                    {
                        for (j=columnFrom; j<=columnTo; j++)
                        {
                            if (i<0 || i>info.imageHeight-1 || j<0 || j>info.imageWidth-1)
                                windowImage[(i-rowFrom)+info.windowHeight*(j-columnFrom)] = 0;
                            else
                                windowImage[(i-rowFrom)+info.windowHeight*(j-columnFrom)] = inputImage[i+info.imageHeight*j];
                        }
                    }
                }
                else
                {
                    for (i=rowFrom; i<=rowTo; i++)
                    {
                        for (j=columnFrom; j<=columnTo; j++)
                        {
                            if (i<0 || i>info.imageHeight-1 || j<0 || j>info.imageWidth-1)
                            {
                                for (k=0; k<3; k++)
                                    windowImage[(i-rowFrom)+info.windowHeight*((j-columnFrom)+info.windowWidth*k)] = 0;
                            }
                            else
                            {
                                for (k=0; k<3; k++)
                                    windowImage[(i-rowFrom)+info.windowHeight*((j-columnFrom)+info.windowWidth*k)] = inputImage[i+info.imageHeight*(j+info.imageWidth*k)];
                            }
                        }
                    }
                }

                // Compute descriptor of window
                ZhuRamananHOGdescriptor(windowImage,(int)options[10],info.windowSize,descriptorMatrix);

                // Store results
                for (d2=0; d2<info.numberOfBlocksPerWindowHorizontally; d2++)
                    for (d1=0; d1<info.numberOfBlocksPerWindowVertically; d1++)
                        for (d3=0; d3<info.descriptorLengthPerBlock; d3++)
                            WindowsMatrixDescriptorsMatrix[windowIndexVertical+info.numberOfWindowsVertically*(windowIndexHorizontal+info.numberOfWindowsHorizontally*(d1+info.numberOfBlocksPerWindowVertically*(d2+info.numberOfBlocksPerWindowHorizontally*d3)))] = descriptorMatrix[d1+info.numberOfBlocksPerWindowVertically*(d2+info.numberOfBlocksPerWindowHorizontally*d3)];
                WindowsCentersMatrix[windowIndexVertical+info.numberOfWindowsVertically*windowIndexHorizontal] = rowCenter + 1;
                WindowsCentersMatrix[windowIndexVertical+info.numberOfWindowsVertically*(windowIndexHorizontal+info.numberOfWindowsHorizontally)] = columnCenter + 1;
            }
        }
    }
}


// ZHU & RAMANAN: Face Detection, Pose Estimation and Landmark Localization in the Wild
void ZhuRamananHOGdescriptor(double *inputImage, int cellHeightAndWidthInPixels, int *imageSize, double *descriptorMatrix)
{
    // unit vectors used to compute gradient orientation
    double uu[9] = {1.0000,0.9397,0.7660,0.500,0.1736,-0.1736,-0.5000,-0.7660,-0.9397};
    double vv[9] = {0.0000,0.3420,0.6428,0.8660,0.9848,0.9848,0.8660,0.6428,0.3420};

    int imageWidth  = imageSize[1];
    int imageHeight = imageSize[0];
    int x,y,o;

    // memory for caching orientation histograms & their norms
    int blocks[2];
    blocks[0] = (int)round((double)imageSize[0]/(double)cellHeightAndWidthInPixels);
    blocks[1] = (int)round((double)imageSize[1]/(double)cellHeightAndWidthInPixels);
    double *hist = (double *)calloc(blocks[0]*blocks[1]*18, sizeof(double));
    double *norm = (double *)calloc(blocks[0]*blocks[1], sizeof(double));

    // memory for HOG features
    int out[3];
    out[0] = max(blocks[0]-2, 0); //You can change this to out[0] = max(blocks[0]-1, 0); and out[1] = max(blocks[1]-1, 0), in order to return the same output size as dalaltriggs
    out[1] = max(blocks[1]-2, 0); //I did the same change in lines 231,232
    out[2] = 27+4;

    int visible[2];
    visible[0] = blocks[0]*cellHeightAndWidthInPixels;
    visible[1] = blocks[1]*cellHeightAndWidthInPixels;

    for (x=1; x<visible[1]-1; x++)
    {
        for (y=1; y<visible[0]-1; y++)
        {
            // first color channel
            double *s = inputImage + min(x,imageSize[1]-2)*imageSize[0] + min(y,imageSize[0]-2);
            double dy = *(s+1) - *(s-1);
            double dx = *(s+imageSize[0]) - *(s-imageSize[0]);
            double v = dx*dx + dy*dy;

            // second color channel
            s += imageSize[0]*imageSize[1];
            double dy2 = *(s+1) - *(s-1);
            double dx2 = *(s+imageSize[0]) - *(s-imageSize[0]);
            double v2 = dx2*dx2 + dy2*dy2;

            // third color channel
            s += imageSize[0]*imageSize[1];
            double dy3 = *(s+1) - *(s-1);
            double dx3 = *(s+imageSize[0]) - *(s-imageSize[0]);
            double v3 = dx3*dx3 + dy3*dy3;

            // pick channel with strongest gradient
            if (v2>v)
            {
                v = v2;
                dx = dx2;
                dy = dy2;
            }
            if (v3>v)
            {
                v = v3;
                dx = dx3;
                dy = dy3;
            }

            // snap to one of 18 orientations
            double best_dot = 0;
            int best_o = 0;
            for (o=0; o<9; o++)
            {
                double dot = uu[o]*dx + vv[o]*dy;
                if (dot>best_dot)
                {
                    best_dot = dot;
                    best_o = o;
                }
                else if (-dot>best_dot)
                {
                    best_dot = -dot;
                    best_o = o + 9;
                }
            }

            // add to 4 histograms around pixel using linear interpolation
            double xp = ((double)x+0.5)/(double)cellHeightAndWidthInPixels - 0.5;
            double yp = ((double)y+0.5)/(double)cellHeightAndWidthInPixels - 0.5;
            int ixp = (int)floor(xp);
            int iyp = (int)floor(yp);
            double vx0 = xp-ixp;
            double vy0 = yp-iyp;
            double vx1 = 1.0-vx0;
            double vy1 = 1.0-vy0;
            v = sqrt(v);

            if (ixp>=0 && iyp>=0)
                *(hist + ixp*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) += vx1*vy1*v;


            if (ixp+1 < blocks[1] && iyp >= 0)
                *(hist + (ixp+1)*blocks[0] + iyp + best_o*blocks[0]*blocks[1]) += vx0*vy1*v;

            if (ixp >= 0 && iyp+1 < blocks[0])
                *(hist + ixp*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) += vx1*vy0*v;

            if (ixp+1 < blocks[1] && iyp+1 < blocks[0])
                *(hist + (ixp+1)*blocks[0] + (iyp+1) + best_o*blocks[0]*blocks[1]) += vx0*vy0*v;
        }
    }

    // compute energy in each block by summing over orientations
    for (int o = 0; o < 9; o++)
    {
        double *src1 = hist + o*blocks[0]*blocks[1];
        double *src2 = hist + (o+9)*blocks[0]*blocks[1];
        double *dst = norm;
        double *end = norm + blocks[1]*blocks[0];
        while (dst < end)
        {
            *(dst++) += (*src1 + *src2) * (*src1 + *src2);
            src1++;
            src2++;
        }
    }

    // compute features
    for (x=0; x<out[1]; x++)
    {
        for (y=0; y<out[0]; y++)
        {
            double *dst = descriptorMatrix + x*out[0] + y;
            double *src, *p, n1, n2, n3, n4;

            p = norm + (x+1)*blocks[0] + y+1;
            n1 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
            p = norm + (x+1)*blocks[0] + y;
            n2 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
            p = norm + x*blocks[0] + y+1;
            n3 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
            p = norm + x*blocks[0] + y;
            n4 = 1.0 / sqrt(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);

            double t1 = 0;
            double t2 = 0;
            double t3 = 0;
            double t4 = 0;

            // contrast-sensitive features
            src = hist + (x+1)*blocks[0] + (y+1);
            for (int o = 0; o < 18; o++)
            {
                double h1 = min(*src * n1, 0.2);
                double h2 = min(*src * n2, 0.2);
                double h3 = min(*src * n3, 0.2);
                double h4 = min(*src * n4, 0.2);
                *dst = 0.5 * (h1 + h2 + h3 + h4);
                t1 += h1;
                t2 += h2;
                t3 += h3;
                t4 += h4;
                dst += out[0]*out[1];
                src += blocks[0]*blocks[1];
            }

            // contrast-insensitive features
            src = hist + (x+1)*blocks[0] + (y+1);
            for (int o = 0; o < 9; o++)
            {
                double sum = *src + *(src + 9*blocks[0]*blocks[1]);
                double h1 = min(sum * n1, 0.2);
                double h2 = min(sum * n2, 0.2);
                double h3 = min(sum * n3, 0.2);
                double h4 = min(sum * n4, 0.2);
                *dst = 0.5 * (h1 + h2 + h3 + h4);
                dst += out[0]*out[1];
                src += blocks[0]*blocks[1];
            }

            // texture features
            *dst = 0.2357 * t1;
            dst += out[0]*out[1];
            *dst = 0.2357 * t2;
            dst += out[0]*out[1];
            *dst = 0.2357 * t3;
            dst += out[0]*out[1];
            *dst = 0.2357 * t4;
        }
    }
    free(hist);
    free(norm);
}



// DALAL & TRIGGS: Histograms of Oriented Gradients for Human Detection
void DalalTriggsHOGdescriptor(double *inputImage, double *params, int *imageSize, double *descriptorVector, unsigned int grayscale)
{

    const float pi = 3.1415926536;

    int numberOfOrientationBins       = (int) params[0];
    double cellHeightAndWidthInPixels     =  params[1];
    int blockHeightAndWidthInCells    = (int) params[2];
    int signedOrUnsignedGradients        = (int) params[3];
    double l2normClipping   = params[4];

    int imageWidth  = imageSize[1];
    int imageHeight = imageSize[0];

    int hist1= 2+ceil(-0.5 + imageHeight/cellHeightAndWidthInPixels);
    int hist2= 2+ceil(-0.5 + imageWidth/cellHeightAndWidthInPixels);

    double binsSize = (1+(signedOrUnsignedGradients==1))*pi/numberOfOrientationBins;

    float dx[3], dy[3], gradientOrientation, gradientMagnitude, tempMagnitude;
    float Xc, Yc, Oc, blockNorm;
    int x1, x2, y1, y2, bin1, bin2;
    int descriptorIndex = 0;

    vector<vector<vector<double> > > h(hist1, vector<vector<double> > (hist2, vector<double> (numberOfOrientationBins, 0.0) ) );
    vector<vector<vector<double> > > block(blockHeightAndWidthInCells, vector<vector<double> > (blockHeightAndWidthInCells, vector<double> (numberOfOrientationBins, 0.0) ) );

    //Calculate gradients (zero padding)
    for(unsigned int y=0; y<imageHeight; y++) {
        for(unsigned int x=0; x<imageWidth; x++) {
            if (grayscale == 1){
                if(x==0) dx[0] = inputImage[y +(x+1)*imageHeight];
                else{
                    if (x==imageWidth-1) dx[0] = -inputImage[y + (x-1)*imageHeight];
                    else dx[0] = inputImage[y+(x+1)*imageHeight] - inputImage[y + (x-1)*imageHeight];
                }
                if(y==0) dy[0] = -inputImage[y+1+x*imageHeight];
                else{
                    if (y==imageHeight-1) dy[0] = inputImage[y-1+x*imageHeight];
                    else dy[0] = -inputImage[y+1+x*imageHeight] + inputImage[y-1+x*imageHeight];
                }
            }
            else{
                if(x==0){
                    dx[0] = inputImage[y +(x+1)*imageHeight];
                    dx[1] = inputImage[y +(x+1)*imageHeight + imageHeight*imageWidth];
                    dx[2] = inputImage[y +(x+1)*imageHeight + 2*imageHeight*imageWidth];
                }
                else{
                    if (x==imageWidth-1){
                        dx[0] = -inputImage[y + (x-1)*imageHeight];
                        dx[1] = -inputImage[y + (x-1)*imageHeight + imageHeight*imageWidth];
                        dx[2] = -inputImage[y + (x-1)*imageHeight + 2*imageHeight*imageWidth];
                    }
                    else{
                        dx[0] = inputImage[y+(x+1)*imageHeight] - inputImage[y + (x-1)*imageHeight];
                        dx[1] = inputImage[y+(x+1)*imageHeight + imageHeight*imageWidth] - inputImage[y + (x-1)*imageHeight + imageHeight*imageWidth];
                        dx[2] = inputImage[y+(x+1)*imageHeight + 2*imageHeight*imageWidth] - inputImage[y + (x-1)*imageHeight + 2*imageHeight*imageWidth];

                    }
                }
                if(y==0){
                    dy[0] = -inputImage[y+1+x*imageHeight];
                    dy[1] = -inputImage[y+1+x*imageHeight + imageHeight*imageWidth];
                    dy[2] = -inputImage[y+1+x*imageHeight + 2*imageHeight*imageWidth];
                }
                else{
                    if (y==imageHeight-1){
                        dy[0] = inputImage[y-1+x*imageHeight];
                        dy[1] = inputImage[y-1+x*imageHeight + imageHeight*imageWidth];
                        dy[2] = inputImage[y-1+x*imageHeight + 2*imageHeight*imageWidth];
                    }
                    else{
                        dy[0] = -inputImage[y+1+x*imageHeight] + inputImage[y-1+x*imageHeight];
                        dy[1] = -inputImage[y+1+x*imageHeight + imageHeight*imageWidth] + inputImage[y-1+x*imageHeight + imageHeight*imageWidth];
                        dy[2] = -inputImage[y+1+x*imageHeight + 2*imageHeight*imageWidth] + inputImage[y-1+x*imageHeight + 2*imageHeight*imageWidth];
                    }
                }
            }

            gradientMagnitude = sqrt(dx[0]*dx[0] + dy[0]*dy[0]);
            gradientOrientation= atan2(dy[0], dx[0]);

            if (grayscale == 0){
                tempMagnitude = gradientMagnitude;
                for (unsigned int cli=1;cli<3;++cli){
                    tempMagnitude= sqrt(dx[cli]*dx[cli] + dy[cli]*dy[cli]);
                    if (tempMagnitude>gradientMagnitude){
                        gradientMagnitude=tempMagnitude;
                        gradientOrientation= atan2(dy[cli], dx[cli]);
                    }
                }
            }

            if (gradientOrientation<0) gradientOrientation+=pi + (signedOrUnsignedGradients==1) * pi;

            // trilinear interpolation

            bin1 = (int)floor(0.5 + gradientOrientation/binsSize) - 1;
            bin2 = bin1 + 1;
            //mexPrintf("x=%d, y=%d, bin1=%d, bin2=%d\n",x,y,bin1,bin2);
            x1   = (int)floor(0.5+ x/cellHeightAndWidthInPixels);
            x2   = x1+1;
            y1   = (int)floor(0.5+ y/cellHeightAndWidthInPixels);
            y2   = y1 + 1;

            Xc = (x1+1-1.5)*cellHeightAndWidthInPixels + 0.5;
            Yc = (y1+1-1.5)*cellHeightAndWidthInPixels + 0.5;

            Oc = (bin1+1+1-1.5)*binsSize;

            if (bin2==numberOfOrientationBins){
                bin2=0;
            }
            if (bin1<0){
                bin1=numberOfOrientationBins-1;
            }

            h[y1][x1][bin1]= h[y1][x1][bin1] + gradientMagnitude * (1-((x+1-Xc)/cellHeightAndWidthInPixels)) * (1-((y+1-Yc)/cellHeightAndWidthInPixels))*(1-((gradientOrientation-Oc)/binsSize));
            h[y1][x1][bin2]= h[y1][x1][bin2] + gradientMagnitude * (1-((x+1-Xc)/cellHeightAndWidthInPixels)) * (1-((y+1-Yc)/cellHeightAndWidthInPixels))*(((gradientOrientation-Oc)/binsSize));
            h[y2][x1][bin1]= h[y2][x1][bin1] + gradientMagnitude * (1-((x+1-Xc)/cellHeightAndWidthInPixels)) * (((y+1-Yc)/cellHeightAndWidthInPixels))*(1-((gradientOrientation-Oc)/binsSize));
            h[y2][x1][bin2]= h[y2][x1][bin2] + gradientMagnitude * (1-((x+1-Xc)/cellHeightAndWidthInPixels)) * (((y+1-Yc)/cellHeightAndWidthInPixels))*(((gradientOrientation-Oc)/binsSize));
            h[y1][x2][bin1]= h[y1][x2][bin1] + gradientMagnitude * (((x+1-Xc)/cellHeightAndWidthInPixels)) * (1-((y+1-Yc)/cellHeightAndWidthInPixels))*(1-((gradientOrientation-Oc)/binsSize));
            h[y1][x2][bin2]= h[y1][x2][bin2] + gradientMagnitude * (((x+1-Xc)/cellHeightAndWidthInPixels)) * (1-((y+1-Yc)/cellHeightAndWidthInPixels))*(((gradientOrientation-Oc)/binsSize));
            h[y2][x2][bin1]= h[y2][x2][bin1] + gradientMagnitude * (((x+1-Xc)/cellHeightAndWidthInPixels)) * (((y+1-Yc)/cellHeightAndWidthInPixels))*(1-((gradientOrientation-Oc)/binsSize));
            h[y2][x2][bin2]= h[y2][x2][bin2] + gradientMagnitude * (((x+1-Xc)/cellHeightAndWidthInPixels)) * (((y+1-Yc)/cellHeightAndWidthInPixels))*(((gradientOrientation-Oc)/binsSize));
        }
    }



    //Block normalization
    for(unsigned int x=1; x<hist2-blockHeightAndWidthInCells; x++)
    {
        for (unsigned int y=1; y<hist1-blockHeightAndWidthInCells; y++)
        {

            blockNorm=0;
            for (unsigned int i=0; i<blockHeightAndWidthInCells; i++)
            {
                for(unsigned int j=0; j<blockHeightAndWidthInCells; j++)
                {
                    for(unsigned int k=0; k<numberOfOrientationBins; k++)
                    {
                        blockNorm+=h[y+i][x+j][k]*h[y+i][x+j][k];
                    }
                }
            }

            blockNorm=sqrt(blockNorm);
            for (unsigned int i=0; i<blockHeightAndWidthInCells; i++)
            {
                for(unsigned int j=0; j<blockHeightAndWidthInCells; j++)
                {
                    for(unsigned int k=0; k<numberOfOrientationBins; k++)
                    {
                        if (blockNorm>0)
                        {
                            block[i][j][k]=h[y+i][x+j][k]/blockNorm;
                            if (block[i][j][k]>l2normClipping)
                                block[i][j][k]=l2normClipping;
                        }
                    }
                }
            }

            blockNorm=0;
            for (unsigned int i=0; i<blockHeightAndWidthInCells; i++)
            {
                for(unsigned int j=0; j<blockHeightAndWidthInCells; j++)
                {
                    for(unsigned int k=0; k<numberOfOrientationBins; k++)
                    {
                        blockNorm+=block[i][j][k]*block[i][j][k];
                    }
                }
            }

            blockNorm=sqrt(blockNorm);
            for (unsigned int i=0; i<blockHeightAndWidthInCells; i++)
            {
                for(unsigned int j=0; j<blockHeightAndWidthInCells; j++)
                {
                    for(unsigned int k=0; k<numberOfOrientationBins; k++)
                    {
                        if (blockNorm>0)
                            descriptorVector[descriptorIndex]=block[i][j][k]/blockNorm;
                        else
                            descriptorVector[descriptorIndex]=0.0;
                        descriptorIndex++;
                    }
                }
            }
        }
    }
}

// WINDOWS INFORMATION
struct wins WindowsInformation(double *options, int imageHeight, int imageWidth, unsigned int inputImageIsGrayscale)
{
    struct wins info;

    // Load window related options for dense case
    if (options[0]==2)
    {
        // Load windowSize-related options
        if (options[3]==1)
        {
            int windowHeightInBlocks = (int)options[1];
            int windowWidthInBlocks = (int)options[2];
            info.windowHeight = (int)windowHeightInBlocks*options[11]*options[10];
            info.windowWidth = (int)windowWidthInBlocks*options[11]*options[10];
        }
        else if (options[3]==2)
        {
            info.windowHeight = (int)options[1];
            info.windowWidth = (int)options[2];
        }
        // Load windowStep-related options
        if (options[6]==1)
        {
            int windowStepHorizontalInCells = (int)options[4];
            int windowStepVerticalInCells = (int)options[5];
            info.windowStepHorizontal = (int)windowStepHorizontalInCells*options[10];
            info.windowStepVertical = (int)windowStepVerticalInCells*options[10];
        }
        else if (options[6]==2)
        {
            info.windowStepHorizontal = (int)options[4];
            info.windowStepVertical = (int)options[5];
        }
        // Return olny windows within image limits
        info.returnOnlyWindowsWithinImageLimits = (unsigned int)options[7];
    }
    else if (options[0]==1)
    {
        info.windowHeight = imageHeight;
        info.windowWidth = imageWidth;
        info.windowStepHorizontal = round(imageWidth/2);
        info.windowStepVertical = round(imageHeight/2);
        info.returnOnlyWindowsWithinImageLimits = 1;
    }

    // Find number of windows based on options and initialize window matrix
    if (info.returnOnlyWindowsWithinImageLimits==1)
    {
        info.numberOfWindowsHorizontally = 1+floor((imageWidth-info.windowWidth)/info.windowStepHorizontal);
        info.numberOfWindowsVertically = 1+floor((imageHeight-info.windowHeight)/info.windowStepVertical);
    }
    else
    {
        info.numberOfWindowsHorizontally = ceil(imageWidth/info.windowStepHorizontal);
        info.numberOfWindowsVertically = ceil(imageHeight/info.windowStepVertical);
    }
    info.numberOfWindows = info.numberOfWindowsHorizontally*info.numberOfWindowsVertically;

    info.inputImageIsGrayscale = inputImageIsGrayscale;
    info.imageWidth = imageWidth;
    info.imageHeight = imageHeight;
    info.imageSize[0] = imageHeight;
    info.imageSize[1] = imageWidth;

    info.windowSize[0] = info.windowHeight;
    info.windowSize[1] = info.windowWidth;

    return info;
}

// PRINT INFORMATION
void PrintInformation(double *options, struct wins info)
{
    if (options[14]==1)
    {
        printf("Input image: %d x %d pixels, ",info.imageHeight,info.imageWidth);
        if (info.inputImageIsGrayscale==1)
            printf("Grayscale\n");
        else
            printf("RGB\n");
        if (options[0]==1)
            printf("SPARSE HOGs\n");
        else
            printf("DENSE HOGs\n");
        printf("Windows Size = %d x %d pixels\n",info.windowHeight,info.windowWidth);
        printf("Windows Step = %d x %d pixels\n",info.windowStepVertical,info.windowStepHorizontal);
        printf("Number of Windows = %d x %d = %d\n",info.numberOfWindowsVertically,info.numberOfWindowsHorizontally,info.numberOfWindows);
        printf("Descriptor per window = %d x %d x %d = %d x 1\n",info.numberOfBlocksPerWindowVertically,info.numberOfBlocksPerWindowHorizontally,info.descriptorLengthPerBlock,info.descriptorLengthPerWindow);
        if (options[8]==1)
            printf("Method of Dalal & Triggs\n");
        else if (options[8]==2)
            printf("Method of Zhu & Ramanan\n");
    }
}