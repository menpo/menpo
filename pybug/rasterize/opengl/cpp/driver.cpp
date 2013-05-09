#include <iostream>
#include <stdint.h>
#include "MM3DRenderer.h"


void generateTextureVector(double* texture)
{
    // texture
    //c1
    texture[0]  =  1.00f;
    texture[1]  =  0.00f;
    texture[2]  =  0.00f;
    texture[3]  =  1.00f;
    //c2
    texture[4]  =  0.00f;
    texture[5]  =  1.00f;
    texture[6]  =  0.00f;
    texture[7]  =  1.00f;
    //c3
    texture[8]  =  0.00f;
    texture[9]  =  0.00f;
    texture[10] =  1.00f;
    texture[11] =  1.00f;
    //c4
    texture[12]  =  1.00f;
    texture[13]  =  0.00f;
    texture[14]  =  0.00f;
    texture[15]  =  1.00f;
    //c5
    texture[16]  =  0.00f;
    texture[17]  =  1.00f;
    texture[18]  =  0.00f;
    texture[19]  =  1.00f;
    //c6
    texture[20]  =  0.00f;
    texture[21]  =  0.00f;
    texture[22] =  1.00f;
    texture[23] =  1.00f;
}

void generateTpsCoord(double* tpsCoord)
{
    //v1
    tpsCoord[0]  =  0.f;
    tpsCoord[1]  =  0.f;
    tpsCoord[2]  =  0.000f;
    tpsCoord[3]  =  1.00f;
    //v2
    tpsCoord[4]  =  0.f;
    tpsCoord[5]  =  0.75f;
    tpsCoord[6]  =  0.00f;
    tpsCoord[7]  =  1.00f;
    //v3
    tpsCoord[8]  =  0.75f;
    tpsCoord[9]  =  0.00f;
    tpsCoord[10] =  0.00f;
    tpsCoord[11] =  1.00f;
}

void generateTextureImage(uint8_t *textureImage)
{
    for(int i = 0; i < 64*64; i = i + 4)
    {
        textureImage[i] =   255;
        textureImage[i+1] = 0;
        textureImage[i+2] = 0;
        textureImage[i+3] = 255;
    }
    for(int i = 64*64; i < 64*64*2; i = i + 4)
    {
        textureImage[i] =   0;
        textureImage[i+1] = 255;
        textureImage[i+2] = 0;
        textureImage[i+3] = 255;
    }
    for(int i = 64*64*2; i < 64*64*3; i = i + 4)
    {
        textureImage[i] =   0;
        textureImage[i+1] = 0;
        textureImage[i+2] = 255;
        textureImage[i+3] = 255;
    }
    for(int i = 64*64*3; i < 64*64*4; i = i + 4)
    {
        textureImage[i] =   255;
        textureImage[i+1] = 255;
        textureImage[i+2] = 0;
        textureImage[i+3] = 255;
    }
}

void generateTexCoord(float* tc)
{
    tc[0]  = 0.;
    tc[1]  = 0.;

    tc[2]  = 0.0;
    tc[3]  = 1.0;

    tc[4]  = 1.0;
    tc[5]  = 0.0;
}

void generateCoord(float* c)
{
    c[0]  = 0.;
    c[1]  = 0.;
    c[2]  = 1.0;

    c[3]  = 0.0;
    c[4]  = 0.0;
    c[5]  = 1.0;

    c[6]  = 0.0;
    c[7]  = 0.0;
    c[8]  = 1.0;
}

int main(int argc, char** argv)
{
    double * tpsCoord = new double[24];
    float* coord = new float[9];
    double * textureVector = new double[24];
    uint8_t* textureImage = new uint8_t[64*64*4];
    float* texCoord = new float[12];

    unsigned int indexData[] =
    {
        0, 2, 1,
    };
    size_t numCoord = 3;
    size_t numTriangle = 1;
    size_t tW = 64;
    size_t tH = 64;

    generateTpsCoord(tpsCoord);
    generateCoord(coord);
    generateTexCoord(texCoord);
    generateTextureImage(textureImage);
    generateTextureVector(textureVector);

    MM3DRenderer renderer(tpsCoord, coord, numCoord, indexData, numTriangle, texCoord, textureImage, tW,tH,true);

    int width = 128;
    int height = 128;
    uint8_t* pixels = new uint8_t[width*height*4];
    float* coordResult = new float[width*height*3];
    renderer.render(argc, argv);

    int count = 0;
    for(int i = 0; i < height*width*3; i++){
        if(coordResult[i] > 0.1)
            count++;
    }
    double ratio  = count*1.0/(height*width*3.0);
    std::cout << "Proportion non-black: " << ratio << std::endl;
    delete[] tpsCoord;
    delete[] coord;
    delete[] textureVector;
    delete[] textureImage;
    delete[] texCoord;
    delete[] pixels;
    delete[] coordResult;
    return(EXIT_SUCCESS);
}

