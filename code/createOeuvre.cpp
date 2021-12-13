#include <iostream> 
#include <string>

#include "BasePermutation.h"
#include "imageanalysis.h"

#include <opencv2/core/types.hpp>
#include <opencv2/core/fast_math.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>


// ======================================================================================

void createPattern(cv::Mat& pattern, int carreSize, int carreNumber)
{
    if (carreNumber % 2 == 0)
    {
        std::cout << " Only take odd number. Will take " << carreNumber + 1 << std::endl; 
        carreNumber++; 
    }

    cv::Mat carreWhite = cv::Mat::ones(carreSize, carreSize, CV_8UC1)*255; 
    pattern = cv::Mat::zeros(carreNumber * carreSize, carreNumber * carreSize, CV_8UC1); 

    cv::Mat matRoi = pattern(cv::Rect(0, 0, carreSize, carreSize));

    for (int kx = 0 ; kx < carreNumber; kx++)
    {
        for (int ky = 0 ; ky < carreNumber; ky++)
        {
            
            if ( (kx+ky) %2 == 1) // white  
            {
                // Create ROI and Copy A to it
                matRoi = pattern(cv::Rect(ky*carreSize, kx*carreSize, carreSize, carreSize));
                carreWhite.copyTo(matRoi); 
            }

        }
    }
}



// ======================================================================================


int main( int argc, char** argv )
{
    cv::Mat image;
    const char* source_window = "Source image";

    if ( argc != 7 )
    {
        printf("usage: CreateOeuvre.out <EncryptedImage> <SecretKey> <Oeuvre> <Oeuvre2> <Oeuvre3> <exportedPattern> \n");
        return -1;
    }
    
    image = cv::imread( argv[1], cv::IMREAD_GRAYSCALE );
    int secretKey = std::stoi(argv[2]); 
    const char* name  = argv[3]; 
    const char* name2 = argv[4]; 
    const char* name3 = argv[5]; 
    const char* exportedPattern = argv[6];

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    //initSeed(secretKey); 

    int dpi = 200; 
    int pixA4_width; 
    int pixA4_height; 

    if (dpi == 72) 
    {
        pixA4_width = 595; 
        pixA4_height = 842; 
    }
    else if (dpi == 200)
    {
        pixA4_width = 1654; 
        pixA4_height = 2339; 
    }
    else if (dpi == 300) // 300 dpi 
    {
        pixA4_width = 2480; 
        pixA4_height = 3508; 
    }
    else 
    {
        printf("Invalid DPI : 72, 200, 300 dpi supported only.\n");
        return -1;
    }
    

    cv::Mat imageFinal = cv::Mat::zeros(pixA4_height, pixA4_width, CV_8UC1); 
    cv::Mat imagePatternFinal = cv::Mat::ones(pixA4_height, pixA4_width, CV_8UC1)*255; 
    cv::Mat imageNoPattern = cv::Mat::ones(pixA4_height, pixA4_width, CV_8UC1)*255; 

    if  (image.size[0] > pixA4_height || image.size[1] > pixA4_width)
    {
        std::string message = "Image too big, please make it smaller than A4 " + std::to_string(dpi) + " DPI, or increase DPI.\n"; 
        printf(message.c_str()); 
        return -1; 
    }

    // Pattern Creation
    cv::Mat pattern, imageWithPattern; 
    int carreSize = 72; // in pixels 
    int number = 3; // 3 carres
    unsigned int patternSize = number * carreSize; 
    createPattern(pattern, carreSize, number); 
    // add liseret 
    cv::Mat liseret = cv::Mat::ones(patternSize+8, patternSize+8, CV_8UC1)*255; 
    cv::Mat roiLis = liseret(cv::Rect(4, 4, patternSize, patternSize));
    pattern.copyTo(roiLis); 
    cv::imwrite(exportedPattern, liseret); 

    // Image at the center of the A4 
    int middleX = (int)(pixA4_width / 2.0f);
    int middleY = (int)(pixA4_height / 2.0f); 

    int boundX = middleX - (int)((float)image.size[1] / 2.0f); 
    int boundY = middleY - (int)((float)image.size[0] / 2.0f); 

    std::cout << boundX << "," << boundY << std::endl; 

    // ========================================================================================================
    // ROI technic 
    cv::Mat roiForImage = imagePatternFinal(cv::Rect(boundX, boundY, image.cols, image.rows)); 
    image.copyTo(roiForImage);
    // place patterns 
    roiForImage = imagePatternFinal(cv::Rect(boundX - patternSize, boundY - patternSize,  patternSize, patternSize)); 
    pattern.copyTo(roiForImage); 
    roiForImage = imagePatternFinal(cv::Rect(boundX + image.size[1], boundY- patternSize,  patternSize, patternSize)); 
    pattern.copyTo(roiForImage); 
    roiForImage = imagePatternFinal(cv::Rect( boundX - patternSize, boundY + image.size[0], patternSize, patternSize)); 
    pattern.copyTo(roiForImage); 
    roiForImage = imagePatternFinal(cv::Rect(boundX + image.size[1], boundY + image.size[0],  patternSize, patternSize)); 
    pattern.copyTo(roiForImage); 

    cv::imwrite(name2, imagePatternFinal); 
    // ========================================================================================================
    // 2nd type of image :  with lines 

    for (int ky = 0; ky < pixA4_height; ky++)
    {

        for (int kx = 0; kx < pixA4_width; kx++)
        {
            bool inside= (kx >= boundX) && (ky >= boundY) && (kx < (boundX + image.size[1]  -1 )) && (ky < (boundY + image.size[0]-1)); 
            bool line1 = (kx == (boundX-3) || kx == (boundX-2) || kx == (boundX-1)); 
            bool line2 = (ky == (boundY-3) || ky == (boundY-2) || ky == (boundY-1)); 
            bool line3 = (kx == (boundX+image.size[1]+3) || kx == (boundX+image.size[1]+2) || kx == (boundX+image.size[1]+1)); 
            bool line4 = (ky == (boundY+image.size[0]+3) || ky == (boundY+image.size[0]+2) || ky == (boundY+image.size[0]+1));
            
            
            if (inside)
            {
                imageFinal.at<IMAGEMAT_TYPE>(ky, kx) = image.at<IMAGEMAT_TYPE>( ky -  boundY,kx - boundX); 
            }
            
            else if (line1 || line2 || line3 || line4)
            {
                imageFinal.at<IMAGEMAT_TYPE>(ky, kx) = 0; 
            }

            else 
            {
                imageFinal.at<IMAGEMAT_TYPE>(ky, kx) = 255; 
            }


        }

    }


    cv::imwrite(name, imageFinal); 

    // ===============================================================================================
    // 3rd type of image : no pattern 
    roiForImage = imageNoPattern(cv::Rect(boundX, boundY, image.cols, image.rows)); 
    image.copyTo(roiForImage);

    cv::imwrite(name3, imageNoPattern);

    // ===============================================================================================

    return EXIT_SUCCESS; 
}
