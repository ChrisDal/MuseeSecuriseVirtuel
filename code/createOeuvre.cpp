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


int main( int argc, char** argv )
{
    cv::Mat image;
    int thresh = 154;
    int max_thresh = 255;
    const char* source_window = "Source image";

    if ( argc != 3 )
    {
        printf("usage: CreateOeuvre.out <EncryptedImage> <Oeuvre>\n");
        return -1;
    }
    
    image = cv::imread( argv[1], cv::IMREAD_GRAYSCALE );
    const char* name = argv[2]; 

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }


    const unsigned int pixA4_width = 2480; 
    const unsigned int pixA4_height = 3508; 


    cv::Mat imageFinal = cv::Mat::zeros(pixA4_height, pixA4_width, CV_8UC1); 

    if  (image.size[0] > pixA4_height || image.size[1] > pixA4_width)
    {
        printf("Image too big, please make it smaller than A4 300 DPI.\n"); 
        return -1; 
    }

    int middleX = (int)(pixA4_width / 2.0f);
    int middleY = (int)(pixA4_height / 2.0f); 

    int boundX = middleX - (int)((float)image.size[1] / 2.0f); 
    int boundY = middleY - (int)((float)image.size[0] / 2.0f); 

    std::cout << boundX << "," << boundY << std::endl; 

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

    return EXIT_SUCCESS; 
}
