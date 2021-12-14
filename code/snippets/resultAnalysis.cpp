#include <iostream>
#include <string>

#include "../toolFunctions.hpp"
#include "../imageanalysis.h"


// Simple script to compare 3 images 

int main(int argc, char** argv )
{

    if ( argc != 4 )
    {
        printf("usage: Analyse.out <ImagedeReference> <ImageBlocky> <ImageDechiffreeBlocky>\n");
        return -1;
    }

    cv::Mat imageref, imageblocky, imagedechiffree;
    imageref = cv::imread( argv[1], cv::IMREAD_GRAYSCALE);
    imageblocky = cv::imread( argv[2], cv::IMREAD_GRAYSCALE);
    imagedechiffree = cv::imread( argv[3], cv::IMREAD_GRAYSCALE);
    

    if ( !imageref.data || !imageblocky.data || !imagedechiffree.data )
    {
        printf("No image data \n");
        return -1;
    }
    else 
    {
        std::cout << "Images Loaded."  << std::endl; 
    }


    // ===================================================
    // PSNR : Ref Blocky 
    cv::Mat MSE(cv::Size(imageref.cols, imageref.rows), imageref.type(), cv::Scalar::all(0));
    double imgpsnr = processPSNR(MSE, imageref, imageblocky); 
    std::cout << "PSNR IMAGE reference versus blocky = " << imgpsnr << std::endl; 

    cv::Mat MSE2(cv::Size(imageblocky.cols, imageblocky.rows), 
                imageblocky.type(), cv::Scalar::all(0));
    double imgpsnr2 = processPSNR(MSE2, imageblocky, imagedechiffree); 
    std::cout << "PSNR IMAGE blocky versus dechiffree = " << imgpsnr2 << std::endl; 

    cv::Mat MSE3(cv::Size(imageref.cols, imageref.rows), 
                imageref.type(), cv::Scalar::all(0));
    double imgpsnr3 = processPSNR(MSE2, imageref, imagedechiffree); 
    std::cout << "PSNR IMAGE reference versus dechiffree = " << imgpsnr3 << std::endl; 


    return EXIT_SUCCESS; 


}