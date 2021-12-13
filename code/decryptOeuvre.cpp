#include "toolFunctions.hpp"
#include "imageanalysis.h"
#include "BasePermutation.h"


int main( int argc, char** argv )
{
    
    const char* source_window = "Canny Edge";

    if ( argc != 4 )
    {
        printf("usage: decryptOeuvre.out <ImageToDecrypt> <SecretKey> <Dyrectory/ImageDecrypted> \n");
        return -1;
    }


    // 0. Read Data 
    cv::Mat image = cv::imread( argv[1], cv::IMREAD_GRAYSCALE );
    const int secretKey = std::stoi(argv[2]); 
    const char* name  = argv[3]; 

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    
    

    // =================================================== 
    // downsampled 
    float pixelSet = 102.0f; 
    const float valuePattern = 72.0f; 
    float pixel_size = pixelSet /  valuePattern; 

    float width = (float)image.size[1]/pixel_size; 
    float height = (float)image.size[0]/pixel_size;

    width = std::powf(2, std::round(std::log2(width))); 
    height = std::powf(2, std::round(std::log2(height))); 
    width = 512.f; 
    height = 512.f; 
    

    cv::Point2f src[4] = {cv::Point2f(0.0f, 0.0f), 
                        cv::Point2f((float)image.size[1], 0.0f), 
                        cv::Point2f(0.0f, (float)image.size[0]), 
                        cv::Point2f((float)image.size[1], (float)image.size[0])}; 

    cv::Point2f dst[4] = {cv::Point2f(0.0f, 0.0f), 
                          cv::Point2f(width, 0.0f), 
                            cv::Point2f(0.0f, height),  cv::Point2f(width, height)};

    cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(src, dst); 
    cv::Mat rescaleROI = cv::Mat::zeros(cv::Size((int)(image.size[1]/pixel_size), 
                                                (int)(image.size[0]/pixel_size)), CV_8UC1);   

    cv::warpPerspective(image, rescaleROI, perspectiveMatrix,  cv::Size((int)width,  (int)height)); 
    std::cout << "W,H = " << width  << "," << height << std::endl; 
    show_wait_destroy("Rescaled", rescaleROI); 
    exportImage(name, "_imageRescaled.png", rescaleROI); 


    // -------------------------------------
    // Init data 
    // -------------------------------------
    const int nrows = (int)rescaleROI.size[0]; 
    const int ncols = (int)rescaleROI.size[1]; 
    const int NM = nrows * ncols;
    const int blocksize = 16; // number of pixels by block 
    int sqrtblockSize = (int)std::sqrt(blocksize);
    const int dpi = 200; 
    unsigned int pixA4_width; 
    unsigned int pixA4_height; 

    // display grid 
    int test = (int) (nrows / (float)sqrtblockSize); 
    cv::Mat testMat = rescaleROI.clone(); 
    testMat.convertTo(testMat, CV_8UC3); 

    for (int k = 0; k < test ; k++)
    {
        cv::line(testMat, cv::Point(sqrtblockSize * k, 0), 
                        cv::Point(sqrtblockSize * k, testMat.size[0]), 
                        cv::Scalar(255,0,0,255), 1); 
        cv::line(testMat, cv::Point(0, sqrtblockSize * k), 
                        cv::Point(testMat.size[1], sqrtblockSize * k), 
                        cv::Scalar(255,0,0,255), 1); 
    }

    show_wait_destroy("Test MAT ", testMat); 

    cv::Mat subsampled(nrows / sqrtblockSize,  ncols/sqrtblockSize, image.type());
    cv::Mat subrecpermutedImage = cv::Mat::zeros(subsampled.size[0], subsampled.size[1], subsampled.type());
    cv::Mat subreconstructedImage = cv::Mat::zeros(subsampled.size[0], subsampled.size[1], subsampled.type());
    cv::Mat reconstructedImg = cv::Mat::eye(nrows, ncols, image.type());

    
    // 1. Key to decrypt 
    std::cout << std::endl << "Secret Key is " << secretKey << std::endl; 
    // initSeed(secretKey); 


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

    // ===================================================
    // PERMUTATION 

    // sequence for permutation
    std::vector<unsigned int> sequence(subsampled.size[0]* subsampled.size[1]);
    std::iota(sequence.begin(), sequence.end(), 0);

    // generate a permutation sequence 
    permuteSequence(sequence);

    

    for (int ki = 0; ki < subsampled.size[0]; ki++)
    {
        for (int kj = 0; kj < subsampled.size[1]; kj++)
        {
            
            int ky = ki*sqrtblockSize; 
            int kx = kj*sqrtblockSize; 
            // mean pixel 
            float pixel = 0.0f; 
            uchar base = rescaleROI.at<IMAGEMAT_TYPE>(ki*sqrtblockSize, kj*sqrtblockSize);

            pixel += (float)rescaleROI.at<IMAGEMAT_TYPE>(ky+1,kx+1); 
            pixel += (float)rescaleROI.at<IMAGEMAT_TYPE>(ky+1,kx+2); 
            pixel += (float)rescaleROI.at<IMAGEMAT_TYPE>(ky+2,kx+1); 
            pixel += (float)rescaleROI.at<IMAGEMAT_TYPE>(ky+2,kx+2); 
            pixel /= 4.0f; 

            subrecpermutedImage.at<IMAGEMAT_TYPE>(ki, kj) =  (IMAGEMAT_TYPE)pixel;
        }
    }

    cv::imshow("Main", subrecpermutedImage);
    cv::waitKey(0);

    // permute data 
    invPermuteData(subrecpermutedImage, subreconstructedImage, sequence); 

    cv::imshow("Main", subreconstructedImage);
    cv::waitKey(0);

    return EXIT_SUCCESS; 
}