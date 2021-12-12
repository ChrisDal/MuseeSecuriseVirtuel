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

    // -------------------------------------
    // Init data 
    // -------------------------------------
    const int nrows = (int)image.size[0]; 
    const int ncols = (int)image.size[1]; 
    const int NM = nrows * ncols;
    const int blocksize = 16; // number of pixels by block 
    int sqrtblockSize = (int)std::sqrt(blocksize);
    const int dpi = 200; 
    unsigned int pixA4_width; 
    unsigned int pixA4_height; 

    // display grid 
    int test = (int) (nrows / (float)sqrtblockSize); 
    cv::Mat testMat = image.clone(); 
    testMat.convertTo(testMat, CV_8UC3); 

    for (int k = 0; k < test ; k++)
    {
        cv::circle(testMat, cv::Point(10, sqrtblockSize * test), 10, cv::Scalar(0,0,255,255)); 
    }

    show_wait_destroy("Test MAT ", testMat); 

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


    cv::Mat subsampled(nrows / sqrtblockSize,  ncols/sqrtblockSize, image.type());
    cv::Mat subrecpermutedImage = cv::Mat::zeros(subsampled.size[0], subsampled.size[1], subsampled.type());
    cv::Mat subreconstructedImage = cv::Mat::zeros(subsampled.size[0], subsampled.size[1], subsampled.type());
    cv::Mat reconstructedImg = cv::Mat::eye(nrows, ncols, image.type());
    
    // 1. Key to decrypt 
    std::cout << std::endl << "Secret Key is " << secretKey << std::endl; 
    // initSeed(secretKey); 

    // ===================================================
    // PERMUTATION 

    // sequence for permutation
    std::vector<unsigned int> sequence(subsampled.size[0]* subsampled.size[1]);
    std::iota(sequence.begin(), sequence.end(), 0);

    // generate a permutation sequence 
    permuteSequence(sequence);

    // =================================================== 
    // downsampled 
    for (int ki = 0; ki < subsampled.size[0]; ki++)
    {
        for (int kj = 0; kj < subsampled.size[1]; kj++)
        {
            subrecpermutedImage.at<IMAGEMAT_TYPE>(ki, kj) = image.at<IMAGEMAT_TYPE>(ki*sqrtblockSize, kj*sqrtblockSize); 
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