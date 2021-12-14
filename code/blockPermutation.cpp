#include <iostream> 
#include <string>
#include "BasePermutation.h"
#include "imageanalysis.h"
#include "toolFunctions.hpp"

#include <opencv2/core/types.hpp>
#include <opencv2/core/fast_math.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

// A symmetric image encryption with naive permutation by blocks 


int main(int argc, char** argv )
{
    /*
    CV_8U - 8-bit unsigned integers ( 0..255 )
    CV_8S - 8-bit signed integers ( -128..127 )
    CV_16U - 16-bit unsigned integers ( 0..65535 )
    CV_16S - 16-bit signed integers ( -32768..32767 )
    CV_32S - 32-bit signed integers ( -2147483648..2147483647 )
    CV_32F - 32-bit floating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )
    CV_64F - 64-bit floating-point numbers ( -DBL_MAX..DBL_MAX, INF, NAN 


    If matrix is of type CV_8U then use Mat.at<uchar>(y,x).
    If matrix is of type CV_8S then use Mat.at<schar>(y,x).
    If matrix is of type CV_16U then use Mat.at<ushort>(y,x).
    If matrix is of type CV_16S then use Mat.at<short>(y,x).
    If matrix is of type CV_32S then use Mat.at<int>(y,x).
    If matrix is of type CV_32F then use Mat.at<float>(y,x).
    If matrix is of type CV_64F then use Mat.at<double>(y,x).

    */

    if ( argc != 5 )
    {
        printf("usage: DisplayImage.out <Image_Path> <SecretKey> <Encrypted Image> <directory> \n");
        return -1;
    }

    // SecretKey 
    const int secretKey = std::stoi(argv[2]); 
    //initSeed(secretKey); 

    cv::Mat image;
    image = cv::imread( argv[1], cv::IMREAD_GRAYSCALE);
    const char* filepathEncrypted = argv[3]; 
    const char* exportedDirectory = argv[4]; 
    

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }else 
    {
        std::cout << "Image Loaded : " << argv[1] << std::endl; 
    }

    const int nrows = (int)image.size[0]; 
    const int ncols = (int)image.size[1]; 
    const int NM = nrows * ncols;
    const int blocksize = 64; // number of pixels by block 
    int sqrtblockSize = (int)std::sqrt(blocksize);

    // Image Matrix 
    cv::Mat smoothedImage = cv::Mat::zeros(image.size[0], image.size[1], image.type()); 
    cv::Mat subsampled(nrows / sqrtblockSize,  ncols/sqrtblockSize, image.type());
    cv::Mat subpermutedImage = cv::Mat::zeros(subsampled.size[0], subsampled.size[1], subsampled.type());
    cv::Mat permutedImage = cv::Mat::zeros(image.size[0], image.size[1], image.type());
    cv::Mat subrecpermutedImage = cv::Mat::zeros(subsampled.size[0], subsampled.size[1], subsampled.type());
    cv::Mat subreconstructedImage = cv::Mat::zeros(subsampled.size[0], subsampled.size[1], subsampled.type());
    cv::Mat reconstructedImg = cv::Mat::eye(nrows, ncols, image.type());

    // Split by blocs 
    int nblocks = (int) (float(nrows)/ float(blocksize)); 

    // mean calculation 
    for (int ki = 0; ki < image.size[0]; ki+=sqrtblockSize)
    {
        for (int kj = 0; kj < image.size[1]; kj+=sqrtblockSize)
        {
            
            float meanpixel = 0.0f; 

            for (int ti = 0; ti < sqrtblockSize; ti++)
            {
                for (int tj = 0; tj < sqrtblockSize; tj++)
                {
                    meanpixel += (float)image.at<IMAGEMAT_TYPE>(ki + ti , kj + tj); 
                }
            }

            meanpixel /= (float)blocksize; 
            for (int ti = 0; ti < sqrtblockSize; ti++)
            {
                for (int tj = 0; tj < sqrtblockSize; tj++)
                {
                    smoothedImage.at<IMAGEMAT_TYPE>(ki + ti , kj + tj)= static_cast<IMAGEMAT_TYPE>(meanpixel);   
                }
            }
        }
    }


    show_wait_destroy("Main", smoothedImage);
    exportImage(exportedDirectory, "/01-smoothed_image.png", smoothedImage); 


    // by bloc 
    for (int ki = 0; ki < subsampled.size[0]; ki++)
    {
        for (int kj = 0; kj < subsampled.size[1]; kj++)
        {
            subsampled.at<IMAGEMAT_TYPE>(ki, kj) = smoothedImage.at<IMAGEMAT_TYPE>(ki*sqrtblockSize, kj*sqrtblockSize); 
        }
    }

    show_wait_destroy("Main", subsampled);
    exportImage(exportedDirectory, "/02-subsampled.png", subsampled);

    // ===================================================
    // PERMUTATION 

    // sequence for permutation
    std::vector<unsigned int> sequence(subsampled.size[0]* subsampled.size[1]);
    std::iota(sequence.begin(), sequence.end(), 0);

    // generate a permutation sequence 
    permuteSequence(sequence);

    // permute data 
    permuteData(subsampled, subpermutedImage, sequence); 

    show_wait_destroy("Main", subpermutedImage);
    exportImage(exportedDirectory, "/03-subpermutedImage.png", subpermutedImage);

    // Reconstructed blocks 
    for (int ki = 0; ki < subreconstructedImage.size[0]; ki++)
    {
        for (int kj = 0; kj < subreconstructedImage.size[1]; kj++)
        {
            
            for (int ti = 0; ti < sqrtblockSize; ti++)
            {
                for (int tj = 0; tj < sqrtblockSize; tj++)
                {
                    permutedImage.at<IMAGEMAT_TYPE>(ki*sqrtblockSize + ti, kj*sqrtblockSize +tj) = subpermutedImage.at<IMAGEMAT_TYPE>(ki, kj);
                }

            }
        }
    }

    show_wait_destroy("Main", permutedImage);
    exportImage(exportedDirectory, "/04-permutedImage.png", permutedImage);

    // ===================================================
    // RECONSTRUCTION 

    // downsampled 
    for (int ki = 0; ki < subsampled.size[0]; ki++)
    {
        for (int kj = 0; kj < subsampled.size[1]; kj++)
        {
            subrecpermutedImage.at<IMAGEMAT_TYPE>(ki, kj) = permutedImage.at<IMAGEMAT_TYPE>(ki*sqrtblockSize, kj*sqrtblockSize); 
        }
    }

    show_wait_destroy("Main", subrecpermutedImage);
    exportImage(exportedDirectory, "/05-subrecpermutedImage.png", subrecpermutedImage);

    
    // permute data 
    invPermuteData(subrecpermutedImage, subreconstructedImage, sequence); 

    show_wait_destroy("Main", subreconstructedImage);
    exportImage(exportedDirectory, "/06-subreconstructedImage.png", subreconstructedImage);

    // oversampled 
    for (int ki = 0; ki < subreconstructedImage.size[0]; ki++)
    {
        for (int kj = 0; kj < subreconstructedImage.size[1]; kj++)
        {
            
            for (int ti = 0; ti < sqrtblockSize; ti++)
            {
                for (int tj = 0; tj < sqrtblockSize; tj++)
                {
                    reconstructedImg.at<IMAGEMAT_TYPE>(ki*sqrtblockSize + ti, kj*sqrtblockSize +tj) = subreconstructedImage.at<IMAGEMAT_TYPE>(ki, kj);
                }

            }
        }
    }

    show_wait_destroy("Main", reconstructedImg);
    exportImage(exportedDirectory, "/07-reconstructedImg.png", reconstructedImg);

    // ===================================================
    // PSNR 
    cv::Mat MSE(cv::Size(image.cols, image.rows), image.type(), cv::Scalar::all(0));
    double imgpsnr = processPSNR(MSE, smoothedImage, reconstructedImg); 
    std::cout << "PSNR IMAGE on blocky = " << imgpsnr << std::endl; 

    cv::Mat MSE2(cv::Size(image.cols, image.rows), image.type(), cv::Scalar::all(0));
    double imgpsnr2 = processPSNR(MSE2, image, reconstructedImg); 
    std::cout << "PSNR IMAGE original versus blocky = " << imgpsnr2 << std::endl; 


    show_wait_destroy("Main", reconstructedImg);
    exportImage(std::string(), filepathEncrypted, permutedImage);
    cv::destroyAllWindows(); 

    return EXIT_SUCCESS; 
}