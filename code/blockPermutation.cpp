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

    cv::Mat image;
    image = cv::imread( argv[1], cv::IMREAD_GRAYSCALE );
    

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    const int nrows = image.size[0]; 
    const int ncols = image.size[1]; 
    const int NM = nrows * ncols;
    const int blocksize = 16; // number of pixels by block 
    int sqrtblockSize = std::sqrt(blocksize);

    cv::Mat smoothedImage = image.clone(); 
    cv::Mat subsampled(nrows / sqrtblockSize,  ncols/sqrtblockSize, image.type());
    cv::Mat permutedImage = cv::Mat(image);
    cv::Mat reconstructedImg = cv::Mat::eye(nrows, ncols, image.type());

    // Split by blocs 
    int nblocks = (int) float(nrows)/ float(blocksize); 

    // mean calculation 
    cv::Mat kernel = cv::Mat::ones(sqrtblockSize, sqrtblockSize, CV_32F) / (float)blocksize; 
    cv::filter2D(image, smoothedImage, -1 , kernel); 
    cv::imshow("Main", smoothedImage);
    cv::waitKey(0); 
    // by bloc 
    for (unsigned int ki = 0; ki < subsampled.size[0]; ki++)
    {
        for (unsigned int kj = 0; kj < subsampled.size[1]; kj++)
        {
            subsampled.at<IMAGEMAT_TYPE>(ki, kj) = smoothedImage.at<IMAGEMAT_TYPE>(ki*sqrtblockSize, kj*sqrtblockSize); 
        }
    }

    cv::imshow("Main", subsampled);
    // permutation on subsampled image 









    // End 
    cv::waitKey(0);
    cv::destroyAllWindows(); 



    return EXIT_SUCCESS; 
}