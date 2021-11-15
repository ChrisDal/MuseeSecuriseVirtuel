#include <stdio.h>
#include <iostream> 
#include <vector> 
#include <numeric> 
#include "BasePermutation.h"
#include "imageanalysis.h"

#include <opencv2/core/types.hpp>
#include <opencv2/core/fast_math.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#define IMAGEMAT_TYPE uchar 


/*===========================================================================*/

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    /*
    IMREAD_COLOR loads the image in the BGR 8-bit format. This is the default that is used here.
    IMREAD_UNCHANGED loads the image as is (including the alpha channel if present)
    IMREAD_GRAYSCALE loads the image as an intensity one

    In the case of color images, the decoded images will have the channels stored in B G R order.
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

    cv::Mat permutedImage = cv::Mat::eye(nrows, ncols, image.type()); 


    // ===================================================
    // PERMUTATION 

    // sequence for permutation
    std::vector<unsigned int> sequence(NM);
    std::iota(sequence.begin(), sequence.end(), 0);

    // generate a permutation sequence 
    permuteSequence(sequence);

    // permute data 
    permuteData(image, permutedImage, sequence); 

    // ===================================================

    // RECONSTRUCTION 
    cv::Mat reconstructedImg = cv::Mat::eye(nrows, ncols, image.type());
    // permute data 
    invPermuteData(permutedImage, reconstructedImg, sequence); 


    // ===================================================
    // HISTOGRAMS 
    cv::Mat histoBarHimg = processBarHistogram(image, cv::Scalar( 255, 0, 0)); 
    cv::Mat histoBarHperm = processBarHistogram(permutedImage, cv::Scalar( 0, 255, 0)); 
    cv::Mat histoBarHrec = processBarHistogram(reconstructedImg, cv::Scalar( 0, 0, 255)); 

    // ===================================================
    // DISPLAY 
    // =======
    
    // display our images
    cv::Mat displayImages;  
    concatenateFourMat(displayImages, image, cv::Mat(), permutedImage, reconstructedImg); 

    // display our histograms  
    cv::Mat displayHistos;  
    concatenateFourMat(displayHistos, histoBarHimg, cv::Mat(), histoBarHperm, histoBarHrec); 


    cv::imshow("Display Histogram: A Original, C Permuted Image, D Reconstructed ", displayHistos );
    cv::imshow("images: A Original, C Permuted Image, D Reconstructed", displayImages );

    // End 
    cv::waitKey(0);
    cv::destroyAllWindows(); 

    return EXIT_SUCCESS;
}