#include <stdio.h>
#include <iostream> 
#include <vector> 
#include <numeric> 
#include "BasePermutation.h"

#include <opencv2/core/types.hpp>
#include <opencv2/core/fast_math.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#define IMAGEMAT_TYPE uchar 

/*===========================================================================*/
// histogram on greyscale
cv::Mat processBarHistogram(cv::Mat& image, cv::Scalar color = cv::Scalar( 255, 0, 0))
{
    // HISTOGRAMS 
    const int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };

    static bool uniform = true; 
    static bool accumulate = false;

    cv::Mat histo;
    // calculate histogram
    cv::calcHist( &image, 1, 0, cv::Mat(), histo, 1, &histSize, histRange, uniform, accumulate );

    // normalized histogram
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    cv::Mat histoImg( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
    cv::normalize(histo, histo, 0, histoImg.rows, cv::NORM_MINMAX, -1, cv::Mat() );

    for( int i = 0; i < histSize; i++ )
    {
        float y = hist_h - cvRound(histo.at<float>(i)) ; 

        cv::rectangle( histoImg, 
                cv::Point2f(bin_w*i - float(bin_w)/2.0f, hist_h),
                cv::Point2f(bin_w*i + float(bin_w)/2.0f, y),
                color,
                cv::FILLED, 8, 0  
                );
    }

    return histoImg;
}



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
    /*const int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };

    bool uniform = true; 
    bool accumulate = false;

    cv::Mat grey_hist, grey_hist_perm, grey_hist_rec;

    cv::calcHist( &image, 1, 0, cv::Mat(), grey_hist, 1, &histSize, histRange, uniform, accumulate );
    cv::calcHist( &permutedImage, 1, 0, cv::Mat(), grey_hist_perm, 1, &histSize, histRange, uniform, accumulate );
    cv::calcHist( &reconstructedImg, 1, 0, cv::Mat(), grey_hist_rec, 1, &histSize, histRange, uniform, accumulate );

    // normalized histogram
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );

    cv::normalize(grey_hist, grey_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    cv::normalize(grey_hist_perm, grey_hist_perm, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    cv::normalize(grey_hist_rec, grey_hist_rec, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

    for( int i = 1; i < histSize; i++ )
    {

        cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(grey_hist.at<float>(i-1)) ),
              cv::Point( bin_w*(i), hist_h - cvRound(grey_hist.at<float>(i)) ),
              cv::Scalar( 255, 0, 0),
              2, 8, 0  );

        cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(grey_hist_perm.at<float>(i-1)) ),
              cv::Point( bin_w*(i), hist_h - cvRound(grey_hist_perm.at<float>(i)) ),
              cv::Scalar( 0, 255, 0),
              2, 8, 0  );

        cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(grey_hist_rec.at<float>(i-1)) ),
              cv::Point( bin_w*(i), hist_h - cvRound(grey_hist_rec.at<float>(i)) ),
              cv::Scalar( 0, 0, 255), 
              2, 8, 0  );
    }*/


    cv::Mat histoBarHimg = processBarHistogram(image, cv::Scalar( 255, 0, 0)); 
    cv::Mat histoBarHperm = processBarHistogram(permutedImage, cv::Scalar( 0, 255, 0)); 
    cv::Mat histoBarHrec = processBarHistogram(reconstructedImg, cv::Scalar( 0, 0, 255)); 

    // ===================================================



    // display our images 
    /*cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::namedWindow("Display Permuted Image", cv::WINDOW_AUTOSIZE );
    cv::namedWindow("Display Reconstructed Image", cv::WINDOW_AUTOSIZE );*/

    cv::imshow("Display Image", image);
    cv::imshow("Display Permuted image", permutedImage);
    cv::imshow("Display Reconstructed image", reconstructedImg);

    cv::imshow("CalcHist barH Image ", histoBarHimg );
    cv::imshow("CalcHist barH  Permuted Image", histoBarHperm );
    cv::imshow("CalcHist barH Reconstructed image", histoBarHrec );

    // End 
    cv::waitKey(0);
    cv::destroyAllWindows(); 

    return EXIT_SUCCESS;
}