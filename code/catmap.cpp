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

// A symmetric image encryption scheme based on 3D chaotic cat map


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
    cv::Mat A_chaotic = cv::Mat::zeros(3,3, CV_32S); 
    int a[3] = {1, 1, 1}; 
    int b[3] = {1, 1, 1}; 



    // set chaotic map 
    A_chaotic.at<int>(0,0) = 1 + a[0]*a[2]*b[1]; 
    A_chaotic.at<int>(0,1) = a[2]; 
    A_chaotic.at<int>(0,2) = a[1] + a[0]*a[2] + a[0]*a[1]*a[2]*b[1];

    A_chaotic.at<int>(1,0) = b[2] + a[0]*b[1] + a[0]*a[2]*b[1]*b[2]; 
    A_chaotic.at<int>(1,1) = a[2]*b[2] + 1; 
    A_chaotic.at<int>(1,2) = a[1]*a[2] + a[0]*a[1]*a[2]*b[1]*b[2] + a[0]*a[2]*b[2] + a[0]*a[1]*b[1] + a[0];

    A_chaotic.at<int>(2,0) = a[0]*b[0]*b[1] + b[1]; 
    A_chaotic.at<int>(2,1) = b[0]; 
    A_chaotic.at<int>(2,2) = a[0]*a[1]*b[0]*b[1] + a[0]*b[0] + a[1]*b[1] + 1;


    // Basic Arnold Cat Map 
    // ======================
    /*
    nx = (2 * x + y) % width
    ny = (x + y) % height
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
    unsigned int iterations = 3*nrows+1;  
    unsigned int counter = 0; 

    cv::Mat empty(image); 
    cv::Mat permutedImage = cv::Mat(image);
    cv::Mat itImage = cv::Mat(image); 

    while (counter < iterations )
    {
        
        for (int i=0; i < nrows; i++) // y 
        {
            for (int j=0; j < ncols; j++) // x 
            {
                int nx = (j + i) % ncols;
                int ny = (j + 2*i) % nrows;
                permutedImage.at<IMAGEMAT_TYPE>(i, j) = itImage.at<IMAGEMAT_TYPE>(ny, nx); 
            }
        }

        cv::imshow("Main", permutedImage);
        cv::waitKey(0);

        std::cout << "iterations i=" << counter << std::endl;
        /*std::string filepath = "/home/e20210011486/Documents/ImageSecu/Projet-Test/test/imagetest" ;
        filepath +=  std::to_string(counter);
        filepath += ".pgm"; 
        cv::imwrite(filepath, permutedImage);
        itImage = cv::imread(filepath, cv::IMREAD_GRAYSCALE); 
        */

        counter++;
        itImage = permutedImage.clone();
    }


    // End 
    cv::waitKey(0);
    cv::destroyAllWindows(); 



    return EXIT_SUCCESS; 
}