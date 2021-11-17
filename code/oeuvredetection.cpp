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

// Detection of oeuvre in image 

float dissim(cv::Point p1, cv::Point p2)
{
    return static_cast<float>(std::sqrt((p2.x - p1.x) *(p2.x - p1.x) +  (p2.y - p1.y) *(p2.y - p1.y))); 
}

cv::Point getGravityCenter(std::vector<cv::Point> class1)
{
    float x = 0.0f; 
    float y = 0.0f; 

    for (cv::Point& p : class1)
    {
        x += p.x; 
        y += p.y;  
    }

    x /= class1.size(); 
    y /= class1.size(); 

    return cv::Point(x, y); 
}

float wardDistance(std::vector<cv::Point> c1, std::vector<cv::Point> c2)
{
    float n1 = c1.size(); 
    float n2 = c2.size(); 

    float dw = 0.0f; 

    dw = ((n1 * n2) / (n1 + n2)) * dissim(getGravityCenter(c1), getGravityCenter(c2)); 

    return dw; 
}



// ======================================================================================


int main( int argc, char** argv )
{
    cv::Mat image;
    int thresh = 154;
    int max_thresh = 255;
    const char* source_window = "Source image";


    image = cv::imread( argv[1], cv::IMREAD_GRAYSCALE );
    cv::Mat pattern = cv::imread(argv[2], cv::IMREAD_GRAYSCALE );
    

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    cv::Mat dst = cv::Mat::zeros( image.size(), CV_32FC1 );

    cv::cornerHarris( image, dst, blockSize, apertureSize, k );
    cv::Mat dst_norm; 
    cv::Mat dst_norm_scaled;

    cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    cv::convertScaleAbs( dst_norm, dst_norm_scaled );
    
    // detect and display image corner 
    std::vector<cv::Point> harrisCorners; 
    for( int i = 0; i < dst_norm.rows ; i++ )
    {
        for( int j = 0; j < dst_norm.cols; j++ )
        {
            if( (int) dst_norm.at<float>(i,j) > thresh )
            {
                harrisCorners.push_back(cv::Point(j,i));
                cv::circle( image, cv::Point(j,i), 5,  cv::Scalar(0, 0, 0, 255), 2, 8, 0 );
            }
        }
    }

    


    // Hierarchical Clustering 
    cv::Mat hierarchicalClustering = cv::Mat::zeros(image.size[0], image.size[1], CV_8UC3); 
    for( int i = 0; i < harrisCorners.size() ; i++ )
    {       
        cv::circle( hierarchicalClustering, harrisCorners.at(i) , 5,  cv::Scalar(255, 0, 0, 255), 2, 8, 0 );
    }

    cv::namedWindow( source_window );
    cv::imshow( source_window, image );
    cv::imshow( "Main", hierarchicalClustering );


    
    cv::waitKey(0);
    
    return EXIT_SUCCESS;
}


