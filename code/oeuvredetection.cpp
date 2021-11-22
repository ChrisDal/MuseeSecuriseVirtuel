#include <iostream> 
#include <string>
#include <algorithm>
#include <functional>
#include <array>
#include <cfloat>

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
    if (class1.size() == 1)
    {
        return class1.at(0); 
    }
    
    float x = 0.0f; 
    float y = 0.0f; 

    for (cv::Point& p : class1)
    {
        x += p.x; 
        y += p.y;  
    }

    x /= (float)class1.size(); 
    y /= (float)class1.size(); 

    return cv::Point(x, y); 
}

float wardDistance(std::vector<cv::Point> c1, std::vector<cv::Point> c2)
{
    float n1 = (float)c1.size(); 
    float n2 = (float)c2.size(); 

    float dw = 0.0f; 

    dw = ((n1 * n2) / (n1 + n2)) * dissim(getGravityCenter(c1), getGravityCenter(c2)); 

    return dw; 
}

// min distance between the gravity center 
float gravityDistance(std::vector<cv::Point> c1, std::vector<cv::Point> c2)
{
    return dissim(getGravityCenter(c1), getGravityCenter(c2)); 
}

void hierarchicalCluster(const std::vector<cv::Point>& vecTocluster, int& Nclasses, std::vector<int>& vecClustered)
{
    int N = vecTocluster.size(); 
    vecClustered.clear(); 
    vecClustered.reserve(N); 

    bool reached_end = false; 

    std::vector< std::vector<cv::Point> > vecClasses; // index = class 

    for (unsigned  int k = 0; k < N; k++)
    {
        std::vector<cv::Point> a = {vecTocluster.at(k)}; 
        vecClasses.push_back(a); 
        vecClustered.push_back(k); 
    }


    while (!reached_end)
    {
        const unsigned int Ndissim = vecClasses.size(); 
        std::cout << "N class = " << vecClasses.size() << std::endl; 
        std::vector< std::vector < float > > matdissim(Ndissim, std::vector<float>(Ndissim, FLT_MAX));  
        
        for (unsigned int i = 0; i < vecClasses.size(); i++)
        {
            for (unsigned int j = i; j < vecClasses.size(); j++)
            {
                if (i==j) { continue; }
                matdissim[i][j] = gravityDistance(vecClasses.at(i), vecClasses.at(j)); 
            } 
        }


        // find min 
        float mindissim = FLT_MAX; 
        int classA =-1; 
        int classB =-1; 

        for (unsigned  int i = 0; i < vecClasses.size() ; i++)
        {
            for (unsigned  int j = i; j < vecClasses.size() ; j++)
            {
                if (i==j) { continue; }
                if (matdissim[i][j] < mindissim)
                {
                    classA = i; 
                    classB = j; 
                    mindissim = matdissim[i][j]; 

                    if (vecClasses.at(j).size() > vecClasses.at(i).size())
                    {
                        classA = j; 
                        classB = i; 
                    }
                }; 

            } 
        }

        // Fusion de classes
        std::cout << "Fusion A <= B " << classA  << "<= " << classB << std::endl; 
        for (unsigned int k = 0; k < vecClasses.at(classB).size(); k++)
        {
            cv::Point ab = vecClasses.at(classB).at(k); 
            vecClasses.at(classA).push_back(ab); 

            // change Class in final vector 
            auto itPoint = std::find(vecTocluster.begin(), vecTocluster.end(), ab); 
            if (itPoint != vecTocluster.end()){
                std::cout << "P " << *itPoint << " class " << classB << " to " << classA << std::endl; 
                vecClustered.at(itPoint - vecTocluster.begin()) = classA;
            }
                 

        }

        // Remove
        auto itClass = vecClasses.begin(); 
        std::cout << "Remove Class " << classB << "/" << vecClasses.size()-1 << std::endl; 
        vecClasses.erase(itClass + classB); 
        std::cout << "Number of class =" << vecClasses.size() << std::endl; 
        
        
         
        reached_end = vecClasses.size() < 5;

    }

    Nclasses = vecClasses.size(); 

}


// clustered with 4 classes 
void unconformedClustering(const std::vector<cv::Point>& vecTocluster, std::vector<int>& vecClustered, const cv::Point& center, float radius=0.0f)
{
    vecClustered.clear(); 
    vecClustered.reserve(vecTocluster.size()); 

    for (unsigned int k= 0; k < vecTocluster.size(); k++)
    {
        
        int x = vecTocluster.at(k).x;
        int y = vecTocluster.at(k).y;

        if ( (x < (center.x - radius) && y < (center.y - radius)))
        {
            vecClustered.push_back(0); 
        }
        else if (x >= (center.x+radius) && y < (center.y-radius))
        {
            vecClustered.push_back(1); 
        }
        else if (x < (center.x -radius) && y >= (center.y + radius))
        {
            vecClustered.push_back(2); 
        }
        else if (x >= (center.x +radius) && y >= (center.y + radius)) 
        {
            vecClustered.push_back(3); 
        }
        else
        {
            vecClustered.push_back(4);
        }
    }

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

    

    
    // ===============================================================================================
    // filtering 
    float threshold_filter = 4.0f; 
    std::vector<int> index_todelete; 
    float dist_mean = 0.0f; 
    for (unsigned int k = 0; k < harrisCorners.size(); k++ )
    {
        for (unsigned int j = k; j < harrisCorners.size(); j++ )
        {   
            float dist = 0.0f;
            if (j==k) { 
                continue; 
            }

            dist = std::sqrt(std::pow(harrisCorners.at(j).x - harrisCorners.at(k).x, 2)  + std::pow(harrisCorners.at(j).y - harrisCorners.at(k).y, 2)); 
            dist_mean += dist; 
            //std::cout << "Distance between point " << j << "," << k << " : " << dist; 
            if (dist < threshold_filter)
            {
                std::cout << " Distance between point " << j << "," << k << " : " << dist << std::endl; 
                index_todelete.push_back(j); 
            }
            
        }

    }

    dist_mean /= (float)harrisCorners.size(); 

    // filter unique value 
    std::sort(index_todelete.begin(), index_todelete.end()); 
    index_todelete.erase(std::unique(index_todelete.begin(), index_todelete.end()),  index_todelete.end()); 
    std::sort(index_todelete.begin(), index_todelete.end(), std::greater<int>()); 
    // erase duplicate 
    auto it = harrisCorners.begin(); 
    for ( unsigned int k = 0; k < index_todelete.size(); k++)
    {
        std::cout << " Remove point " << index_todelete.at(k) << std::endl;
        harrisCorners.erase(harrisCorners.begin() + index_todelete.at(k) ); 
    }


    // ===============================================================================================

    cv::Point centerImage = { image.cols/2, image.rows/2 }; 
    std::vector<int> vecClustered; 
    unconformedClustering(harrisCorners, vecClustered, centerImage, image.cols/6.0f ); 

    cv::Mat hierarchicalClustering = cv::Mat::zeros(image.size[0], image.size[1], CV_8UC3); 
    cv::imshow( "Main", hierarchicalClustering );
    for( int i = 0; i < harrisCorners.size() ; i++ )
    {       
        cv::Scalar color = cv::Scalar(vecClustered.at(i)*20, 0,vecClustered.at(i)*20,255); 

        if (vecClustered.at(i) == 0)
        {
            color = cv::Scalar(255,0,0,255); 
        }
        else if (vecClustered.at(i) == 1)
        {
            color = cv::Scalar(0,255,0,255); 
        }
        else if (vecClustered.at(i) == 2)
        {
            color = cv::Scalar(0,0,255,255);
        }
        else if (vecClustered.at(i) == 3)
        {
            color = cv::Scalar(36,180,240,255); // BGR order 
        }
        
        cv::circle( hierarchicalClustering, harrisCorners.at(i) , 5,  color, 2, 8, 0 );
        std::cout << "Point "<< i << std::endl; 
        cv::imshow( "Main", hierarchicalClustering );
        cv::waitKey(0);
        
    }

    cv::namedWindow( source_window );
    cv::imshow( source_window, image );
    cv::imshow( "Main", hierarchicalClustering );


    
    cv::waitKey(0);
    
    return EXIT_SUCCESS;
}


