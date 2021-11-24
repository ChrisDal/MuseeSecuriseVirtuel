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
#include "opencv2/features2d.hpp"

// Detection of oeuvre in image 
template <typename T>
float distance(T p1, T p2)
{
    return static_cast<float>(std::sqrt(std::pow(p2.x - p1.x,2) +  std::pow(p2.y - p1.y, 2))); 
}

template <typename T> 
T getGravityCenter(std::vector<T> class1)
{
    if (class1.size() == 1)
    {
        return class1.at(0); 
    }
    
    typename T::value_type x = 0.0f; 
    typename T::value_type y = 0.0f; 

    for (T& p : class1)
    {
        x += p.x; 
        y += p.y;  
    }

    x /= (float)class1.size(); 
    y /= (float)class1.size(); 

    return T(x, y); 
}

template < typename T> 
float wardDistance(std::vector<T> c1, std::vector<T> c2)
{
    float n1 = (float)c1.size(); 
    float n2 = (float)c2.size(); 

    float dw = 0.0f; 

    dw = ((n1 * n2) / (n1 + n2)) * distance(getGravityCenter(c1), getGravityCenter(c2)); 

    return dw; 
}

// min distance between the gravity center
template <typename T>  
float gravityDistance(std::vector<T> c1, std::vector<T> c2)
{
    return distance(getGravityCenter(c1), getGravityCenter(c2)); 
}

template <typename T> 
void hierarchicalCluster(const std::vector<T>& vecTocluster, int& Nclasses, std::vector<int>& vecClustered)
{
    int N = vecTocluster.size(); 
    vecClustered.clear(); 
    vecClustered.reserve(N); 

    bool reached_end = false; 

    std::vector< std::vector<T> > vecClasses; // index = class 

    for (unsigned  int k = 0; k < N; k++)
    {
        std::vector<T> a = {vecTocluster.at(k)}; 
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
template <typename T> 
void unconformedClustering(const std::vector<T>& vecTocluster, std::vector<int>& vecClustered, const T& center, float radius=0.0f)
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

// remove duplicate points  
template <typename T> 
void filterPoints(std::vector<T>& features, float threshold = std::sqrt(2))
{
    std::vector<int> index_todelete; 
    int Npoints = 0; 
    int overall = features.size(); 

    float dist_mean = 0.0f; 
    for (unsigned int k = 0; k < features.size(); k++ )
    {
        for (unsigned int j = k; j < features.size(); j++ )
        {   
            float dist = 0.0f;
            if (j==k) { continue; }

            dist = distance(features.at(j), features.at(k)); 


            if (dist < threshold)
            {
                //std::cout << " Distance between point " << j << "," << k << " : " << dist << std::endl; 
                index_todelete.push_back(j); 
            } 
        }
    }


    // filter unique value to remove 
    std::sort(index_todelete.begin(), index_todelete.end()); 
    index_todelete.erase(std::unique(index_todelete.begin(), index_todelete.end()),  index_todelete.end());
    // remove from end to start  
    std::sort(index_todelete.begin(), index_todelete.end(), std::greater<int>()); 

    // erase duplicate on features vector 
    auto it = features.begin(); 
    for ( unsigned int k = 0; k < index_todelete.size(); k++)
    {
        //std::cout << " Remove point " << index_todelete.at(k) << std::endl;
        Npoints++; 
        features.erase(features.begin() + index_todelete.at(k) ); 
    }

    std::cout << "Filtering removed " << Npoints << "/" << overall << " points." << std::endl; 

}

template <typename T> 
void harrisCornerDetection(const cv::Mat& img, std::vector<T>& harrisCorners, float threshHarris, float threshFilter=std::sqrt(2))
{
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    harrisCorners.clear(); 
    cv::Mat dst = cv::Mat::zeros( img.size(), CV_32FC1 );

    cv::cornerHarris( img, dst, blockSize, apertureSize, k );
    cv::Mat dst_norm; 
    cv::Mat dst_norm_scaled;

    cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    cv::convertScaleAbs( dst_norm, dst_norm_scaled );
    
    // detect and display image corner 
    for( int i = 0; i < dst_norm.rows ; i++ )
    {
        for( int j = 0; j < dst_norm.cols; j++ )
        {
            if( (int) dst_norm.at<float>(i,j) > threshHarris )
            {
                harrisCorners.push_back( T(j,i) );
                //cv::circle( image, cv::Point(j,i), 5,  cv::Scalar(0, 0, 0, 255), 2, 8, 0 );
            }
        }
    }

    
    // FILTERING 
    filterPoints(harrisCorners,  threshFilter); 
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

    // ===============================================================================================
    // FEATURES DETECTION : Harris Corner 
    std::vector<cv::Point2f> harrisCornersimg; 
    harrisCornerDetection(image, harrisCornersimg, thresh, 4.0f); 
    
    // SET CLASS "CLUSTERING"
    cv::Point2f centerImage = { image.cols/2.0f, image.rows/2.0f }; 
    std::vector<int> vecClustered; 
    unconformedClustering(harrisCornersimg, vecClustered, centerImage, image.cols/6.0f ); 

    // FEATURES DETECTION : Harris Corner Pattern 
    std::vector<cv::Point2f> harrisCornerspattern; 
    harrisCornerDetection(pattern, harrisCornerspattern, thresh, 4.0f); 
  
    // ===============================================================================================
    // trying to find a good match for each clustered class 0 1 2 3
    std::vector<cv::Point2f> pointofClass; 
    int classWanted = 0; 
    for (unsigned int k=0; k < vecClustered.size(); k++)
    {
        if (vecClustered.at(k) == classWanted)
        {
            pointofClass.push_back(harrisCornersimg.at(k)); 
        }
    }

    // determine ROI 
    cv::Point2f topleft = cv::Point2f(FLT_MAX, FLT_MAX); 
    cv::Point2f bottomRight = cv::Point2f(FLT_MIN, FLT_MIN);
    for (const cv::Point2f& p : pointofClass )
    {
        if (p.x <= topleft.x && p.y < topleft.y)
        {
            topleft = cv::Point2f(p);
        }

        if (p.x >= bottomRight.x && p.y > bottomRight.y)
        {
            bottomRight = cv::Point2f(p); 
        }
    } 

    std::cout << " Point TopLeft " << topleft << "Point bottomRight " << bottomRight << std::endl; 
    float xi = topleft.x - topleft.x*0.05f > 0 ? topleft.x - topleft.x*0.05f : topleft.x; 
    float yi = topleft.y - topleft.y*0.05f > 0 ? topleft.y - topleft.y*0.05f : topleft.y;
    float wi = (bottomRight.x*1.05f - xi) < image.cols ?  (bottomRight.x*1.05f  - xi) : bottomRight.x ; 
    float hi = (bottomRight.y*1.05f - yi) < image.rows ?  (bottomRight.y*1.05f  - yi) : bottomRight.y ; 

    cv::Rect roi = cv::Rect(xi, yi, wi, hi);
    cv::Mat image_roi = image(roi); 

    cv::imshow("Image ROI", image_roi);  

    // Match all features from pattern etendu to ROI
    //cv::Mat mathom = cv::findHomography(harrisCornerspattern, pointofClass, 1); 






    // display 
    cv::Mat hierarchicalClustering = cv::Mat::zeros(image.size[0], image.size[1], CV_8UC3); 
    for( int i = 0; i < harrisCornersimg.size() ; i++ )
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
        
        cv::circle( hierarchicalClustering, harrisCornersimg.at(i) , 5,  color, 2, 8, 0 );
        cv::circle( image, harrisCornersimg.at(i), 5,  cv::Scalar(0, 0, 0, 255), 2, 8, 0 ); 
        // interactive display clustering
        /*std::cout << "Point "<< i << std::endl; 
        cv::imshow( "Main", hierarchicalClustering );
        cv::waitKey(0);*/
        
    }

    for( int i = 0; i < harrisCornerspattern.size() ; i++ )
    {
        cv::circle( pattern, harrisCornerspattern.at(i), 5,  cv::Scalar(0, 0, 0, 255), 2, 8, 0 ); 
    }

    cv::namedWindow( source_window );
    cv::imshow( source_window, image );
    cv::imshow( "Main", hierarchicalClustering );
    cv::imshow("Pattern", pattern); 


    
    cv::waitKey(0);
    
    return EXIT_SUCCESS;
}


