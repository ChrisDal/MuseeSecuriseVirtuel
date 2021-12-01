#include <iostream> 
#include <string>
#include <algorithm>
#include <functional>
#include <array>
#include <cfloat>
#include <map>

#include "BasePermutation.h"
#include "imageanalysis.h"

#include <opencv2/core/types.hpp>
#include <opencv2/core/fast_math.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>


#define DEBUG_ON_DISPLAY 1

// Detection of oeuvre in image 
// Euclidian Distance 
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

template <typename T> 
std::vector<T> getVecPoint(const std::vector<int>& classifiedVec, const std::vector<T>&  harrispoints, unsigned int classwanted)
{
    // trying to find a good match for each clustered class 0 1 2 3
    std::vector<T> pointofClass;
    for (unsigned int k=0; k < classifiedVec.size(); k++)
    {
        if (classifiedVec.at(k) == classwanted)
        {
            pointofClass.push_back(harrispoints.at(k)); 
            
        }
    }

    return pointofClass; 
}


// =====================================================
// Histograms per axis 
cv::Point2f getIndexOfPoint(const std::vector<cv::Point2f>& classifiedPoint, int width, int height, int cornerType=0, float dsize = 5.0f)
{
    std::cout << "Class c=" << cornerType << std::endl; 
    int krows = (int)(width/dsize);  // image.size[1]
    int kcols = (int)(height/dsize); // image.size[0]
    unsigned int numberMinPoints = 2; 

    std::vector<unsigned int> historows(krows,  0.f); 
    std::vector<unsigned int> histocols(kcols,  0.f);

    if (cornerType > 3) {
        std::cout << "Invalid cornerType, return 0 0 "; 

        return cv::Point2f(0.0, 0.0); 
    }

    for (const cv::Point& p : classifiedPoint)
    {
        int kx = (int)(p.x /dsize); 
        int ky = (int)(p.y /dsize); 

        historows[kx] += 1; 
        histocols[ky] += 1; 
    }

    // Point that keep index of corner pattern that interest us  
    cv::Point2f index; 
    switch(cornerType)
    {
        case 0 : index = cv::Point2f(0.0, 0.0); break;
        case 1 : index = cv::Point2f(FLT_MAX, 0.0); break;
        case 2 : index = cv::Point2f(0.0, FLT_MAX); break;
        case 3 : index = cv::Point2f(FLT_MAX, FLT_MAX); break; 
        default: index = cv::Point2f(0.0, 0.0); break;
    }


 
    for (unsigned int k = 0; k< historows.size(); k++)
    {
        // top left , bottom left 
        if (cornerType == 0 || cornerType == 2)
        {
            if (historows[k] > numberMinPoints) {
                index.x = k; 
            }
        } 
        else if (cornerType == 1 || cornerType == 3)
        {
            if (historows[k] > numberMinPoints && k < index.x) {
                index.x = k; 
            }
        }
        
    }


    for (unsigned int k = 0; k< histocols.size(); k++)
    {
        if (cornerType == 0 || cornerType == 1)
        {
            if (histocols[k] > numberMinPoints) {
                index.y = k; 
            }
        }
        else if (cornerType == 2 || cornerType == 3)
        {
            if (histocols[k] > numberMinPoints && k < index.y) {
                index.y = k; 
            }
        }
    }

    // Point coord in image
    index.x *= dsize; 
    index.y *= dsize; 

    return index; 

}

std::vector<std::pair<float, float>> houghAnalysis(const std::vector<cv::Point2f>& vecClassified, int width, int height, 
                                                    float dtheta=1.5f, float drho = 3.2f, float maxtheta=180.f)
{

    //int imgWidth = image.size[1]; 
    //int imgHeight = image.size[0]; 

    double maxRho = std::sqrt(width*width + height*height); 

    int Ntheta = (int)(maxtheta / dtheta) + 1;
    int Nrho   = (int)(maxRho / drho) + 1; 

    // Hough transformation  
    std::vector<std::vector<unsigned int> > houghvec(Ntheta, std::vector<unsigned int>(Nrho, 0));   

    // pour chaque point 
    for (const cv::Point2f& p : vecClassified)
    {
        for (unsigned int k=0; k < Ntheta; k++)
        {
            float theta = dtheta * (float)k * 3.14f / 180.0f; 
            float rho = p.x * std::cos(theta) + p.y * std::sin(theta); 

            int krho = (int) std::abs(rho / drho); 
            houghvec[k][krho] += 1; 
        }
    }

    cv::Mat testMat = cv::Mat::zeros(Nrho, Ntheta, CV_8UC1); 
    const unsigned int minVotes = 3; 
    std::vector<std::pair<float, float>> winnerTuple; 
    for (unsigned int k = 0; k < Ntheta; k++)
    {
        for (unsigned int i = 0; i < Nrho; i++)
        {
            
            testMat.at<uchar>(i, k) = 50*houghvec[k][i]; 
            if (houghvec[k][i] >= minVotes) {
                winnerTuple.emplace_back((float)k*dtheta, i*drho); 
            }
        }
    }

    for (const std::pair<float, float>& wint : winnerTuple)
    {
        std::cout << "Pair : <" << wint.first << "," << wint.second << ">\n"; 
    }

    if (DEBUG_ON_DISPLAY)
    {
        cv::imshow(" Hough Mat" , testMat); 
        cv::waitKey(0); 
    }

    return winnerTuple; 
}


void displayHoughLines(const std::vector< std::pair<float, float> >& analysisresult, cv::Mat& imageOndisp)
{
    for (const std::pair<float, float>& lines : analysisresult)
    {
        float rho = lines.second; 
        float theta = lines.first * 3.14f / 180.0f; // theta = degree

        float a = std::cos(theta); 
        float b = std::sin(theta); 
        cv::Point2f p0(rho * a, rho * b); 

        cv::Point2f p1(p0.x + (int)(10000 * (-b)), p0.y + (int)(10000*a)); 
        cv::Point2f p2(p0.x - (int)(10000 * (-b)), p0.y - (int)(10000*a)); 

        cv::line(imageOndisp ,p1, p2, cv::Scalar(0,0,255, 255)); 

        cv::imshow("Line Houghs", imageOndisp); 
        cv::waitKey(0); 

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

    int pattern_width = pattern.size[1]; 
    int pattern_height = pattern.size[0]; 


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
    cv::Mat pattern_extracted = image(roi); 
    cv::imshow("Image ROI", pattern_extracted);  

    // =========================================================
    // HOUGH TRANFORMATION 
    // 0 à 2pi 

    int imgWidth = image.size[1]; 
    int imgHeight = image.size[0]; 
    float diag = std::sqrt(imgWidth*imgWidth + imgHeight*imgHeight); 
    float drho = diag*0.000705f ; 
    
    std::vector<std::pair<float, float>> houghlines = houghAnalysis(pointofClass, imgWidth, imgHeight, 
                                                                    2.0f, drho); 
    displayHoughLines(houghlines, image); 

    // Lines Detection done => each line must define our pattern template 
    // All points that belongs to a line are valid 
    // Valid only these points 

    std::vector<cv::Point2f> validPointsbyHough; 
    std::vector<unsigned int> validIndexbyHough; 
    /* theta, rho */
    
    for (unsigned int k = 0; k < pointofClass.size(); k++)
    {
        
        for (const auto& line : houghlines)
        {
            float rho = line.second; 
            float theta = line.first; 

            float rhop = pointofClass[k].x*std::cos(theta * 3.14f /180.0f) + pointofClass[k].y*std::sin(theta * 3.14f /180.0f); 

            if (std::abs(std::abs(rhop) - rho) < drho)
            {
                validIndexbyHough.push_back(k);
                if (DEBUG_ON_DISPLAY){

                    std::cout << "Point valid by Hough : " <<  pointofClass[k].x  << "," << pointofClass[k].y << std::endl; 
                }
                
                break;
            }
            
        }
    }

    // Remove invalid points
    std::vector <unsigned int> removeIndex;  
    for (unsigned int k= 0; k < pointofClass.size(); k++)
    {
        if (std::find(validIndexbyHough.begin(),validIndexbyHough.end(), k) == std::end(validIndexbyHough))
        {
            removeIndex.push_back(k); 
        }
    }

    std::sort(removeIndex.begin(), removeIndex.end(), std::greater<unsigned int>()); 
    for (const unsigned int& index : removeIndex)
    {
        pointofClass.erase(pointofClass.begin() + index); 
    }

    if (DEBUG_ON_DISPLAY){ 
        std::cout << "Number of valid points by Hough = " << pointofClass.size() << std::endl; 
    }
                


    

    // =========================================================

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
        //cv::circle(img_withmatchesROI, harrisCornersimg.at(i) , 5,  color, 2, 8, 0 ); 
        // interactive display clustering
        //std::cout << "Point "<< i << std::endl; 
        //cv::imshow( "Main", hierarchicalClustering );
        //cv::waitKey(0) 
        
    }

    for (const cv::Point2f& p : validPointsbyHough)
    {
        cv::circle( hierarchicalClustering, p , 5,  cv::Scalar(255,255,255,255), 2, 8, 0 );
    }




    // =====================================================
    // iterate through class 
    std::vector<cv::Point2f> intersections;
    intersections.reserve(4); 
    for (unsigned int c = 0; c < 4; c++)
    {
        std::vector<cv::Point2f> pointsclass = getVecPoint(vecClustered, harrisCornersimg, c);
        // Histogram per line 
        cv::Point2f cornerlineindex = getIndexOfPoint(pointsclass, image.size[1], image.size[0], c);
        
        // vertical line 
        cv::line(hierarchicalClustering, 
                cv::Point2f(cornerlineindex.x, 0.0f),   
                cv::Point2f(cornerlineindex.x, hierarchicalClustering.size[0]), 
                cv::Scalar(60 + 40*c, 60 + 40*c,60 + 40*c,255));
        // horizontal Line 
        cv::line(hierarchicalClustering, 
                cv::Point2f(0.0f, cornerlineindex.y),   
                cv::Point2f(hierarchicalClustering.size[1], cornerlineindex.y), 
                cv::Scalar(60 + 40*c,60 + 40*c,60 + 40*c,255));

        
        intersections.emplace_back(cornerlineindex.x, cornerlineindex.y); 
        
        cv::imshow( "Main", hierarchicalClustering );
        cv::waitKey(0);
    }

    // =====================================================
    // IMAGE DETECTION  

    cv::Point2f topLeft; 
    
    if ( std::abs(intersections[0].x - intersections[2].x) > (image.size[1]*0.10f))
    {
        std::cout << "Variations > 10% de la taille de l'image\n"; 
        topLeft.x = std::min(intersections[0].x, intersections[2].x);
    }
    else 
    {
        topLeft.x = std::min(intersections[0].x , intersections[2].x);
    }

    if ( std::abs(intersections[0].y - intersections[1].y) > (image.size[0]*0.10f))
    {
        std::cout << "Variations > 10% de la taille de l'image\n"; 
        topLeft.y = std::min(intersections[0].y, intersections[1].y);
    }
    else 
    {
        topLeft.y = std::max(intersections[0].y ,intersections[1].y);
    }

    cv::Point2f botRight;
    if ( std::abs(intersections[1].x - intersections[3].x) > (image.size[1]*0.10f))
    {
        std::cout << "Variations > 10% de la taille de l'image\n"; 
        botRight.x = std::min(intersections[1].x, intersections[3].x);
    }
    else 
    {
        botRight.x = std::max(intersections[1].x , intersections[3].x);
    }

    if ( std::abs(intersections[2].y - intersections[3].y) > (image.size[0]*0.10f))
    {
        std::cout << "Variations > 10% de la taille de l'image\n"; 
        botRight.y = std::min(intersections[2].y, intersections[3].y);
    }
    else 
    {
        botRight.y = std::max(intersections[2].y, intersections[3].y);
    }


    cv::Rect roiImage = cv::Rect((int)topLeft.x, (int)topLeft.y,
                                std::ceil(botRight.x-topLeft.x),
                                std::ceil(botRight.y-topLeft.y));
    std::cout << topLeft.x << std::endl; 
    cv::Mat outDetectedImage = image(roiImage);

    // =====================================================

    for( int i = 0; i < harrisCornerspattern.size() ; i++ )
    {
        cv::circle( pattern, harrisCornerspattern.at(i), 5,  
                    cv::Scalar(0, 0, 0, 255), 2, 8, 0 ); 
    }

    

    cv::namedWindow( source_window );
    cv::imshow( source_window, image );
    cv::imshow( "Main", hierarchicalClustering );
    cv::imshow("Pattern", pattern); 
    cv::imshow("Detected Image",outDetectedImage ); 


    
    cv::waitKey(0);
    
    return EXIT_SUCCESS;
}


