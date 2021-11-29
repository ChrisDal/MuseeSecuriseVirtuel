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
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>


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
    int numberMinPoints = 2; 

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

    // Find the mean center 
    cv::Point2f ancragePoint(0.0f, 0.0f); 

    if (pointofClass.size() == 16)
    {
        // lucky we got all points
        for (unsigned int k=0; k <  pointofClass.size(); k++)
        {
            ancragePoint.x += pointofClass[k].x; 
            ancragePoint.y += pointofClass[k].y; 
        }

        ancragePoint.x /= (float)pointofClass.size(); 
        ancragePoint.y /= (float)pointofClass.size(); 
    }




    cv::Rect roi = cv::Rect(xi, yi, wi, hi);
    cv::Mat image_roi = image(roi); 
    cv::imshow("Image ROI", image_roi);  

    // Match all features from pattern etendu to ROI 
    // take one keypoint descriptor - ref = pattern 
    std::vector< std::vector<std::pair < float , cv::Point2f> > > matches; 
    matches.reserve(pointofClass.size()); 
    std::vector< std::pair<float, int> > distPoints = {}; 
    std::vector<bool> pointChoosen(pointofClass.size(), false);
    
    for (unsigned int k = 0; k < pointofClass.size(); k++)
    {
        distPoints.clear(); 
        std::vector<std::pair< float, cv::Point2f>> pointMatch; 
         

        //distance calculations 

        for (unsigned int i = 0; i < pointofClass.size(); i++)
        {
            if (pointChoosen[i])
            {
                continue; 
            }

            float scale_factorx = (float)(pattern.cols)/ (float)wi ;   
            float scale_factory = (float)(pattern.rows)/ (float)hi ;   
            cv::Point2f point_harris_scaled = (pointofClass[i] - cv::Point2f(xi, xi)); 
            point_harris_scaled.x *= scale_factorx; 
            point_harris_scaled.y *= scale_factory; 

            float di = distance(harrisCornerspattern[k], point_harris_scaled); 
             
            distPoints.push_back(std::make_pair(di, i)); 
        }

        // Sort by distance 
        std::sort(distPoints.begin(), distPoints.end());
        std::cout << "Distance between Pattern[" << k << "] & distance p1=" << distPoints[0].first << " & distance p2=" <<  distPoints[1].first << "\n" << std::endl; 
        // take the first 2 
        pointMatch.push_back(std::make_pair(distPoints[0].first, pointofClass[distPoints[0].second])); 
        pointMatch.push_back(std::make_pair(distPoints[1].first, pointofClass[distPoints[1].second]));
        // invalidate first one to tak e
        pointChoosen[distPoints[0].second] = true; 
        matches.push_back(pointMatch);  
        pointMatch.clear(); 
    }


 
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.998f;
    std::vector< cv::Point2f > good_matches;

    for (int i = 0; i < matches.size(); i++)
    {
        std::cout << " Match 0=" << matches[i][0].first << " Match 1=" <<matches[i][1].first << std::endl;
        if (matches[i][0].first < ratio_thresh * matches[i][1].first) {
            good_matches.push_back(matches[i][0].second);
            std::cout << "Find Match for point " << harrisCornerspattern[i] << " => " << matches[i][0].second <<" \n"; 
            continue; 
        }

        good_matches.push_back(cv::Point2f(-1,-1));

    }



    // Draw matches
    cv::Mat img_matches;
    realConcatenateTwoMat(img_matches, pattern, image); 

    cv::Rect roi2 = cv::Rect(pattern.cols, 0, image.cols, image.rows);
    cv::Mat img_withmatchesROI = img_matches(roi2);

    // --------------------------------------------------------
    // HOUGH TRANFORMATION 
    // 0 à 2pi 
    
    float dtheta = 5.0f; 
    float maxtheta = 180.0f; 
    int Ntheta  = (int) (maxtheta / dtheta);
    float drho = 10.f;  
    int houghcols = (int)(image.size[1]/drho); 

    // Hough transformation  
    std::vector<std::vector<unsigned int> > houghvec(Ntheta, std::vector<unsigned int>(houghcols, 0));   

    // pour chaque point 
    for (const cv::Point& p : pointofClass)
    {
        float xk = p.x; 
        float yk = p.y; 

        float theta = 0.0f; 

        for (int i = 0; i < Ntheta; i++)
        {
            theta += dtheta; 
            float theta_rad = theta * 3.14f / 180.0f; 
            float rho = xk*cos(theta_rad) + yk*sin(theta_rad); 
            int j = (int)(rho / drho);    

            if (j >= 0)
            {
                houghvec[i][j] += 1; 
                std::cout << "Rho = " << rho << " theta = " << theta << std::endl; 
                std::cout << "j = " << j << " i="<< i << std::endl;
            }
                       
        }
    }
    // --------------------------------------------------------

    



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
        cv::circle(img_withmatchesROI, harrisCornersimg.at(i) , 5,  color, 2, 8, 0 ); 
        // interactive display clustering
        //std::cout << "Point "<< i << std::endl; 
        //cv::imshow( "Main", hierarchicalClustering );
        //cv::waitKey(0);

        
        
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

    // mean per corner 
    cv::Point2f topLeft; 
    
    if ( std::abs(intersections[0].x - intersections[2].x) > (image.size[1]*0.10f))
    {
        std::cout << "Variations > 10% de la taille de l'image\n"; 
        topLeft.x = std::min(intersections[0].x, intersections[2].x);
    }
    else 
    {
        topLeft.x = (intersections[0].x +  intersections[2].x) / 2.0f;
    }

    if ( std::abs(intersections[0].y - intersections[1].y) > (image.size[0]*0.10f))
    {
        std::cout << "Variations > 10% de la taille de l'image\n"; 
        topLeft.y = std::min(intersections[0].y, intersections[1].y);
    }
    else 
    {
        topLeft.y = (intersections[0].y +  intersections[1].y) / 2.0f;
    }

    cv::Point2f botRight;
    if ( std::abs(intersections[1].x - intersections[3].x) > (image.size[1]*0.10f))
    {
        std::cout << "Variations > 10% de la taille de l'image\n"; 
        botRight.x = std::min(intersections[1].x, intersections[3].x);
    }
    else 
    {
        botRight.x = (intersections[1].x +  intersections[3].x) / 2.0f;
    }

    if ( std::abs(intersections[2].y - intersections[3].y) > (image.size[0]*0.10f))
    {
        std::cout << "Variations > 10% de la taille de l'image\n"; 
        botRight.y = std::min(intersections[2].y, intersections[3].y);
    }
    else 
    {
        botRight.y = (intersections[2].y +  intersections[3].y) / 2.0f;
    }


    cv::Rect roiImage = cv::Rect(topLeft.x, topLeft.y,
                                botRight.x-topLeft.x,
                                botRight.y-topLeft.y);
    std::cout << topLeft.x << std::endl; 
    cv::Mat outDetectedImage = image(roiImage);



    

 
    // =====================================================




    roi2 = cv::Rect(0, 0, pattern.cols, pattern.rows);
    cv::Mat pattern_withmatchesROI = img_matches(roi2);

    for( int i = 0; i < harrisCornerspattern.size() ; i++ )
    {
        cv::circle( pattern, harrisCornerspattern.at(i), 5,  
                    cv::Scalar(0, 0, 0, 255), 2, 8, 0 ); 
        cv::circle( pattern_withmatchesROI,harrisCornerspattern.at(i), 5,  
                    cv::Scalar(0, 0, 0, 255), 2, 8, 0 ); 
        
    }

    // draw matches 
    for (unsigned int k = 0; k < good_matches.size(); k++ )
    {   
        // translated point 
        cv::Point2f gm = good_matches[k] + cv::Point2f(pattern.rows, 0.0f);  
        if (good_matches[k] == cv::Point2f(-1, -1))
        {
            std::cout << "No Match for point " << harrisCornerspattern[k] << "\n"; 
            continue; 
        }
        cv::line(img_matches, harrisCornerspattern[k],  gm, cv::Scalar(120,120,120,255), 3);

        /*cv::line(img_matches, scene_corners[0] + cv::Point2f((float)pattern.cols, 0),
                scene_corners[1] + cv::Point2f((float)pattern.cols, 0), cv::Scalar(0, 255, 0), 4); 
        cv::line( img_matches, scene_corners[1] + cv::Point2f((float)pattern.cols, 0),
          scene_corners[2] + cv::Point2f((float)pattern.cols, 0), cv::Scalar( 0, 255, 0), 4 );
        cv::line( img_matches, scene_corners[2] + cv::Point2f((float)pattern.cols, 0),
            scene_corners[3] + cv::Point2f((float)pattern.cols, 0), cv::Scalar( 0, 255, 0), 4 );
        cv::line( img_matches, scene_corners[3] + cv::Point2f((float)pattern.cols, 0),
            scene_corners[0] + cv::Point2f((float)pattern.cols, 0), cv::Scalar( 0, 255, 0), 4 );*/
    }

    cv::namedWindow( source_window );
    cv::imshow( source_window, image );
    cv::imshow( "Main", hierarchicalClustering );
    cv::imshow("Pattern", pattern); 
    cv::imshow("Good Matches", img_matches );
    cv::imshow("Detected Image",outDetectedImage ); 


    
    cv::waitKey(0);
    
    return EXIT_SUCCESS;
}


