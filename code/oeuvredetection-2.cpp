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
float distanceNumber(T p1, T p2)
{
    return static_cast<float>(std::sqrt(std::pow(p2- p1,2))); 
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

template < typename T > 
void printPoint(const char* message, const T& point )
{
    std::cout << message << " "; 
    std::cout << "[" << point.x << "," << point.y << "]" << std::endl; 
}

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( cv::Point pt1, cv::Point pt2, cv::Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


static double distance(const cv::Point p0, const cv::Point p1)
{
    return std::sqrt(std::pow(p1.x - p0.x, 2) +  std::pow(p1.y - p0.y, 2)); 
}

// ======================================================================================

void show_wait_destroy(const char* winname, cv::Mat img);

int lowThreshold = 50;
const int max_lowThreshold = 300;
const int ratio = 3;
const int kernel_size = 3;
const char* window_name = "Edge Map";


double getEntropie(const cv::Mat& Img, unsigned int width, unsigned int height, unsigned int nbsymb)
{
    double h = 0.0; 
    unsigned int size = width * height;

    // histogramme 
    unsigned int histo[256] = {0};
    for (unsigned int k=0; k < width; k++)
    {
        for (unsigned int j=0; j < height; j++)
        {
            histo[Img.at<IMAGEMAT_TYPE>(j, k)] += 1;  
        }
    }

    // entropie
    for (unsigned int k=0; k <nbsymb; k++ )
    {
        float palpha = float(histo[k]) / float(size); 

        if (palpha > 0.0) 
            h += (palpha) * std::log2(palpha); 
    }

    return -h; 

}


// ======================================================================================


int main( int argc, char** argv )
{
    
    const char* source_window = "Canny Edge";

    if ( argc != 3 )
    {
        printf("usage: detectSheet.out <PictureToAnalyse> <ExtractedData> \n");
        return -1;
    }
    
    cv::Mat image = cv::imread( argv[1], cv::IMREAD_GRAYSCALE );
    const char* name  = argv[2]; 

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    unsigned int dxy = 64; // pixels 
    int nx = (int) (image.size[1] / (float)dxy); 
    int ny = (int) (image.size[0] / (float)dxy); 

    cv::Mat entropimage = cv::Mat::zeros(dxy*ny,dxy*nx, CV_8UC1); 

    for (unsigned int k = 0 ; k < nx ; k++)
    {
        for (unsigned int j = 0; j < ny ; j++)
        {
            cv::Rect patch = cv::Rect(k * dxy, j *dxy, dxy, dxy); 
            cv::Mat roi = image(patch); 

            double entropie =  getEntropie(roi, dxy, dxy, 256);
            
            cv::Mat entRoi = entropimage(patch); 
            cv::Mat valueEntropy = cv::Mat::ones(dxy, dxy, CV_8UC1) * cvRound(31*entropie); 
            valueEntropy.copyTo(entRoi); 

        }
    }
    cv::namedWindow("Test Entropie ", cv::WINDOW_NORMAL); 
    cv::imshow("Test Entropie ", entropimage); 
    cv::waitKey(0); 

    cv::Mat thresholdEnt;
    float minentropy = 6.0f;  
    cv::threshold(entropimage, thresholdEnt, cvRound(31*minentropy), 255, cv::THRESH_BINARY);
    show_wait_destroy("Threshold entropy", thresholdEnt) ; 

    
    return EXIT_SUCCESS; 
}


void show_wait_destroy(const char* winname, cv::Mat img) {
    cv::namedWindow(winname, cv::WINDOW_NORMAL);
    cv::imshow(winname, img);
    cv::moveWindow(winname, 500, 0);
    cv::waitKey(0);
    cv::destroyWindow(winname);
}
