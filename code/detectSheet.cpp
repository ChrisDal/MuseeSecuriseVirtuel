#include <iostream> 
#include <string>
#include <vector>
#include <cfloat>
#include "imageanalysis.h"

#define DEBUG_ON_DISPLAY 1

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

/* detectSheet.cpp 
Objectif : Detect the sheet of paper where is the oeuvre 
Return a png file where the oeuvre is corrected from distortion 
*/ 

cv::Mat image, blurred;
cv::Mat cannyEdge;

// ======================================================================================

std::vector<std::pair<float, float>> houghAnalysis(const std::vector<cv::Point2f>& vecClassified, int width, int height, 
                                                    float dtheta=1.5f, float drho = 3.2f, float maxtheta=180.f)
{

    double maxRho = std::sqrt(width*width + height*height); 

    int Ntheta = (int)(maxtheta / dtheta) + 1;
    int Nrho   = (int)(maxRho / drho) + 1; 

    // Hough transformation  
    std::vector<std::vector<unsigned int> > houghvec(Ntheta, std::vector<unsigned int>(Nrho, 0));   

    // pour chaque point 
    for (const cv::Point2f& p : vecClassified)
    {
        for (int k=0; k < Ntheta; k++)
        {
            float theta = dtheta * (float)k * 3.14f / 180.0f; 
            float rho = p.x * std::cos(theta) + p.y * std::sin(theta); 

            int krho = (int) std::abs(rho / drho); 
            houghvec[k][krho] += 1; 
        }
    }

    cv::Mat testMat = cv::Mat::zeros(Nrho, Ntheta, CV_8UC1); 
    const unsigned int minVotes = 20; 
    std::vector<std::pair<float, float>> winnerTuple; 
    for (int k = 0; k < Ntheta; k++)
    {
        for (int i = 0; i < Nrho; i++)
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

        if (DEBUG_ON_DISPLAY)
        {
            cv::imshow("Line Houghs", imageOndisp); 
            cv::waitKey(0); 
        }

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

void show_wait_destroy(const char* winname, cv::Mat img) {
    //cv::namedWindow(winname, cv::WINDOW_NORMAL);
    cv::imshow(winname, img);
    cv::moveWindow(winname, 500, 0);
    cv::waitKey(0);
    cv::destroyWindow(winname);
}

int lowThreshold = 50;
const int max_lowThreshold = 300;
const int ratio = 3;
const int kernel_size = 3;
const char* window_name = "Edge Map";

static void CannyThreshold(int, void*)
{
    cannyEdge = cv::Mat::zeros(cv::Size(image.size[1], image.size[0]) , CV_8UC1); 
    cv::Canny(blurred, cannyEdge, lowThreshold, lowThreshold*ratio, kernel_size);
    cv::imshow( window_name, cannyEdge );
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
    
    image = cv::imread( argv[1], cv::IMREAD_GRAYSCALE );
    const char* name  = argv[2]; 

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }


    // Enhanced contrast 
    float alpha = 1.5f; 
    float beta = 0.0f; 
    image.convertTo(image, -1, alpha, beta);
    cv::imshow("Image Enhanced", image); 
    cv::waitKey(0); 

    // blurred image => contours 
    image.copyTo(blurred);
    cv::medianBlur(image, blurred, 9);

    cannyEdge = cv::Mat::zeros(cv::Size(image.size[1], image.size[0]) , CV_8UC1); 
    // Find 250 and 250*3 
    int cannyThreshFound = 100; 
    cv::Canny(blurred, cannyEdge, cannyThreshFound, cannyThreshFound*3, 3);

    cv::namedWindow( window_name, cv::WINDOW_NORMAL );
    cv::imshow(window_name, cannyEdge);
    cv::waitKey(0); 
    std::string tr_name = std::string(name) + std::to_string(cannyThreshFound) + ".png"; 
    cv::imwrite(tr_name.c_str(), cannyEdge); 
    std::cout << "Image Exported : " << tr_name << std::endl; 

    // ====================================
    // Find contours 
    // -------------
    std::vector< std::vector <cv::Point> > contours; 
    cv::findContours(cannyEdge, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> approximations; 
    std::vector<std::vector<cv::Point>> rectangles;
    for( size_t i = 0; i < contours.size(); i++ )
    {
        cv::approxPolyDP(contours[i], approximations, cv::arcLength(contours[i], true)*0.02, true);
        if( approximations.size() == 4 && std::fabs(cv::contourArea(approximations)) > 1000 &&
            cv::isContourConvex(approximations) )
        {
            double maxCosine = 0;
            for( int j = 2; j < 5; j++ )
            {
                // find the maximum cosine of the angle between joint edges
                double cosine = std::fabs(angle(approximations[j%4], approximations[j-2], approximations[j-1]));
                maxCosine = MAX(maxCosine, cosine);
            }
            // if cosines of all angles are small
            // (all angles are ~90 degree) then write quandrange
            // vertices to resultant sequence
            if( maxCosine < 0.3 ){
                rectangles.push_back(approximations);
            }
        }

    }

    // ====================================
    // Find the biggest surface  
    std::vector<cv::Point> ourSheet;
    ourSheet.reserve(4);  
    double maxSurface = FLT_MIN; 
    
    for (std::vector<cv::Point>& rect : rectangles)
    {
        if (std::fabs(cv::contourArea(rect)) > maxSurface){
            ourSheet = rect; 
            maxSurface = std::fabs(cv::contourArea(rect)); 
        }
    }

    cv::Mat imageColor =  cv::imread( argv[1]);
    cv::polylines(imageColor, ourSheet, true, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);

    show_wait_destroy("Edge Detection carre", imageColor); 
    // export image
    std::string foundCarrename = std::string(name) + "_found.png"; 
    cv::imwrite(foundCarrename.c_str(), imageColor); 
    std::cout << "Image Exported : " << foundCarrename << std::endl;


    // ====================================
    // Warp image
    // -----------
    unsigned int pixA4_width = 2480; 
    unsigned int pixA4_height = 3508; 

    // ordering topLeft, topRight, bottomLeft, bottomRight
    cv::Point2f topLeft, topRight, bottomLeft, bottomRight;
    double mindist = FLT_MAX; 
    double maxdist = FLT_MIN; 

    for (const cv::Point& p : ourSheet)
    {
        double d = distance(cv::Point(0,0), p); 
        if (d < mindist) { 
            mindist = d; 
            topLeft =cv::Point2f( p); 
        }

        if (d > maxdist)
        {
            bottomRight =cv::Point2f( p); 
            maxdist = d; 
        }
    }

    double disty = FLT_MAX; 
    double distx = FLT_MAX; 
    for (const cv::Point& p : ourSheet)
    {
        if (cv::Point2f(p) == topLeft || cv::Point2f(p) == bottomRight)
        {
            continue; 
        }

        double dx = std::fabs(topLeft.x - p.x); 
        double dy = std::fabs(topLeft.y - p.y); 
 
        if (dx < distx) { 
            distx = dx; 
            bottomLeft = cv::Point2f(p); 
        }

        if (dy < disty)
        {
            disty = dy; 
            topRight = cv::Point2f(p); 
        }
    }

    printPoint("topLeft", topLeft); 
    printPoint("topRight", topRight); 
    printPoint("bottomLeft", bottomLeft); 
    printPoint("bottomRight", bottomRight); 

    cv::Point2f src[4] = {topLeft, topRight, bottomLeft, bottomRight}; 
    cv::Point2f dst[4] = {cv::Point2f(0.0f, 0.0f), 
                        cv::Point2f((float)pixA4_width, 0.0f), 
                          cv::Point2f(0.0f, (float)pixA4_height),
                          cv::Point2f((float)pixA4_width, (float)pixA4_height)}; 
    cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(src, dst); 

    cv::Mat warpImage = cv::Mat::zeros(cv::Size(pixA4_height, pixA4_width), CV_8UC1);   

    cv::warpPerspective(image, warpImage, perspectiveMatrix, cv::Size(pixA4_width, pixA4_height)); 
    std::string perspectiveimage = std::string(name) + "_perspective.png"; 
    cv::imwrite(perspectiveimage.c_str(), warpImage); 
    show_wait_destroy("Warp Image ", warpImage); 

    // histogram 
    cv::Mat histo = processBarHistogram(warpImage, cv::Scalar(0, 255, 0)); 
    show_wait_destroy("Histogram", histo); 


    return EXIT_SUCCESS; 
}



