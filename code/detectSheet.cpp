#include <iostream> 
#include <string>
#include <vector>
#include <cfloat>
#include "imageanalysis.h"
#include "toolFunctions.hpp"

#define DEBUG_ON_DISPLAY 1

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

/* detectSheet.cpp 
Objectif : Detect the sheet of paper where is the oeuvre 
Return a png file where the oeuvre is corrected from distortion 
*/ 


// ======================================================================================

int main( int argc, char** argv )
{
    
    const char* source_window = "Canny Edge";

    if ( argc != 3 )
    {
        printf("usage: detectSheet.out <PictureToAnalyse> <ExtractedData> \n");
        return -1;
    }
    cv::Mat image, blurred;
    cv::Mat cannyEdge;

    image = cv::imread( argv[1], cv::IMREAD_GRAYSCALE );
    const char* name  = argv[2]; 
    const char* window_name = "Edge Map";

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    // ---------------------------------
    // Preprocessing : Enhanced contrast 
    // ---------------------------------
    cv::Mat enhanced_image = image.clone(); 
    float alpha = 1.5f; 
    float beta = 0.0f; 
    
    enhanced_image.convertTo(enhanced_image, -1, alpha, beta);
    cv::imshow("Image Enhanced", enhanced_image); 
    cv::waitKey(0); 

    // blurred image => contours 
    enhanced_image.copyTo(blurred);
    cv::medianBlur(image, blurred, 9);

    cannyEdge = cv::Mat::zeros(cv::Size(image.size[1], image.size[0]) , CV_8UC1); 
    // Find 250 and 250*3 
    int cannyThreshFound = 100; 
    cv::Canny(blurred, cannyEdge, cannyThreshFound, cannyThreshFound*3, 3);
    show_wait_destroy( window_name, cannyEdge);

    // Exported
    std::string tr_name = std::to_string(cannyThreshFound) + ".png"; 
    exportImage(name, tr_name, cannyEdge);

    // ====================================
    // Find contours 
    // -------------
    double minCypherImageSize = 0.25 * double(image.size[0] * image.size[1]); 
    std::vector< std::vector <cv::Point> > contours; 
    std::vector<cv::Vec4i> hierarchy; 
    // dilate canny output to remove potential holes between edge segments
    cv::dilate(cannyEdge, cannyEdge, cv::UMat(), cv::Point(-1,-1));
    cv::findContours(cannyEdge, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> approximations; 
    std::vector<std::vector<cv::Point>> rectangles;
    for( size_t i = 0; i < contours.size(); i++ )
    {
        cv::approxPolyDP(contours[i], approximations, cv::arcLength(contours[i], true)*0.08, true);
        if( approximations.size() == 4 && std::fabs(cv::contourArea(approximations)) > minCypherImageSize &&
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
    cv::Mat imageColor = cv::imread(argv[1]);
    std::vector<cv::Point> ourSheet;
    ourSheet.reserve(4);  
    double maxSurface = FLT_MIN; 

    for (int idx=0; idx >=0; idx=hierarchy[idx][0])
    {
        cv::Scalar color( rand()&255, rand()&255, rand()&255, 80 );
        cv::drawContours( imageColor, contours, idx, color, cv::FILLED, 8, hierarchy );
    }
    
    for (std::vector<cv::Point>& rect : rectangles)
    {
        std::cout << "Process Rectangle ... " << std::endl; 
        cv::polylines(imageColor, ourSheet, true, cv::Scalar(0, 255, 0, 255), 3, cv::LINE_AA);
        
        
        if (std::fabs(cv::contourArea(rect)) > maxSurface){
            ourSheet = rect; 
            maxSurface = std::fabs(cv::contourArea(rect)); 
        }
 
    }

    std::cout << "End Processing Rectangle ... " << std::endl; 
    cv::polylines(imageColor, ourSheet, true, cv::Scalar(0, 0, 255, 255), 3, cv::LINE_AA);
    show_wait_destroy("Edge Detection carre", imageColor); 
    // Export Image 
    exportImage(name, "_found.png", imageColor); 
    

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
    // Export Image 
    exportImage(name, "_perspective.png", warpImage);  
    show_wait_destroy("Warp Image ", warpImage); 

    // histogram 
    cv::Mat histo = processBarHistogram(warpImage, cv::Scalar(0, 255, 0)); 
    show_wait_destroy("Histogram", histo); 


    return EXIT_SUCCESS; 
}



