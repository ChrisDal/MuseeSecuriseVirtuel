#include <iostream> 
#include <string>
#include <algorithm>
#include <functional>
#include <array>
#include <cfloat>
#include <map>

#include "BasePermutation.h"
#include "imageanalysis.h"
#include "toolFunctions.hpp"

#include <opencv2/core/types.hpp>
#include <opencv2/core/fast_math.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>


#define DEBUG_ON_DISPLAY 1

// ==========================================================================================
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

// ======================================================================================


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

void detectEdge(const cv::Mat& src, cv::Mat& edges, int cannyThreshFound)
{
    // Detected edge 
    // ----------------

    // blurred image => contours 
    cv::Mat blurred; 
    src.copyTo(blurred);
    cv::medianBlur(src, blurred, 9);

    // Threshold canny 
    edges = cv::Mat::zeros(cv::Size(src.size[1], src.size[0]) , CV_8UC1);
    cv::Canny(blurred, edges, cannyThreshFound, cannyThreshFound*3, 3);

    const char* window_name = "Edge Map";
    cv::namedWindow( window_name, cv::WINDOW_NORMAL);
    cv::imshow(window_name, edges); 
    cv::waitKey(0); 
}

// Process histogram per axis on binary image (0 or 255)
void processAxisHistograms(std::vector<int>& histox, std::vector<int>& histoy, const cv::Mat& src, int threshold=200)
{
    histox.clear(); 
    histox.resize(src.size[1], 0);

    histoy.clear(); 
    histoy.resize(src.size[0], 0); 


    for (int j = 0; j < src.size[1]; j++ )
    {
        for (int k = 0; k < src.size[0]; k++)
        {
            if (src.at<uchar>(k,j) > threshold)
            {
                histox[j] += 1; 
            }
        }
    }

    for (int k = 0; k < src.size[0]; k++)
    {
        for (int j = 0; j < src.size[1]; j++ )
    {
            if (src.at<uchar>(k,j) > threshold)
            {
                histoy[k] += 1; 
            }
        }
    }
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
    cv::Mat blurred, image; 
    image = cv::imread( argv[1], cv::IMREAD_GRAYSCALE );
    const char* name  = argv[2]; 
    

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    // Detected edge 
    // ----------------
    cv::Mat cannyEdge;  
    int threshold_canny = 35; 
    detectEdge(image, cannyEdge, threshold_canny); 


    // Histograms
    // -------------
    std::vector<int> histocannyx; 
    std::vector<int> histocannyy; 
    processAxisHistograms(histocannyx, histocannyy, cannyEdge, 200); 

    int thresholdN = 150; 
    cv::Point2f xHist = cv::Point2f(FLT_MAX, FLT_MIN); 
    cv::Point2f yHist = cv::Point2f(FLT_MAX, FLT_MIN);

    cv::Mat realHisto = cv::Mat::zeros(cannyEdge.size[0], cannyEdge.size[1], CV_8UC1); 
    for (int k = 0; k < cannyEdge.size[1] ; k++)
    {
        if (k < 20 || k > cannyEdge.size[1] - 20)
        {
            continue; 
        }

        if (histocannyx[k] > thresholdN) {
            
            if (k < xHist.x) {
                xHist.x =(float) k; 
            }

            if (k > xHist.y) {
                xHist.y = (float)k; 
            }
        }

        for (int j = 0; j < histocannyx[k] ; j++)
        {
            realHisto.at<uchar>(j,k) = 255; 
        }
    }

    for (int k = 0; k < cannyEdge.size[0] ; k++)
    {
        if (k < 20 || k > cannyEdge.size[0] - 20)
        {
            continue; 
        }

        if (histocannyy[k] > thresholdN) {
            
            if (k < yHist.x) {
                yHist.x = (float)k; 
            }

            if (k > yHist.y) {
                yHist.y = (float)k; 
            }
        }


        for (int j = 0; j < histocannyy[k] ; j++)
        {
            realHisto.at<uchar>(k,j) = 255; 
        }
    }

    printPoint("xhist", xHist); 
    printPoint("yhist", yHist); 
    cv::Mat rgbrealHisto; 
    cv::cvtColor(realHisto, rgbrealHisto, cv::COLOR_GRAY2BGR); 
    cv::line(rgbrealHisto, cv::Point2f(xHist.x, 150.f), cv::Point2f(xHist.y, 150.f), cv::Scalar(255,0,0,255), 15); 
    cv::line(rgbrealHisto, cv::Point2f(150.f, yHist.x), cv::Point2f(150.f, yHist.y), cv::Scalar(0,0,255,255), 15); 
    show_wait_destroy("Histogram of Canny Edge", rgbrealHisto); 

    std::string histoname = name + std::string("_histograms.png"); 
    cv::imwrite(histoname, rgbrealHisto); 

    

    unsigned int dxy = 32; // pixels 
    int nx = (int) (image.size[1] / (float)dxy); 
    int ny = (int) (image.size[0] / (float)dxy); 

    cv::Mat entropimage = cv::Mat::zeros(image.size[0],image.size[1], CV_8UC1); 

    for (int k = 0 ; k < nx ; k++)
    {
        for (int j = 0; j < ny ; j++)
        {
            cv::Rect patch = cv::Rect(k * dxy, j *dxy, dxy, dxy); 
            cv::Mat roi = image(patch); 

            double entropie =  getEntropie(roi, dxy, dxy, 256);
            
            cv::Mat entRoi = entropimage(patch); 
            cv::Mat valueEntropy = cv::Mat::ones(dxy, dxy, CV_8UC1) * cvRound(31*entropie); 
            valueEntropy.copyTo(entRoi); 

        }
    }
    show_wait_destroy("Test Entropie ", entropimage); 
    std::string entropiename = name + std::string("_entropie.png"); 
    cv::imwrite(entropiename, entropimage); 

    cv::Mat thresholdEnt;
    float minentropy = 6.0f;  
    cv::threshold(entropimage, thresholdEnt, cvRound(31*minentropy), 255, cv::THRESH_BINARY);
    show_wait_destroy("Threshold entropy", thresholdEnt) ; 

    // Find carre Entropy 
    // ---------------
    cv::Mat new_image; 
    cv::bitwise_and(image, image, new_image, thresholdEnt); 
    show_wait_destroy("Threshold entropy image", new_image);

    cv::Point2f topLeft = cv::Point2f(FLT_MAX, FLT_MAX); 
    cv::Point2f bottomRight = cv::Point2f(FLT_MIN, FLT_MIN); 

    for (int kx = 0; kx <  thresholdEnt.size[1]; kx++)
    {
        for (int ky = 0; ky < thresholdEnt.size[0] ; ky++)
        {
            if (thresholdEnt.at<IMAGEMAT_TYPE>(ky, kx) == 255)
            {
                // 
                if (kx <= topLeft.x)
                {
                    topLeft.x = (float)kx; 
                }

                if (ky <= topLeft.y )
                {
                    topLeft.y = (float)ky; 
                }

                if (kx >= bottomRight.x) 
                {
                    bottomRight.x = (float) kx; 
                }
                if (  ky >= bottomRight.y)
                {
                    bottomRight.y = (float) ky; 
                }


            }
        }
    }

    cv::Mat imageDetection; 
    cv::cvtColor(image,imageDetection, cv::COLOR_GRAY2BGR); 
    cv::rectangle(imageDetection, topLeft, bottomRight, cv::Scalar(0,0,255,255), 5); 
    cv::circle(imageDetection, topLeft , 10,  cv::Scalar(255,0,0,255), cv::FILLED, 8, 0 );
    cv::circle(imageDetection, bottomRight , 10,  cv::Scalar(0,0,255,255), cv::FILLED, 8, 0 );

    cv::line(imageDetection, cv::Point2f(xHist.x, yHist.x), cv::Point2f(xHist.y, yHist.x), cv::Scalar(255,0,255,180), 15); 
    cv::line(imageDetection, cv::Point2f(xHist.y, yHist.x), cv::Point2f(xHist.y, yHist.y), cv::Scalar(255,0,255,180), 15); 

    printPoint("TopLeft", topLeft);  
    printPoint("bottomRight", bottomRight);

    show_wait_destroy("Image Detection Carrre", imageDetection);

    cv::Mat imageRGB; 
    cv::cvtColor(image,imageRGB, cv::COLOR_GRAY2BGR); 
    cv::Mat roi = imageRGB(cv::Rect(cv::Point2f(xHist.x, yHist.x), cv::Point2f(xHist.y, yHist.y))); 

    show_wait_destroy("DetectedImage", roi);

    std::string imageDetectedname = name + std::string("_imagedetected.png"); 
    cv::imwrite(imageDetectedname, roi); 
    std::cout << "Image Detected written at " << imageDetectedname << std::endl; 

    

    return EXIT_SUCCESS; 
}
