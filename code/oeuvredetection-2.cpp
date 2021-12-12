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

// returns sequence of squares detected on the image.
static void findSquares( const cv::Mat& image, std::vector<std::vector<cv::Point> >& squares )
{
    
    int thresh = 50;  
    int N = 11; // levels 
    
    
    squares.clear();
    cv::Mat pyr, timg, gray0(image.size(), CV_8U), gray;
    // down-scale and upscale the image to filter out the noise
    cv::pyrDown(image, pyr, cv::Size(image.cols/2, image.rows/2));
    cv::pyrUp(pyr, timg, image.size());
    std::vector<std::vector<cv::Point> > contours;

    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        cv::mixChannels(&timg, 1, &gray0, 1, ch, 1);
    
        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                cv::Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                cv::dilate(gray, gray, cv::Mat(), cv::Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }
        
            // find contours and store them all as a list
            cv::findContours(gray, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
            
            std::vector<cv::Point> approx;
            
            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                cv::approxPolyDP(contours[i], approx, cv::arcLength(contours[i], true)*0.02, true);
                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    std::fabs(cv::contourArea(approx)) > 1000 &&
                    cv::isContourConvex(approx) )
                {
                    double maxCosine = 0;
                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = std::fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // cosines of all angles are small
                    if( maxCosine < 0.07 )
                        squares.push_back(approx);
                }
            }
        }
    }

}

// Filter squares by median area : 
// if abs(area - median area) > 10 % => remove
void filterSquares(std::vector<std::vector<cv::Point>>& squares, std::vector<double>& areas)
{

    // Process Areas 
    areas.clear(); 
    areas.reserve(squares.size()); 

    for (int k = 0; k < squares.size(); k++)
    {
        areas.push_back(cv::contourArea(squares[k])); 
    }

    // mediane 
    int idmed = areas.size() % 2 == 0 ? (int) ((float)areas.size() / 2.0f) : int((float)areas.size() / 2.0f) + 1; 
    double medianArea = areas.at(idmed); 
    
    std::cout << "Median Area = " << medianArea << " pix²." << std::endl; 

    // Determine outliers
    std::vector<int> indexRemove; 
    for (int k = 0; k < squares.size(); k++)
    {
        if (std::fabs(areas[k] - medianArea) > 0.10 * medianArea)
        {
            indexRemove.insert(indexRemove.begin(), k); 
        }
    }

    // Remove outliers 
    for (const int& c : indexRemove)
    {
        squares.erase(squares.begin() + c); 
        areas.erase(areas.begin() + c); 
    }

}

// Get the square side pixel = sqrt(meanArea) 
double getSquareSide(const std::vector<double>& areas)
{
    double sumarea = 0.0; 
    for (const double& area : areas) {
        sumarea += area; 
    }
    double meanArea = sumarea / (double)areas.size(); 

    std::cout << " Square Size Side : " << std::sqrt(meanArea) << "pixels." <<  std::endl; 

    return std::sqrt(meanArea); 

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
    cv::Mat blurred, image, imgcolor; 
    image = cv::imread( argv[1], cv::IMREAD_GRAYSCALE );
    imgcolor = cv::imread(argv[1], cv::IMREAD_COLOR); 

    const char* name  = argv[2]; 
    

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }


    // Find Square Pattern Size 
    // -------------------------
    std::vector<std::vector<cv::Point> > squares;
    std::vector<double> areas; 

    findSquares(imgcolor, squares);
    filterSquares(squares, areas); 
    
    float squareSize = (float)getSquareSide(areas); 

    // Draw Filter Squares
    cv::polylines(imgcolor, squares, true, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    
    // Draw one square of square side to vizually verify 
    cv::Rect2f OneSquare = cv::Rect2f((float)squares[0][0].x , (float)squares[0][0].y, squareSize, squareSize); 
    cv::rectangle(imgcolor, OneSquare, cv::Scalar(255, 0, 0, 125), -1); 

    // Process Mean of all points to find image "center" at least a point in image 
    cv::Point meanPoint = {0, 0}; 
    int npoints = 0; 
    for (const std::vector<cv::Point>& s : squares)
    {
        for (const cv::Point& p : s)
        {
            meanPoint += p; 
            npoints++; 
        }
    } 
    meanPoint.x = int((float)meanPoint.x /  float(npoints)); 
    meanPoint.y = int((float)meanPoint.y /  float(npoints)); 

    cv::circle(imgcolor, meanPoint, 10, cv::Scalar(0,0,255, 255), cv::FILLED); 

    // image
    exportImage(name, "_patternSquareSize.png", imgcolor); 
    show_wait_destroy("PatternSquare", imgcolor); 


    // ==============================================
    // Methode Entropie 
    // ==============================================
    
    // Entropie Detection 
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

    
    printPoint("TopLeft", topLeft);  
    printPoint("bottomRight", bottomRight);

    // Comparison between the Two methodes 
    /*cv::line(imageDetection, cv::Point2f(xHist.x, yHist.x), cv::Point2f(xHist.y, yHist.x), cv::Scalar(255,0,255,180), 15); 
    cv::line(imageDetection, cv::Point2f(xHist.y, yHist.x), cv::Point2f(xHist.y, yHist.y), cv::Scalar(255,0,255,180), 15); 
    */

    show_wait_destroy("Image Detection Carrre Entropie", imageDetection);



    // ==============================================
    // Methode histogrammes 
    // ==============================================

    // Enhanced image 
    // ----------------
    float alpha = 1.f; 
    float beta = 0.0f; 
    cv::Mat enhancedImage = image.clone(); 
    enhancedImage.convertTo(enhancedImage, -1, alpha, beta);



    // Detected edge 
    // ----------------
    cv::Mat cannyEdge;  
    int threshold_canny = 100; 
    detectEdge(enhancedImage, cannyEdge, threshold_canny); 

    show_wait_destroy("Canny Edge on enhanced image", cannyEdge); 
    exportImage(name, "_cannyEdge_100_perspectivecorrection.png", cannyEdge); 


    // Histograms
    // -------------
    std::vector<int> histocannyx; 
    std::vector<int> histocannyy; 
    processAxisHistograms(histocannyx, histocannyy, cannyEdge, 200); 

    int thresholdN = 125; 
    cv::Point2f xHist = cv::Point2f(FLT_MAX, FLT_MIN); 
    cv::Point2f yHist = cv::Point2f(FLT_MAX, FLT_MIN);

    cv::Mat realHisto = cv::Mat::zeros(cannyEdge.size[0], cannyEdge.size[1], CV_8UC1); 
    for (int k = 0; k < cannyEdge.size[1] ; k++)
    {
        if (k < 20 || k > cannyEdge.size[1] - 20)
        {
            continue; 
        }

        // filter with entropie
        if (k < topLeft.x || k > bottomRight.x) {
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
        // filter with entropie
        if (k < topLeft.y || k > bottomRight.y) {
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
    cv::line(rgbrealHisto, cv::Point2f(xHist.x, (float)thresholdN), cv::Point2f(xHist.y, (float)thresholdN), cv::Scalar(255,0,0,255), 15); 
    cv::line(rgbrealHisto, cv::Point2f((float)thresholdN, yHist.x), cv::Point2f((float)thresholdN, yHist.y), cv::Scalar(0,0,255,255), 15); 
    show_wait_destroy("Histogram of Canny Edge", rgbrealHisto); 

    std::string histoname = name + std::string("_histograms.png"); 
    cv::imwrite(histoname, rgbrealHisto); 



    

    cv::Mat imageRGB; 
    cv::cvtColor(image,imageRGB, cv::COLOR_GRAY2BGR); 
    cv::Mat roi = imageRGB(cv::Rect(cv::Point2f(xHist.x, yHist.x), cv::Point2f(xHist.y, yHist.y))); 

    show_wait_destroy("DetectedImage", roi);

    std::string imageDetectedname = name + std::string("_imagedetected.png"); 
    cv::imwrite(imageDetectedname, roi); 
    std::cout << "Image Detected written at " << imageDetectedname << std::endl; 

    

    return EXIT_SUCCESS; 
}
