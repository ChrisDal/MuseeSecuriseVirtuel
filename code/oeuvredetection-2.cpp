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

// remove duplicate points  
template <typename T> 
void filterPoints(std::vector<T>& features, float threshold = (float)std::sqrt(2))
{
    std::vector<int> index_todelete; 
    int Npoints = 0; 
    int overall = (int)features.size(); 

    float dist_mean = 0.0f; 
    for (int k = 0; k < overall; k++ )
    {
        for (int j = k; j < overall; j++ )
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
    for ( int k = 0; k < (int)index_todelete.size(); k++)
    {
        //std::cout << " Remove point " << index_todelete.at(k) << std::endl;
        Npoints++; 
        features.erase(features.begin() + index_todelete.at(k) ); 
    }

    std::cout << "Filtering removed " << Npoints << "/" << overall << " points." << std::endl; 

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

// clustered with 4 classes 
template <typename T> 
void unconformedClustering(const std::vector<T>& vecTocluster, std::vector<int>& vecClustered, const T& center, float radius=0.0f)
{
    vecClustered.clear(); 
    vecClustered.reserve(vecTocluster.size()); 

    for (int k= 0; k < vecTocluster.size(); k++)
    {
        
        float x = (float)vecTocluster.at(k).x;
        float y = (float)vecTocluster.at(k).y;

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

template <typename T> 
std::vector<T> getVecPoint(const std::vector<int>& classifiedVec, const std::vector<T>&  harrispoints, int classwanted)
{
    // trying to find a good match for each clustered class 0 1 2 3
    std::vector<T> pointofClass;
    for (unsigned int k=0; k < (unsigned int)classifiedVec.size(); k++)
    {
        if (classifiedVec.at(k) == classwanted)
        {
            pointofClass.push_back(harrispoints.at(k)); 
            
        }
    }

    return pointofClass; 
}

std::vector<std::pair<float, float>> houghAnalysis(const std::vector<cv::Point2f>& vecClassified, int width, int height, 
                                                    float dtheta=1.5f, float drho = 3.2f, float maxtheta=180.f,
                                                    const std::string& name = "")
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
        for (int k=0; k < Ntheta; k++)
        {
            float theta = dtheta * (float)k * 3.14f / 180.0f; 
            float rho = p.x * std::cos(theta) + p.y * std::sin(theta); 

            int krho = (int) std::abs(rho / drho); 
            houghvec[k][krho] += 1; 
        }
    }

    cv::Mat testMat = cv::Mat::zeros(Nrho, Ntheta, CV_8UC1); 
    const unsigned int minVotes = 6; 
    std::vector<std::pair<float, float>> winnerTuple; 
    for (int k = 0; k < Ntheta; k++)
    {
        for (int i = 0; i < Nrho; i++)
        {
            
            testMat.at<uchar>(i, k) = 50*houghvec[k][i]; 
            float ktheta = (float)k*dtheta; 
            //bool validAngle = true; 
            bool validAngle = ktheta > 177.f || ktheta < 3.5f || ( ktheta < 93.5f && ktheta > 86.5f); 
            validAngle =  (ktheta == 180.0f) || ktheta == 0.0f || ktheta == 90.f; 
            if (houghvec[k][i] >= minVotes && validAngle) {
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
        if ( ! name.empty())
        {
            exportImage(name, "_accumulatorhough.png", testMat); 
        }
        
    }

    return winnerTuple; 
}


void displayHoughLines(const std::vector< std::pair<float, float> >& analysisresult, const cv::Mat& imageOndisp, const std::string& name)
{
    cv::Mat displayImage = imageOndisp.clone(); 

    for (const std::pair<float, float>& lines : analysisresult)
    {
        float rho = lines.second; 
        float theta = lines.first * 3.14f / 180.0f; // theta = degree

        float a = std::cos(theta); 
        float b = std::sin(theta); 
        cv::Point2f p0(rho * a, rho * b); 

        cv::Point2f p1(p0.x + (int)(10000 * (-b)), p0.y + (int)(10000*a)); 
        cv::Point2f p2(p0.x - (int)(10000 * (-b)), p0.y - (int)(10000*a)); 

        cv::line(displayImage ,p1, p2, cv::Scalar(0,0,255, 255)); 
    }

    show_wait_destroy("Line Houghs", displayImage); 
    exportImage(name, "_linehoughs.png", displayImage); 
    
}

// 
cv::Point2f getIndexOfPoint(const std::vector<cv::Point2f>& classifiedPoint, int width, int height, int cornerType=0, float dsize = 5.0f)
{
    std::cout << "Class c=" << cornerType << std::endl; 
    int krows = (int)(width/dsize);  // image.size[1]
    int kcols = (int)(height/dsize); // image.size[0]
    unsigned int numberMinPoints = 2; 

    std::vector<unsigned int> historows(krows,  0); 
    std::vector<unsigned int> histocols(kcols,  0);

    if (cornerType > 3 || cornerType <  0) {
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
                index.x = (float)k; 
            }
        } 
        else if (cornerType == 1 || cornerType == 3)
        {
            if (historows[k] > numberMinPoints && (float)k < index.x) {
                index.x = (float)k; 
            }
        }
        
    }


    for (unsigned int k = 0; k< histocols.size(); k++)
    {
        if (cornerType == 0 || cornerType == 1)
        {
            if (histocols[k] > numberMinPoints) {
                index.y = (float)k; 
            }
        }
        else if (cornerType == 2 || cornerType == 3)
        {
            if (histocols[k] > numberMinPoints && (float)k < index.y) {
                index.y = (float)k; 
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
    
    const char* source_window = "Canny Edge";

    if ( argc != 3 )
    {
        printf("usage: detection.out <PictureToAnalyse> <ExtractedData> \n");
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
    std::vector<cv::Point2f> pointsToclass; 
    std::vector<double> areas; 
    findSquares(imgcolor, squares);
    filterSquares(squares, areas); 
    
    double squareSize = getSquareSide(areas); 

    // set error 
    double margeErreur = 0.03; 
    squareSize = std::ceil(squareSize); 
    std::cout << " Square Size Side : " << squareSize << "pixels." <<  std::endl;

    // Draw Filter Squares
    cv::polylines(imgcolor, squares, true, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    
    // Draw one square of square side to vizually verify 
    cv::Rect2f OneSquare = cv::Rect2f((float)squares[0][0].x , (float)squares[0][0].y, 
                                    (float)squareSize, (float)squareSize); 
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
            pointsToclass.push_back(cv::Point2f(p)); 
            cv::circle(imgcolor, p, 3, cv::Scalar(255,0,0, 255), cv::FILLED);
        }
    } 
    meanPoint.x = int((float)meanPoint.x /  float(npoints)); 
    meanPoint.y = int((float)meanPoint.y /  float(npoints)); 

    cv::circle(imgcolor, meanPoint, 10, cv::Scalar(0,0,255, 255), cv::FILLED);
     

    // image
    exportImage(name, "_patternSquareSize.png", imgcolor); 
    show_wait_destroy("PatternSquare", imgcolor); 

    // =========================================================
    // Hough Method 
    // =========================================================
    // 0 à 2pi 
    std::vector<int> vecClustered; 
    unconformedClustering(pointsToclass, vecClustered, cv::Point2f(meanPoint), (float)image.cols/10.f ); 

    std::vector<cv::Point2f> pointsClasswanted =  getVecPoint(vecClustered, pointsToclass, 0); 
    
    /*int imgWidth = imgcolor.size[1]; 
    int imgHeight = imgcolor.size[0]; 
    float diag = (float)std::sqrt(imgWidth*imgWidth + imgHeight*imgHeight); 
    float drho = diag*0.00055f ; 

    // Filter points
    //filterPoints(pointsToclass, 4.0f); 
    std::vector<std::pair<float, float>> houghlines = houghAnalysis(pointsToclass, imgWidth, imgHeight, 
                                                                   2.0f, drho, 180.0f, name); 
    displayHoughLines(houghlines, imgcolor, name); */


    // display 
    cv::Mat hierarchicalClustering = cv::imread(argv[1], cv::IMREAD_COLOR); 
    std::vector<cv::Scalar> clustercolors = {cv::Scalar(255,0,0,255), cv::Scalar(0,255,0,255), 
                                            cv::Scalar(0,0,255,255), cv::Scalar(36,180,240,255), 
                                            cv::Scalar(255,0,255,255) }; 

    for( int i = 0; i < pointsToclass.size() ; i++ )
    {       
        cv::Scalar color = cv::Scalar(vecClustered.at(i)*20, 0,vecClustered.at(i)*20,255); 

        if (vecClustered.at(i) < 4 && vecClustered.at(i) > -1)
        {
            color = clustercolors[vecClustered.at(i)]; 
        }
        else
        {
            color = clustercolors[4]; 
        }
        
        cv::circle( hierarchicalClustering, pointsToclass.at(i) , 5,  color, 2, 8, 0 );
        
    }
    // =====================================================
    // iterate through class 
    std::vector<cv::Point2f> intersections;
    intersections.reserve(4); 
    for (unsigned int c = 0; c < 4; c++)
    {
        std::vector<cv::Point2f> pointsclass = getVecPoint(vecClustered, pointsToclass, c);
        // Histogram per line 
        cv::Point2f cornerlineindex = getIndexOfPoint(pointsclass, image.size[1], image.size[0], c);

        std::cout << " Class c=" << c << " Find intersection at " << cornerlineindex << std::endl; 
        
        // vertical line 
        cv::line(hierarchicalClustering, 
                cv::Point2f(cornerlineindex.x, 0.0f),   
                cv::Point2f(cornerlineindex.x, (float)hierarchicalClustering.size[0]), 
                clustercolors[c]);
        // horizontal Line 
        cv::line(hierarchicalClustering, 
                cv::Point2f(0.0f, cornerlineindex.y),   
                cv::Point2f((float)hierarchicalClustering.size[1], cornerlineindex.y), 
                clustercolors[c]);

        
        intersections.push_back(cornerlineindex); 
        cv::namedWindow("Main", cv::WINDOW_NORMAL); 
        cv::imshow( "Main", hierarchicalClustering );
        cv::waitKey(0);
    }

    // ordering 

    std::cout << "Polylines Size " << intersections.size() << "\n"; 
    for (const auto& p : intersections){
        printPoint("Elem = " , p); 
    }

    cv::line(hierarchicalClustering, intersections[0], intersections[1], cv::Scalar(255,0,255,255), 1); 
    cv::line(hierarchicalClustering, intersections[1], intersections[3], cv::Scalar(255,0,255,255), 1); 
    cv::line(hierarchicalClustering, intersections[2], intersections[3], cv::Scalar(255,0,255,255), 1); 
    cv::line(hierarchicalClustering, intersections[0], intersections[2], cv::Scalar(255,0,255,255), 1); 
    
    //cv::polylines(hierarchicalClustering, intersections, true, cv::Scalar(200, 200, 200, 125), 5); 
    exportImage(name, "_clusteringpoints.png", hierarchicalClustering); 
    show_wait_destroy("Clustering Points",hierarchicalClustering ); 
    float width = intersections[1].x -  intersections[0].x; 
    float height = intersections[2].y -  intersections[0].y;

    cv::Point2f src[4] = {intersections[0], 
                        intersections[1], 
                        intersections[2], 
                        intersections[3]}; 

    cv::Point2f dst[4] = {cv::Point2f(0.0f, 0.0f), 
                          cv::Point2f(width, 0.0f), 
                            cv::Point2f(0.0f, height),  cv::Point2f(width, height)};

    cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(src, dst); 
    cv::Mat rescaleROI = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);   

    cv::warpPerspective(image, rescaleROI, perspectiveMatrix,  cv::Size((int)width,  (int)height)); 
    show_wait_destroy("Test Perspective", rescaleROI); 
    exportImage(name, "_image_detected_clustering.png", rescaleROI); 



    // ==============================================
    // Methode Entropie 
    // ==============================================
    
    // Entropie Detection 
    unsigned int dxy = 16; // pixels 
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

    float ratio = (float)squareSize / 72.0f; 
    





    

    return EXIT_SUCCESS; 
}
