// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <numeric>
using namespace cv;
using namespace std;


static void help(const char* programName)
{
    cout <<
    "\nA program using pyramid scaling, Canny, contours and contour simplification\n"
    "to find squares in a list of images (pic1-6.png)\n"
    "Returns sequence of squares detected on the image.\n"
    "Call:\n"
    "./" << programName << " [file_name (optional)]\n"
    "Using OpenCV version " << CV_VERSION << "\n" << endl;
}
int thresh = 50, N = 11;
const char* wndname = "Square Detection Demo";
// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}
// returns sequence of squares detected on the image.
static void findSquares( const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();
    cv::Mat pyr, timg, gray0(image.size(), CV_8U), gray;
    // down-scale and upscale the image to filter out the noise
    cv::pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    cv::pyrUp(pyr, timg, image.size());
    std::vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);
    
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
                cv::dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }
        
            // find contours and store them all as a list
            cv::findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
            
            std::vector<cv::Point> approx;
            
            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                cv::approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.02, true);
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
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }
                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.07 )
                        squares.push_back(approx);
                }
            }
        }
    }

}


static double intersectionOverUnion(const std::vector<cv::Point>& s1, const std::vector<cv::Point>& s2)
{

}

int main(int argc, char** argv)
{
    const char* names[] = { "pic1.png", "pic2.png", "pic3.png",
        "pic4.png", "pic5.png", "pic6.png", 0 };
    help(argv[0]);

    if( argc > 1)
    {
     names[0] =  argv[1];
     names[1] =  0;
    }

    for( int i = 0; names[i] != 0; i++ )
    {
        string filename = samples::findFile(names[i]);

        cv::Mat image = imread(filename, cv::IMREAD_COLOR); 

        if( image.empty() )
        {
            cout << "Couldn't load " << filename << endl;
            continue;
        }

        std::vector<std::vector<cv::Point> > squares;
        findSquares(image, squares);

        // sort 
        std::vector<double> areas; 
        for (int k = 0; k < squares.size(); k++)
        {
            areas.push_back(cv::contourArea(squares[k])); 
        }

        // mediane 
        int idmed = areas.size() % 2 == 0 ? areas.size() / 2.0f : int((float)areas.size() / 2.0f) + 1; 
        double medianArea = areas.at(idmed); 
        
        std::cout << "Median Area = " << medianArea << " pix²." << std::endl; 

        std::vector<int> indexRemove; 
        for (int k = 0; k < squares.size(); k++)
        {
            if (std::fabs(areas[k] - medianArea) > 0.1 * medianArea)
            {
                indexRemove.insert(indexRemove.begin(), k); 
            }
        }
        for (const int& c : indexRemove)
        {
            squares.erase(squares.begin() + c); 
            areas.erase(areas.begin() + c); 
        }

        // mean area 
        double sumarea = 0.0; 
        for (const double& area : areas)
        {
            sumarea+=area; 
        }
        double meanArea = sumarea / (double)areas.size(); 
        float squareSize = (float)std::sqrt(meanArea); 
        std::cout << squareSize << std::endl; 
        cv::Rect2f OneSquare = cv::Rect2f(squares[0][0].x , squares[0][0].y, squareSize, squareSize); 
        cv::rectangle(image, OneSquare, cv::Scalar(255, 0, 0, 255), -1); 


        cv::polylines(image, squares, true, Scalar(0, 255, 0), 1, LINE_AA);
        cv::imwrite("test_windows.png", image); 
        cv::namedWindow(wndname, cv::WINDOW_NORMAL); 
        cv::imshow(wndname, image);

        // ESCAPE 
        int c = cv::waitKey();
        if( c == 27 ){
            break;
        }
            
    }
    return 0;
}