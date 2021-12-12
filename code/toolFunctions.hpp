#include <iostream>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

void show_wait_destroy(const char* winname, cv::Mat img) {
    cv::namedWindow(winname, cv::WINDOW_NORMAL);
    cv::imshow(winname, img);
    cv::moveWindow(winname, 500, 0);
    cv::waitKey(0);
    cv::destroyWindow(winname);
}


// helper function: OpenCV
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

template < typename T > 
void printPoint(const char* message, const T& point )
{
    std::cout << message << " "; 
    std::cout << "[" << point.x << "," << point.y << "]" << std::endl; 
}

// Export Image : 
void exportImage(const std::string& directory, const char* filename, const cv::Mat& matToWrite)
{
    std::string filepathname = directory + filename; 
    cv::imwrite(filepathname.c_str(), matToWrite); 
    std::cout << "[EXPORTATION] Image Exported : " << filepathname << std::endl;
}

void exportImage(const std::string& directory, const std::string& filename, const cv::Mat& matToWrite)
{
    std::string filepathname = directory + filename; 
    cv::imwrite(filepathname.c_str(), matToWrite); 
    std::cout << "[EXPORTATION] Image Exported : " << filepathname << std::endl;
}