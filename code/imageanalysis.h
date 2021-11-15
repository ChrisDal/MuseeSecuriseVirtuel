#include <math.h>
#include <float.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>


#define IMAGEMAT_TYPE uchar 

// ===================================================
// HISTOGRAMS 
/*const int histSize = 256;
float range[] = { 0, 256 }; //the upper boundary is exclusive
const float* histRange[] = { range };

bool uniform = true; 
bool accumulate = false;

cv::Mat grey_hist, grey_hist_perm, grey_hist_rec;

cv::calcHist( &image, 1, 0, cv::Mat(), grey_hist, 1, &histSize, histRange, uniform, accumulate );
cv::calcHist( &permutedImage, 1, 0, cv::Mat(), grey_hist_perm, 1, &histSize, histRange, uniform, accumulate );
cv::calcHist( &reconstructedImg, 1, 0, cv::Mat(), grey_hist_rec, 1, &histSize, histRange, uniform, accumulate );

// normalized histogram
int hist_w = 512;
int hist_h = 400;
int bin_w = cvRound( (double) hist_w/histSize );

cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );

cv::normalize(grey_hist, grey_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
cv::normalize(grey_hist_perm, grey_hist_perm, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
cv::normalize(grey_hist_rec, grey_hist_rec, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

for( int i = 1; i < histSize; i++ )
{

    cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(grey_hist.at<float>(i-1)) ),
            cv::Point( bin_w*(i), hist_h - cvRound(grey_hist.at<float>(i)) ),
            cv::Scalar( 255, 0, 0),
            2, 8, 0  );

    cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(grey_hist_perm.at<float>(i-1)) ),
            cv::Point( bin_w*(i), hist_h - cvRound(grey_hist_perm.at<float>(i)) ),
            cv::Scalar( 0, 255, 0),
            2, 8, 0  );

    cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(grey_hist_rec.at<float>(i-1)) ),
            cv::Point( bin_w*(i), hist_h - cvRound(grey_hist_rec.at<float>(i)) ),
            cv::Scalar( 0, 0, 255), 
            2, 8, 0  );
}*/



/*===========================================================================*/
// HISTOGRAM ON GRAYSCALE 

// Input : Mat Image , Histogram Color 
cv::Mat processBarHistogram(cv::Mat& image, cv::Scalar color = cv::Scalar( 255, 0, 0))
{
    // HISTOGRAMS 
    const int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };

    static bool uniform = true; 
    static bool accumulate = false;

    cv::Mat histo;
    // calculate histogram
    cv::calcHist( &image, 1, 0, cv::Mat(), histo, 1, &histSize, histRange, uniform, accumulate );

    // normalized histogram
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    cv::Mat histoImg( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
    cv::normalize(histo, histo, 0, histoImg.rows, cv::NORM_MINMAX, -1, cv::Mat() );

    for( int i = 0; i < histSize; i++ )
    {
        float y = hist_h - cvRound(histo.at<float>(i)) ; 

        cv::rectangle( histoImg, 
                cv::Point2f(bin_w*i - float(bin_w)/2.0f, hist_h),
                cv::Point2f(bin_w*i + float(bin_w)/2.0f, y),
                color,
                cv::FILLED, 8, 0  
                );
    }

    return histoImg;
}

/*===========================================================================*/
// PSNR 

// return an image of PSNR greyscale and a psnr 
double processPSNR(cv::Mat& exportMSE, const cv::Mat& img, const cv::Mat& processed)
{
    cv::Scalar minColor = cv::Scalar(0);   // black 
    cv::Scalar maxColor = cv::Scalar(255); // white
    cv::Mat doubleMSE = cv::Mat::zeros(cv::Size(exportMSE.cols * 2, exportMSE.rows * 2), CV_64F);

    // 
    double mse = 0.0;
    double result = 0.0;
    double pixmse = 0.0; 

    double maxmse = -DBL_MAX; 
    double minmse = DBL_MAX; 


    for(int i=0; i <img.rows; i++)
    {
        for(int j=0; j < img.cols; j++)
        {
            double a = (double)img.at<IMAGEMAT_TYPE>(i,j); 
            double b = (double)processed.at<IMAGEMAT_TYPE>(i,j); 

            pixmse = std::pow(a - b, 2); 

            mse += pixmse; 
            doubleMSE.at<double>(i, j) = pixmse; 

            if (pixmse > maxmse)
            {
                maxmse = pixmse;  
            }

            if (pixmse < minmse)
            {
                minmse = pixmse; 
            }
        }

        
    }

    mse /= (double)img.rows*img.cols; 
    result =  10.0*log10(255.0*255.0/mse); 

    std::cout << "Min MSE=" << minmse << std::endl; 
    std::cout << "Max MSE=" << maxmse << std::endl; 


    // normalized between 0.0 and 1.0
    for(int i=0; i <img.rows; i++)
    {
        for(int j=0; j < img.cols; j++)
        {
            exportMSE.at<IMAGEMAT_TYPE>(i, j) = (IMAGEMAT_TYPE)(255*((doubleMSE.at<double>(i, j) - minmse) / (maxmse - minmse))); 
        }   
    }

    return result; 

}


/*===========================================================================*/
// DISPLAY Pack of images 
// same size as A required for all images 
void concatenateFourMat(cv::Mat& concatdisplayImage, const cv::Mat& A, const cv::Mat& B , const cv::Mat& C,const cv::Mat& D)
{
   // display our images 
    concatdisplayImage = cv::Mat(cv::Size(A.cols * 2, A.rows * 2), A.type(), cv::Scalar::all(0)); 
    // Create ROI and Copy A to it
    cv::Mat matRoi = concatdisplayImage(cv::Rect(0,0,A.cols, A.rows));
    A.copyTo(matRoi);
    // Copy B to ROI
    matRoi = concatdisplayImage(cv::Rect(A.cols, 0, A.cols, A.rows));
    B.copyTo(matRoi);
    // Copy C to ROI 
    matRoi = concatdisplayImage(cv::Rect(0,A.rows,A.cols,A.rows));
    C.copyTo(matRoi);
    // Copy D to ROI 
    matRoi = concatdisplayImage(cv::Rect(A.cols,A.rows,A.cols,A.rows));
    D.copyTo(matRoi);

}





