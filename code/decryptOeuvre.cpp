#include "toolFunctions.hpp"
#include "imageanalysis.h"
#include "BasePermutation.h"

// =======================================================



// ========================================================

int main( int argc, char** argv )
{
    
    const char* source_window = "Canny Edge";

    if ( argc != 5 )
    {
        printf("usage: decryptOeuvre.out <ImageToDecrypt> <SecretKey> <Directory/ImageDecrypted> <blockSize>\n");
        return -1;
    }


    // 0. Read Data 
    cv::Mat image = cv::imread( argv[1], cv::IMREAD_GRAYSCALE );
    const int secretKey = std::stoi(argv[2]); 
    const char* name  = argv[3]; 
    const int blocksize = std::stoi(argv[4]); // number of pixels by block 

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    
    

    // =================================================== 
    // downsampled 
    float pixelSet = 102.0f; 
    const float valuePattern = 72.0f; 
    float pixel_size = pixelSet /  valuePattern; 

    float width = (float)image.size[1]/pixel_size; 
    float height = (float)image.size[0]/pixel_size;

    width = std::powf(2, std::round(std::log2(width))); 
    height = std::powf(2, std::round(std::log2(height))); 
    

    cv::Point2f src[4] = {cv::Point2f(0.0f, 0.0f), 
                        cv::Point2f((float)image.size[1], 0.0f), 
                        cv::Point2f(0.0f, (float)image.size[0]), 
                        cv::Point2f((float)image.size[1], (float)image.size[0])}; 

    cv::Point2f dst[4] = {cv::Point2f(0.0f, 0.0f), 
                          cv::Point2f(width, 0.0f), 
                            cv::Point2f(0.0f, height),  cv::Point2f(width, height)};

    cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(src, dst); 
    cv::Mat rescaleROI = cv::Mat::zeros(cv::Size((int)(image.size[1]/pixel_size), 
                                                (int)(image.size[0]/pixel_size)), CV_8UC1);   

    cv::warpPerspective(image, rescaleROI, perspectiveMatrix,  cv::Size((int)width,  (int)height)); 
    std::cout << "W,H = " << width  << "," << height << std::endl; 
    show_wait_destroy("Rescaled", rescaleROI); 
    exportImage(name, "_imageRescaled.png", rescaleROI); 


    // -------------------------------------
    // Init data 
    // -------------------------------------
    const int nrows = (int)rescaleROI.size[0]; 
    const int ncols = (int)rescaleROI.size[1]; 
    const int NM = nrows * ncols;
    int sqrtblockSize = (int)std::sqrt(blocksize);
    const int dpi = 200; 
    unsigned int pixA4_width; 
    unsigned int pixA4_height; 

    // display grid 
    int test = (int) (nrows / (float)sqrtblockSize); 
    cv::Mat testMat = rescaleROI.clone(); 
    testMat.convertTo(testMat, CV_8UC3); 

    for (int k = 0; k < test ; k++)
    {
        cv::line(testMat, cv::Point(sqrtblockSize * k, 0), 
                        cv::Point(sqrtblockSize * k, testMat.size[0]), 
                        cv::Scalar(255,0,0,255), 1); 
        cv::line(testMat, cv::Point(0, sqrtblockSize * k), 
                        cv::Point(testMat.size[1], sqrtblockSize * k), 
                        cv::Scalar(255,0,0,255), 1); 
    }

    show_wait_destroy("Test MAT ", testMat); 
    exportImage(name, "_gridOverImage.png", testMat); 

    cv::Mat subsampled(nrows / sqrtblockSize,  ncols/sqrtblockSize, image.type());
    cv::Mat subrecpermutedImage = cv::Mat::zeros(subsampled.size[0], subsampled.size[1], subsampled.type());
    cv::Mat subreconstructedImage = cv::Mat::zeros(subsampled.size[0], subsampled.size[1], subsampled.type());
    cv::Mat reconstructedImg = cv::Mat::eye(nrows, ncols, image.type());

    
    // 1. Key to decrypt 
    std::cout << std::endl << "Secret Key is " << secretKey << std::endl; 
    // initSeed(secretKey); 


    if (dpi == 72) 
    {
        pixA4_width = 595; 
        pixA4_height = 842; 
    }
    else if (dpi == 200)
    {
        pixA4_width = 1654; 
        pixA4_height = 2339; 
    }
    else if (dpi == 300) // 300 dpi 
    {
        pixA4_width = 2480; 
        pixA4_height = 3508; 
    }
    else 
    {
        printf("Invalid DPI : 72, 200, 300 dpi supported only.\n");
        return -1;
    }

    // ===================================================
    // PERMUTATION 

    // sequence for permutation
    std::vector<unsigned int> sequence(subsampled.size[0]* subsampled.size[1]);
    std::iota(sequence.begin(), sequence.end(), 0);

    // generate a permutation sequence 
    permuteSequence(sequence);

    

    for (int ki = 0; ki < subsampled.size[0]; ki++)
    {
        for (int kj = 0; kj < subsampled.size[1]; kj++)
        {
            
            int ky = ki*sqrtblockSize; 
            int kx = kj*sqrtblockSize; 
            // mean pixel 
            float pixel = 0.0f; 
            int dk = (int)(sqrtblockSize / 2.0f)-1; 
            // Take Kernel at SideBlock / 2 + voisinage +1
            pixel += (float)rescaleROI.at<IMAGEMAT_TYPE>(ky+dk,kx+dk); 
            pixel += (float)rescaleROI.at<IMAGEMAT_TYPE>(ky+dk,kx+dk+1); 
            pixel += (float)rescaleROI.at<IMAGEMAT_TYPE>(ky+dk+1,kx+dk); 
            pixel += (float)rescaleROI.at<IMAGEMAT_TYPE>(ky+dk+1,kx+dk+1); 
            pixel /= 4.0f; 

            subrecpermutedImage.at<IMAGEMAT_TYPE>(ki, kj) =  (IMAGEMAT_TYPE)pixel;
        }
    }

    show_wait_destroy("Sub reconstructed permuted image", subrecpermutedImage);
    exportImage(name, "02-subrecpermutedImage.png",  subrecpermutedImage); 

    // permute data 
    invPermuteData(subrecpermutedImage, subreconstructedImage, sequence); 

    show_wait_destroy("Main", subreconstructedImage);
    exportImage(name, "03-subreconstructedImage.png",  subreconstructedImage); 


    cv::Rect rectBlock; 

    // resampled image 
    // oversampled 
    for (int ki = 0; ki < subreconstructedImage.size[0]; ki++)
    {
        for (int kj = 0; kj < subreconstructedImage.size[1]; kj++)
        {
            rectBlock = cv::Rect(kj*sqrtblockSize, ki*sqrtblockSize, sqrtblockSize, sqrtblockSize); 
            reconstructedImg(rectBlock).setTo(subreconstructedImage.at<IMAGEMAT_TYPE>(ki, kj)); 
            //reconstructedImg.at<IMAGEMAT_TYPE>(ki*sqrtblockSize + ti, kj*sqrtblockSize +tj) = subreconstructedImage.at<IMAGEMAT_TYPE>(ki, kj);
        }
    }

    show_wait_destroy("Main", reconstructedImg);
    exportImage(name, "04-reconstructedImg.png",  reconstructedImg); 


    return EXIT_SUCCESS; 
}