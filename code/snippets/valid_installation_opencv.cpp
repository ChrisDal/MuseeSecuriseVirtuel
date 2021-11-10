#include <stdio.h>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#define IMAGEMAT_TYPE uchar 

using namespace cv;

/*===========================================================================*/
// TP5 Chiffrement Multimédia 

// use keycph 
void initSeed(int keycph) { srand(keycph); }


// Output : 0 or 1, uniform
int getRandBinary()
{
    float n = (float)rand() / (double)RAND_MAX; 
    if (n > 0.5)
    {
        return 1; 
    }
    return 0; 
}


// Output : 0:255, uniform
int getRandOctet()
{
    float n = (float)(rand() / (double)(RAND_MAX)) *(255.0f - 0.0f);  
    return (int)n; 
}


// ===============================
// get a point to linear position to 2D position 
template <class T> 
void setPosition(cv::Point& ppt, const T& ki, const int ncols)
{
    ppt.y = static_cast<int>(float(ki) / float(ncols)); // row 
    ppt.x = ki % ncols;                                 // col

}



/*===========================================================================*/

// Algorithme de Fisher-Yates 
/*
pour i de n - 1 descendant_à 1 :
      j ← nombre aléatoire entier 0 ≤ j ≤ i
      échanger a[j] et a[i]
*/



// generateur de nombre pseudo aleatoire 
int GNPA(int maxN)
{
    return static_cast<int>(((float)rand() / (double)RAND_MAX) * maxN); 
}

// permutation with Fisher-Yates Algorithm on OpenCV Matrice
void permuteFY(cv::Mat& matrice, int nrows, int ncols)
{
    int newpos;
    cv::Point p = cv::Point(0,0); 
    int maType = matrice.type(); // voir le type => int 

    for (unsigned int i = nrows*ncols -1; i > 0; i--)
    {   
        newpos = GNPA(i);
        
        setPosition(p, i, ncols); 
        std::cout << p << std::endl; 

        // Swap position
        std::swap(matrice.at<IMAGEMAT_TYPE>(p.y, p.x), matrice.at<IMAGEMAT_TYPE>(p.y, p.x)); 
    }
}


// Call permutation on matrice
void permuteSequence(cv::Mat& matrice)
{
    permuteFY(matrice, matrice.size[0], matrice.size[1]); 
}



void permuteData(const cv::Mat& data,  cv::Mat& permdata, const std::vector<unsigned int>& sequence)
{
    
    cv::Point pseq   = cv::Point(0,0); 
    cv::Point pdata  = cv::Point(0,0);
    int rows = data.size[0]; 
    int cols = data.size[1];

    for (unsigned int k = 0; k < sequence.size(); k++)
    {
        setPosition(pdata, k, cols); 
        setPosition(pseq, sequence[k], cols); 

        std::cout << pdata << " <=> " << pseq << std::endl; 

        permdata.at<IMAGEMAT_TYPE>(pdata.y, pdata.x) = data.at<IMAGEMAT_TYPE>(pseq.y, pseq.x); 
    }
}

// From permuted data and sequence permutation 
// Rebuild the original dat = unpermuted data 
void invPermuteData(const cv::Mat& permdata,  cv::Mat& data, const std::vector<unsigned int>& sequence)
{

    cv::Point pseq   = cv::Point(0,0); 
    cv::Point pdata  = cv::Point(0,0);
    int rows = permdata.size[0]; 
    int cols = permdata.size[1];

    for (unsigned int k = 0; k < sequence.size(); k++)
    {
        setPosition(pdata, k, cols); 
        setPosition(pseq, sequence[k], cols); 

        std::cout << pseq << " <=> " << pdata << std::endl; 
        data.at<IMAGEMAT_TYPE>(pseq.y , pseq.x) = permdata.at<IMAGEMAT_TYPE>(pdata.y, pdata.x); 
    }
    
}




template <class T>
void printData(const std::vector<T>& vec, std::string prefix= "")
{
    
    std::cout << prefix << std::endl;
    for (int x = 0; x < vec.size(); x++)
    {
        std::cout << vec[x] << ", "; 
    }

    std::cout << std::endl; 
}

/*===========================================================================*/

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    /*
    IMREAD_COLOR loads the image in the BGR 8-bit format. This is the default that is used here.
    IMREAD_UNCHANGED loads the image as is (including the alpha channel if present)
    IMREAD_GRAYSCALE loads the image as an intensity one

    In the case of color images, the decoded images will have the channels stored in B G R order.
    */
    
    Mat image;
    image = imread( argv[1], IMREAD_GRAYSCALE );

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }


    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    waitKey(0);

    return EXIT_SUCCESS;
}