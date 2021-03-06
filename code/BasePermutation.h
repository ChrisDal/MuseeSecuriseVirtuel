#include <vector> 
#include <numeric> 
#include <opencv2/opencv.hpp>

#define IMAGEMAT_TYPE uchar 

/*===========================================================================*/
// TP5 Chiffrement Multimédia 

// use keycph 
void initSeed(int keycph) { srand(keycph); }


// Output : 0 or 1, uniform
int getRandBinary()
{
    float n = (float)rand() / (float)RAND_MAX; 
    if (n > 0.5f)
    {
        return 1; 
    }
    return 0; 
}


// Output : 0:255, uniform
int getRandOctet()
{
    float n = (float)(rand() / (float)(RAND_MAX)) *(255.0f - 0.0f);  
    return (int)n; 
}



/*===========================================================================*/
// get a point to linear position to 2D position 
template <class T> 
void setPosition(cv::Point& ppt, const T& ki, const int ncols)
{
    ppt.y = static_cast<int>(float(ki) / float(ncols)); // row 
    ppt.x = ki % ncols;                                 // col

}


// ===============================
// generateur de nombre pseudo aleatoire 
int GNPA(int maxN)
{
    return static_cast<int>(((float)rand() / (float)RAND_MAX) * maxN); 
}


/*===========================================================================*/
// permutation with Fisher-Yates Algorithm
// Algorithme de Fisher-Yates 
/*
pour i de n - 1 descendant_à 1 :
    j ← nombre aléatoire entier 0 ≤ j ≤ i
    échanger a[j] et a[i]
*/

template <class T>
void permuteFY(std::vector<T>& vecToShuffle, unsigned int maxN)
{
    int newpos; 
    for (unsigned int i = maxN-1; i > 0; i--)
    {   
        newpos = GNPA(i);
        std::swap(vecToShuffle[i], vecToShuffle[newpos]); 
    }
}


// Permute the sequence of indices
void permuteSequence(std::vector<unsigned int>& sequence)
{
    permuteFY(sequence, (unsigned int)sequence.size()); 
}


/*===========================================================================*/

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

        //std::cout << pdata << " <=> " << pseq << std::endl; 

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

        //std::cout << pseq << " <=> " << pdata << std::endl; 
        data.at<IMAGEMAT_TYPE>(pseq.y , pseq.x) = permdata.at<IMAGEMAT_TYPE>(pdata.y, pdata.x); 
    }
    
}

/*===========================================================================*/
