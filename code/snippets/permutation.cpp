#include <iostream> 
#include <stdio.h>
#include <vector> 
#include <numeric> 
#include <string>

/*===========================================================================*/

// TP5 Chiffrement Multimédia 

// use keycph 
void initSeed(int keycph)
{
    srand(keycph);
}


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

// process permutation on vector 
template <class T>
void permuteFY(std::vector<T>& vecToShuffle, int ncols, int nrows)
{
    int newpos; 
    for (unsigned int i = nrows*ncols -1; i > 0; i--)
    {   
        newpos = GNPA(i);
        std::cout << i << " <=> " << newpos << std::endl; 
        std::swap(vecToShuffle[i], vecToShuffle[newpos]); 
    }
}

// permutation with Fisher-Yates Algorithm
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


void permuteSequence(std::vector<unsigned int>& sequence)
{
    permuteFY(sequence, sequence.size()); 
}

template <class T>
void permuteData(const std::vector<T>& data,  std::vector<T>& permdata, const std::vector<unsigned int>& sequence)
{
    for (unsigned int k = 0; k < sequence.size(); k++)
    {
        permdata[k] = data[sequence[k]]; 
    }
}



template <class T>
void invPermuteData(const std::vector<T>& permdata,  std::vector<T>& data, const std::vector<unsigned int>& sequence)
{
    for (unsigned int k = 0; k < sequence.size(); k++)
    {
        data[sequence[k]] = permdata[k]; 
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

int main()
{
    int M = 5; 
    int N = 5; 

    // clé secrète 
    int K = 456489; 

    // Init Seed 
    initSeed(K);

    /*{
        std::vector<int> test_data(M*N); 
        std::iota(test_data.begin(), test_data.end(), 0);

        // ===================================================
    
        printData(test_data); 
        // generate une nouvelle position 
        permuteFY(test_data, M, N); 
        // look at the permuted content 
        printData(test_data); 

        // ===================================================

    }*/


    {
        std::vector<int> test_data(M*N); 
        std::iota(test_data.begin(), test_data.end(), 0);
        // permuted data 
        std::vector<int> permuted_data(test_data.size()); 
        // sequence for permutation
        std::vector<unsigned int> sequence(test_data.size());
        std::iota(sequence.begin(), sequence.end(), 0);


        // ===================================================
        // PERMUTATION 
        printData(sequence, "Raw Sequence"); 
        // generate a permutation sequence 
        permuteSequence(sequence);
        // look at the permuted sequence 
        printData(sequence, "Sequence for permutation Data"); 
        // permute data 
        permuteData(test_data, permuted_data, sequence); 
        printData(permuted_data, "Permuted Data");

        // ===================================================
        // INV PERMUTATION
        // retrieve original data 
        std::vector<int> original_data(M*N); 
        invPermuteData(permuted_data, original_data, sequence);
        printData(original_data, "Retrieve Original Data :");


    }

    
    

    return EXIT_SUCCESS; 
}