#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <time.h>
#include "omp.h"

using namespace std;

const int maxNum = 100;
const int arrSize = 1500;

//Writes an input Array to an external text file
void writeFile(long array[arrSize][arrSize])
{
    int i, j;
    ofstream ArrayFile;
    ArrayFile.open("Output.txt");
    ArrayFile << "Result: " << endl;
    
    for (i = 0; i < arrSize; ++i)
        for (j = 0; j < arrSize; ++j)
        {
            ArrayFile << " " << array[i][j];
            if (j == arrSize - 1)
                ArrayFile << endl;
        }

    ArrayFile.close();
}

//Adding random integers to the Array
void initArray(int array[arrSize][arrSize], int size)
{
    int i, j;
    //#pragma omp_set_num_threads(omp_get_num_procs());
    for (i = 0; i < size; ++i)
        for (j = 0; j < size; ++j)
        {
            array[i][j] = rand() % maxNum + 1;
        }
}

int main()
{
    int i, j, k;
    srand(time(NULL));
    cout << "Array Size:  " << arrSize << endl;
    static int arrA[arrSize][arrSize], arrB[arrSize][arrSize];
    static long int arrC[arrSize][arrSize];
    initArray(arrA, arrSize);
    initArray(arrB, arrSize);

    cout << "Starting Matrix Multiplication..." << endl;

    clock_t start = clock(); //Timer begins

   // omp_set_num_threads(2);

    //Matrix Multiplication using OpenMP parallel for loop
#pragma omp parallel for private(i, j, k) shared(arrA, arrB, arrC) num_threads(8)
    for (i = 0; i < arrSize; ++i)
    {
        for (j = 0; j < arrSize; ++j)
        {
            for (k = 0; k < arrSize; ++k)
            {
                arrC[i][j] += arrA[i][k] * arrB[k][j];
            }
        }
    }

    clock_t end = clock(); //Timer Stops
    cout << "Array Multiplication Completed!" << endl;

    double timeTaken = double(end - start) / CLOCKS_PER_SEC;

    cout << "Processing Time: " << timeTaken << " secs" << endl;

    cout << "Task Completed! Please check external file to find the output Array." << endl;

    writeFile(arrC); //Writes the resulting array to textfile

    return 0;
}