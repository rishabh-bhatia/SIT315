#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <time.h>
#include <thread> //Initialising thread library

using namespace std;

const int maxNum = 100;
const int arrSize = 1000;
const int maxThreads = 8;
static int arrA[arrSize][arrSize], arrB[arrSize][arrSize];
static long int arrC[arrSize][arrSize];
thread T[arrSize];

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
    for (i = 0; i < size; ++i)
        for (j = 0; j < size; ++j)
        {
            array[i][j] = rand() % maxNum + 1;
        }
}

// Multiplying array a and b and storing in array c by dividing multiplication steps propotionally into a different threads.
void multiplication(int threadNo)
{
    int i, j , k;

    for (i = threadNo * arrSize / maxThreads; i < (threadNo + 1) * arrSize / maxThreads; i++)
    {
        for (j = 0; j < arrSize; j++)
        {
            for (k = 0; k < arrSize; k++)
            {
                arrC[i][j] += arrA[i][k] * arrB[k][j];
            }
        }
    }
}

int main()
{
    int i, j;
    srand(time(NULL));
    cout << "Array Size:  " << arrSize << endl;
    static int arrayA[arrSize][arrSize], arrayB[arrSize][arrSize];
    static long int arrayC[arrSize][arrSize];
    initArray(arrayA, arrSize);
    initArray(arrayB, arrSize);

    cout << "Starting Matrix Multiplication..." << endl;

    clock_t a = clock();//Timer begins

    for (i = 0; i < maxThreads; i++)
    {
        T[i] = thread(multiplication, i);
    }

    for (j = 0; j < maxThreads; j++)
    {
        T[j].join();
    }


    clock_t b = clock();//Stopping the timer
    cout << "Array Multiplication Completed!" << endl;

    double timeTaken = double(b - a) / CLOCKS_PER_SEC;

    cout << "Processing Time: " << timeTaken << " secs" << endl;

    cout << "Task Completed! Please check external file to find the output Array." << endl;

    writeFile(arrC); //Writes the resulting array to textfile

    return 0;
}