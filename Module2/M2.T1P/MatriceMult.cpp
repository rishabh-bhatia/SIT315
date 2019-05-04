#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <time.h>

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
	static int arrayA[arrSize][arrSize], arrayB[arrSize][arrSize];
	static long int arrayC[arrSize][arrSize];
	initArray(arrayA, arrSize);
	initArray(arrayB, arrSize);

	cout << "Starting Matrix Multiplication..." << endl;

	clock_t a = clock();//Timer begins

	//Matrix Multiplication
	for (i = 0; i < arrSize; ++i)
	{
		for (j = 0; j < arrSize; ++j)
		{
			for (k = 0; k < arrSize; ++k)
			{
				arrayC[i][j] += arrayA[i][k] * arrayB[k][j];
			}
		}
	}

	clock_t b = clock();//Timer Stops
	cout << "Array Multiplication Completed!" << endl;

	double timeTaken = double(b - a)/ CLOCKS_PER_SEC;//Time taken from time a to b

	cout << "Processing Time: " << timeTaken << " secs" << endl;

	cout << "Task Completed! Please check external file to find the output Array." << endl;

	writeFile(arrayC);//Writes the resulting array to textfile

	return 0;
}