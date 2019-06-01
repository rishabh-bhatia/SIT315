#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

using namespace std;

#define MAX 4
const int maxNum = 3;

int main()
{


	MPI_Init(NULL, NULL);

	//Calculating the number of chunks the matrice is getting divided into
	int np = 1;
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	//Calculating numbr of elements per chunk
	int rows = MAX / np;
	int elementsInSlave = rows * MAX;

	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Create arrays
	int arrA[MAX][MAX];
	int arrB[MAX][MAX];
	int arrC[MAX][MAX];

	arrA[MAX][MAX] = arrB[MAX][MAX] = arrC[MAX][MAX] = { 0 };

	clock_t timerStart;
	//double start ;

	// Master will distribute to it's slaves
	if (rank == 0) {

		// Initialise Array A and B with random values
		for (int i = 0; i < MAX; i++) {
			for (int j = 0; j < MAX; j++) {
				arrA[i][j] = rand() % maxNum + 1;
			}
		}

		for (int i = 0; i < MAX; i++) {
			for (int j = 0; j < MAX; j++) {
				arrB[i][j] = rand() % maxNum + 1;
			}
		}

		// Print A and B
		cout << "A: "<< endl;
			for (int i = 0; i < MAX; i++) {
				for (int j = 0; j < MAX; j++) {
					cout << arrA[i][j] << " ";
				}
				cout << endl;
			}
			cout << "B: "<< endl;
			for (int i = 0; i < MAX; i++) {
				for (int j = 0; j < MAX; j++) {
					cout << arrB[i][j] << " ";
				}
				cout << endl;
			}

		timerStart = clock(); //Starting the timer
			//auto start = chrono::steady_clock::now();
	}

	// Broadcast matrix B to all slave processes
	MPI_Bcast(&arrB, MAX * MAX, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	//Prefill the parts of A and the array which will produce the final result i.e. arrC or resultC
	int* partOfA = new int[rows * MAX];
	int* resultC = new int[rows * MAX]{ 0 };

	// Scatter the A matrix 
	MPI_Scatter(arrA, elementsInSlave, MPI_INT, partOfA, elementsInSlave, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	//Multiply the matrice in each process and save in resultC
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < MAX; j++) {
			for (int k = 0; k < MAX; k++) {
				resultC[i * MAX + j] = resultC[i * MAX + j] + (partOfA[i * MAX + k] * arrB[k][j]);
			}
		}
	}


	// MPI_GATHER the sol from each process
	MPI_Gather(resultC, elementsInSlave, MPI_INT, arrC, elementsInSlave, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	// Master processes the received parts (prints values)
	if (rank == 0) {
		//Stopping the timer
		clock_t end = clock();

		double time_taken = (double)(end) - (double)(timerStart) / CLOCKS_PER_SEC;

		//auto end = chrono::steady_clock::now();
		//auto time_taken = end - start;
		cout << "Time Taken :" << time_taken / 10 << " ms" << endl;


		cout << "C: " << endl;
		for (int i = 0; i < MAX; i++) {
			for (int j = 0; j < MAX; j++) {
				cout << arrC[i][j] << " ";
			}
			cout << endl;
		}
	}

	MPI_Finalize();

	return 0;
}